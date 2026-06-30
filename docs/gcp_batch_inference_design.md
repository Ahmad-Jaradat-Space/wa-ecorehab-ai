# GCP Batch Inference Design

**Status:** Design only — not built. This document specifies how the WA EcoRehab AI
pipeline that runs today *locally and offline* would be lifted onto Google Cloud
Platform (GCP) for landscape-scale, scheduled, tiled inference over real Digital
Earth Australia (DEA) Sentinel-2 data. It maps every cloud component back to a
concrete piece of the existing repository so the design stays honest about what
already works versus what is proposed.

> **Honesty note.** The repository today ships a *synthetic-but-georeferenced*
> demo AOI: pixel reflectances are fabricated, but the CRS, affine transforms,
> windowed reads, polygon area maths, and the whole code architecture are real
> and operate on standard Cloud Optimized GeoTIFFs (COGs) and GeoPackages. This
> document describes how to run the *same code paths* against real DEA/WA data at
> scale. It does **not** claim the cloud system has been deployed, nor that the
> model achieves field-validated ecological accuracy. The system produces a
> native-vegetation / disturbance / water **proxy** and a rehabilitation
> **proxy**, not a species map or a mine-rehabilitation compliance verdict.

---

## 1. What we are scaling, and what we already have locally

The local entry point is `python -m ecorehab.inference.batch_predict --config
configs/inference.yaml`. Its real behaviour (see
[`src/ecorehab/inference/batch_predict.py`](../src/ecorehab/inference/batch_predict.py)
and [`tile_inference.py`](../src/ecorehab/inference/tile_inference.py)):

1. Load a segmentation checkpoint (`unet.pt`) and its metadata (band order,
   feature indices, per-feature `mean`/`std`, class ids/names, `ignore_index`).
2. Open one composite COG and run **windowed, overlapping inference**:
   `predict_raster_windowed` reads `tile_size=256` windows with `overlap=32`
   via `rasterio.windows.Window`, builds the **13-feature** stack (7 Sentinel-2
   bands + 6 indices: NDVI, EVI, NDWI, MNDWI, NBR, BSI), standardises with the
   training mean/std, runs the U-Net, softmaxes, and **feather-blends** tiles
   into a seamless probability mosaic via `ProbabilityAccumulator` +
   `tile_weights` (`stitch.py`).
3. Mark nodata/all-NaN pixels as `ignore_index=255`.
4. Write outputs through `utils.io.write_raster` (CRS/transform/nodata preserved,
   tiled + compressed):
   - `predictions/prediction_mask.tif`
   - `predictions/native_vegetation_probability.tif`
   - `predictions/uncertainty.tif` (predictive entropy)
   - `vector_outputs/predicted_polygons.gpkg` (polygonised, min 0.5 ha)
   - `vector_outputs/area_summary.csv` (per-class hectares with 95% block-bootstrap CI)
   - `vector_outputs/area_summary_by_polygon.csv` (per mining tenement)

The whole thing already runs inside the repo `Dockerfile` (`python:3.11-slim`,
`pip install ".[dl,boost,rs]"`, entrypoint `python -m ecorehab.cli`).

**The local design constraint we are removing:** `predict_raster_windowed`
materialises one full-AOI `ProbabilityAccumulator` of shape `(C, H, W)` in RAM.
For a 768×768 demo AOI that is trivial. For a multi-thousand-km² WA AOI it is
not. The cloud design's central job is to **fan that single in-memory mosaic out
across many workers, each owning a spatial block**, then merge — without changing
the per-tile maths.

---

## 2. Architecture overview (ASCII)

```
                         ┌───────────────────────────────────────────────┐
                         │  Cloud Scheduler  (cron: monthly composite)    │
                         └───────────────────────┬───────────────────────┘
                                                 │ triggers
                                                 ▼
                         ┌───────────────────────────────────────────────┐
                         │  Cloud Workflows  (orchestration / DAG)        │
                         │  ingest → composite → fan-out → merge → report │
                         └───┬───────────┬──────────────┬────────────┬────┘
                             │           │              │            │
              ┌──────────────▼──┐  ┌─────▼───────┐ ┌────▼────────┐ ┌─▼─────────────┐
   DEA STAC   │ Ingest job      │  │ Composite   │ │ Tiled infer │ │ Report job    │
   (S3/STAC)─▶│ (Cloud Run job) │  │ (Vertex /   │ │ FAN-OUT     │ │ (Cloud Run    │
              │ stac.py         │  │  Cloud Run) │ │ per tile    │ │  job)         │
              └────────┬────────┘  └─────┬───────┘ └──────┬──────┘ └──────┬────────┘
                       │ raw COGs        │ composite COG  │ tile probs    │ html/csv
                       ▼                 ▼                ▼               ▼
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                         GCS  gs://wa-ecorehab-<env>/                       │
        │  raw/   composites/   tiles/   predictions/   vector_outputs/   reports/   │
        └──────────────────────────────────────────────────────────────────────────┘
                       ▲                       ▲                         │
                       │ pull image            │ load model             │ metrics
              ┌────────┴────────┐    ┌─────────┴──────────┐   ┌─────────▼──────────┐
              │ Artifact        │    │ Vertex AI Model    │   │ Monitoring +       │
              │ Registry        │    │ Registry (versions)│   │ Drift report       │
              │ (repo Docker)   │    │ unet@v3, rf@v2     │   │ (Cloud Monitoring) │
              └─────────────────┘    └────────────────────┘   └────────────────────┘
                       ▲
              ┌────────┴────────┐
              │ Vertex AI       │
              │ Custom Training │  (train_segmentation.py on GPU; writes unet.pt → Model Registry)
              └─────────────────┘
```

The **tiled-inference fan-out** is the only genuinely new distributed-systems
piece. Everything else is a managed wrapper around code that already exists.

---

## 3. GCS bucket layout

One regional bucket per environment (`-dev`, `-stg`, `-prod`), in
`australia-southeast1` (Sydney) or `australia-southeast2` (Melbourne) to keep
data inside Australia and close to DEA's AWS `ap-southeast-2` egress. Prefixes
mirror the local `data/processed/<aoi>/` and `outputs/` layout so the code's
`artifacts.py` path helpers translate one-to-one to object keys.

```
gs://wa-ecorehab-prod/
  raw/                                  # immutable source pulls (lifecycle → coldline 30d)
    s2_ard/<aoi>/<date>/B02.tif ...     # DEA Sentinel-2 ARD scenes (ga_s2am_ard_3)
    vectors/wa_native_veg/<snapshot>.gpkg
    vectors/wa_tenements/<snapshot>.gpkg
  composites/                           # cloud-free seasonal median composites (COG)
    <aoi>/<period>/composite.tif        # ← maps to artifacts.composite_path(cfg)
    <aoi>/<period>/data_manifest.json   # provenance (artifacts.write_manifest)
  models/                               # checkpoints mirrored from Vertex Model Registry
    unet/v3/unet.pt                     # state_dict + bands/mean/std/class ids (see ckpt schema)
    random_forest/v2/model.joblib
  tiles/                               # transient per-run tile probability shards (fan-out)
    <aoi>/<run_id>/tile_00042.npz       # (C, h, w) probs + window offsets; TTL 7d
  predictions/                          # merged rasters (COG) — the deliverables
    <aoi>/<run_id>/prediction_mask.tif
    <aoi>/<run_id>/native_vegetation_probability.tif
    <aoi>/<run_id>/uncertainty.tif
  vector_outputs/
    <aoi>/<run_id>/predicted_polygons.gpkg
    <aoi>/<run_id>/area_summary.csv
    <aoi>/<run_id>/area_summary_by_polygon.csv
  reports/
    <aoi>/<run_id>/site_report.html     # build_site_report.py output
    <aoi>/<run_id>/model_card.md
  drift/
    <aoi>/<run_id>/feature_stats.json   # per-band/index distribution snapshot for drift checks
```

**Layout principles**

- `raw/` is write-once and immutable; everything downstream is reproducible from
  it plus a config + model version. Lifecycle rule demotes `raw/` to Coldline
  after 30 days and `tiles/` (transient shards) is deleted after 7 days.
- `<run_id>` (e.g. `20260630T1200Z`) namespaces every run so a re-run never
  clobbers a delivered product and so the monitoring job can diff consecutive
  runs.
- COGs everywhere downstream of `raw/` so `rasterio` windowed reads work directly
  off GCS via the `/vsigs/` or `gs://` virtual filesystem — no full download.
- Object versioning ON for `models/`, `predictions/`, `vector_outputs/`,
  `reports/` (audit trail for anything that could inform a decision).

---

## 4. Container image (Artifact Registry)

Reuse the repository `Dockerfile` unchanged for the application logic; build two
tags from it:

| Tag | Base | Purpose | Extras |
|---|---|---|---|
| `wa-ecorehab:cpu-<sha>` | `python:3.11-slim` (current Dockerfile) | ingest, composite, **tiled inference**, polygonise, reporting | `.[dl,boost,rs]` |
| `wa-ecorehab:gpu-<sha>` | CUDA `devel` base (variant noted in Dockerfile comments) | Vertex custom **training** only | `.[dl,boost,rs]` + CUDA torch |

```
australia-southeast1-docker.pkg.dev/<project>/wa-ecorehab/wa-ecorehab:cpu-<gitsha>
```

- Build with Cloud Build on push to `main`; tag with the git SHA (immutable) and
  also `:latest-cpu` / `:latest-gpu` (moving). Vertex/Cloud Run pin the **SHA**,
  never `latest`, so a run is fully reproducible.
- The image already exposes every stage via `ENTRYPOINT ["python", "-m",
  "ecorehab.cli"]`, so a Cloud Run job is just `args: ["inference",
  "--config", "configs/inference.yaml"]` (or `data`, `composite`, `report`).
- The only image change needed for cloud: ensure `GDAL`/`rasterio` is built with
  the GCS virtual filesystem (the manylinux `rasterio` wheel already includes
  `/vsigs/`; set `GS_NO_SIGN_REQUEST` off and rely on the job's service-account
  ADC). No code change — `rasterio.open("gs://...")` works in-process.

CPU inference is the default and the honest baseline: the U-Net is only **7.77M
parameters**, and inference is embarrassingly parallel over tiles, so horizontal
CPU fan-out is cheaper and simpler than a few big GPUs (see §11).

---

## 5. Model training: Vertex AI Custom Training

Training stays a deliberate, human-triggered job — not part of the scheduled
inference loop. It wraps the existing
`python -m ecorehab.training.train_segmentation --config configs/unet.yaml` CLI.

- **Job type:** Vertex AI `CustomJob` using the `wa-ecorehab:gpu-<sha>` image, one
  `g2-standard` / `a2` machine with a single L4 or A100 (the 7.77M-param U-Net
  with CE+Dice and inverse-frequency weights fits comfortably; early stopping hit
  epoch 19 in the demo).
- **Inputs:** tiles under `tiles/` (or generated in-job from a composite +
  vectors), config from the repo image, seed pinned (`DEFAULT_SEED=42`).
- **Output / hand-off:** the checkpoint `unet.pt` is the contract between training
  and inference. `load_segmentation_checkpoint` requires `arch`, `encoder`,
  `base_channels`, `in_channels`, `num_classes`, `mean`, `std`, `bands`,
  `feature_indices`, `include_bands`, `class_ids`, `class_names`, `ignore_index`.
  The job uploads `unet.pt` to `models/unet/v<n>/` **and** registers it in the
  Vertex Model Registry (§7).
- **Experiment tracking:** Vertex AI Experiments (or the MLflow already noted in
  the repo) logs the same spatial-hold-out metrics the local run reports so cloud
  and local numbers are comparable: U-Net test macro-F1 **0.926**, mean IoU
  **0.866**, balanced accuracy **0.954**; RF baseline macro-F1 **0.897**, mean
  IoU **0.814**. **Spatial-block split is mandatory** (18 train / 7 val / 8 test
  blocks) — a random split leaks and is not used to register a model.
- **Calibration guardrail:** the U-Net uses **GroupNorm, not BatchNorm**, on
  purpose (small/few batches make BN running stats unreliable and flatten the
  softmax). Any cloud retrain must preserve this; the model card records that
  probabilities are peaked (max-prob median ~0.83, entropy median ~0.52), which
  the uncertainty raster and area CIs depend on.

---

## 6. Tiled inference at scale: the fan-out

This section is the heart of the design. The goal: run exactly the per-tile
computation in `predict_raster_windowed` (window read → `build_feature_stack` →
`standardize` → U-Net → softmax → feather weights) across many workers, each
owning a subset of tiles, and merge their feathered contributions into the same
mosaic `ProbabilityAccumulator` produces today.

### 6.1 Tile plan (driver)

A small driver step (Cloud Run job, or the first Workflows step) opens the
composite COG header only, reads `height/width/transform/crs/nodata`, and calls
the existing `generate_tiles((h, w), tile_size, stride=tile_size-overlap)` from
`labels/tiling.py` — the *same* function the local path uses. It writes a tile
manifest (`tiles/<aoi>/<run_id>/plan.json`): one record per tile with
`row_off, col_off, width, height`. No pixels are read here.

### 6.2 Fan-out (two interchangeable options)

**Option A — Cloud Run jobs with task indices (recommended for v1).**
A single Cloud Run *job* launched with `--tasks N`; `CLOUD_RUN_TASK_INDEX` /
`CLOUD_RUN_TASK_COUNT` deterministically shard the tile manifest. Each task:

1. `rasterio.open("gs://.../composite.tif")` and, for each of its tiles, reads
   **only that window** off GCS (`/vsigs/` range reads — this is the windowed
   read, now remote instead of local-disk).
2. Runs the unchanged `_window_features` + model + softmax + `tile_weights`.
3. Writes its weighted `(C, h, w)` tile probability + offsets + per-tile weight
   sum to `tiles/<aoi>/<run_id>/tile_<idx>.npz`.

No worker holds the full `(C, H, W)` mosaic — that is the key change from local.

**Option B — Dataflow (Apache Beam) for very large / many-AOI batches.**
Same per-tile DoFn, but Beam handles dynamic work rebalancing, autoscaling, and
fusion. Preferred when one AOI is hundreds of tiles or many AOIs run together.
The DoFn body is byte-for-byte the local per-tile code.

Either way the per-tile maths is **identical to the local implementation**; only
the *scheduler* and the *destination* of each tile's result differ.

### 6.3 Merge / stitch (reducer)

A reducer step streams the tile shards and reconstructs the mosaic exactly as
`ProbabilityAccumulator.add` / `.finalize` does: accumulate `prob * weights` into
`prob_sum` and `weights` into `weight_sum`, then divide. Two ways to keep this
bounded in memory:

- **Block reducer:** partition the AOI into large super-blocks; each reducer task
  owns one super-block and only merges the shards overlapping it (overlap halo of
  `overlap=32` px included so the feather blend at block seams is correct), then
  writes its super-block as a COG tile. The driver finalises a single COG via
  `gdal_merge`/`rio cogeo`. This caps reducer RAM regardless of AOI size.
- **Single reducer** (small AOIs): one task replays all shards into one
  `ProbabilityAccumulator` — literally the local code path.

The reducer then runs the rest of `batch_predict` **unchanged**:
`argmax → prediction_mask.tif`, native-class probability band, `predictive_entropy`
→ `uncertainty.tif`, `polygonize_prediction` (min 0.5 ha) →
`predicted_polygons.gpkg`, `bootstrap_area_ci` (n_boot=300, seed from config) →
`area_summary.csv`, and `area_by_polygon` over tenements →
`area_summary_by_polygon.csv`. All writes go through `write_raster`/`write_vector`
to `predictions/` and `vector_outputs/`, as COG and GeoPackage on GCS.

### 6.4 Why not Vertex AI Batch Prediction?

Vertex AI Batch Prediction is built for **per-record/per-instance** scoring
(rows, images-as-instances) with a managed request/response contract. Our unit of
work is a *georeferenced window of a COG with a feather-weighted spatial merge* —
the I/O is windowed range-reads from GCS and the output is a stitched raster +
polygons, not independent predictions. Forcing that through Batch Prediction
would mean re-encoding tiles as instances and re-implementing the stitch
downstream anyway. So:

- **Recommended:** Cloud Run jobs (Option A) for the fan-out — simplest, cheapest,
  reuses the container and CLI directly.
- **Scale-up:** Dataflow (Option B) when throughput/autoscaling matters.
- **GKE batch Jobs:** a valid third option if the org already standardises on GKE
  (Indexed `Job` with `completions=N`, `parallelism=K`, node pool sized to the
  image); functionally equivalent to Cloud Run task indices but more ops.
- **Vertex Batch Prediction:** intentionally **not** used for the raster fan-out;
  it remains a reasonable choice only for the *classical* pixel model
  (`model_kind: classical`), where scoring really is per-pixel-feature-row.

---

## 7. Model Registry and versioning

- **Vertex AI Model Registry** holds each trained model as `unet@v<n>` /
  `random_forest@v<n>`, with the checkpoint in `models/<name>/v<n>/` as the
  artifact URI and the spatial-hold-out metrics + model card attached as
  metadata.
- **Promotion gate:** a candidate is promoted to the `production` alias only if it
  (a) beats the incumbent on spatial-block macro-F1 / mean IoU, and (b) does not
  regress the honest failure metric — **native precision must stay reported and
  be ≥ the incumbent's 0.78** (the over-prediction problem must not get worse).
- **Inference pins a version:** `configs/inference.yaml` already names
  `checkpoint: outputs/metrics/unet.pt`; in cloud this resolves to a registry
  alias (`unet@production`) → a concrete `models/unet/v3/unet.pt`. The `run_id`
  records which model version produced each delivered product.
- **Rollback** is an alias move, not a redeploy, because inference loads the
  checkpoint at job start.

---

## 8. Scheduled inference (Cloud Scheduler → Workflows)

```
Cloud Scheduler (cron, e.g. monthly)
   → Cloud Workflows DAG:
       1. ingest      (Cloud Run job: stac.py → raw/ )        [skip if cached]
       2. composite   (Cloud Run/Vertex: build_composites → composites/<aoi>/<period>/)
       3. plan        (driver: generate_tiles → tiles/.../plan.json)
       4. infer       (Cloud Run job, --tasks N  OR  Dataflow) → tiles/.../tile_*.npz
       5. merge       (reducer → predictions/ + vector_outputs/)
       6. report      (Cloud Run job: build_site_report → reports/)
       7. drift       (Cloud Run job: §9)  → drift/ + Cloud Monitoring metrics
       8. notify      (Pub/Sub / email: link to site_report.html + CI table)
```

- **Cloud Scheduler** is the only cron; **Workflows** owns the DAG, retries (per
  step), and step-level timeouts. Each step is the repo container with a different
  `ecorehab <stage>` arg, so the DAG is a thin YAML over existing CLIs.
- **Idempotency:** steps key off `composites/<aoi>/<period>/composite.tif` and the
  `<run_id>`; re-running a failed DAG resumes without recomputing cached steps.
- **Manual / event triggers:** the same Workflow can be invoked ad hoc (new AOI,
  re-run with a new model version) — Scheduler is just one caller.

---

## 9. Monitoring and data / concept drift

Two layers: **operational** (is the pipeline healthy?) and **scientific** (is the
model still meaningful?).

**Operational (Cloud Monitoring / Logging):**
- Per-step success/failure, duration, retries, cost from Workflows + Cloud Run
  metrics; alert on step failure or a 2× runtime regression.
- Data-quality gates inside the run: fraction of `ignore_index=255` pixels per
  composite (the demo runs ~15% ignore / ~0.4% nodata); alert if a composite is
  mostly nodata/cloud (mirrors the local `min_valid_fraction 0.5` tile drop).

**Data drift (input distribution):**
- The drift step snapshots per-feature statistics over valid pixels for all **13
  features** (7 bands + NDVI/EVI/NDWI/MNDWI/NBR/BSI): mean, std, and histogram, to
  `drift/<aoi>/<run_id>/feature_stats.json`.
- Compare against the **training** feature distribution stored with the model
  (the checkpoint already carries `mean`/`std`). Flag drift with population
  stability index (PSI) / KS distance per feature. Seasonal phenology and sensor
  changes (e.g. an S2 processing-baseline shift) are the expected real-world
  drivers; a large PSI on NDVI/NIR is the trigger to consider a retrain.

**Concept drift / prediction monitoring (no labels at inference time):**
- Track the distribution of **predicted areas with their CIs** run-over-run:
  disturbed, native, water hectares. A step change outside the bootstrap CI band
  that is not explained by a real land-cover event is a concept-drift signal.
- Track **uncertainty**: median predictive entropy and the share of low-confidence
  pixels. Rising entropy ⇒ the model is extrapolating off its training
  distribution.
- **Surface the known failure mode every run.** The headline honest finding from
  evaluation is that native vegetation is **over-predicted: 1910 ha [1723, 2096]
  vs 1344 ha truth (+42%), and the CI does not cover truth**, because
  managed-green agriculture is spectrally confused with native vegetation
  (**native precision 0.78**). The drift report explicitly flags AOIs with high
  agricultural fraction as elevated false-positive risk for the native class and
  routes their low-confidence polygons for human (ecologist) review rather than
  treating the area number as ground truth. Truth-vs-prediction comparison is only
  possible in demo mode (synthetic `truth_labels.tif`); in real mode this becomes
  a *consistency* check against WA Native Vegetation Extent polygons and DEA Land
  Cover, **not** an accuracy claim.

---

## 10. Security, IAM, and data governance

- One **service account per stage** (ingest, train, infer, report) with least
  privilege: ingest writes only `raw/`; infer reads `composites/`+`models/`,
  writes `tiles/`+`predictions/`; report reads predictions, writes `reports/`.
- Workload Identity (GKE) or job-attached SAs (Cloud Run/Vertex); no keys in the
  image. `rasterio`/GCS use Application Default Credentials.
- Data residency: keep buckets and compute in Australian regions. DEA Sentinel-2
  ARD and WA government vectors are open (CC BY 4.0) — record source URLs and
  snapshot dates in `data_manifest.json` (the local `artifacts.write_manifest`
  already does this) for provenance and licence compliance.

---

## 11. Cost notes

Indicative only (australia-southeast region, list prices change — treat as
order-of-magnitude, not a quote). The design is biased toward **cheap horizontal
CPU** because the model is tiny and the work is embarrassingly parallel.

| Component | Driver of cost | Lever to control it |
|---|---|---|
| **Inference compute** | CPU-seconds × tiles; U-Net is 7.77M params → CPU is fine | Cloud Run jobs scale to zero between runs; right-size `--tasks`; prefer many small CPU tasks over a GPU |
| **Training** | GPU-hours (occasional, human-triggered) | Single L4/A100 spot/Vertex, early stopping (demo stopped at epoch 19); train rarely |
| **GCS storage** | `raw/` Sentinel-2 ARD is the bulk | Lifecycle: `raw/`→Coldline 30d; delete `tiles/` shards after 7d; COG compression on all derived rasters |
| **GCS egress / reads** | windowed `/vsigs/` range reads per tile | Keep compute in the same region as the bucket (zero egress); COG internal tiling minimises bytes read per window |
| **Dataflow (Option B)** | worker-hours + shuffle | Only use for large/many-AOI batches; Cloud Run jobs are cheaper for v1 |
| **Orchestration** | Workflows steps + Scheduler | Negligible; per-step billing |
| **Monitoring/logging** | log volume | Sample logs; keep metrics, drop debug |

**Rule of thumb:** for a single Swan-Coastal-Plain-sized real AOI run monthly,
the dominant recurring cost is `raw/` storage, not compute. Inference of a
7.77M-param model over a few hundred CPU-bound tiles is cents-to-low-dollars per
run; the engineering choice that saves the most money is *not* reaching for GPUs
or Vertex Batch Prediction when fan-out CPU jobs suffice.

---

## 12. What changes from the local windowed inference — explicit mapping

| Concern | Local (today, real code) | Cloud (this design) | Code change? |
|---|---|---|---|
| Composite source | local COG via `rasterio.open(path)` | `rasterio.open("gs://.../composite.tif")` (`/vsigs/`) | **None** (path/URI) |
| Tile plan | `generate_tiles((h,w),tile_size,stride)` in-process | same function in a driver step → `plan.json` | **None** |
| Window read | `src.read(window=Window(...))` from disk | same call, GCS range read | **None** |
| Feature stack | `_window_features` → `build_feature_stack` + `standardize` | identical, per worker | **None** |
| Model | `load_segmentation_checkpoint("unet.pt")` | checkpoint from `models/<v>/` via Model Registry alias | **None** (path) |
| Tile compute | softmax + `tile_weights` per tile | same, per worker task | **None** |
| Mosaic accumulation | one in-RAM `ProbabilityAccumulator(C,H,W)` | sharded: workers write `tile_*.npz`; reducer replays `.add`/`.finalize` per super-block | **New glue** (shard write + block reducer); maths unchanged |
| Stitch / blend | `acc.finalize()` once | block reducer + final COG merge | **New glue**, same algorithm |
| Outputs | `write_raster`/`write_vector` to `outputs/` | same functions → GCS `predictions/`+`vector_outputs/` | **None** (path) |
| Area + CI | `bootstrap_area_ci`, `area_by_polygon` | identical, in reducer | **None** |
| Trigger | `python -m ecorehab.inference.batch_predict --config ...` | Cloud Scheduler → Workflows → Cloud Run job (same CLI) | **None** (orchestration) |
| Memory ceiling | full `(C,H,W)` must fit in RAM | bounded per worker (one tile) and per reducer (one super-block) | **The reason the new glue exists** |
| Scale unit | one AOI, one process | tiles fanned across N tasks / Dataflow / GKE | Orchestration only |

**Summary:** the per-tile science (read window → 13 features → standardise →
U-Net → softmax → feather) and every output writer are reused *as-is*. The only
net-new engineering is (1) splitting the single in-memory probability mosaic into
GCS tile shards and (2) a memory-bounded block reducer that replays the existing
`ProbabilityAccumulator` logic — plus the managed wrappers (Artifact Registry,
Vertex training/registry, Cloud Run/Dataflow fan-out, Scheduler/Workflows,
Monitoring). Nothing in the design requires changing the model, the metrics, or
the honesty guardrails: the system remains a **native-vegetation / disturbance /
water proxy with calibrated uncertainty and CI'd hectares**, and it keeps
surfacing — not hiding — the agriculture-confusion over-prediction of native
vegetation.

---

## 13. Open data sources (real mode)

| Source | Use | URL |
|---|---|---|
| DEA Sentinel-2 ARD (`ga_s2am_ard_3`) | imagery input via STAC | https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/ |
| DEA Land Cover | weak labels / masks / context | https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover |
| DEA Fractional Cover | recovery (PV/NPV/BS) signal | https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/ |
| WA Native Vegetation Extent (CC BY 4.0) | native-veg label polygons | https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent |
| WA Mining Tenements DMIRS-003 (CC BY 4.0) | per-tenement area summaries | https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003 |
| Atlas of Living Australia | optional biodiversity overlay | https://docs.ala.org.au/ |
| TERN / GeoNadir drone data | optional high-res validation | https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/ |

> Demo mode needs none of these — it generates a synthetic-but-georeferenced AOI
> offline. Real mode (`aoi.demo: false`) consumes these sources; this GCP design
> is what carries that real-mode pipeline to landscape scale.
