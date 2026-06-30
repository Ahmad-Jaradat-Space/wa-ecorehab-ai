# Reproducibility

This document describes how to reproduce every result, raster, vector, and figure
in **WA EcoRehab AI** from a clean checkout, and what guarantees (and caveats)
back that reproducibility.

The defaults run the pipeline in **demo mode**: a deterministic
synthetic-but-georeferenced Area of Interest (AOI) over the Swan Coastal Plain
near Perth. This means anyone can reproduce the full data → labels → tiles →
models → inference → report chain **offline, with no API keys and no
downloads**.

> **Honesty note — what is and isn't real.** In demo mode the *pixel values* of
> the Sentinel-2-style composite are fabricated by a deterministic generator. The
> *geospatial machinery is real*: the working CRS (EPSG:3577, Australian Albers,
> equal-area metres), the affine transform, the NaN nodata corner (~0.4% of
> pixels), the GeoPackage vegetation/tenement polygons, the rasterisation, the
> spatial-block split, the windowed inference, and the hectare/area arithmetic all
> behave exactly as they would on real imagery. The same code paths run on real
> [Digital Earth Australia](https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/)
> Sentinel-2 and [WA Government](https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent)
> vectors when you set `aoi.demo: false` and install the `[rs]` extra. Because the
> imagery is synthetic, the quoted metrics are a **demonstration of the validation
> methodology**, not a field-validated accuracy claim. See
> [`limitations.md`](limitations.md) and [`model_card.md`](model_card.md).

---

## 1. Determinism and seeds

A single integer seed propagates through every stochastic step. It is defined
once and threaded everywhere:

- **`constants.DEFAULT_SEED = 42`** (`src/ecorehab/constants.py`) is the package
  default.
- **`project.seed`** in any config (default `42`) is the value the CLIs actually
  read; the shipped configs set `seed: 42` explicitly.

Every source of randomness is keyed off that seed:

| Stochastic step | Where | Seeded by |
|---|---|---|
| Synthetic AOI generation | `data/demo.py` (`DemoConfig.seed`) | `42` |
| Spatial-block train/val/test split | `labels/tiling.py` (`SplitConfig.seed`) | `42` |
| Balanced pixel sampling (classical) | `labels/sampling.py`, `training/train_classical.py` | `cfg.project.seed` |
| RandomForest / XGBoost / LightGBM | `models/classical.py` (`random_state`) | `cfg.project.seed` |
| U-Net training | `training/train_segmentation.py` | `torch.manual_seed(cfg.project.seed)` **and** `np.random.seed(cfg.project.seed)` |
| Augmentation noise | `training/augmentations.py` | `cfg.project.seed` |
| Block-bootstrap area confidence intervals | `evaluation/area_estimation.py` | `cfg.project.seed` |

So both **NumPy** and **PyTorch** RNGs are seeded before training, the synthetic
data is regenerated bit-for-bit, and the spatial split is stable across runs.

**Caveats (honest determinism).** Exact bitwise reproduction assumes the same
versions of NumPy / PyTorch / scikit-learn and the same hardware/BLAS backend.
Floating-point reduction order can differ across CPU architectures and thread
counts, and CUDA introduces additional non-determinism (some cuDNN kernels are
non-deterministic by default). The from-scratch U-Net uses **GroupNorm, not
BatchNorm**, partly for this reason: with few small batches, BatchNorm running
statistics are unstable and flatten the softmax. Expect metrics to be stable to
within small floating-point tolerances, not identical to the last decimal across
machines. Train on CPU (the default, and what CI uses) for the most repeatable
numbers.

---

## 2. Configuration system

Everything is config-driven; there are **no hard-coded paths or
hyperparameters**. Each CLI takes `--config path/to.yaml`.

**Typed + validated.** Configs are parsed into a pydantic v2 `Config` object
(`src/ecorehab/utils/config.py`). The models use `extra="forbid"`, so a typo in a
YAML key fails loudly instead of being silently ignored. A `model_validator`
enforces invariants such as *class ids must be contiguous from 0* (they map
directly onto model logits) and *`ignore_index` must not collide with a class id*.

**`extends` mechanism.** A config can inherit from a base config via an
`extends:` key and override only what it needs, via a recursive deep-merge (child
wins). This keeps the AOI block in one place:

```yaml
# configs/unet.yaml
extends: aoi_swan_coastal_plain.yaml
segmentation:
  arch: unet
  base_channels: 32
  loss: ce_dice
  epochs: 20
```

`aoi_swan_coastal_plain.yaml` is the base; `unet.yaml`, `random_forest.yaml`,
`inference.yaml`, `report.yaml`, `ci_smoke.yaml`, `aoi_wheatbelt.yaml`, and
`aoi_mining_proxy.yaml` all extend it. Circular `extends` chains are detected and
rejected. Paths resolve relative to the repo root regardless of your working
directory, and the equal-area working CRS defaults to `EPSG:3577`.

The shipped configs live in `configs/`:

| Config | Purpose |
|---|---|
| `aoi_swan_coastal_plain.yaml` | Base AOI (project seed/CRS/resolution, demo settings, 3-class schema, label buffers, tiling + spatial split, features) |
| `aoi_wheatbelt.yaml`, `aoi_mining_proxy.yaml` | Alternate AOIs (override the base) |
| `random_forest.yaml` | RandomForest pixel baseline (300 trees) |
| `unet.yaml` | From-scratch U-Net segmentation baseline |
| `segformer.yaml`, `xgboost.yaml` | Optional models (need `[dl]` / `[boost]` extras) |
| `inference.yaml` | Tiled/windowed batch inference |
| `report.yaml` | Decision-grade HTML site report |
| `ci_smoke.yaml` | Tiny AOI + cheap models so CI runs the whole pipeline in well under a minute |

---

## 3. Data provenance: `data_manifest.json`

Every stage that writes processed artifacts also writes/updates a
`data_manifest.json` under `data/processed/<aoi>/` (`src/ecorehab/artifacts.py`,
`write_manifest`). It records, on every run:

- `aoi` name and `demo_mode` flag (so you always know whether outputs came from
  synthetic or real data),
- working `crs` (`EPSG:3577`) and `resolution_m` (10.0),
- `generated_utc` timestamp (UTC, second precision),
- plus per-stage provenance entries the caller passes in (inputs, parameters,
  counts).

All stages also agree on canonical on-disk paths through `artifacts.py`
(`composite_path`, `label_path`, `truth_path`, `tile_index_path`, etc.), so
`build_composites` writes the composite exactly where `build_labels` and
`build_tiles` expect to read it. This is what makes the chain reproducible and
inspectable rather than a pile of ad-hoc files.

The hidden `truth_labels.tif` exists **only in demo mode** and is used **only for
honest evaluation, never for training** — weak labels are used for training, the
synthetic ground truth is held out to score against.

---

## 4. Environment setup

The **pip path defined in `pyproject.toml` is the primary, CI-tested route.**
`requires-python = ">=3.10"`. The core dependency set is deliberately small so
the full demo and the test suite run **without** PyTorch, STAC, or boosting
libraries (those are lazy-imported optional extras with clear install hints).

### Option A — pip (recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: demo pipeline + tests (no torch)
# or, with extras:
pip install -e ".[dev]"     # + pytest, ruff, black, pre-commit, jupyter, nbconvert
```

Optional extras (combine as needed, e.g. `.[dev,dl]`):

| Extra | Installs | Enables |
|---|---|---|
| `dev` | pytest, ruff, black, pre-commit, nbconvert, jupyter | Tests, linting, formatting, notebooks |
| `dl` | torch, torchvision, albumentations, segmentation-models-pytorch, timm | U-Net training, SegFormer / smp_unet |
| `boost` | xgboost, lightgbm, shap | Gradient-boosted baselines + SHAP |
| `rs` | pystac-client, odc-stac, rasterstats, rtree | Real DEA/WA data acquisition (`aoi.demo: false`) |
| `viz` | folium, leafmap, contextily, plotly | Interactive maps / reporting extras |
| `mlops` | mlflow | Optional experiment tracking |
| `all` | everything above | Full install |

> The from-scratch U-Net needs **only `torch`** (covered by `[dl]`). `segmentation-models-pytorch`/`timm` are required only for `smp_unet`/`segformer`. MLflow is **optional** — the pipeline runs and reports without any experiment tracker; install `[mlops]` only if you want runs logged to `mlruns/`.

Convenience `make` targets wrap these: `make install`, `make install-dev`,
`make install-all`.

### Option B — conda

For users who prefer conda-forge geospatial builds, `environment.yml` pins
Python 3.11 and pulls GDAL/GEOS/PROJ-backed geospatial packages plus CPU PyTorch
from conda-forge (pip-only extras like `segmentation-models-pytorch`, `timm`,
`shap` are layered on top):

```bash
conda env create -f environment.yml
conda activate wa-ecorehab-ai
pip install -e .
```

### Option C — Docker (CPU)

The `Dockerfile` builds a slim CPU image (`python:3.11-slim`). GDAL/PROJ/GEOS
ship inside the rasterio/geopandas/pyproj wheels, so no system GDAL is needed.
PyTorch is **CPU-only** here; build a CUDA variant from a devel base for GPU
training.

```bash
docker build -t wa-ecorehab-ai .
# default command prints the CLI help; mount outputs and run a stage, e.g.:
docker run --rm -v "$PWD/outputs:/app/outputs" wa-ecorehab-ai \
    python -m ecorehab.inference.batch_predict --config configs/inference.yaml
```

The image installs `".[dl,boost,rs]"` so it can run the deep-learning, boosting,
and real-data paths out of the box.

---

## 5. End-to-end reproduction

### Fast path (one command)

```bash
make install-dev
make pipeline       # data -> composites -> labels -> tiles -> RF -> U-Net -> predict -> report
```

`make pipeline` runs, in order: `demo-data composites labels tiles train-rf
train-unet predict report` — fully offline on the synthetic AOI. (To run the U-Net
and inference you also need `pip install -e ".[dl]"`, or `make install-all`.)

### Explicit per-stage sequence

Every stage takes `--config`. This is exactly what `make pipeline` expands to,
plus the example-figure script:

```bash
# 1. Generate the deterministic synthetic-but-georeferenced AOI (raster + vectors)
python -m ecorehab.data.make_demo            --config configs/aoi_swan_coastal_plain.yaml
# (real mode only: python -m ecorehab.data.download_vectors --config ...; demo generates its own)

# 2. Build the Sentinel-2-style surface-reflectance composite
python -m ecorehab.data.build_composites     --config configs/aoi_swan_coastal_plain.yaml

# 3. Weak labels (3-class: disturbed/native/water; ignore=255) from polygons + spectral gates
python -m ecorehab.labels.build_labels       --config configs/aoi_swan_coastal_plain.yaml

# 4. Tile into 128 px tiles with a SPATIAL-BLOCK split (18 train / 7 val / 8 test)
python -m ecorehab.labels.build_tiles        --config configs/aoi_swan_coastal_plain.yaml

# 5. Train the RandomForest pixel baseline (300 trees)
python -m ecorehab.training.train_classical  --config configs/random_forest.yaml

# 6. Train the from-scratch U-Net (GroupNorm, CE+Dice, inverse-freq weights, early stop)
python -m ecorehab.training.train_segmentation --config configs/unet.yaml

# 7. Tiled/windowed batch inference -> prediction + probability + uncertainty GeoTIFFs + polygons
python -m ecorehab.inference.batch_predict   --config configs/inference.yaml

# 8. Decision-grade HTML site report (per-class metrics + block-bootstrap area CIs)
python -m ecorehab.reporting.build_site_report --config configs/report.yaml

# 9. Regenerate the committed example figures
python scripts/make_example_figures.py       --config configs/aoi_swan_coastal_plain.yaml
```

A unified dispatcher (`ecorehab <stage> --config ...`, installed via the
`ecorehab` console script) wraps the same stages.

### What you should get

Reproduced artifacts land under `outputs/` (predictions, vector outputs, metrics,
reports). The headline numbers, all evaluated on the **spatial-block test split
against the hidden synthetic truth**:

| Model | macro-F1 | mean IoU | balanced acc. | overall acc. |
|---|---|---|---|---|
| RandomForest (300 trees) | 0.897 | 0.814 | 0.895 | 0.906 |
| U-Net (7.77M params, from scratch) | **0.926** | **0.866** | **0.954** | 0.915 |

Per-class U-Net: disturbed P 1.00 / R 0.87 / F1 0.93 / IoU 0.87; native P 0.78 /
R 0.99 / F1 0.87 / IoU 0.77; water P 0.96 / R 0.99 / F1 0.98 / IoU 0.95.

**Area estimates (U-Net inference, EPSG:3577, 95% block-bootstrap CI) vs truth:**

| Class | Predicted (ha) | 95% CI | Truth (ha) |
|---|---|---|---|
| disturbed | 3718 | [3530, 3901] | 4285 |
| native | 1910 | [1723, 2096] | 1344 |
| water | 269 | [213, 339] | 249 |

> **Headline honest finding (the result to actually read).** Native vegetation is
> **over-predicted — 1910 ha vs 1344 ha truth (+42%), and the bootstrap CI does
> not cover the truth.** The cause is a deliberate confuser built into the AOI:
> managed-green agriculture is spectrally close to native vegetation and gets
> misclassified as native (native **precision 0.78**, recall ~1.0). Aggregate
> accuracy looks high; the class-specific view exposes the real failure mode. This
> is exactly the kind of error that matters for area-based ecological reporting,
> which is why we report per-class metrics and confidence intervals rather than
> overall accuracy.

Probabilities are peaked (max-prob median ~0.83, entropy median ~0.52), a benefit
of GroupNorm over BatchNorm for small/few batches.

---

## 6. What is and isn't committed

The repo commits **code, configs, docs, tests, and a handful of small example
PNGs** — never bulk data or model weights. From `.gitignore`:

- **Gitignored:** `data/raw/`, `data/interim/`, `data/processed/`,
  `data/external/` (all generated/downloaded data), all of `outputs/*`, model
  checkpoints (`*.pt`, `*.pth`, `*.joblib`, `*.pkl`, `checkpoints/`), and
  experiment-tracking dirs (`mlruns/`, `wandb/`).
- **Committed:** `outputs/examples/*.png` and `data/sample/` are explicit
  un-ignore exceptions, plus `.gitkeep` files that preserve the directory layout.

The five committed example figures (regenerated by step 9 above) are:

```
outputs/examples/01_data_audit.png
outputs/examples/02_labels_and_split.png
outputs/examples/03_evaluation.png
outputs/examples/04_prediction_and_uncertainty.png
outputs/examples/05_model_comparison.png
```

Because everything else is regenerated deterministically from code + configs,
keeping data and weights out of git is safe: a clean checkout plus
`make pipeline` reproduces them.

---

## 7. Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) gates every push to `main` and every
pull request with two jobs:

1. **`lint-and-test`** — matrix over Python **3.11 and 3.12**:
   - `ruff check src tests` (lint),
   - `black --check src tests` (format gate),
   - `pytest -q` (the suite, **50 tests passing**),
   - installs `".[dev]"` plus CPU-only PyTorch from the official index.

2. **`pipeline-smoke`** — proves the whole pipeline runs end-to-end **offline**
   using the tiny `configs/ci_smoke.yaml` (small AOI, cheap models). It runs
   `make_demo → build_labels → build_tiles → train_classical →
   train_segmentation → batch_predict → build_site_report`, then asserts the key
   outputs exist (`prediction_mask.tif`, `area_summary.csv`, `site_report.html`)
   and uploads the site report as a build artifact.

Locally, the same gates are `make fmt-check` (black + ruff) and `make test`.

---

## 8. Validated stack

The pipeline is intentionally version-tolerant (`requires-python >=3.10`; nothing
pinned tighter than needed). CI exercises Python 3.11/3.12; the full chain has
also been validated end-to-end on the following modern stack:

| Component | Validated version |
|---|---|
| Python | 3.10 – 3.14 |
| NumPy | 2.5 |
| rasterio | 1.5 |
| PyTorch | 2.12 |
| geopandas | 1.1 |

Most users will be on Python 3.10–3.12; 3.14 is validated but not required.
Optional/heavy dependencies (torch beyond core, smp/timm, xgboost/lightgbm,
pystac/odc-stac, mlflow) are lazy-imported, so the **core demo and the test suite
run without any of them**.

---

## 9. Real-data mode (notes)

Setting `aoi.demo: false` and installing the `[rs]` extra switches the same
pipeline onto real [DEA Sentinel-2 ARD](https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/)
(`ga_s2am_ard_3` via STAC) and WA Government vectors
([Native Vegetation Extent](https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent),
[Mining Tenements DMIRS-003](https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003),
both CC BY 4.0). This path is implemented but requires network access and has not
been exercised in CI; it will not produce field-validated accuracy without
independent ground truth. See [`data_sources.md`](data_sources.md) for the full
source list and licences.
