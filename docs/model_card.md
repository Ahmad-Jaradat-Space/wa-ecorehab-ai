# Model Card — WA EcoRehab AI

A model card in the style of Mitchell et al. (2019), *"Model Cards for Model Reporting"*
([arXiv:1810.03596](https://arxiv.org/abs/1810.03596)).

This card documents **two** models that share one task, one label schema, one
feature stack, and one spatial evaluation protocol:

1. **RandomForest pixel baseline** — a classical, per-pixel classifier.
2. **U-Net (from-scratch)** — a convolutional semantic-segmentation network.

> **Read this first — what the numbers mean.** Every metric below was produced on
> a **synthetic-but-georeferenced demo AOI**: the pixel *values* are fabricated,
> but the CRS, affine transform, polygon geometry, area arithmetic, tiling,
> spatial-split logic and model code are all real and run unchanged on real
> Digital Earth Australia (DEA) / Western Australia (WA) government data
> (`aoi.demo: false`). These results therefore demonstrate that the **pipeline
> behaves correctly and reports honestly**; they are **not** a measurement of
> real-world ecological accuracy. See [Intended use](#intended-use) and
> [Caveats & recommendations](#caveats--recommendations).

---

## 1. Model details

### Overview

| Field | Value |
|---|---|
| Project | `wa-ecorehab-ai` — open geospatial ML for native-vegetation extent, disturbance and rehabilitation-proxy monitoring in Western Australia |
| Task | 3-class semantic segmentation / pixel classification |
| Classes | `0 = disturbed_or_cleared`, `1 = native_remnant_vegetation`, `2 = water_or_wetland`; `255 = ignore` (never predicted as a target, used for boundary/nodata/ambiguous pixels) |
| Working CRS | **EPSG:3577** (GDA94 / Australian Albers — equal-area, metres). All labels, tiling and hectare reporting use this CRS; area is never computed in EPSG:4326. |
| Inputs | **13 channels** = 7 Sentinel-2 surface-reflectance bands (`blue, green, red, rededge, nir, swir1, swir2`) + 6 spectral indices (`ndvi, evi, ndwi, mndwi, nbr, bsi`) |
| Spatial resolution | 10 m (Sentinel-2 native) |
| Output | Per-pixel class mask, native-vegetation probability raster, entropy uncertainty raster, polygonised vector, hectare area summary |
| Frameworks | scikit-learn (RF); PyTorch (U-Net) |
| Determinism | Fixed seeds throughout (`DEFAULT_SEED = 42`) |
| License / data | Code open-source; intended for open DEA Sentinel-2 ARD + WA government vectors (CC BY 4.0) |
| Version | Demo-mode evaluation, 3-class schema |

### Model A — RandomForest pixel baseline

| Field | Value |
|---|---|
| Type | `sklearn.ensemble.RandomForestClassifier` |
| Estimators | 300 trees |
| Unit of prediction | Single pixel (no spatial context) |
| Feature vector | The 13 channels above, sampled per pixel from training tiles |
| Purpose | Honest, fast, interpretable baseline; feature-importance sanity check; reference point the segmentation model must beat |
| Implementation | `src/ecorehab/models/classical.py`, trained via `ecorehab.training.train_classical` |

### Model B — U-Net (from-scratch)

| Field | Value |
|---|---|
| Type | Encoder–decoder CNN with skip connections, implemented from scratch (no pretrained encoder) |
| Parameters | **7.77M** |
| Normalisation | **GroupNorm** (deliberately *not* BatchNorm — see below) |
| Loss | **Cross-entropy + Dice**, with **inverse-frequency class weights** (handles the disturbed≫native≫water imbalance) |
| Optimizer | **AdamW** |
| LR schedule | **Cosine** decay |
| Regularisation / stopping | **Early stopping on validation macro-F1** |
| Best epoch | Epoch 19 (best val macro-F1 **0.988**) |
| Input | 13-channel tiles of 128×128 px |
| Implementation | `src/ecorehab/models/unet.py`, losses in `models/losses.py`, trained via `ecorehab.training.train_segmentation` |

**Why GroupNorm, not BatchNorm.** Training uses few, small batches over a small
tiled AOI. BatchNorm's running statistics become unreliable in that regime and
tend to *flatten* the softmax, degrading both accuracy and calibration. GroupNorm
removes the batch dependence; an observed side effect is that output
probabilities are **peaked** (see [Calibration](#calibration-note)). This is a
conscious design choice and should not be reverted.

---

## 2. Intended use

### Primary intended uses

- A **production-style ecological monitoring pipeline** demonstrated end-to-end:
  open data → weak labels → classical + deep models → spatial validation →
  calibrated, uncertainty-aware, hectare-level area reports.
- An **open-data case study** and **portfolio artefact** showing CRS-correct
  geospatial ML engineering for Western Australia.
- A **native-vegetation / disturbance / water mapping *proxy*** and a
  **rehabilitation-monitoring *proxy*** (vegetation-recovery signal over disturbed
  or tenement areas), with explicit uncertainty.
- A **methods demonstration** of spatial-block validation, class-imbalance
  handling, probability calibration and bootstrap area confidence intervals.

### Primary intended users

Geospatial data scientists, ML engineers, environmental analysts and reviewers
evaluating the engineering approach. Any ecological decision-maker must treat
outputs as **screening / triage**, subordinate to field validation.

### Out-of-scope uses (DO NOT use this model for)

These limits are firm. The model and its evidence base do **not** support:

- **Species-level vegetation classification.** The schema is 3 coarse cover
  classes; there is no taxonomic signal.
- **Mine-rehabilitation *compliance* assessment.** "Rehabilitation" here is a
  remote-sensing *proxy*, not a regulatory or contractual determination.
- **Drone- or sub-metre-level accuracy.** Inputs are 10 m Sentinel-2; outputs
  cannot resolve fine structure and must not be presented as such.
- **Operational ecological-approval readiness.** Nothing here is calibrated or
  validated for statutory/environmental-approval workflows.
- **Ground-truth accuracy claims without independent field validation.** Demo
  metrics are computed against a *synthetic* hidden-truth raster; even on real
  data, weak labels are not ground truth (see below).

---

## 3. Training data

> The model is trained **only on weak labels**, **never** on the demo truth raster.
> The demo `truth_labels.tif` exists solely for honest evaluation
> ([Section 4](#4-evaluation-data)) and is firewalled from training.

### Area of interest (demo mode)

| Property | Value |
|---|---|
| AOI | Swan Coastal Plain — synthetic AOI, centred near Perth (~115.86°E, 32.05°S) |
| Extent | 768 × 768 px @ 10 m = **7.68 km × 7.68 km** |
| CRS | EPSG:3577 |
| Georeferencing | Real affine transform; NaN nodata corner (~0.4% of pixels); GeoPackage polygons |
| Scene content | 24 synthetic native-vegetation polygons; 2 mining tenements; a sinuous river + water blobs; a managed-green agriculture grid intentionally built as a **spectral confuser** for native vegetation |

The pixel reflectances are **fabricated**. The geometry, transform, nodata
handling and vector overlays are real. The same code paths consume real DEA
Sentinel-2 ARD and WA vectors when `aoi.demo: false`.

### Weak-label construction (3-class schema, `ignore = 255`)

Labels are generated by rasterising the WA-style native-vegetation polygons and
applying spectral gates — **not** hand-annotated ground truth:

- **Native** (`1`): NDVI ≥ 0.45 **inside** inward-eroded native polygons.
- **Water** (`2`): NDWI > 0.10 **outside** polygons.
- **Disturbed** (`0`): NDVI < 0.25 **outside** polygons.
- **Everything else → `ignore = 255`** (never guessed).

Buffering / filtering to suppress boundary and area noise:

| Parameter | Value |
|---|---|
| Positive inward buffer | 20 m |
| Boundary exclusion buffer | 20 m |
| Minimum polygon area | 0.5 ha |

### Resulting weak-label class distribution

| Class | Pixel count | Share of labelled |
|---|---:|---:|
| `0` disturbed_or_cleared | 360,551 | ~71.8% |
| `1` native_remnant_vegetation | 123,190 | ~24.5% |
| `2` water_or_wetland | 18,759 | ~3.7% |
| `255` ignore | 87,324 | ~15% of all 589,824 px |

The class imbalance (disturbed ≫ native ≫ water) is the reason the U-Net uses
inverse-frequency-weighted CE+Dice.

### Tiling and spatial split

| Property | Value |
|---|---|
| Tiles | 33 tiles of 128 px |
| Split type | **Spatial-block split** (256 px blocks) — *not* a random split |
| Split sizes | **18 train / 7 val / 8 test** |
| Tile filter | `min_valid_fraction = 0.5` (drops mostly-nodata/ignore tiles) |

A spatial-block split is used as the primary protocol because a naive random
tile split leaks spatially-autocorrelated pixels between train and test and
inflates apparent performance.

---

## 4. Evaluation data

- **Test set:** the **8 held-out spatial-block test tiles** described above.
- **Reference:** a **hidden synthetic truth raster** (`truth_labels.tif`) that
  is *never* seen during training or labelling. All metrics in
  [Section 5](#5-metrics) and area figures in
  [Section 6](#6-quantitative-analyses) are computed against it.
- **Why this design:** evaluating on spatially-disjoint tiles against an
  independent truth layer is the honest way to estimate generalisation for
  spatially-autocorrelated imagery. It mirrors how the pipeline would be assessed
  on real DEA/WA data, where the "truth" would instead be field plots or a
  higher-quality reference map.

**Important:** because the demo truth is synthetic, these numbers measure
*pipeline correctness and internal consistency*, not real ecological accuracy.

---

## 5. Metrics

All metrics are computed on the **spatial-block test split vs. hidden truth**.
Per-class numbers are precision (P), recall (R), F1 and IoU.

### Model A — RandomForest (300 trees)

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| `0` disturbed_or_cleared | 0.98 | 0.88 | 0.93 | 0.86 |
| `1` native_remnant_vegetation | **0.78** | 1.00 | 0.87 | 0.78 |
| `2` water_or_wetland | 1.00 | 0.81 | 0.89 | 0.81 |

| Aggregate | Value |
|---|---:|
| Macro-F1 | 0.897 |
| Mean IoU | 0.814 |
| Weighted-F1 | 0.908 |
| Balanced accuracy | 0.895 |
| Overall accuracy | 0.906 |

### Model B — U-Net (from-scratch, GroupNorm, 7.77M params)

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| `0` disturbed_or_cleared | 1.00 | 0.87 | 0.93 | 0.87 |
| `1` native_remnant_vegetation | **0.78** | 0.99 | 0.87 | 0.77 |
| `2` water_or_wetland | 0.96 | 0.99 | 0.98 | 0.95 |

| Aggregate | Value |
|---|---:|
| Macro-F1 | 0.926 |
| Mean IoU | 0.866 |
| Weighted-F1 | 0.917 |
| Balanced accuracy | 0.954 |
| Overall accuracy | 0.915 |
| Best validation macro-F1 (epoch 19) | 0.988 |

### Head-to-head

| Metric | RandomForest | U-Net | Winner |
|---|---:|---:|:--|
| Macro-F1 | 0.897 | **0.926** | U-Net |
| Mean IoU | 0.814 | **0.866** | U-Net |
| Balanced accuracy | 0.895 | **0.954** | U-Net |
| Weighted-F1 | **0.908** | 0.917 | U-Net |
| Overall accuracy | 0.906 | **0.915** | U-Net |

The U-Net beats the RF baseline on macro-F1, mean IoU and balanced accuracy. The
biggest gain is on **water** (F1 0.89 → 0.98, IoU 0.81 → 0.95): spatial context
helps the CNN resolve the thin sinuous river that a per-pixel classifier
fragments. **Crucially, both models share the same native-class ceiling
(precision 0.78)** — the limitation is in the *labels and spectral separability*,
not the model family (see [Section 6](#6-quantitative-analyses)).

### Calibration note

The U-Net's outputs are **peaked**, not flat:

| Calibration statistic | Value |
|---|---:|
| Median max-probability | **~0.83** |
| Median predictive entropy | ~0.52 |

This sharpness is attributable to **GroupNorm** (BatchNorm running stats are
unreliable with few/small batches and tend to flatten the softmax). Peaked
probabilities make the entropy uncertainty raster more discriminative for
flagging review areas, but they also mean **confidence is not a substitute for
calibration against field data**; on real data, calibration (reliability
diagram, ECE/Brier) should be re-checked and temperature-scaling considered.

---

## 6. Quantitative analyses

### Area reporting (U-Net inference, EPSG:3577, 95% block-bootstrap CI)

| Class | Predicted area (ha) | 95% CI (ha) | Truth (ha) | Error vs truth |
|---|---:|---:|---:|---:|
| disturbed_or_cleared | 3,718 | [3,530, 3,901] | 4,285 | **−13%** (under) |
| native_remnant_vegetation | **1,910** | **[1,723, 2,096]** | **1,344** | **+42% (over)** |
| water_or_wetland | 269 | [213, 339] | 249 | +8% |

### Headline failure mode — native vegetation is *over-predicted*

This is the single most important honest finding in the project:

> **Native vegetation is over-predicted: 1,910 ha vs. 1,344 ha truth (+42%), and
> the 95% bootstrap CI [1,723, 2,096] does NOT cover the true value.**

**Cause.** The AOI deliberately contains a **managed-green agriculture** grid that
is spectrally close to native remnant vegetation at 10 m. Both models classify
much of that managed green as native, which is exactly why **native precision is
capped at 0.78** for *both* RF and U-Net while native *recall* is ~1.0 (almost all
true native is found, but the predicted-native pool is contaminated with
agriculture). High recall + moderate precision → systematic **area inflation**.

**Why it matters.** Per-class precision and an honest CI surface a real,
directional bias (over-counting native vegetation) that a single
overall-accuracy figure (0.915) would completely hide. For any decision about
native-vegetation extent, this means the headline hectare figure should be read
as an **upper-leaning estimate**, and agriculture/native confusion must be
resolved before the number is trusted — e.g. via DEA Land Cover / Fractional
Cover masking, seasonal (multi-date) composites that exploit crop phenology, or a
dedicated agriculture class.

**Generality.** Because the ceiling appears identically across model families,
the fix is in the *data and labels* (better masks, more spectral/temporal
separation, an explicit agriculture class), not in swapping the architecture.

### Disturbed and water

Disturbed area is mildly **under**-predicted (−13%), the mirror image of native
over-prediction (managed-green pixels that should be disturbed/agriculture are
absorbed into native). Water is close to truth (+8%) and the only class the
U-Net materially improves over RF.

---

## 7. Ethical considerations & caveats

- **Synthetic demo data.** The reported accuracy is on fabricated pixels. It
  evidences *engineering correctness and honest reporting*, **not** real
  ecological performance. Do not quote these figures as measured field accuracy.
- **Weak labels are not ground truth.** Even on real data, labels come from
  polygon rasterisation + spectral gates with explicit `ignore` regions. Boundary
  pixels, nodata and ambiguous pixels are excluded, never guessed.
- **Directional bias has consequences.** Over-predicting native vegetation could,
  if taken at face value, *understate* clearing/disturbance or *overstate*
  conservation extent. The bias is disclosed precisely so it is not relied upon.
- **No regulatory standing.** Outputs are not fit for compliance, approvals, or
  enforcement and must not be presented as such.
- **Human-in-the-loop required.** Treat predictions as screening; an ecologist
  should review low-confidence (high-entropy) areas and the agriculture/native
  boundary before any decision.
- **Data licensing.** Real-mode inputs (DEA Sentinel-2 ARD; WA Native Vegetation
  Extent and Mining Tenements, CC BY 4.0) carry attribution obligations
  ([Section 9](#9-references--data-sources)).

---

## 8. Caveats & recommendations

### Caveats

- Single small AOI (7.68 km square ≈ 59 km²), 33 tiles, 8 test tiles — a *small* spatial sample;
  CIs are correspondingly wide and a few tiles drive aggregate metrics.
- Single-date composite: no seasonal/phenological signal, which is exactly the
  signal that would separate cropland from native vegetation.
- 3-class schema collapses agriculture, built-up and bare ground into
  "disturbed", which is part of why managed green leaks into "native".
- Peaked probabilities (median max-prob ~0.83) can read as overconfident; they
  are not yet calibrated against any field reference.

### Recommendations before any real-world use

1. **Resolve agriculture/native confusion** — add an explicit
   `agriculture_or_managed_green` class and bring in **DEA Land Cover** /
   **DEA Fractional Cover** masks.
2. **Use multi-date / seasonal composites** to exploit crop phenology vs. the
   stable signature of native remnant vegetation.
3. **Validate against field data** (or a higher-quality reference map) and
   **re-check calibration** (reliability diagram, ECE, Brier); consider
   temperature scaling.
4. **Report per-class metrics and area CIs, not overall accuracy** — keep the
   native over-prediction caveat attached to every native hectare figure.
5. **Keep the human in the loop** for high-entropy regions and tenement-level
   decisions.

---

## 9. References & data sources

**Methodology**

- M. Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019 —
  <https://arxiv.org/abs/1810.03596>

**Real-mode data sources** (demo mode requires none)

| Source | Use | URL |
|---|---|---|
| DEA Sentinel-2 ARD (`ga_s2am_ard_3`) via STAC | Primary 10 m multispectral imagery | <https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/> |
| DEA Land Cover | Masking, change context | <https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover> |
| DEA Fractional Cover | Green / non-green / bare recovery signal | <https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/> |
| WA Native Vegetation Extent (CC BY 4.0) | Native-vegetation label polygons | <https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent> |
| WA Mining Tenements DMIRS-003 (CC BY 4.0) | AOI selection, tenement overlays | <https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003> |
| Atlas of Living Australia | Optional biodiversity/weed overlay | <https://docs.ala.org.au/> |
| TERN / GeoNadir drone data | Optional high-resolution module | <https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/> |

---

*Generated for the `wa-ecorehab-ai` open-data case study. All metrics are from the
verified demo-mode evaluation on synthetic-but-georeferenced data; they
demonstrate pipeline correctness and honest reporting, not field-validated
ecological accuracy.*
