# Limitations

This document is the honest counterweight to the README. It states, without
softening, what `wa-ecorehab-ai` does **not** demonstrate, where the numbers
come from, and which results are real evidence versus proxy or weak signal.

If you read only one section, read [§3 Failure modes](#3-failure-modes-and-the-headline-finding)
and [§6 What we explicitly do not claim](#6-what-we-explicitly-do-not-claim).
The headline: in demo mode the model **over-predicts native vegetation by ~42 %**
(1910 ha predicted vs 1344 ha of truth) because managed-green agriculture is
spectrally confused with native vegetation, and — critically — the bootstrap
confidence interval does **not** cover the true value.

---

## 1. The demo data is synthetic

The default pipeline (`aoi.demo: true`) runs fully offline against a
**deterministic synthetic-but-georeferenced** Area of Interest on the Swan
Coastal Plain. This is a deliberate engineering choice (reproducibility, no
network, no credentials, CI can run the whole thing), but it has a sharp
consequence: **the spectral pixel values are fabricated**, so every accuracy
number in this repository is measured against synthetic truth, not the real
landscape.

What is **real** vs **fabricated** in demo mode:

| Component | Status | Notes |
|---|---|---|
| CRS | **Real** | EPSG:3577 (GDA94 / Australian Albers), equal-area, metres — the correct CRS for Australian area reporting. |
| Affine transform / georeferencing | **Real** | 768×768 px @ 10 m = 7.68 km square; real transform written on every raster. |
| Nodata handling | **Real** | NaN nodata corner (~0.4 % of pixels) exercised end-to-end. |
| Vector geometry | **Real format** | GeoPackage polygons (24 synthetic native-veg polygons, 2 mining tenements, river + water blobs, an agriculture grid) — geometry, attributes, topology, rasterisation are real; the *locations/extents are invented*. |
| Area mathematics | **Real** | Hectare computation in an equal-area CRS is genuine and correct. |
| **Sentinel-2 reflectance values** | **FABRICATED** | The 7 bands and 6 indices are synthesised, not observed. They are internally consistent (NDVI separates "vegetation", NDWI separates "water") but they are not Earth. |
| **Truth labels** | **FABRICATED** | `truth_labels.tif` is generated alongside the synthetic AOI and is used **only for honest evaluation, never for training**. |
| Model accuracy figures | **Real maths on fabricated inputs** | The metrics are computed correctly; what they measure is a synthetic world. |

The same code paths support real DEA/WA data (`aoi.demo: false`); that mode is
implemented but has not been run in this environment (it needs network access
and the `.[rs]` extras). **No accuracy claim in this repo has been validated on
real imagery.**

> **What you may conclude from the demo numbers:** that the *architecture,
> spatial-split methodology, calibration, uncertainty, and area-reporting
> machinery* are correct and behave sensibly. **What you may not conclude:** any
> statement about how well the system maps real native vegetation in Western
> Australia.

---

## 2. The labels are weak, not field truth

Even in the intended real-data mode, the supervision signal is **weak labels**,
not ecological ground truth. There is no open, pixel-level, field-validated
native-vegetation label dataset for WA at the resolution and quality this task
would ideally need. We do not hide that — we treat it as a first-class design
constraint.

Weak labels are derived from **generalised government polygons + spectral
decision rules**:

- **3-class schema** (`0 = disturbed_or_cleared`, `1 = native_remnant_vegetation`,
  `2 = water_or_wetland`), with an explicit **`ignore = 255`** class for anything
  ambiguous.
- **Polygon erosion / boundary exclusion**: positive native labels only come from
  pixels **20 m inside** native polygons (`positive_inward_buffer_m = 20`); a
  **20 m boundary exclusion** ring (`boundary_exclusion_buffer_m = 20`) is set to
  ignore; polygons below `min_polygon_area_ha = 0.5` are dropped.
- **Spectral gates** decide the rest: native requires `NDVI ≥ 0.45` *inside* eroded
  polygons; water requires `NDWI > 0.10` *outside* polygons; disturbed requires
  `NDVI < 0.25` *outside* polygons; **everything else → ignore**.

This produces a heavily ignored label field — **~15 % of pixels (87,324 of
589,824) are `ignore`** — and a strongly imbalanced one: disturbed 360,551 px,
native 123,190 px, water 18,759 px.

The honest implications:

1. **The labels inherit the polygons' errors.** Government native-vegetation
   extent polygons are generalised, can be out of date, and were not drawn to
   align with 10 m pixels. Erosion reduces edge contamination but cannot fix a
   polygon that is wrong.
2. **The spectral gates encode our assumptions, not nature.** A fixed
   `NDVI ≥ 0.45` threshold *defines* "native" partly by greenness. That is
   exactly the assumption that breaks against green agriculture (see §3). The
   model is, to a degree, learning to reproduce the rule that built its labels.
3. **High scores against weak labels can be partly circular.** Because both the
   labels and (implicitly) the evaluation reflect the same rules and the same
   synthetic world, the headline F1/IoU numbers should be read as *internal
   consistency of the pipeline*, not as field accuracy.

---

## 3. Failure modes (and the headline finding)

These are the failure modes from the spec's evaluation design (§9), with the
ones we can actually observe in the demo results marked as **evidenced** and the
ones that are expected-in-real-data marked as **anticipated**.

### 3.1 Seasonal / managed-green agriculture confused with native vegetation — EVIDENCED, headline

This is the most important limitation in the project, and it is visible in the
numbers.

In demo inference (U-Net, EPSG:3577), native vegetation area is **over-predicted**:

| Class | Predicted area | 95 % block-bootstrap CI | Truth | Error |
|---|---|---|---|---|
| disturbed | 3718 ha | [3530, 3901] | 4285 ha | −13 % (CI excludes truth) |
| **native** | **1910 ha** | **[1723, 2096]** | **1344 ha** | **+42 % (CI excludes truth)** |
| water | 269 ha | [213, 339] | 249 ha | +8 % (CI covers truth) |

The cause is spectral: a **managed-green agriculture grid** (deliberately built
into the synthetic AOI as a confuser) is dense, green, and reflects much like
remnant native vegetation at 10 m. The model labels it native. This shows up
directly as **native precision 0.78** (≈22 % of pixels called "native" are not),
against **native recall ≈ 1.00** — i.e. the model finds essentially all true
native pixels but pays for it with a large pool of false positives drawn from
agriculture. High recall + modest precision is exactly the signature of an
over-predicted class.

Both the RandomForest and U-Net baselines show the *same* native precision (0.78),
which is strong evidence that this is a **data/feature limitation (agriculture is
spectrally close to native), not a model-capacity problem.** A better
architecture will not fix it; better labels, more bands/seasons, or an explicit
agriculture class would.

### 3.2 Polygon edge pixels — partly MITIGATED, residual risk

Native polygon boundaries do not align to the Sentinel-2 grid. We mitigate with
inward erosion (20 m) and a boundary-exclusion ignore ring (20 m), plus a
minimum-area filter. This *reduces* edge label noise but **shrinks the effective
training footprint** and cannot recover the true class of the excluded ring.
Residual edge error remains wherever a polygon is geometrically wrong.

### 3.3 Mixed pixels at 10 m — anticipated, structural

A single 10 m Sentinel-2 pixel routinely contains a mixture (e.g. scattered trees
over bare soil, a track through bush, a thin riparian strip). Hard 3-class
labels force a mixed pixel into one class. This is a **resolution limit**, not a
model defect, and it interacts with every other failure mode below.

### 3.4 Sparse vegetation vs bare ground — anticipated

Sparsely vegetated native ground and bare/disturbed ground sit close in index
space (low-to-moderate NDVI, elevated BSI). The fixed `NDVI < 0.25 → disturbed`
gate will mislabel sparse native as disturbed, and vice versa, in the transition
zone. In real WA dryland this is a known hard boundary; here it is bounded by the
synthetic design but the mechanism is real.

### 3.5 Water / wetland confusion — minor in demo, anticipated in real data

Water area is the best-behaved class (predicted 269 ha vs 249 ha truth, **CI
covers truth**), helped by an NDWI gate. But **ephemeral wetlands, turbid water,
and shallow flooded vegetation** are genuinely ambiguous in real imagery and are
not exercised by the clean synthetic water blobs. Expect water/wetland precision
to fall on real data, especially seasonally.

### 3.6 Cloud, cloud-shadow, and bare-soil confounds — anticipated, NOT exercised

The synthetic AOI has **no clouds and no cloud shadow**. Real Sentinel-2 scenes
do. Unmasked cloud reads as bright/high-reflectance (can mimic bare/built);
cloud shadow darkens pixels (can mimic water or dense canopy). Likewise, **mine
infrastructure and natural bare ground** are spectrally similar. None of these
confounds are present in demo mode, so the demo over-states robustness. Real
deployment requires explicit cloud/shadow masking (see §5).

---

## 4. Area uncertainty: the bootstrap CI understates total uncertainty

The repo reports 95 % **block-bootstrap** confidence intervals on hectare
estimates. These are honest about **one** source of uncertainty and silent about
the larger one. The native result makes this concrete: the CI is **[1723, 2096]
ha**, the truth is **1344 ha**, and **the interval does not contain the truth**.

That is not a bug in the bootstrap — it is the bootstrap working as designed and
revealing its own scope:

- **What the block-bootstrap CI captures:** *sampling variability* — how much the
  area estimate would wobble if we resampled spatial blocks of the AOI. It
  answers "how precise is this estimate given this model and these pixels?"
- **What it does NOT capture:** *systematic model error / bias*. If the model
  consistently mislabels agriculture as native across the whole AOI, **every**
  bootstrap resample inherits the same bias. The CI then forms tightly **around
  the biased estimate (1910 ha)**, not around the truth. A precise interval
  around a wrong centre is still wrong.

In estimator terms: the bootstrap quantifies **variance**, not **bias**, and the
native error here is dominated by bias (the +566 ha / +42 % offset is far larger
than the ±~190 ha interval half-width). Total error ≈ bias + variance; reporting
only the variance term makes the estimate look more trustworthy than it is.

The methodologically correct fix is **design-based, area-adjusted accuracy
estimation** (Olofsson et al. style: a probability sample of reference points,
an error matrix, and confidence intervals on *bias-corrected* areas). That
requires real reference data (§5) and is **not** done here — the demo has no
independent reference sample, only the synthetic full-coverage truth used for
diagnosis. Until that exists, treat the area CIs as **lower bounds on
uncertainty**.

---

## 5. What would be required before real deployment

This is a **production-style pipeline and an open-data case study**, not a
deployable monitoring product. Before any real ecological or rehabilitation
decision could rest on it, at minimum:

1. **Field validation / independent reference data.** Ground or high-resolution
   (drone, see TERN/GeoNadir) reference at a statistically valid sample of
   locations. Nothing in this repo has been checked against the real landscape.
2. **Independent test AOIs (site holdout).** The current spatial-block split
   (18 train / 7 val / 8 test tiles within one AOI) controls *within-AOI*
   leakage but does **not** prove transfer. The spec's Wheatbelt and
   mining-proxy AOIs exist precisely so models can be tested on geography never
   seen in training. Single-AOI numbers do not establish generalisation.
3. **A proper accuracy-assessment design.** A probability-based reference sample
   and an error matrix, yielding **area-adjusted estimates with bias-corrected
   confidence intervals** (§4) — replacing the bootstrap-only CIs.
4. **Cloud and cloud-shadow masking.** Real Sentinel-2 needs an explicit
   cloud/shadow/cirrus mask (e.g. s2cloudless / Fmask / the DEA ARD masks)
   before compositing. The demo skips this entirely.
5. **Temporal compositing and seasonality.** Multi-date, season-aware composites
   (and/or DEA Fractional Cover time series) are the most plausible route to
   separating **managed-green agriculture from native vegetation** — the §3.1
   headline failure — because the two differ over time (crop phenology, harvest)
   even when they look alike in a single snapshot. A single static composite
   cannot make this distinction.
6. **A richer class schema.** Promoting `agriculture_or_managed_green` and
   `built_or_infrastructure` to first-class labels (spec §5) so the model is
   *allowed* to call agriculture "agriculture" instead of being forced to choose
   between native and disturbed.
7. **Calibration on real data and human-in-the-loop review.** The peaked,
   GroupNorm-calibrated probabilities (max-prob median ~0.83, entropy median
   ~0.52) are calibrated to the synthetic world; calibration must be re-checked
   on real imagery, and low-confidence / high-entropy areas should be routed to
   an ecologist for review rather than auto-accepted.

---

## 6. What we explicitly do not claim

To be unambiguous, this project does **NOT** demonstrate or claim:

- **No species-level vegetation classification.** The schema is 3 broad classes;
  there is no taxonomic identification.
- **No mine-rehabilitation compliance assessment.** Mining tenements are used
  only as area-reporting overlays. "Rehabilitation" here is a **public-data
  proxy**, not a compliance or approval determination.
- **No drone-level accuracy from Sentinel-2.** 10 m optical imagery cannot
  deliver sub-metre or canopy-individual accuracy, regardless of model.
- **No operational / ecological-approval readiness.** This is a portfolio and
  methodology demonstration, not an accredited monitoring service.
- **No ground-truth accuracy.** All accuracy figures are against **synthetic**
  truth (demo mode) or, in principle, weak labels — never against independent
  field validation. There is currently **no field-validated accuracy number** in
  this repository.

### Safe framing (what this project *is*)

- A **production-style ecological monitoring pipeline** with real geospatial
  machinery (CRS, tiling, rasterisation, windowed inference, COG/GeoPackage/CSV
  outputs, HTML site report).
- An **open-data case study** built to run on DEA Sentinel-2 ARD and WA
  government vectors.
- A **native-vegetation mapping proxy** and a **rehabilitation-monitoring proxy**.
- A **spatial-validation and uncertainty-aware area-reporting demonstration** —
  including the honest finding that **native vegetation is over-predicted and the
  bootstrap CI fails to cover truth.**

---

## Sources

- DEA Sentinel-2 ARD (STAC): https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/
- DEA Land Cover: https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover
- DEA Fractional Cover: https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/
- WA Native Vegetation Extent (CC BY 4.0): https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent
- WA Mining Tenements DMIRS-003 (CC BY 4.0): https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003
- Atlas of Living Australia: https://docs.ala.org.au/
- TERN / GeoNadir drone data: https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/

See also: [`docs/spatial_validation.md`](spatial_validation.md),
[`docs/label_strategy.md`](label_strategy.md),
[`docs/data_sources.md`](data_sources.md).
