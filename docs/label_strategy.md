# Label Strategy: Weak Supervision for WA Native Vegetation

This document explains how `wa-ecorehab-ai` produces the *training labels* it never
had. There is no off-the-shelf, open, pixel-level ecological ground-truth raster
for Western Australia at Sentinel-2 resolution, so the pipeline **synthesises a
deliberately weak label** from open vector boundaries plus transparent spectral
rules. The honest position is that this is *weak supervision*, and we treat that
as a design feature rather than a defect: every assumption is explicit, every
uncertain pixel is abstained on, and every label carries a confidence value.

> **Scope and honesty note.** In demo mode the *pixel reflectance values are
> synthetic* (deterministically fabricated for a 768×768 Swan Coastal Plain AOI).
> Everything else — the CRS (EPSG:3577, Australian Albers, equal-area metres),
> the affine transform, the GeoPackage polygons, the buffering geometry, the
> area maths — is real, and the identical code path runs on real DEA/WA data with
> `aoi.demo: false`. The labels described here are a **proxy** for native-vegetation
> extent, disturbance and a rehabilitation signal. They are **not** species maps,
> **not** a compliance assessment, and **not** field-validated ground truth.
>
> Source of the rules described below: `src/ecorehab/labels/weak_labels.py` and
> `src/ecorehab/labels/rasterize.py`. Threshold/buffer defaults:
> `src/ecorehab/utils/config.py` (`LabelsConfig`). Class schema:
> `src/ecorehab/constants.py` (`DEFAULT_CLASSES`).

---

## 1. Why there is no perfect open label (and why weak supervision is the right move)

Supervised segmentation wants a clean, per-pixel `y`. For native-vegetation
monitoring in WA, that label does not exist in open form:

- **Vector extent layers are polygons, not pixels.** The
  [WA Native Vegetation Extent](https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent)
  layer (CC BY 4.0) tells you *roughly where* remnant vegetation is, as polygons
  digitised at a coarser scale than a 10 m pixel. Its boundaries are generalised,
  its currency lags real disturbance, and a polygon labelled "vegetation" can
  contain internal clearings, tracks, and bare patches.
- **Land-cover products are coarse or modelled.**
  [DEA Land Cover](https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover)
  and [DEA Fractional Cover](https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/)
  are excellent national products but are themselves model outputs at their own
  resolutions and ontologies; using them directly as "truth" would launder one
  model's errors into ours.
- **Field/drone truth is sparse and local.**
  [TERN/GeoNadir drone data](https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/)
  and [Atlas of Living Australia](https://docs.ala.org.au/) occurrence records are
  invaluable but cover a tiny, non-random fraction of any AOI — useful for
  *validation spot-checks*, not for wall-to-wall pixel labels.
- **Species-level ecological classes are not separable from Sentinel-2 alone.**
  Even with perfect labels, 10 m multispectral pixels cannot resolve species; we
  therefore restrict ourselves to a coarse, defensible 3-class schema.

The pragmatic, honest response is **weak supervision**: combine the strongest
*spatial* prior we have (the extent polygons) with the strongest *spectral*
priors we have (vegetation/water indices), encode them as simple, auditable
rules, and — critically — **abstain** wherever the two priors disagree or are
ambiguous. A noisy label we understand beats a clean label we have to invent.

---

## 2. The 3-class schema (+ ignore)

The minimum-viable schema (`constants.DEFAULT_CLASSES`, overridable in config) uses
contiguous integer ids from 0 so they map directly onto model logits and sklearn
class indices.

| id  | name (constant)              | meaning (proxy)                                | colour    |
| --- | ---------------------------- | ---------------------------------------------- | --------- |
| 0   | `disturbed_or_cleared`       | non-vegetated / cleared / bare ground          | `#d9a05b` |
| 1   | `native_remnant_vegetation`  | trusted green interior of remnant-veg polygons | `#1b7837` |
| 2   | `water_or_wetland`           | open water / wet surfaces                       | `#2c7fb8` |
| 255 | *ignore* (`IGNORE_INDEX`)    | abstain — not used for training or metrics       | —         |

`255` is the sentinel for "do not supervise here". It is passed straight through
to PyTorch `CrossEntropyLoss(ignore_index=255)` and filtered out before any
sklearn metric, so abstained pixels contribute **zero gradient and zero score**.
This is what makes principled abstention cheap: we are never forced to guess.

---

## 3. The exact rules (as implemented)

All thresholds are config defaults (`LabelsConfig`); indices come from
`features/indices.py`:

- **NDVI** = (NIR − Red) / (NIR + Red) — greenness.
- **NDWI** = (Green − NIR) / (Green + NIR) — McFeeters open-water index.

The label raster starts as **all-ignore** and classes are written in this order.
nodata is detected first (any non-finite band → `nodata` mask; reflectance nodata
is converted to `NaN` so indices propagate it), and indices are filled with a
sentinel `−1.0` where NaN so they fail every "vegetated/wet" test by construction.

```text
native (1)    := inside_core           AND NDVI >= 0.45    AND not nodata
water  (2)    := (not inside_any)      AND NDWI >  0.10    AND not nodata
disturbed (0) := (not inside_any)      AND (not water)     AND NDVI < 0.25  AND not nodata
ignore (255)  := boundary ring  OR  nodata  OR  anything not matched above
```

where

- `inside_any`  = pixels inside the (unbuffered) vegetation polygons;
- `inside_core` = pixels inside polygons **eroded inward by 20 m**
  (`positive_inward_buffer_m`), i.e. the trusted interior;
- `boundary ring` = a band around each polygon boundary, **20 m in and 20 m out**
  (`boundary_exclusion_buffer_m`), applied as a hard override to `ignore`.

Two override steps run *after* the class assignments, so they win:

1. `label[boundary_ring] = 255`, `confidence = 0` — edge pixels are the noisiest.
2. `label[nodata] = 255`, `confidence = 0` — nodata is never a class.

### What deliberately becomes `ignore`

- The **inward 20 m margin** of every polygon (inside `inside_any` but outside
  `inside_core`): too close to a generalised boundary to trust.
- The **outer 20 m ring** outside each polygon: forced to ignore *even if it would
  otherwise pass the water or disturbed test*.
- **Intermediate-NDVI pixels outside polygons** (0.25 ≤ NDVI < 0.45 and not wet):
  sparse vegetation and — importantly — **managed-green agriculture**. We refuse
  to label these. See §7: this is exactly the class the model later struggles with.
- **Degraded/cleared patches *inside* a polygon** (NDVI < 0.45 but inside `inside_core`):
  not native (fails the greenness gate), but also not eligible for `disturbed` or
  `water` (those require `not inside_any`). They fall through to `ignore`. This is a
  conservative choice — we do not assert "disturbance inside a remnant" from a
  single composite — and it is a known limitation for the rehabilitation proxy.

### Why each gate is one-directional

- Native requires **both** the spatial prior (inside the trusted interior) **and**
  the spectral prior (green). A green pixel outside known remnant polygons is *not*
  promoted to native — that would invent extent.
- Water and disturbed are only asserted **outside** polygons, so spectral noise
  inside a remnant cannot flip a vegetation pixel to water/bare.

---

## 4. Boundary-noise handling

Weak labels are dominated by *edge* and *ambiguity* noise. Five mechanisms
(see `rasterize.py` + `weak_labels.py`) contain it:

| Mechanism                       | Parameter / default                       | What it does |
| ------------------------------- | ----------------------------------------- | ------------ |
| Inward erosion (positive buffer)| `positive_inward_buffer_m = 20.0`         | Native is taken only from the polygon *core*, two pixels in from the generalised boundary. Implemented as `polygons_to_mask(..., buffer_m=-20)`. |
| Boundary exclusion ring         | `boundary_exclusion_buffer_m = 20.0`      | A 40 m-wide ring (±20 m about each boundary) is hard-set to `ignore`. Implemented as `outer(+20) AND NOT inner(−20)` in `boundary_exclusion_mask`. |
| Minimum mappable unit           | `min_polygon_area_ha = 0.5`               | The project's minimum mappable polygon (≈50 pixels at 10 m). Enforced where polygons are *created/consumed* as polygons: synthetic native polygons in `data/demo.py`, and predicted polygons in `inference/polygonize.py` (sub-0.5 ha specks dropped). The weak-label generator itself does not re-filter the input layer by area — it trusts the source vector's own MMU. |
| Ignore mask / abstention        | `ignore_index = 255`                       | nodata, edge rings, and all unmatched pixels are excluded from loss and metrics. |
| Per-pixel confidence raster     | `label_confidence.tif`, float32 ∈ [0, 1]  | Records how trustworthy each label is (see §6). |

Because `inside_core` and the ring's inner edge are both the −20 m erosion, the
ring **never overwrites a native pixel** — native sits strictly inside the ring's
inner boundary. The ring only ever demotes would-be water/disturbed pixels (and
the inner margin that was already ignore) near a boundary.

---

## 5. Verified weak-label pixel counts (demo AOI)

For the 768×768 = **589,824-pixel** synthetic Swan Coastal Plain AOI:

| class                          | id  | pixels   | % of AOI | % of labelled |
| ------------------------------ | --- | -------- | -------- | ------------- |
| disturbed_or_cleared           | 0   | 360,551  | 61.1%    | 71.8%         |
| native_remnant_vegetation      | 1   | 123,190  | 20.9%    | 24.5%         |
| water_or_wetland               | 2   | 18,759   | 3.2%     | 3.7%          |
| **ignore**                     | 255 | 87,324   | **14.8%**| —             |
| **total**                      |     | 589,824  | 100%     | —             |

Roughly **one pixel in seven is abstained on**, and the labelled classes are
strongly imbalanced (water is ~4% of labelled pixels). This imbalance is why
training uses **inverse-frequency class weights** and why we report **class-specific**
F1/IoU rather than overall accuracy. (Pixel counts are emitted to
`data_manifest.json` by `labels/build_labels.py`.)

---

## 6. The confidence raster

Alongside `labels.tif`, `build_labels` writes `label_confidence.tif` — a float32
raster in [0, 1] giving a per-pixel trust score. Confidence scales with how
decisively a pixel clears its threshold, so marginal pixels are down-weighted:

| class      | confidence formula                                  | floor → ceiling |
| ---------- | --------------------------------------------------- | --------------- |
| native     | `clip(0.7 + 0.6·(NDVI − 0.45), 0.7, 1.0)`           | 0.70 → 1.00     |
| water      | `clip(0.6 + 2.0·(NDWI − 0.10), 0.6, 1.0)`           | 0.60 → 1.00     |
| disturbed  | `clip(0.6 + 1.5·(0.25 − NDVI), 0.6, 1.0)`           | 0.60 → 1.00     |
| ignore (ring/nodata) | `0.0`                                     | 0.0             |

A pixel sitting right on its threshold gets the floor (0.6–0.7); a pixel far into
the confident region saturates at 1.0. This raster is the natural hook for
confidence-weighted sampling/loss and for honest visual QA of where the labels are
shaky — it makes label noise *visible and quantified* rather than hidden.

---

## 7. Honest discussion of label noise and the failure it predicts

Weak labels carry several irreducible noise sources, and we are explicit about each:

1. **Boundary generalisation.** The extent polygons are coarser than 10 m. Erosion
   + the exclusion ring remove the worst of it at the cost of abstaining on a
   sizeable fraction of edge pixels (a chunk of the 14.8% ignore).
2. **Spectral confusion (the headline limitation).** A single greenness threshold
   cannot separate *managed-green agriculture* from *native vegetation*. The label
   logic side-steps this at training time by sending intermediate-NDVI pixels
   outside polygons to `ignore` — but the model is never explicitly taught that
   agriculture is *not* native, because we never give it that negative. At
   inference the model therefore tends to call green farmland "native".
3. **Temporal/internal degradation.** Cleared or degraded patches *inside* a
   remnant polygon are abstained on, not labelled disturbed, so the rehabilitation
   proxy is conservative.

The downstream evidence of (2) is the project's headline honest finding. Evaluated
on the spatial-block **test** split against held-out truth, the U-Net achieves
strong aggregate scores (macro-F1 0.926, mean IoU 0.866) but **over-predicts native
vegetation**:

| class     | precision | recall | predicted area (95% CI) | truth area |
| --------- | --------- | ------ | ----------------------- | ---------- |
| disturbed | 1.00      | 0.87   | 3,718 ha [3,530–3,901]  | 4,285 ha   |
| native    | **0.78**  | 0.99   | **1,910 ha [1,723–2,096]** | **1,344 ha** |
| water     | 0.96      | 0.99   | 269 ha [213–339]        | 249 ha     |

Native is over-predicted by **+42% (1,910 vs 1,344 ha)** and the bootstrap CI does
**not** cover truth. The native **precision of 0.78** is the fingerprint of
agriculture-as-native confusion — exactly the ambiguity the label rules flagged by
abstaining. This is reported as a limitation, not papered over.

### How the design mitigates (but does not eliminate) the noise

- **Ignore index** keeps the noisiest pixels (edges, nodata, ambiguous greens) out
  of the loss and the metrics, so they neither train nor flatter the model.
- **Confidence raster** quantifies per-pixel trust for weighting and QA.
- **Conservative one-directional gates** prevent spectral noise from inventing extent.
- **Spatial-block train/val/test split** (never random) means metrics are measured
  on geographically held-out ground, so the agriculture-confusion error shows up in
  the numbers instead of leaking away.
- **Honest evaluation only.** The demo `truth_labels.tif` is used *exclusively* for
  evaluation and **never** for training; the gap between the weak label and that
  truth is itself a measured noise source, not a hidden one.

None of this makes the labels ground truth. The correct mitigation for the
remaining error is real negatives for agriculture (a crop/pasture mask or a
land-use layer) and field/drone validation — both supported by the real-data path,
neither claimed as done here.

---

## 8. Worked example — decision table

Six representative pixels through the rules (`inside_core` = inside the 20 m-eroded
polygon interior; `inside_any` = inside the raw polygon; "ring" = within ±20 m of a
boundary):

| # | inside_core | inside_any | in ring | nodata | NDVI | NDWI | → label | confidence | why |
| - | ----------- | ---------- | ------- | ------ | ---- | ---- | ------- | ---------- | --- |
| 1 | ✓ | ✓ | ✗ | ✗ | 0.72 | 0.05 | **native (1)** | 0.86 | trusted green interior |
| 2 | ✓ | ✓ | ✗ | ✗ | 0.30 | −0.10 | **ignore (255)** | 0.00 | inside remnant but fails greenness gate; not eligible for disturbed/water |
| 3 | ✗ | ✗ | ✗ | ✗ | −0.05 | 0.35 | **water (2)** | 1.00 | wet, outside polygons (NDWI capped at 1.0) |
| 4 | ✗ | ✗ | ✗ | ✗ | 0.08 | −0.20 | **disturbed (0)** | 0.86 | clearly non-vegetated, outside polygons |
| 5 | ✗ | ✗ | ✗ | ✗ | 0.55 | −0.05 | **ignore (255)** | 0.00 | managed-green **agriculture** — green but outside known remnant ⇒ abstain (the §7 confuser) |
| 6 | ✗ | — | ✓ | ✗ | 0.60 | 0.40 | **ignore (255)** | 0.00 | boundary ring override wins over any class |
| 7 | — | — | — | ✓ | n/a | n/a | **ignore (255)** | 0.00 | nodata corner |

Confidence checks: #1 `0.7 + 0.6·(0.72−0.45) = 0.86`; #3 `0.6 + 2.0·(0.35−0.10) = 1.1 → 1.0`;
#4 `0.6 + 1.5·(0.25−0.08) = 0.86`.

Pixel **#5** is the crux of the whole strategy: the label rules are honest enough to
*decline* to call green farmland "native", which is why the training signal is
clean — and why the *unaided* model, lacking an explicit agriculture negative,
still over-predicts native at inference (§7).

---

## 9. Where this lives in the pipeline

| Artefact / module | Role |
| ----------------- | ---- |
| `src/ecorehab/labels/rasterize.py` | `rasterize_geometries`, `polygons_to_mask` (metric buffers), `boundary_exclusion_mask` (the ±20 m ring) |
| `src/ecorehab/labels/weak_labels.py` | `build_weak_labels` → `WeakLabelResult(label, confidence, class_counts)` |
| `src/ecorehab/labels/build_labels.py` (CLI) | writes `labels.tif` (nodata = 255) + `label_confidence.tif`, records counts in `data_manifest.json` |
| `src/ecorehab/utils/config.py` (`LabelsConfig`) | thresholds, buffers, class schema, `ignore_index` |
| `src/ecorehab/constants.py` | `DEFAULT_CLASSES`, `IGNORE_INDEX = 255`, `REFLECTANCE_NODATA = −9999` |

Run it:

```bash
python -m ecorehab.labels.build_labels --config configs/aoi_swan_coastal_plain.yaml
```

---

### Real-data label sources (used when `aoi.demo: false`)

- WA Native Vegetation Extent (CC BY 4.0): https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent
- WA Mining Tenements DMIRS-003 (CC BY 4.0): https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003
- DEA Sentinel-2 ARD (`ga_s2am_ard_3`, STAC): https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/
- DEA Land Cover: https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover
- DEA Fractional Cover: https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/
- Atlas of Living Australia (validation): https://docs.ala.org.au/
- TERN / GeoNadir drone data (validation): https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/
