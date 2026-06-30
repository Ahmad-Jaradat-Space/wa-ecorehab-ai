# Spatial validation

How `wa-ecorehab-ai` measures model performance without fooling itself.

> **TL;DR** — Satellite imagery pixels are spatially autocorrelated: a tile and
> its neighbour are near-duplicates. A naive *random* tile split therefore puts
> almost-identical tiles in both train and test, and the model is effectively
> graded on data it has already seen. That inflates every metric. In this repo
> the **spatial-block split is the primary validation** (whole 256 px blocks are
> assigned to train/val/test so no block straddles the boundary), and the random
> split exists **only as a diagnostic to expose the gap**. All headline numbers
> in this project (U-Net macro-F1 **0.926**, RF **0.897**) are reported on the
> spatial-block **test** split against held-out truth.
>
> **Honesty note.** The demo AOI uses *synthetic* pixel values. The CRS,
> affine transform, tile geometry, block assignment, centroid distances and area
> maths are all real, and the same code runs on real
> [DEA Sentinel-2 ARD](https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/)
> and [WA government vectors](https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent).
> The synthetic data was designed to reproduce the *failure mode* of real
> imagery (managed-green agriculture spectrally confused with native vegetation),
> not to produce flattering scores.

Source code referenced throughout:

- `src/ecorehab/labels/tiling.py` — tile generation, `spatial_block_split`, `random_split`, `build_tile_index`
- `src/ecorehab/evaluation/spatial_cv.py` — `assign_kfold_blocks`, `nearest_train_distance`, `summarise_split_comparison`
- `src/ecorehab/evaluation/metrics.py` — `segmentation_metrics` (per-class precision/recall/F1/IoU + aggregates)
- Figure: `outputs/examples/02_labels_and_split.png` (train/val/test block map)

---

## 1. The problem: spatial autocorrelation

Earth observation data violates the assumption that most ML evaluation quietly
relies on: that the test set is independent of the training set. Two facts make
satellite pixels strongly dependent on their neighbours:

1. **The ground is continuous.** A patch of native vegetation, a paddock, or a
   river does not stop at a pixel boundary. Adjacent 10 m Sentinel-2 pixels
   usually belong to the same land cover and have nearly identical spectra.
2. **The sensor blurs.** The point-spread function and atmospheric scattering
   smear signal across neighbouring pixels, so even at a true land-cover edge the
   transition is gradual rather than crisp.

The consequence is **spatial autocorrelation**: the value at one location is
predictable from nearby locations (Tobler's first law of geography — "everything
is related to everything else, but near things are more related than distant
things"). Two tiles that are neighbours are, statistically, close to duplicates.

This is not a nuisance to be cleaned away — it is the structure of the data, and
it is exactly what an honest validation design has to respect.

---

## 2. Why a naive random split leaks

The natural thing to do is shuffle all tiles and randomly assign 60/20/20 to
train/val/test. With spatially autocorrelated data this **leaks**:

- A tile placed in *test* very often has an immediate neighbour placed in
  *train*. Because neighbours are near-duplicates, the model has effectively
  been trained on the test tile.
- The test score then measures **memorisation / interpolation between known
  neighbours**, not the thing we actually care about: **generalisation to a new
  place the model has never seen**.
- The result is a systematic **optimistic bias**. Every metric — accuracy,
  F1, IoU, calibration — looks better than the model deserves, and the bias is
  largest for exactly the spatially clustered rare classes (water, native
  remnants) that matter most ecologically.

The deployment question for an ecological monitoring product is *"how will this
model do over the next tenement / catchment we have never labelled?"* A random
split cannot answer that question. It answers a different, easier one and quietly
substitutes the answer.

`random_split` (`tiling.py`) is implemented faithfully and kept in the codebase
**on purpose** — not because it is a defensible way to grade the model, but
because contrasting it with the spatial split is the most convincing way to
*demonstrate* the leakage to a reviewer.

```python
def random_split(n_tiles, val_fraction=0.2, test_fraction=0.2, seed=...):
    """Naive per-tile random split (for the leakage comparison, NOT primary use)."""
```

The docstring states the intent explicitly, and it is a core project
convention: *the spatial split is primary; train/val/test come from spatial
blocks, never a naive random split (which leaks).*

---

## 3. The three split strategies (weakest to strongest)

| Strategy | Unit assigned to a split | Independence of test set | Role in this repo |
|---|---|---|---|
| **Random tile split** | individual tile | none — neighbours leak across splits | diagnostic only, to expose the gap |
| **Spatial-block split** | whole block of contiguous tiles | good — adjacency contained inside a block | **PRIMARY validation** |
| **AOI / site holdout** | a whole region / tenement | strongest — different landscape entirely | best option; needs ≥2–3 AOIs |

The three configs `configs/aoi_swan_coastal_plain.yaml`,
`configs/aoi_wheatbelt.yaml`, and `configs/aoi_mining_proxy.yaml` are the basis
for AOI holdout when real multi-AOI data is wired in.

### 3.1 Why block beats random

A spatial-block split keeps a *buffer of guaranteed independence* between train
and test: because an entire block goes to one split, every test tile is at least
one block (256 px = 2.56 km here) away from any *other-split* boundary in the
direction of its own block, and most test tiles are surrounded by same-split
neighbours. Leakage can now only occur at block edges, and is far smaller than
the wall-to-wall leakage of a random split.

### 3.2 Why AOI/site holdout is strongest (and not yet exercised here)

Block splitting still trains and tests within the *same landscape*, so it cannot
catch failures that come from a genuinely new region — different soils, phenology,
sun angle, atmospheric state, or sensor-processing baseline. **AOI/site holdout**
trains on some AOIs and tests on entirely separate ones, which is the closest
proxy to real deployment. It is the recommended target once multiple real AOIs
are ingested; the demo ships a single 7.68 km AOI, so the demo's primary
reported validation is the block split.

---

## 4. Spatial-block split implementation

The mechanism is deliberately simple and auditable.

### 4.1 Tiling

The AOI is **768 × 768 px @ 10 m** (7.68 km square,
[EPSG:3577](https://epsg.io/3577) Australian Albers, equal-area, metres). With
`tile_size = 128` and `stride = 128` (no overlap for training; see
`configs/aoi_swan_coastal_plain.yaml`), `generate_tiles` lays down a 6 × 6 grid
= **36 candidate tiles**. The last row/column is shifted inward so every tile is
exactly 128 px with no ragged edges.

`build_tile_index` then drops any tile whose **valid fraction <
`min_valid_fraction` (0.5)** — i.e. tiles that are mostly nodata (the NaN corner)
or `ignore=255`. That removes 3 tiles, leaving **33 tiles**.

### 4.2 Blocking and assignment

`spatial_block_split` maps each tile to the coarse **block** it falls in
(`block_size_px = 256`), then assigns *whole blocks* to a split:

```python
block_ids = [(t.row_off // block_size_px, t.col_off // block_size_px) for t in tiles]
unique_blocks = sorted(set(block_ids))
perm = rng.permutation(len(unique_blocks))      # deterministic: seed=42
# rank < n_test -> "test"; next n_val -> "val"; remainder -> "train"
```

Each 256 px block contains a 2 × 2 patch of 128 px tiles, so the 768 px AOI has a
3 × 3 grid = **9 blocks**. With `test_fraction = val_fraction = 0.2`:

- `n_test = max(1, round(0.2 × 9)) = 2` blocks
- `n_val  = max(1, round(0.2 × 9)) = 2` blocks (clamped to keep ≥1 train block)
- `n_train = 5` blocks

Because every tile inherits its **block's** label, **no block straddles two
splits** — that is the whole point. After the `min_valid_fraction` filter, the
verified tile counts are:

| Split | Blocks | Tiles |
|---|---:|---:|
| train | 5 | **18** |
| val   | 2 | **7** |
| test  | 2 | **8** |
| **total** | **9** | **33** |

The assignment is deterministic (`seed = 42`), so the split is reproducible and
can be drawn on a map (`outputs/examples/02_labels_and_split.png`) and audited:
every tile in the index carries its CRS bounds plus `block_row`/`block_col`,
which is what makes the leakage diagnostics below possible.

### 4.3 The tile index, not thousands of files

Tiles are stored as a **GeoDataFrame index** of windows + split assignment, not
as exported chips. The training dataset reads pixel windows directly from the
composite/label GeoTIFFs. This keeps the split transparent (one table you can
plot and inspect) and avoids silently baking a stale split into duplicated files.

---

## 5. Leakage diagnostic: nearest-train distance

`nearest_train_distance` (`spatial_cv.py`) quantifies the leakage directly. For
each evaluation tile it computes the Euclidean distance from its centroid to the
nearest **train** tile centroid, in CRS units (metres for EPSG:3577):

```python
dists = np.sqrt(((ev_pts[:, None, :] - train_pts[None, :, :]) ** 2).sum(axis=2))
return dists.min(axis=1)
```

How to read it:

- **Random split** → a histogram piled up at small distances (often one tile
  width, ~1.28 km, or less). Many eval tiles sit immediately next to a train
  tile. That short distance *is* the leakage, made visible.
- **Spatial-block split** → the distribution shifts right; the minimum
  separation is bounded below by the block geometry, so the near-zero spike
  disappears.

This is the cleanest single picture for explaining why the two splits give
different scores: the gap in metrics is explained by the gap in train/test
proximity. (Plotted in the spatial-validation notebook.)

---

## 6. Spatial k-fold cross-validation

A single block test split uses only ~8 tiles, so the test estimate is noisy.
`assign_kfold_blocks` provides **spatial k-fold**: it assigns a fold id to each
*block* (not each tile), so all tiles in a block share a fold and folds remain
spatially disjoint.

```python
blocks = list(tile_index.groupby(["block_row", "block_col"]).groups.keys())
fold_of_block = {b: int(rng.integers(0, k)) for b in blocks}
```

Rotating the held-out fold gives a distribution of scores instead of one number,
which is the honest way to report performance on a small AOI. Note the practical
ceiling: with only **9 blocks**, large `k` leaves very few blocks per fold, so
fold estimates are high-variance — another reason real deployment wants more AOIs
and AOI-level holdout rather than squeezing more folds out of one small scene.

---

## 7. Comparing random vs spatial metrics

`summarise_split_comparison` tabulates headline metrics across split strategies
for a side-by-side read:

```python
results = {"random": random_metrics, "spatial_block": spatial_metrics}
summarise_split_comparison(results)
# -> columns: split, macro_f1, mean_iou, weighted_f1, overall_accuracy, balanced_accuracy
```

Each `*_metrics` dict comes from `segmentation_metrics` (label vs prediction,
`ignore=255` removed). The protocol:

1. Build **two tile indices from the same tiles** — one with
   `split_method="random"`, one with `"spatial_block"` (both `seed=42`).
2. Train (or evaluate the same trained model) and score on each split's test
   tiles.
3. Compare. The **expected, honest direction is that the random split scores
   higher** than the spatial split, and the difference is the optimism that
   spatial validation removes.

The repository **reports the spatial-block result as the model's performance**;
the random-split number is presented next to it strictly as a cautionary
contrast, never as the headline. (The exact random-split figures are produced by
the spatial-validation notebook on each run and are not hard-coded here, to avoid
implying a fixed result.)

---

## 8. Results on the spatial-block test split

These are the **primary, honestly-validated** numbers: evaluated on the
spatial-block **test** split (8 tiles, 2 held-out blocks) against the hidden
`truth_labels.tif`, which is used **only** for evaluation and never for training.
Class schema: `0 = disturbed_or_cleared`, `1 = native_remnant_vegetation`,
`2 = water_or_wetland`.

### 8.1 Aggregate metrics

| Metric | RandomForest (300 trees) | U-Net (from scratch, 7.77M params) |
|---|---:|---:|
| macro-F1 | 0.897 | **0.926** |
| mean IoU | 0.814 | **0.866** |
| weighted-F1 | 0.908 | 0.917 |
| balanced accuracy | 0.895 | **0.954** |
| overall accuracy | 0.906 | 0.915 |

The U-Net (GroupNorm, CE + Dice loss, inverse-frequency class weights, early stop
at epoch 19, best val macro-F1 0.988) beats the RandomForest baseline on macro-F1,
mean IoU, and balanced accuracy. GroupNorm is used deliberately: with few small
batches, BatchNorm running statistics are unreliable and flatten the softmax;
GroupNorm keeps probabilities peaked (max-prob median ~0.83, entropy median ~0.52).

### 8.2 Per-class metrics

**RandomForest**

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| disturbed | 0.98 | 0.88 | 0.93 | 0.86 |
| native | **0.78** | 1.00 | 0.87 | 0.78 |
| water | 1.00 | 0.81 | 0.89 | 0.81 |

**U-Net**

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| disturbed | 1.00 | 0.87 | 0.93 | 0.87 |
| native | **0.78** | 0.99 | 0.87 | 0.77 |
| water | 0.96 | 0.99 | 0.98 | 0.95 |

We report **class-specific metrics, not overall accuracy** (a golden rule of the
project), because overall accuracy can look high while the ecologically
important rare classes fail.

---

## 9. The honest finding: native vegetation is over-predicted

Both models show the same signature, and it survives spatial validation rather
than being hidden by it: **native precision is only 0.78 while native recall is
~1.00**. The model finds essentially all real native vegetation, but it also
labels a lot of non-native green as native — i.e. it **over-predicts** native
vegetation.

This propagates straight into the hectare report (U-Net inference, EPSG:3577,
95% block-bootstrap CI):

| Class | Predicted area (ha) | 95% CI | Truth (ha) |
|---|---:|---:|---:|
| disturbed | 3718 | [3530, 3901] | 4285 |
| native | **1910** | **[1723, 2096]** | **1344** |
| water | 269 | [213, 339] | 249 |

**Native vegetation is over-predicted by ~42% (1910 vs 1344 ha), and the
confidence interval does not cover the truth.** The cause is a real and
well-known remote-sensing failure mode that the synthetic AOI was built to
reproduce: **managed-green agriculture is spectrally close to native vegetation
at 10 m and gets misclassified as native** (hence native precision 0.78). The
block-bootstrap CI captures sampling variability but **not** this systematic
class-confusion bias — which is precisely why we surface it in words rather than
trusting the interval alone.

This is the headline limitation of the project, and stating it plainly is the
point: a trustworthy monitoring system reports its own bias.

---

## 10. Scope and limitations of this validation

- **Synthetic pixels.** The reported scores describe model behaviour on a
  controlled synthetic AOI. They are **not** field-validated accuracy and must
  not be read as ground truth. The pipeline, geometry, and statistics are real;
  the radiometry is fabricated.
- **No species-level claims.** This is a 3-class native / disturbed / water
  *proxy*, not species mapping, not a mine-rehabilitation compliance assessment,
  and not operationally approval-ready.
- **Single AOI.** Block splitting controls *within-scene* leakage but not
  cross-region transfer. AOI/site holdout across multiple real AOIs is the
  stronger test and is the recommended next step.
- **Small test set.** ~8 test tiles; prefer the spatial k-fold distribution over
  any single split number.
- **Weak labels are weak.** Training labels are derived from buffered government
  polygons + spectral gates with an `ignore=255` band for ambiguous pixels;
  boundary rings and nodata are never guessed. See `docs/label_strategy.md`.

### What real data would change

Running with `aoi.demo: false` against real
[DEA Sentinel-2 ARD](https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/),
[WA Native Vegetation Extent](https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent)
(CC BY 4.0), and
[WA Mining Tenements DMIRS-003](https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003)
(CC BY 4.0) exercises the *same* splitting and diagnostic code. The validation
design does not change — only the data does — which is the property that makes
this a credible production-style template rather than a one-off demo.

---

## See also

- `docs/label_strategy.md` — weak-label construction, buffers, and the ignore band
- `docs/limitations.md` — full honesty register
- `outputs/examples/02_labels_and_split.png` — train/val/test block map
- `outputs/examples/03_evaluation.png`, `05_model_comparison.png` — metrics figures
