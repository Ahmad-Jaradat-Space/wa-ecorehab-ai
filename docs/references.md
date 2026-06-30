# References

A curated, annotated bibliography for **WA EcoRehab AI** — an open-data,
production-style geospatial ML pipeline for native-vegetation extent,
disturbance, and rehabilitation-*proxy* monitoring in Western Australia.

This list does two things:

1. Points to the **exact open-data products** the pipeline ingests (real mode)
   or emulates (demo mode), with licences and access URLs.
2. Grounds each **methodological choice** in the literature, with a short note
   on *where in the repo* that choice shows up and *why* it was made.

A note on honesty, repeated throughout the docs: the committed demo AOI is
**synthetic** — pixel values are fabricated, but the CRS, affine transform,
GeoPackage polygons, area arithmetic, and the entire code path are real and
designed to run unchanged on real DEA/WA data (`aoi.demo: false`). Results in
this repo are therefore a **methods demonstration**, not a validated ecological
product. No species-level classification, no mine-rehabilitation compliance, no
drone-level accuracy from Sentinel-2, and no ground-truth accuracy claims are
made without independent field validation.

---

## 1. Data products

These are the open Australian Earth-observation and government vector products
the pipeline is built around. In demo mode none are required (the synthetic AOI
stands in); in real mode (`aoi.demo: false`) they are fetched via STAC and the
WA open-data catalogue.

| Product | Role in pipeline | Licence / access | URL |
|---|---|---|---|
| **Digital Earth Australia — Sentinel-2 ARD** (`ga_s2am_ard_3`) | Primary imagery input: surface-reflectance bands used to build the 7-band + 6-index, 13-feature stack at 10 m | Open (CC BY 4.0); accessed via DEA STAC / `odc-stac` | https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/ |
| **Digital Earth Australia — Land Cover** | Auxiliary weak-label evidence and masking (water, built, agriculture context); annual Landsat-derived land cover, 1988–present | Open (CC BY 4.0) | https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover |
| **Digital Earth Australia — Fractional Cover** | Recovery / rehabilitation-proxy signal: green vegetation, non-green vegetation, bare soil fractions (Landsat, 30 m) | Open (CC BY 4.0) | https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/ |
| **WA Native Vegetation Extent** | Primary label source for the `native_remnant_vegetation` class (polygons → weak raster labels) | CC BY 4.0 | https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent |
| **WA Mining Tenements (DMIRS-003)** | AOI selection and per-tenement area summaries (the rehabilitation-proxy framing) | CC BY 4.0 | https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003 |
| **Atlas of Living Australia (ALA)** | Optional biodiversity / weed-occurrence overlay (occurrence API + downloads) | Open; see ALA terms | https://docs.ala.org.au/ |
| **TERN / GeoNadir public drone data** | Optional high-resolution module (scale-mismatch comparison vs Sentinel-2); not a core dependency | Public Australian drone mapping data, where available | https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/ |

**Repo grounding.** The weak-label schema is driven directly by these products:
the WA Native Vegetation Extent polygons define class 1
(`native_remnant_vegetation`); DEA Land Cover / Fractional Cover and
spectral gates define classes 0 (`disturbed_or_cleared`) and 2
(`water_or_wetland`); everything ambiguous becomes `ignore=255` (~15% of the
589,824 demo pixels). The deliberate spectral confuser in the synthetic AOI —
a managed-green agriculture grid — exists to reproduce the single most
important real-world failure mode of these products at 10 m: managed green
crops/pasture being spectrally close to native vegetation (see §6, the
over-prediction finding).

---

## 2. Methods — semantic segmentation, normalisation, losses, class imbalance

### Architectures

- **Ronneberger, Fischer & Brox (2015).** *U-Net: Convolutional Networks for
  Biomedical Image Segmentation.* MICCAI. https://arxiv.org/abs/1505.04597
  > The from-scratch segmentation model in `models/unet.py` is a U-Net.
  > Encoder–decoder with skip connections is a strong, explainable baseline for
  > dense prediction on small datasets — well-matched to a single-AOI,
  > weak-label setting. Our U-Net (7.77M params) reaches test **macro-F1 0.926
  > / mean IoU 0.866** on the spatial hold-out, beating the RandomForest
  > baseline (0.897 / 0.814).

- **Chen, Zhu, Papandreou, Schroff & Adam (2018).** *Encoder-Decoder with
  Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+).*
  ECCV. https://arxiv.org/abs/1802.02611
  > Atrous (dilated) convolutions enlarge the receptive field without losing
  > resolution — relevant for landscape-scale context at 10 m. Listed as a
  > planned/alternative decoder; the SegFormer path (`models/segformer.py`, via
  > `segmentation-models-pytorch`) is the modern transformer alternative.

- **Xie, Wang, Yu, Anandkumar, Alvarez & Luo (2021).** *SegFormer: Simple and
  Efficient Design for Semantic Segmentation with Transformers.* NeurIPS.
  https://arxiv.org/abs/2105.15203
  > Hierarchical transformer encoder + lightweight all-MLP decoder; strong
  > accuracy/efficiency trade-off and good robustness. Wired up as the optional
  > `models/segformer.py` (lazy `smp`/`timm` import) to show a current-practice
  > extension beyond the U-Net baseline.

### Normalisation

- **Wu & He (2018).** *Group Normalization.* ECCV.
  https://arxiv.org/abs/1803.08494
  > **This is a deliberate, load-bearing choice.** The U-Net uses GroupNorm,
  > **not** BatchNorm. With few small batches (33 tiles of 128 px;
  > 18 train / 7 val / 8 test), BatchNorm running statistics are unreliable and
  > tend to flatten the softmax. GroupNorm is batch-size-independent and keeps
  > probabilities well-shaped (max-prob median ~0.83, entropy median ~0.52),
  > which is what makes the downstream calibration and entropy-uncertainty
  > rasters meaningful — so the U-Net deliberately uses GroupNorm rather than
  > BatchNorm.

### Losses and class imbalance

- **Milletari, Navab & Ahmadi (2016).** *V-Net: Fully Convolutional Neural
  Networks for Volumetric Medical Image Segmentation* (Dice loss). 3DV.
  https://arxiv.org/abs/1606.04797
  > Introduces the (soft) Dice objective. The U-Net trains with a combined
  > **cross-entropy + Dice** loss (`models/losses.py`). Dice optimises overlap
  > directly, which helps the rarer `water_or_wetland` class (only 18,759
  > weak-label pixels) — reflected in its high test IoU (0.95).

- **Lin, Goyal, Girshick, He & Dollár (2017).** *Focal Loss for Dense Object
  Detection.* ICCV. https://arxiv.org/abs/1708.02002
  > Down-weights easy, well-classified pixels so training focuses on hard/rare
  > ones. Available in `models/losses.py` as the focal option for rare-class
  > sensitivity, complementing the inverse-frequency CE weighting actually used
  > in the headline run.

- **Sudre, Li, Vercauteren, Ourselin & Cardoso (2017).** *Generalised Dice
  Overlap as a Deep Learning Loss Function for Highly Unbalanced
  Segmentations.* DLMIA. https://arxiv.org/abs/1707.03237
  > Background reading for class-imbalance-aware overlap losses; motivates the
  > class-weighting / Dice combination over plain accuracy-driven training.

**Repo grounding for imbalance.** Beyond the loss, imbalance is handled with
**inverse-frequency class weights** in CE, an `ignore=255` mask for ambiguous
pixels, and a `min_valid_fraction = 0.5` filter that drops mostly-nodata/ignore
tiles. Evaluation never leans on overall accuracy alone — class-specific
precision/recall/F1/IoU are the primary metrics (see §5).

---

## 3. Spatial validation / spatial ML

- **Roberts, Bahn, Ciuti, Boyce, Elith, Guillera-Arroita, Hauenstein,
  Lahoz-Monfort, Schröder, Thuiller, Warton, Wintle, Hartig & Dormann (2017).**
  *Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure.* Ecography 40(8): 913–929.
  https://doi.org/10.1111/ecog.02881
  > Canonical argument that random CV over spatially autocorrelated data gives
  > optimistic, leakage-inflated scores, and that **block / spatial CV** is
  > required for honest generalisation estimates. This is the primary
  > justification for our spatial-block split (block = 256 px) over a naive
  > random split — see `labels/tiling.py` and `evaluation/spatial_cv.py`.

- **Ploton, Mortier, Réjou-Méchain, Barbier, Picard, Rossi, Dormann, Cornu,
  Viennois, Bayol, Lyapustin, Gourlet-Fleury & Pélissier (2020).** *Spatial
  validation reveals poor predictive performance of large-scale ecological
  mapping models.* Nature Communications 11: 4540.
  https://doi.org/10.1038/s41467-020-18321-9
  > Direct remote-sensing/ecology evidence that ignoring spatial
  > autocorrelation can make a model look far better than it is. Motivates
  > treating the **spatial split as primary** (a core project convention)
  > and reporting it as the headline number, not the random split.

- **Valavi, Elith, Lahoz-Monfort & Guillera-Arroita (2019).** *blockCV: An R
  package for generating spatially or environmentally separated folds for
  k-fold cross-validation of species distribution models.* Methods in Ecology
  and Evolution 10(2): 225–232. https://doi.org/10.1111/2041-210X.13107
  > Practical reference for constructing spatial blocks; conceptually mirrors
  > our 256-px block assignment that keeps adjacent, autocorrelated tiles in the
  > same fold to prevent train/test bleed.

- **Karasiak, Dejoux, Monteil & Sheeren (2022).** *Spatial dependence between
  training and test sets: another pitfall of classification accuracy assessment
  in remote sensing.* Machine Learning 111: 2715–2740.
  https://doi.org/10.1007/s10994-021-05972-1
  > Reinforces, specifically for remote-sensing classification, that spatially
  > dependent train/test samples inflate reported accuracy — the exact pitfall
  > the block split is designed to avoid.

**Repo grounding.** Tiling and split assignment live in `labels/tiling.py`;
spatial-CV utilities in `evaluation/spatial_cv.py`. The demo `truth_labels.tif`
is used **only** for honest evaluation on the spatial hold-out and never for
training.

---

## 4. Calibration & uncertainty

- **Guo, Pleiss, Sun & Weinberger (2017).** *On Calibration of Modern Neural
  Networks.* ICML. https://arxiv.org/abs/1706.04599
  > Shows modern nets are often miscalibrated and introduces **Expected
  > Calibration Error (ECE)** with reliability diagrams as the standard
  > diagnostics. Implemented in `evaluation/calibration.py`. Our GroupNorm
  > choice (§2) is partly about keeping probabilities trustworthy enough for
  > these diagnostics to be meaningful (peaked but not degenerate: max-prob
  > median ~0.83).

- **Brier (1950).** *Verification of forecasts expressed in terms of
  probability.* Monthly Weather Review 78(1): 1–3.
  https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
  > The Brier score — mean squared error of probabilistic predictions — is one
  > of our probability-quality metrics alongside ECE, reported per the
  > spec's evaluation requirements.

- **Shannon (1948).** *A Mathematical Theory of Communication.* Bell System
  Technical Journal 27: 379–423, 623–656.
  https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
  > Predictive **entropy** of the per-pixel softmax is the basis for the
  > `uncertainty.tif` raster (`evaluation/uncertainty.py`); higher entropy
  > flags low-confidence areas for human review (entropy median ~0.52 in the
  > demo run).

- **Gal & Ghahramani (2016).** *Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning.* ICML.
  https://arxiv.org/abs/1506.02142
  > Background for model (epistemic) uncertainty via MC-dropout; our current
  > uncertainty is the simpler single-pass predictive entropy, with MC-dropout
  > noted as a natural extension.

**Repo grounding.** Calibration metrics (ECE, Brier, reliability diagram) live
in `evaluation/calibration.py`; entropy-based uncertainty in
`evaluation/uncertainty.py`, exported as `uncertainty.tif`.

---

## 5. Area estimation & accuracy assessment in remote sensing

- **Olofsson, Foody, Herold, Stehman, Woodcock & Wulder (2014).** *Good
  practices for estimating area and assessing accuracy of land change.* Remote
  Sensing of Environment 148: 42–57.
  https://doi.org/10.1016/j.rse.2014.02.015
  > The standard reference for **honest area reporting**: report a confusion
  > matrix, class-specific accuracies, and area estimates **with uncertainty**,
  > and recognise that a naive pixel-count of a classified map is a *biased*
  > area estimator. We follow the spirit of this guidance: area is computed in
  > an equal-area CRS (EPSG:3577) and reported with **95% block-bootstrap
  > confidence intervals** (`evaluation/area_estimation.py`), and we surface the
  > native-vegetation bias explicitly (§6) rather than hiding it.

- **Olofsson, Foody, Stehman & Woodcock (2013).** *Making better use of
  accuracy data in land change studies: Estimating accuracy and area and
  quantifying uncertainty using stratified estimation.* Remote Sensing of
  Environment 129: 122–131. https://doi.org/10.1016/j.rse.2012.10.031
  > Companion paper formalising stratified, design-based area estimation and its
  > confidence intervals — the statistical backdrop for treating predicted
  > hectares as estimates-with-error, not exact truths.

- **Stehman & Foody (2019).** *Key issues in rigorous accuracy assessment of
  land cover products.* Remote Sensing of Environment 231: 111199.
  https://doi.org/10.1016/j.rse.2019.05.018
  > Synthesises good-practice accuracy assessment; reinforces reporting
  > class-specific metrics and uncertainty over a single global accuracy figure.

- **Efron & Tibshirani (1993).** *An Introduction to the Bootstrap.* Chapman &
  Hall. https://doi.org/10.1201/9780429246593
  > Methodological basis for the **block-bootstrap** confidence intervals on
  > area. Blocking (rather than i.i.d. resampling) respects spatial
  > autocorrelation among pixels/tiles, consistent with §3.

**Repo grounding — the headline honest finding.** On the spatial-block test
split, the U-Net **over-predicts native vegetation: 1910 ha [1723, 2096]
vs 1344 ha truth (+42%), and the 95% CI does not cover truth.** The cause is
spectral confusion between managed-green agriculture and native vegetation
(native **precision 0.78**, recall ~0.99–1.00). Disturbed area is
under-predicted (3718 ha [3530, 3901] vs 4285 ha truth) and water is close
(269 ha [213, 339] vs 249 ha truth). This is exactly the kind of class-specific,
uncertainty-aware reporting Olofsson et al. argue for — and it is the project's
central limitation, stated plainly rather than smoothed over.

---

## 6. Geospatial tooling & standards

- **rasterio** — raster I/O on top of GDAL.
  https://rasterio.readthedocs.io/
  > All raster reads/writes go through `utils/io.write_raster`, which preserves
  > CRS, affine transform, nodata, and writes tiled+compressed GeoTIFFs.
  > Windowed reads power the tiled/overlap inference in
  > `inference/tile_inference.py`.

- **GeoPandas** — vector dataframes (Shapely + pyogrio/Fiona + pyproj).
  https://geopandas.org/
  > Used to read/reproject WA polygons, rasterize labels, and write the
  > polygonised `predicted_polygons.gpkg` and per-tenement area CSVs.

- **GDAL/OGR** — the underlying geospatial data-abstraction library.
  https://gdal.org/
  > The C/C++ engine beneath rasterio and Fiona/pyogrio; provides the
  > read/write drivers (GeoTIFF, GeoPackage) and warping the pipeline relies on.

- **Shapely** — geometric operations (buffering, erosion, area).
  https://shapely.readthedocs.io/
  > Powers the boundary-noise handling: inward (erosion) buffer of 20 m on
  > positive polygons, a 20 m boundary-exclusion ring, and a 0.5 ha minimum
  > polygon area before rasterization (`labels/weak_labels.py`,
  > `labels/rasterize.py`).

- **pyproj / EPSG:3577 (GDA94 Australian Albers, equal-area)** — projection &
  CRS transforms.
  https://pyproj4.github.io/pyproj/ — EPSG registry: https://epsg.io/3577
  > **A golden rule:** all labels, tiling, and hectare reporting are done in
  > **EPSG:3577**, an equal-area projection in metres. Area is *never* computed
  > in EPSG:4326 (degrees), which would bias hectares. CRS/area helpers live in
  > `utils/crs.py`.

- **STAC (SpatioTemporal Asset Catalog) specification** — standard for
  indexing/querying EO assets.
  https://stacspec.org/
  > DEA exposes Sentinel-2 ARD via STAC; real-mode discovery uses
  > `pystac-client` against the DEA endpoint (see the DEA STAC how-to in §1).

- **odc-stac** (Open Data Cube) — load STAC items into `xarray` datasets.
  https://odc-stac.readthedocs.io/
  > Lazy import in `data/stac.py`; turns STAC query results into analysis-ready
  > arrays for `data/build_composites.py` in real mode (`.[rs]` extra).

- **Open Data Cube / `datacube`** — the broader framework DEA is built on.
  https://www.opendatacube.org/
  > Context for the DEA data model and the odc-stac loaders; not a hard
  > dependency of the demo path.

**Repo grounding.** Optional heavy dependencies (`odc-stac`, `pystac-client`,
torch beyond core, `smp`/`timm`, `xgboost`/`lightgbm`) are **lazy imports** with
clear install hints, so the offline demo and the 50-test pytest suite run
without them. The pipeline is validated on Python 3.10+ (and on 3.14 with
NumPy 2.5, rasterio 1.5, PyTorch 2.12, GeoPandas 1.1).

---

## How these references map to the repo

| Decision in the repo | Grounded in |
|---|---|
| EPSG:3577 for all area maths | pyproj / EPSG:3577; Olofsson et al. (2014) |
| Weak labels + `ignore=255` from gov polygons | §1 data products; spec §6 |
| Spatial-block split as primary validation | Roberts et al. (2017); Ploton et al. (2020); Karasiak et al. (2022) |
| U-Net from scratch | Ronneberger et al. (2015) |
| SegFormer / DeepLabV3+ as extensions | Xie et al. (2021); Chen et al. (2018) |
| GroupNorm instead of BatchNorm | Wu & He (2018) |
| CE + Dice (+ optional focal), inverse-freq weights | Milletari et al. (2016); Lin et al. (2017); Sudre et al. (2017) |
| ECE + reliability + Brier + entropy uncertainty | Guo et al. (2017); Brier (1950); Shannon (1948) |
| Hectares with 95% block-bootstrap CIs | Olofsson et al. (2013, 2014); Efron & Tibshirani (1993) |
| rasterio/GeoPandas/GDAL/Shapely tooling, STAC ingest | §6 tooling references |

---

*Last reviewed: 2026-06-30. URLs accessed at time of writing; DOIs are stable.*
