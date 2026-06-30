# Data Sources

This document is the authoritative reference for every external dataset that
`wa-ecorehab-ai` consumes (or is wired to consume), how the repository uses it,
how it is accessed, and under what licence. It also documents the exact band
mapping the code applies to Digital Earth Australia (DEA) Sentinel-2, and the
difference between the offline **demo mode** and the networked **real mode**.

> **Read this first — honesty note.** The committed demo runs entirely offline on
> a **synthetic** Area of Interest (AOI). The *pixel values are fabricated*; the
> CRS, affine transform, nodata handling, polygon geometries, rasterization, and
> hectare maths are all **real**. None of the open datasets below are downloaded
> in demo mode. They are the datasets the pipeline is built to ingest in real
> mode (`aoi.demo: false`), and the access paths in `src/ecorehab/data/` are the
> real endpoints. Treat any quantitative result in this repo as a
> *methods/architecture demonstration on synthetic data*, not as a measurement of
> real Western Australian vegetation. See [Demo mode vs real mode](#demo-mode-vs-real-mode).

---

## Quick reference

| Dataset | Role in this repo | Access method | Licence | Canonical URL |
|---|---|---|---|---|
| DEA Sentinel-2 ARD (`ga_s2am_ard_3`) | Primary 10 m multispectral imagery input | STAC (`pystac-client` + `odc-stac`) | CC BY 4.0 | https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/ |
| DEA Land Cover | Weak-label support: water/built/agriculture masking, change context | STAC / DEA products | CC BY 4.0 | https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover |
| DEA Fractional Cover | Recovery proxy: green / non-green / bare-soil fractions | STAC / DEA products | CC BY 4.0 | https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/ |
| WA Native Vegetation Extent | Primary positive-label source (native/remnant polygons) | WA SLIP / catalogue (ArcGIS REST / WFS) | CC BY 4.0 | https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent |
| WA Mining Tenements (DMIRS-003) | AOI selection + per-tenement area reporting overlay | WA SLIP / catalogue (ArcGIS REST / WFS) | CC BY 4.0 | https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003 |
| Atlas of Living Australia (ALA) | Optional biodiversity / weed occurrence overlay | ALA occurrence API / downloads | CC BY 4.0 (attribution; check per-record) | https://docs.ala.org.au/ |
| TERN / GeoNadir drone data | Optional high-resolution orthomosaic module (scale comparison) | Public catalogue download | CC BY (per-dataset; verify) | https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/ |

Only the first five rows are needed to reproduce the core pipeline. ALA and
TERN/GeoNadir are optional extensions described in the project spec and are not
exercised by the committed demo.

---

## 1. DEA Sentinel-2 ARD — `ga_s2am_ard_3`

**What it is.** Digital Earth Australia's Analysis Ready Data (ARD) product for
Sentinel-2A, delivered as surface-reflectance (NBART — Nadir BRDF-Adjusted
Reflectance with Terrain correction) at 10 m. It is the standard open,
cloud-screened, terrain-corrected Sentinel-2 source for Australia. (The
companion Sentinel-2B collection is `ga_s2bm_ard_3`; the config defaults to
`ga_s2am_ard_3` and can be changed via `data.collection`.)

**How this repo uses it.** It is the **primary imagery input**. In real mode the
pipeline searches DEA STAC for items over the AOI bounding box and date range,
loads the seven configured bands via `odc-stac`, reprojects to the project CRS
(EPSG:3577), and takes a **median composite over time** to suppress cloud and
acquisition noise. From those bands the feature stack is built: 7 surface
reflectance bands + 6 spectral indices = **13 features** (see
[Band and feature stack](#band-and-feature-stack)).

**Access method.** STAC API via the optional `[rs]` extra
(`pystac-client` + `odc-stac`). The relevant code is
`src/ecorehab/data/stac.py`:

- STAC endpoint (`data.stac_url`): `https://explorer.dea.ga.gov.au/stac`
- Collection (`data.collection`): `ga_s2am_ard_3`
- Default window (`data.date_start`/`data.date_end`): `2023-01-01` .. `2023-03-31`
  (a late-summer composite, deliberately chosen so that senescent/managed-green
  agriculture is *less* likely to be confused with native vegetation — see
  the [limitation](#known-failure-mode-agriculture-vs-native) below)
- Cloud filter (`data.max_cloud_cover`): items with `eo:cloud_cover < 20%`
- Composite (`data.composite`): `median` over the time dimension

If the `[rs]` extras or the network are unavailable, the code raises a clear
`ImportError`/`RuntimeError` with an install hint rather than corrupting the
offline demo path.

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0),
© Commonwealth of Australia (Geoscience Australia / Digital Earth Australia).

**Canonical URL.** https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/

### DEA S2 asset-name → canonical band mapping

DEA names its Sentinel-2 assets with an `nbart_` prefix. This repo maps those
asset names onto its own canonical band order. The mapping is the single source
of truth in `src/ecorehab/data/stac.py` (`DEA_S2_ASSETS`):

| Canonical band (this repo) | DEA STAC asset | Sentinel-2 band | Nominal centre |
|---|---|---|---|
| `blue` | `nbart_blue` | B02 | ~490 nm |
| `green` | `nbart_green` | B03 | ~560 nm |
| `red` | `nbart_red` | B04 | ~665 nm |
| `rededge` | `nbart_red_edge_1` | B05 | ~705 nm |
| `nir` | `nbart_nir_1` | B08 | ~842 nm |
| `swir1` | `nbart_swir_2` | B11 | ~1610 nm |
| `swir2` | `nbart_swir_3` | B12 | ~2190 nm |

Notes on the mapping (intentional, easy to misread):

- The repo's `swir1` maps to DEA's `nbart_swir_2` (B11) and `swir2` maps to
  `nbart_swir_3` (B12). DEA's asset suffixes follow its own SWIR numbering, not
  the repo's `swir1/swir2` labels — do not "fix" this to `nbart_swir_1`.
- Only these seven bands are loaded; the asset list passed to `odc-stac` is
  derived from `data.bands` in order, so changing `data.bands` changes both the
  download and the downstream feature stack.

### Band and feature stack

The 13-dimensional per-pixel feature vector used by every model:

| # | Feature | Type | Source |
|---|---|---|---|
| 1–7 | `blue, green, red, rededge, nir, swir1, swir2` | Surface reflectance | DEA S2 ARD bands above |
| 8 | `ndvi` | Index | (nir − red) / (nir + red) |
| 9 | `evi` | Index | enhanced vegetation index |
| 10 | `ndwi` | Index | (green − nir) / (green + nir) |
| 11 | `mndwi` | Index | modified NDWI (green, swir1) |
| 12 | `nbr` | Index | normalised burn ratio (nir, swir2) |
| 13 | `bsi` | Index | bare soil index |

Indices are computed in `src/ecorehab/features/indices.py`. In demo mode these
are computed from the synthetic bands; in real mode from the DEA composite.

---

## 2. DEA Land Cover

**What it is.** A continental, annual land-cover product derived from the Landsat
archive (covering roughly 1988 to present), providing a consistent categorical
land-cover classification for Australia.

**How this repo uses it.** As **weak-label support**, not as a model input. The
label strategy (see `docs/label_strategy.md`) uses land cover to help mask out
water, built-up areas, and obvious agriculture when deciding which out-of-polygon
pixels are credible "disturbed/cleared" negatives, and to provide multi-year
change context for the AOI. In the committed 3-class demo the equivalent masking
is produced from synthetic spectral gates rather than a downloaded land-cover
raster; the architecture supports substituting the real product in real mode.

**Access method.** DEA STAC / DEA product downloads (same DEA infrastructure as
the Sentinel-2 ARD).

**Licence.** CC BY 4.0, © Commonwealth of Australia (Geoscience Australia / DEA).

**Canonical URL.** https://www.ga.gov.au/scientific-topics/dea/dea-data-and-products/dea-land-cover

---

## 3. DEA Fractional Cover

**What it is.** A Landsat-derived product (~30 m, Australia-wide time series)
that decomposes each pixel into three fractions: **photosynthetic vegetation
(PV / green)**, **non-photosynthetic vegetation (NPV / non-green)**, and **bare
soil (BS)**.

**How this repo uses it.** As the basis for the **rehabilitation/recovery
proxy** and as an optional auxiliary feature. Increasing green fraction and
decreasing bare-soil fraction over disturbed ground is the open-data signal used
to *proxy* vegetation recovery — it is explicitly a proxy, not a measured
rehabilitation outcome. In the demo this signal is illustrated with synthetic
data; the real product would be ingested in real mode.

**Access method.** DEA STAC / DEA product downloads.

**Licence.** CC BY 4.0, © Commonwealth of Australia (Geoscience Australia / DEA).

**Canonical URL.** https://knowledge.dea.ga.gov.au/notebooks/DEA_products/DEA_Fractional_Cover/

---

## 4. WA Native Vegetation Extent

**What it is.** Western Australian Government open spatial data delineating the
extent of native/remnant vegetation as polygons.

**How this repo uses it.** It is the **primary positive-label source**. Native
vegetation polygons are reprojected to EPSG:3577, then *eroded inward* (a 20 m
positive inward buffer) and surrounded by a 20 m boundary-exclusion ring before
rasterization, so that ambiguous edge pixels become `ignore` (255) rather than
noisy positives. Only polygons above `min_polygon_area_ha = 0.5` are used. This
is the layer that defines class `1 = native_remnant_vegetation` in the weak-label
schema. (In the demo, 24 *synthetic* native-vegetation polygons stand in for this
layer.)

**Access method.** WA **SLIP** (Shared Location Information Platform) public
services / the WA data catalogue. The code reads the published ArcGIS REST /
WFS-style feature service with an AOI bbox filter and reprojects to the project
CRS. See `src/ecorehab/data/download_vectors.py` (`WA_NATIVE_VEG_URL`, base
service `https://services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Environment/MapServer`).

> The exact ArcGIS layer ids on the SLIP services change over time. The
> **canonical catalogue page below is the source of truth**; confirm the current
> layer id / endpoint there before a real-mode run, and update the config
> accordingly.

**Licence.** CC BY 4.0, © Government of Western Australia.

**Canonical URL.** https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent

---

## 5. WA Mining Tenements (DMIRS-003)

**What it is.** Western Australian mining-tenement boundary polygons published by
the Department of Energy, Mines, Industry Regulation and Safety (DMIRS), dataset
code DMIRS-003.

**How this repo uses it.** Two roles: (a) **AOI selection** — picking
disturbed/resource landscapes to monitor as a rehabilitation *proxy*; and (b)
**per-tenement area reporting** — predicted class areas are summarised against
tenement polygons to produce `area_summary_by_polygon.csv`. The repo does **not**
claim any mine-rehabilitation compliance assessment; tenements are used purely as
public spatial context. (In the demo, 2 *synthetic* tenement polygons stand in
for this layer.)

**Access method.** WA SLIP public services / catalogue (ArcGIS REST / WFS), AOI
bbox filtered, reprojected to EPSG:3577. See
`src/ecorehab/data/download_vectors.py` (`WA_TENEMENTS_URL`, base service
`https://services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/Mining/MapServer`).
As above, confirm the current layer id at the catalogue page before running.

**Licence.** CC BY 4.0, © Government of Western Australia.

**Canonical URL.** https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003

---

## 6. Atlas of Living Australia (ALA) — optional

**What it is.** Australia's national biodiversity database, aggregating species
occurrence records with an open API.

**How this repo uses it.** As an **optional** overlay: a biodiversity / weed
occurrence-risk layer for selected invasive or conservation-relevant taxa, to
contextualise predictions. It is not part of the core pipeline and is not
exercised by the demo.

**Access method.** ALA occurrence API and bulk downloads (see ALA developer
docs).

**Licence.** Generally CC BY 4.0 with required attribution to ALA and to the
underlying data providers; individual records carry their own licences and
sensitivity flags. **Check per-record licensing** before redistribution.

**Canonical URL.** https://docs.ala.org.au/

---

## 7. TERN / GeoNadir drone data — optional

**What it is.** Publicly released Australian drone/UAV orthomosaics (high
spatial resolution RGB / multispectral) via TERN and the GeoNadir community
platform.

**How this repo uses it.** As an **optional high-resolution module**: read one
orthomosaic, derive a simple vegetation-cover mask (e.g. ExG/VARI for RGB), and
compare canopy/green cover against Sentinel-2 NDVI or DEA Fractional Cover over
the same area to discuss the **scale mismatch** between 10 m satellite pixels and
centimetre-scale drone imagery. Optional; not in the core build, not in the demo.

> Honesty note: this module is for *scale-comparison and discussion only*. It does
> **not** transfer drone-level accuracy to the Sentinel-2 models.

**Access method.** Public catalogue download (per-dataset).

**Licence.** Typically Creative Commons (often CC BY), but **per-dataset** —
verify the licence of any specific orthomosaic before use.

**Canonical URL.** https://www.tern.org.au/news/national-research-drone-data-available-to-the-public/

---

## Demo mode vs real mode

The selector is the **`aoi.demo`** flag in the AOI config
(`configs/aoi_swan_coastal_plain.yaml`).

### Demo mode (`aoi.demo: true`) — the committed default

- **No network and no open data are touched.** `download_vectors` and the
  composite builder call into `src/ecorehab/data/demo.py` to generate a
  deterministic synthetic scene.
- The AOI is a **768 × 768 px @ 10 m = 7.68 km square** over the Swan Coastal
  Plain (centre ≈ 115.86°E, 32.05°S), in **EPSG:3577**.
- **Pixel values are fabricated.** What is *real*: the CRS, the affine transform,
  a NaN nodata corner (~0.4% of pixels), and GeoPackage polygon geometries.
- Synthetic vectors: **24 native-vegetation polygons** and **2 mining tenements**,
  plus a sinuous river, water blobs, and — deliberately — a **managed-green
  agriculture grid** included as a spectral *confuser* to stress-test the
  native-vs-agriculture failure mode.
- A `truth_labels.tif` exists **only for honest evaluation** and is never used for
  training.

This is what lets the full pipeline (`make pipeline`) and the 50-test suite run
offline and reproducibly.

### Real mode (`aoi.demo: false`)

- Requires the optional `[rs]` extras (`pystac-client`, `odc-stac`, geopandas
  vector I/O) and network access.
- Sentinel-2 ARD is pulled via DEA STAC (`stac.py`); WA Native Vegetation Extent
  and Mining Tenements are pulled from WA SLIP services (`download_vectors.py`),
  AOI-bbox filtered and reprojected to EPSG:3577.
- DEA Land Cover and Fractional Cover slot in as weak-label support / recovery
  proxy.
- Implemented but **not run in this environment** — it needs live endpoints and
  the optional extras. If anything is missing the code fails loudly with an
  install/endpoint hint rather than silently degrading.

| Aspect | Demo mode | Real mode |
|---|---|---|
| `aoi.demo` | `true` | `false` |
| Network needed | No | Yes |
| Optional extras | No | `[rs]` (`pystac-client`, `odc-stac`, …) |
| Imagery | Synthetic 768² scene | DEA S2 ARD median composite |
| Vectors | 24 synthetic veg + 2 synthetic tenements | WA SLIP Native Veg + Tenements |
| Pixel values | **Fabricated** | Real surface reflectance |
| CRS / transform / geometry / area maths | Real | Real |

### Known failure mode (agriculture vs native)

The single most important honest result, surfaced here because it is rooted in
the data: on the synthetic AOI the U-Net **over-predicts native vegetation**
(**1910 ha** predicted vs **1344 ha** truth, +42%; the 95% block-bootstrap CI
[1723, 2096] ha does **not** cover the truth). The cause is that **managed-green
agriculture is spectrally close to native vegetation** and is misclassified as
native (native **precision 0.78**). The late-summer (Jan–Mar) composite window is
chosen partly to reduce this confusion in real mode, but it is a genuine,
expected limitation of separating native from managed green cover with 10 m
multispectral data alone. This is exactly why the project frames itself as a
*native-vegetation mapping proxy* with spatial validation and uncertainty — not as
a species- or compliance-grade map.

---

## Attribution and licence compliance

- **DEA products (Sentinel-2 ARD, Land Cover, Fractional Cover):** CC BY 4.0,
  © Commonwealth of Australia (Geoscience Australia / Digital Earth Australia).
  Attribute Geoscience Australia / DEA and link the product pages above.
- **WA Government layers (Native Vegetation Extent, Mining Tenements DMIRS-003):**
  CC BY 4.0, © Government of Western Australia. Attribute the State of Western
  Australia and link the catalogue pages above. Layer ids on the SLIP services
  change over time — treat the catalogue pages as canonical.
- **ALA / TERN / GeoNadir:** Creative Commons, but **per-record / per-dataset**.
  Verify the licence and any attribution or sensitivity requirements of the
  specific records or orthomosaics you use before redistribution.
- **Synthetic demo data in this repo:** generated by `src/ecorehab/data/demo.py`
  and released under the repository's own MIT licence. The data manifest records
  this as `"synthetic (MIT); real layers are CC BY 4.0"` (see
  `download_vectors.py`).
- **CC BY 4.0 in practice:** redistribution and derivative works are permitted
  provided you give appropriate credit, link the licence, and indicate any
  changes. Keep these attributions in `outputs/` reports and any published
  figures derived from real data.

When you switch to real mode, re-confirm each licence at its canonical URL
(licences and endpoints can change), and make sure generated reports carry the
corresponding data attributions.
