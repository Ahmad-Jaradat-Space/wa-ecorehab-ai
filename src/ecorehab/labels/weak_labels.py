"""Weak-label generation from vegetation polygons + spectral gates.

The honest premise (see docs/label_strategy.md): there is no perfect open
pixel-level ecological label for WA. We synthesise a *weak* label by combining
the WA Native Vegetation Extent polygons with simple, transparent spectral rules,
and we are deliberately conservative — pixels we cannot confidently assign are
set to ``ignore`` rather than guessed.

Rules (3-class schema 0=disturbed, 1=native, 2=water; 255=ignore):
  * native (1): inside polygons eroded by ``positive_inward_buffer_m`` AND
    NDVI >= ``native_ndvi_threshold``. (A green polygon interior we trust.)
  * water (2): NDWI > ``water_ndwi_threshold`` AND outside polygons.
  * disturbed (0): outside polygons AND NDVI < ``disturbed_ndvi_threshold``
    (clearly non-vegetated). Intermediate-NDVI pixels outside polygons are
    *ambiguous* (sparse veg / agriculture) -> ignore.
  * ignore (255): polygon boundary rings, nodata, and everything not confidently
    assigned above.

A per-pixel confidence raster records how trustworthy each label is.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np

from ecorehab.features.indices import compute_index
from ecorehab.labels.rasterize import boundary_exclusion_mask, polygons_to_mask
from ecorehab.utils.config import Config
from ecorehab.utils.crs import reproject_gdf
from ecorehab.utils.io import RasterData


@dataclass
class WeakLabelResult:
    label: np.ndarray  # (H, W) uint8, IGNORE_INDEX for ignore
    confidence: np.ndarray  # (H, W) float32 in [0, 1]
    class_counts: dict[int, int]


def build_weak_labels(
    composite: RasterData, vegetation: gpd.GeoDataFrame, cfg: Config
) -> WeakLabelResult:
    """Generate a weak label + confidence raster for one AOI composite.

    Args:
        composite: surface-reflectance raster (bands match ``cfg.data.bands``).
        vegetation: native-vegetation polygons (any CRS; reprojected internally).
        cfg: project config (label thresholds + buffers).
    """
    lab_cfg = cfg.labels
    ignore = lab_cfg.ignore_index
    band_names = composite.band_names or tuple(cfg.data.bands)
    shape = (composite.height, composite.width)
    transform = composite.transform

    # Ensure polygons share the composite CRS (metre-based for buffering).
    veg = reproject_gdf(vegetation, composite.crs) if len(vegetation) else vegetation

    arr = composite.array.astype("float32")
    # nodata -> NaN so indices propagate it.
    if composite.nodata is not None and not np.isnan(composite.nodata):
        arr = np.where(arr == composite.nodata, np.nan, arr)
    ndvi = compute_index(arr, band_names, "ndvi")
    ndwi = compute_index(arr, band_names, "ndwi")
    nodata = np.any(~np.isfinite(arr), axis=0)

    label = np.full(shape, ignore, dtype="uint8")
    conf = np.zeros(shape, dtype="float32")

    if len(veg):
        inside_core = polygons_to_mask(
            veg, transform, shape, buffer_m=-lab_cfg.positive_inward_buffer_m
        )
        edge_ring = boundary_exclusion_mask(
            veg, transform, shape, buffer_m=lab_cfg.boundary_exclusion_buffer_m
        )
        inside_any = polygons_to_mask(veg, transform, shape, buffer_m=0.0)
    else:  # no polygons -> nothing native
        inside_core = np.zeros(shape, dtype=bool)
        edge_ring = np.zeros(shape, dtype=bool)
        inside_any = np.zeros(shape, dtype=bool)

    ndvi_ok = np.nan_to_num(ndvi, nan=-1.0)
    ndwi_ok = np.nan_to_num(ndwi, nan=-1.0)

    # --- native (1): trusted green polygon interiors -----------------------------
    native = inside_core & (ndvi_ok >= lab_cfg.native_ndvi_threshold) & ~nodata
    label[native] = 1
    # confidence scales with how far above the NDVI threshold the pixel sits.
    conf[native] = np.clip(0.7 + 0.6 * (ndvi_ok[native] - lab_cfg.native_ndvi_threshold), 0.7, 1.0)

    # --- water (2): wet pixels outside vegetation --------------------------------
    water = (ndwi_ok > lab_cfg.water_ndwi_threshold) & ~inside_any & ~nodata
    label[water] = 2
    conf[water] = np.clip(0.6 + 2.0 * (ndwi_ok[water] - lab_cfg.water_ndwi_threshold), 0.6, 1.0)

    # --- disturbed (0): clearly non-vegetated, outside vegetation ----------------
    disturbed = ~inside_any & ~water & (ndvi_ok < lab_cfg.disturbed_ndvi_threshold) & ~nodata
    label[disturbed] = 0
    conf[disturbed] = np.clip(
        0.6 + 1.5 * (lab_cfg.disturbed_ndvi_threshold - ndvi_ok[disturbed]), 0.6, 1.0
    )

    # --- ignore overrides --------------------------------------------------------
    # Polygon-edge rings are too noisy to trust.
    label[edge_ring] = ignore
    conf[edge_ring] = 0.0
    # nodata always ignore.
    label[nodata] = ignore
    conf[nodata] = 0.0

    counts = {int(c): int(np.count_nonzero(label == c)) for c in [cd.id for cd in lab_cfg.classes]}
    counts[ignore] = int(np.count_nonzero(label == ignore))
    return WeakLabelResult(label=label, confidence=conf, class_counts=counts)


__all__ = ["WeakLabelResult", "build_weak_labels"]
