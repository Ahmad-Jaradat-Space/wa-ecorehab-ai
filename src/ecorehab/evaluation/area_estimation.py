"""Uncertainty-aware area estimation in hectares (EPSG:3577).

Reporting "1,343 ha of native vegetation" without an interval hides model error.
We provide:
  * deterministic area-by-class and area-by-polygon (tenement/management unit),
  * a spatial **block bootstrap** confidence interval, which respects the spatial
    autocorrelation of prediction errors (a naive pixel bootstrap would give
    absurdly tight intervals because neighbouring pixels are not independent).
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine

from ecorehab import constants
from ecorehab.labels.rasterize import polygons_to_mask
from ecorehab.utils.crs import pixel_area_ha


def area_by_class_ha(
    prediction: np.ndarray,
    transform: Affine,
    class_ids: list[int],
    ignore_index: int = constants.IGNORE_INDEX,
) -> dict[int, float]:
    """Hectares per class id from a prediction raster."""
    px_ha = pixel_area_ha(transform)
    return {
        cid: int(np.count_nonzero(prediction == cid)) * px_ha
        for cid in class_ids
        if cid != ignore_index
    }


@dataclass
class AreaCI:
    class_id: int
    point_ha: float
    lower_ha: float
    upper_ha: float
    confidence_level: float


def bootstrap_area_ci(
    prediction: np.ndarray,
    transform: Affine,
    class_id: int,
    n_boot: int = 500,
    block_size: int = 32,
    confidence_level: float = 0.95,
    ignore_index: int = constants.IGNORE_INDEX,
    seed: int = constants.DEFAULT_SEED,
) -> AreaCI:
    """Spatial block-bootstrap CI for the area of one class.

    The raster is partitioned into ``block_size`` x ``block_size`` blocks; each
    bootstrap replicate resamples blocks with replacement and rescales the class
    pixel count to the full valid area. Percentile interval.
    """
    rng = np.random.default_rng(seed)
    h, w = prediction.shape
    px_ha = pixel_area_ha(transform)
    point = int(np.count_nonzero(prediction == class_id)) * px_ha

    # Build block grid; each block contributes (class_pixels, valid_pixels).
    blocks: list[tuple[int, int]] = []
    for r0 in range(0, h, block_size):
        for c0 in range(0, w, block_size):
            sub = prediction[r0 : r0 + block_size, c0 : c0 + block_size]
            valid = int(np.count_nonzero(sub != ignore_index))
            if valid == 0:
                continue
            pos = int(np.count_nonzero(sub == class_id))
            blocks.append((pos, valid))
    if not blocks:
        return AreaCI(class_id, point, point, point, confidence_level)

    arr = np.array(blocks, dtype="float64")  # (n_blocks, 2)
    total_valid = arr[:, 1].sum()
    nb = len(arr)
    estimates = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.integers(0, nb, size=nb)
        sample = arr[idx]
        frac = sample[:, 0].sum() / max(1.0, sample[:, 1].sum())
        estimates[i] = frac * total_valid * px_ha
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.quantile(estimates, [alpha, 1.0 - alpha])
    return AreaCI(class_id, point, float(lo), float(hi), confidence_level)


def area_by_polygon(
    prediction: np.ndarray,
    transform: Affine,
    crs: str,
    polygons: gpd.GeoDataFrame,
    class_ids: list[int],
    class_names: dict[int, str] | None = None,
    id_field: str | None = None,
    ignore_index: int = constants.IGNORE_INDEX,
) -> pd.DataFrame:
    """Per-polygon class areas in hectares (e.g. by mining tenement).

    Each polygon gets one row per class plus its total valid area.
    """
    px_ha = pixel_area_ha(transform)
    polys = polygons.to_crs(crs) if str(polygons.crs) != crs else polygons
    shape = prediction.shape
    rows = []
    for i, (_, row) in enumerate(polys.iterrows()):
        pid = row[id_field] if id_field and id_field in polys.columns else i
        mask = polygons_to_mask(
            gpd.GeoDataFrame(geometry=[row.geometry], crs=crs), transform, shape
        )
        inside = prediction[mask]
        valid = inside[inside != ignore_index]
        rec: dict[str, object] = {"polygon_id": pid}
        for cid in class_ids:
            name = class_names.get(cid, f"class_{cid}") if class_names else f"class_{cid}"
            rec[f"{name}_ha"] = int(np.count_nonzero(valid == cid)) * px_ha
        rec["valid_ha"] = int(valid.size) * px_ha
        rows.append(rec)
    return pd.DataFrame(rows)


__all__ = [
    "AreaCI",
    "area_by_class_ha",
    "area_by_polygon",
    "bootstrap_area_ci",
]
