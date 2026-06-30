"""Vectorise a class prediction raster into polygons with hectare areas."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import shape

from ecorehab import constants
from ecorehab.utils.crs import SQ_M_PER_HECTARE


def polygonize_prediction(
    prediction: np.ndarray,
    transform: Affine,
    crs: str,
    class_names: dict[int, str] | None = None,
    min_area_ha: float = 0.5,
    ignore_index: int = constants.IGNORE_INDEX,
) -> gpd.GeoDataFrame:
    """Convert a (H, W) class raster into a polygon GeoDataFrame.

    Each polygon carries its class id/name and area in hectares (CRS must be
    metre-based, e.g. EPSG:3577). Polygons below ``min_area_ha`` are dropped.
    """
    pred = prediction.astype("int32")
    valid = (pred != ignore_index).astype("uint8")
    records = []
    geoms = []
    for geom, val in features.shapes(pred, mask=valid.astype(bool), transform=transform):
        cid = int(val)
        if cid == ignore_index:
            continue
        poly = shape(geom)
        area_ha = poly.area / SQ_M_PER_HECTARE
        if area_ha < min_area_ha:
            continue
        name = class_names.get(cid, f"class_{cid}") if class_names else f"class_{cid}"
        records.append({"class_id": cid, "class_name": name, "area_ha": area_ha})
        geoms.append(poly)
    if not records:
        return gpd.GeoDataFrame(
            {"class_id": [], "class_name": [], "area_ha": []}, geometry=[], crs=crs
        )
    return gpd.GeoDataFrame(records, geometry=geoms, crs=crs)


__all__ = ["polygonize_prediction"]
