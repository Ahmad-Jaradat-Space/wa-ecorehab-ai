"""CRS and area helpers centred on EPSG:3577 (Australian Albers, equal-area).

Why this matters: hectare reporting and label/tile generation must happen in an
equal-area, metre-based projection. Computing areas in a geographic CRS
(EPSG:4326, degrees) is wrong, and computing them in a UTM zone introduces
small area distortions far from the central meridian. EPSG:3577 is the standard
choice for Australia-wide equal-area work.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine

from ecorehab import constants

SQ_M_PER_HECTARE: float = 10_000.0


def is_projected_metres(crs: str | CRS) -> bool:
    """True if ``crs`` is projected with linear units of metres."""
    crs = CRS.from_user_input(crs)
    if not crs.is_projected:
        return False
    axis = crs.axis_info[0]
    return axis.unit_name in {"metre", "meter", "m"}


def assert_equal_area(crs: str | CRS) -> None:
    """Raise if ``crs`` is not a projected, metre-based CRS suitable for area."""
    if not is_projected_metres(crs):
        raise ValueError(
            f"CRS {CRS.from_user_input(crs).to_string()!r} is not projected in metres; "
            f"reproject to {constants.EQUAL_AREA_CRS} before area computation."
        )


def to_equal_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the equal-area CRS (EPSG:3577)."""
    return reproject_gdf(gdf, constants.EQUAL_AREA_CRS)


def reproject_gdf(gdf: gpd.GeoDataFrame, dst_crs: str | CRS) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame, validating it has a source CRS."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS; cannot reproject safely.")
    if CRS.from_user_input(gdf.crs) == CRS.from_user_input(dst_crs):
        return gdf
    return gdf.to_crs(dst_crs)


def transform_point(
    lon: float, lat: float, dst_crs: str | CRS, src_crs: str | CRS = constants.GEOGRAPHIC_CRS
) -> tuple[float, float]:
    """Transform a single (lon, lat) point to ``dst_crs``. Returns (x, y)."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def bbox_to_equal_area(
    bbox: tuple[float, float, float, float], src_crs: str | CRS = constants.GEOGRAPHIC_CRS
) -> tuple[float, float, float, float]:
    """Reproject a (minx, miny, maxx, maxy) bbox into the equal-area CRS."""
    transformer = Transformer.from_crs(src_crs, constants.EQUAL_AREA_CRS, always_xy=True)
    xs, ys = transformer.transform(
        [bbox[0], bbox[2], bbox[0], bbox[2]],
        [bbox[1], bbox[1], bbox[3], bbox[3]],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def geodataframe_area_ha(gdf: gpd.GeoDataFrame) -> float:
    """Total geometry area in hectares (reprojects to equal-area first)."""
    ea = to_equal_area(gdf)
    return float(ea.geometry.area.sum() / SQ_M_PER_HECTARE)


def pixel_area_ha(transform: Affine) -> float:
    """Area of a single pixel in hectares, from a raster affine transform.

    Assumes the raster CRS is projected in metres (caller's responsibility).
    """
    # |a| = pixel width, |e| = pixel height (e is negative for north-up rasters).
    area_m2 = abs(transform.a) * abs(transform.e)
    return area_m2 / SQ_M_PER_HECTARE


def class_area_ha(label: np.ndarray, transform: Affine, class_id: int) -> float:
    """Hectares covered by ``class_id`` pixels in a label array."""
    n = int(np.count_nonzero(label == class_id))
    return n * pixel_area_ha(transform)


def class_area_table_ha(
    label: np.ndarray,
    transform: Affine,
    class_ids: list[int],
    ignore_index: int = constants.IGNORE_INDEX,
) -> dict[int, float]:
    """Hectares per class id (ignores ``ignore_index`` pixels)."""
    px_ha = pixel_area_ha(transform)
    out: dict[int, float] = {}
    for cid in class_ids:
        if cid == ignore_index:
            continue
        out[cid] = int(np.count_nonzero(label == cid)) * px_ha
    return out


__all__ = [
    "SQ_M_PER_HECTARE",
    "assert_equal_area",
    "bbox_to_equal_area",
    "class_area_ha",
    "class_area_table_ha",
    "geodataframe_area_ha",
    "is_projected_metres",
    "pixel_area_ha",
    "reproject_gdf",
    "to_equal_area",
    "transform_point",
]
