"""Vector -> raster helpers (thin, correctness-focused wrappers over rasterio)."""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry


def rasterize_geometries(
    geometries: Iterable[BaseGeometry],
    transform: Affine,
    out_shape: tuple[int, int],
    value: int = 1,
    fill: int = 0,
    all_touched: bool = False,
    dtype: str = "uint8",
) -> np.ndarray:
    """Burn geometries into a raster of ``out_shape`` (rows, cols).

    The output array shape exactly matches ``out_shape`` and is aligned to
    ``transform``; callers rely on this for label/image alignment.
    """
    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        return np.full(out_shape, fill, dtype=dtype)
    shapes = ((g, value) for g in geoms)
    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )


def polygons_to_mask(
    gdf: gpd.GeoDataFrame,
    transform: Affine,
    out_shape: tuple[int, int],
    buffer_m: float = 0.0,
    all_touched: bool = False,
) -> np.ndarray:
    """Boolean mask of polygon coverage, with optional metric buffer.

    ``buffer_m`` > 0 dilates polygons, < 0 erodes them (inward buffer). The GDF
    must already be in a metre-based CRS that matches ``transform``.
    """
    geoms = gdf.geometry
    if buffer_m != 0.0:
        geoms = geoms.buffer(buffer_m)
    arr = rasterize_geometries(
        geoms.values, transform, out_shape, value=1, fill=0, all_touched=all_touched
    )
    return arr.astype(bool)


def boundary_exclusion_mask(
    gdf: gpd.GeoDataFrame,
    transform: Affine,
    out_shape: tuple[int, int],
    buffer_m: float,
) -> np.ndarray:
    """Ring mask around polygon boundaries (outward buffer minus inward buffer).

    These edge pixels are the most label-noisy and are set to ignore.
    """
    if buffer_m <= 0:
        return np.zeros(out_shape, dtype=bool)
    outer = polygons_to_mask(gdf, transform, out_shape, buffer_m=buffer_m)
    inner = polygons_to_mask(gdf, transform, out_shape, buffer_m=-buffer_m)
    return outer & ~inner


__all__ = ["boundary_exclusion_mask", "polygons_to_mask", "rasterize_geometries"]
