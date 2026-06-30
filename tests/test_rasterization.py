"""Rasterization shape + transform consistency, and weak-label invariants."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from ecorehab import constants
from ecorehab.labels.rasterize import (
    boundary_exclusion_mask,
    polygons_to_mask,
    rasterize_geometries,
)
from ecorehab.labels.weak_labels import build_weak_labels


def test_rasterize_shape_matches_out_shape():
    transform = from_origin(0, 100, 10.0, 10.0)
    out_shape = (10, 10)
    arr = rasterize_geometries([box(0, 0, 50, 100)], transform, out_shape, value=1)
    assert arr.shape == out_shape
    assert arr.dtype == np.uint8


def test_rasterize_left_half_burned():
    # Raster covers x in [0,100], y in [0,100], 10 m pixels -> 10x10.
    transform = from_origin(0, 100, 10.0, 10.0)
    arr = rasterize_geometries([box(0, 0, 50, 100)], transform, (10, 10), value=1)
    # Left 5 columns inside the polygon -> 1, right 5 -> 0.
    assert arr[:, :5].all()
    assert not arr[:, 5:].any()
    assert arr.sum() == 50


def test_polygons_to_mask_inward_buffer_shrinks():
    transform = from_origin(0, 1000, 10.0, 10.0)
    shape = (100, 100)
    gdf = gpd.GeoDataFrame(geometry=[box(200, 200, 800, 800)], crs=constants.EQUAL_AREA_CRS)
    full = polygons_to_mask(gdf, transform, shape, buffer_m=0.0)
    eroded = polygons_to_mask(gdf, transform, shape, buffer_m=-50.0)
    assert eroded.sum() < full.sum()
    assert (eroded & ~full).sum() == 0  # eroded is a subset


def test_boundary_exclusion_is_a_ring():
    transform = from_origin(0, 1000, 10.0, 10.0)
    shape = (100, 100)
    gdf = gpd.GeoDataFrame(geometry=[box(300, 300, 700, 700)], crs=constants.EQUAL_AREA_CRS)
    ring = boundary_exclusion_mask(gdf, transform, shape, buffer_m=30.0)
    core = polygons_to_mask(gdf, transform, shape, buffer_m=-30.0)
    # Ring and core must not overlap.
    assert (ring & core).sum() == 0
    assert ring.sum() > 0


def test_weak_labels_consistency(composite, scene, small_config):
    res = build_weak_labels(composite, scene.vegetation, small_config)
    assert res.label.shape == (composite.height, composite.width)
    assert res.confidence.shape == res.label.shape
    # Only valid class ids or ignore appear.
    allowed = {c.id for c in small_config.labels.classes} | {small_config.labels.ignore_index}
    assert set(np.unique(res.label)).issubset(allowed)
    # nodata corner must be ignore.
    assert res.label[0, 0] == small_config.labels.ignore_index
    # Native pixels should be a meaningful share (synthetic AOI has veg).
    assert res.class_counts[1] > 0


def test_weak_labels_native_aligns_with_high_ndvi(composite, scene, small_config):
    from ecorehab.features.indices import compute_index

    res = build_weak_labels(composite, scene.vegetation, small_config)
    ndvi = compute_index(composite.array, composite.band_names, "ndvi")
    native_ndvi = ndvi[res.label == 1]
    other_ndvi = ndvi[res.label == 0]
    assert np.nanmedian(native_ndvi) > np.nanmedian(other_ndvi)
