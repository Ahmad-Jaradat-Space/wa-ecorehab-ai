"""CRS handling + equal-area hectare maths."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from ecorehab import constants
from ecorehab.utils import crs


def test_equal_area_is_projected_metres():
    assert crs.is_projected_metres(constants.EQUAL_AREA_CRS)
    assert not crs.is_projected_metres(constants.GEOGRAPHIC_CRS)


def test_assert_equal_area_rejects_geographic():
    with pytest.raises(ValueError):
        crs.assert_equal_area("EPSG:4326")
    crs.assert_equal_area(constants.EQUAL_AREA_CRS)  # should not raise


def test_pixel_area_ha_10m():
    # 10 m x 10 m pixel = 100 m^2 = 0.01 ha
    transform = from_origin(0, 0, 10.0, 10.0)
    assert crs.pixel_area_ha(transform) == pytest.approx(0.01)


def test_class_area_ha_counts():
    transform = from_origin(0, 0, 10.0, 10.0)
    label = np.zeros((10, 10), dtype="uint8")
    label[:5, :] = 1  # 50 pixels of class 1
    # 50 pixels * 0.01 ha = 0.5 ha
    assert crs.class_area_ha(label, transform, 1) == pytest.approx(0.5)


def test_geodataframe_area_ha_known_square():
    # A 1000 m x 1000 m square in EPSG:3577 = 1 km^2 = 100 ha.
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs=constants.EQUAL_AREA_CRS)
    assert crs.geodataframe_area_ha(g) == pytest.approx(100.0, rel=1e-6)


def test_reproject_preserves_area_roughly():
    # Build a square in 3577, reproject to 4326 and back; area should be stable.
    g = gpd.GeoDataFrame(
        geometry=[box(-1_500_000, -3_600_000, -1_499_000, -3_599_000)], crs=constants.EQUAL_AREA_CRS
    )
    a1 = crs.geodataframe_area_ha(g)
    g4326 = g.to_crs("EPSG:4326")
    a2 = crs.geodataframe_area_ha(g4326)  # reprojects back to 3577 internally
    assert a1 == pytest.approx(a2, rel=1e-6)


def test_reproject_requires_source_crs():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError):
        crs.to_equal_area(g)
