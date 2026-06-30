"""Area-in-hectares estimation + bootstrap confidence intervals."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from ecorehab.evaluation.area_estimation import (
    area_by_class_ha,
    area_by_polygon,
    bootstrap_area_ci,
)


def test_area_by_class_known_counts():
    transform = from_origin(0, 1000, 10.0, 10.0)  # 0.01 ha pixels
    pred = np.zeros((100, 100), dtype="uint8")
    pred[:50, :] = 1  # 5000 px class 1
    pred[50:, :] = 2  # 5000 px class 2
    areas = area_by_class_ha(pred, transform, [0, 1, 2])
    assert areas[1] == pytest.approx(50.0)  # 5000 * 0.01
    assert areas[2] == pytest.approx(50.0)
    assert areas[0] == pytest.approx(0.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    transform = from_origin(0, 1280, 10.0, 10.0)
    pred = (rng.random((128, 128)) < 0.3).astype("uint8")  # ~30% class 1
    ci = bootstrap_area_ci(pred, transform, class_id=1, n_boot=200, block_size=16, seed=0)
    assert ci.lower_ha <= ci.point_ha <= ci.upper_ha
    assert ci.upper_ha > ci.lower_ha  # non-degenerate interval


def test_bootstrap_ci_ignores_ignore_pixels():
    transform = from_origin(0, 640, 10.0, 10.0)
    pred = np.full((64, 64), 255, dtype="uint8")
    pred[:32, :32] = 1  # only this quadrant valid, all class 1
    ci = bootstrap_area_ci(pred, transform, class_id=1, n_boot=100, block_size=16, ignore_index=255)
    # 32*32 px * 0.01 ha = 10.24 ha
    assert ci.point_ha == pytest.approx(10.24, rel=1e-6)


def test_area_by_polygon_partitions():
    transform = from_origin(0, 1000, 10.0, 10.0)
    pred = np.zeros((100, 100), dtype="uint8")
    pred[:, :] = 1
    polys = gpd.GeoDataFrame(
        {"tenement_id": ["A"]}, geometry=[box(0, 500, 500, 1000)], crs="EPSG:3577"
    )
    df = area_by_polygon(
        pred,
        transform,
        "EPSG:3577",
        polys,
        [0, 1],
        {0: "other", 1: "native"},
        id_field="tenement_id",
    )
    assert df.loc[0, "polygon_id"] == "A"
    # polygon is 500 m x 500 m = 25 ha, all class 1
    assert df.loc[0, "native_ha"] == pytest.approx(25.0, rel=1e-2)
