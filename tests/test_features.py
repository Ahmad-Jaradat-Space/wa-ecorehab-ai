"""Spectral index correctness + feature stack assembly + IO round-trips."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from ecorehab.features.indices import build_feature_stack, compute_index
from ecorehab.utils.io import RasterData, read_raster, write_raster


def _toy_cube():
    # bands: blue, green, red, rededge, nir, swir1, swir2
    names = ("blue", "green", "red", "rededge", "nir", "swir1", "swir2")
    arr = np.zeros((7, 2, 2), dtype="float32")
    arr[2] = 0.1  # red
    arr[4] = 0.5  # nir
    arr[1] = 0.2  # green
    arr[5] = 0.3  # swir1
    arr[6] = 0.25  # swir2
    arr[0] = 0.05  # blue
    return arr, names


def test_ndvi_formula():
    arr, names = _toy_cube()
    ndvi = compute_index(arr, names, "ndvi")
    expected = (0.5 - 0.1) / (0.5 + 0.1)
    assert np.allclose(ndvi, expected)


def test_ndvi_range_and_nan_propagation():
    arr, names = _toy_cube()
    arr[4, 0, 0] = np.nan  # nir nodata
    ndvi = compute_index(arr, names, "ndvi")
    assert np.isnan(ndvi[0, 0])
    assert np.isfinite(ndvi[1, 1])


def test_build_feature_stack_dimensions():
    arr, names = _toy_cube()
    feats, fnames = build_feature_stack(arr, names, indices=("ndvi", "ndwi", "nbr"))
    assert feats.shape == (7 + 3, 2, 2)
    assert fnames[:7] == list(names)
    assert fnames[7:] == ["ndvi", "ndwi", "nbr"]


def test_unknown_index_raises():
    arr, names = _toy_cube()
    with pytest.raises(ValueError):
        compute_index(arr, names, "not_an_index")


def test_raster_roundtrip_preserves_metadata(tmp_path):
    arr = np.arange(2 * 4 * 4, dtype="float32").reshape(2, 4, 4)
    transform = from_origin(-1_500_000, -3_500_000, 10.0, 10.0)
    path = tmp_path / "r.tif"
    write_raster(path, arr, transform, "EPSG:3577", nodata=-9999.0, band_names=("a", "b"))
    rd = read_raster(path)
    assert rd.array.shape == (2, 4, 4)
    assert rd.crs == "EPSG:3577"
    assert rd.transform.a == pytest.approx(10.0)
    assert rd.nodata == pytest.approx(-9999.0)
    assert np.allclose(rd.array, arr)


def test_with_nodata_as_nan():
    arr = np.array([[[1.0, -9999.0], [3.0, 4.0]]], dtype="float32")
    rd = RasterData(arr, from_origin(0, 0, 10, 10), "EPSG:3577", nodata=-9999.0)
    nan_rd = rd.with_nodata_as_nan()
    assert np.isnan(nan_rd.array[0, 0, 1])
    assert nan_rd.array[0, 0, 0] == 1.0
