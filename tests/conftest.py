"""Shared fixtures: a tiny synthetic AOI usable without network or heavy compute."""

from __future__ import annotations

import numpy as np
import pytest

from ecorehab.data.demo import generate_demo_scene
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import RasterData


@pytest.fixture(scope="session")
def small_config() -> Config:
    """A small, fast demo config (192 px AOI)."""
    cfg = load_config("configs/aoi_swan_coastal_plain.yaml")
    cfg.data.demo.size_px = 192
    cfg.tiling.tile_size = 64
    cfg.tiling.stride = 64
    cfg.tiling.split.block_size_px = 128
    return cfg


@pytest.fixture(scope="session")
def scene(small_config: Config):
    return generate_demo_scene(small_config)


@pytest.fixture(scope="session")
def composite(scene) -> RasterData:
    return RasterData(
        scene.composite, scene.transform, scene.crs, nodata=None, band_names=scene.band_names
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
