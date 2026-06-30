"""Shared loaders that turn on-disk AOI artifacts into arrays for ML stages.

Keeps train_classical / train_segmentation / inference in agreement on how the
composite becomes a feature stack and which array is the evaluation target.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from ecorehab import artifacts
from ecorehab.features.indices import build_feature_stack
from ecorehab.utils.config import Config
from ecorehab.utils.io import RasterData, read_raster, read_vector


def load_composite(cfg: Config) -> RasterData:
    """Read the composite with nodata mapped to NaN."""
    path = artifacts.composite_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"composite not found: {path}. Run make_demo/build_composites.")
    return read_raster(path).with_nodata_as_nan()


def load_feature_stack(cfg: Config) -> tuple[np.ndarray, list[str], RasterData]:
    """Build the (C, H, W) feature stack (bands + indices) from the composite."""
    comp = load_composite(cfg)
    feats, names = build_feature_stack(
        comp.array,
        comp.band_names or tuple(cfg.data.bands),
        cfg.features.indices,
        include_bands=cfg.features.include_bands,
    )
    return feats, names, comp


def load_label(cfg: Config) -> np.ndarray:
    path = artifacts.label_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"labels not found: {path}. Run build_labels.")
    return read_raster(path).array[0].astype("int64")


def load_truth(cfg: Config) -> np.ndarray | None:
    """Hidden per-pixel truth (demo mode only); None in real mode."""
    path = artifacts.truth_path(cfg)
    if not path.exists():
        return None
    return read_raster(path).array[0].astype("int64")


def load_tile_index(cfg: Config) -> gpd.GeoDataFrame:
    path = artifacts.tile_index_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"tile index not found: {path}. Run build_tiles.")
    return read_vector(path)


def evaluation_target(cfg: Config) -> tuple[str, np.ndarray]:
    """Return (kind, array) used to score predictions.

    Prefers the hidden demo truth (honest evaluation); falls back to the weak
    labels in real mode (with the caveat documented in docs/limitations.md).
    """
    truth = load_truth(cfg)
    if truth is not None:
        return "truth", truth
    return "weak_label", load_label(cfg)


def split_mask(tile_index: gpd.GeoDataFrame, split: str, shape: tuple[int, int]) -> np.ndarray:
    """Boolean pixel mask covering all tiles assigned to ``split``."""
    mask = np.zeros(shape, dtype=bool)
    for _, t in tile_index[tile_index["split"] == split].iterrows():
        mask[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] = True
    return mask


__all__ = [
    "evaluation_target",
    "load_composite",
    "load_feature_stack",
    "load_label",
    "load_tile_index",
    "load_truth",
    "split_mask",
]
