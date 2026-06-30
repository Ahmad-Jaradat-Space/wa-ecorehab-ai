"""Spatial cross-validation utilities + random-vs-spatial leakage diagnostics.

The central honesty argument of the project: imagery pixels are spatially
autocorrelated, so a random tile split places near-duplicate neighbours in both
train and test, inflating metrics. These helpers quantify that gap and the
train/test proximity that causes it.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def assign_kfold_blocks(tile_index: gpd.GeoDataFrame, k: int = 5, seed: int = 42) -> np.ndarray:
    """Assign a spatial-block fold id (0..k-1) to each tile.

    Tiles sharing a (block_row, block_col) get the same fold -> spatial CV.
    """
    if not {"block_row", "block_col"}.issubset(tile_index.columns):
        raise ValueError("tile_index must have block_row/block_col columns")
    blocks = list(tile_index.groupby(["block_row", "block_col"]).groups.keys())
    rng = np.random.default_rng(seed)
    fold_of_block = {b: int(rng.integers(0, k)) for b in blocks}
    return tile_index.apply(
        lambda r: fold_of_block[(r["block_row"], r["block_col"])], axis=1
    ).to_numpy()


def nearest_train_distance(
    tile_index: gpd.GeoDataFrame, train_split: str = "train", eval_split: str = "val"
) -> np.ndarray:
    """Distance from each eval tile centroid to the nearest train tile centroid.

    Small distances under a random split reveal the leakage that spatial blocks
    remove. Distances are in CRS units (metres for EPSG:3577).
    """
    train = tile_index[tile_index["split"] == train_split]
    ev = tile_index[tile_index["split"] == eval_split]
    if train.empty or ev.empty:
        return np.array([])
    train_pts = np.array([(g.centroid.x, g.centroid.y) for g in train.geometry])
    ev_pts = np.array([(g.centroid.x, g.centroid.y) for g in ev.geometry])
    dists = np.sqrt(((ev_pts[:, None, :] - train_pts[None, :, :]) ** 2).sum(axis=2))
    return dists.min(axis=1)


def summarise_split_comparison(results: dict[str, dict]) -> pd.DataFrame:
    """Tabulate headline metrics across split strategies for side-by-side comparison.

    Args:
        results: {split_name: metrics_dict} where each metrics_dict comes from
            :func:`ecorehab.evaluation.metrics.segmentation_metrics`.
    """
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "split": name,
                "macro_f1": m.get("macro_f1"),
                "mean_iou": m.get("mean_iou"),
                "weighted_f1": m.get("weighted_f1"),
                "overall_accuracy": m.get("overall_accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "assign_kfold_blocks",
    "nearest_train_distance",
    "summarise_split_comparison",
]
