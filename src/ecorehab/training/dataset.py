"""Torch dataset for tiled segmentation + feature standardisation helpers.

The dataset holds the full AOI feature stack in memory and slices tile windows
on access (the demo AOI is small; for large AOIs this would become windowed
rasterio reads, see docs/gcp_batch_inference_design.md). Features are
standardised with statistics computed on the *train* split only, to avoid
leaking validation/test statistics into preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecorehab import constants
from ecorehab.training.augmentations import AugFn


def compute_feature_stats(
    features: np.ndarray, tiles: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the given tiles' valid (finite) pixels."""
    c = features.shape[0]
    sums = np.zeros(c)
    sqs = np.zeros(c)
    counts = np.zeros(c)
    for _, t in tiles.iterrows():
        win = features[
            :, t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width
        ].reshape(c, -1)
        finite = np.isfinite(win)
        sums += np.where(finite, win, 0.0).sum(axis=1)
        sqs += np.where(finite, win**2, 0.0).sum(axis=1)
        counts += finite.sum(axis=1)
    counts = np.maximum(counts, 1)
    mean = sums / counts
    var = np.maximum(sqs / counts - mean**2, 1e-8)
    return mean.astype("float32"), np.sqrt(var).astype("float32")


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise a (C, H, W) stack; NaN (nodata) -> 0 after scaling."""
    out = (features - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(out, nan=0.0).astype("float32")


class TileSegmentationDataset(Dataset):
    """Yields (image CxHxW float32, mask HxW int64) for tiles in one split."""

    def __init__(
        self,
        features: np.ndarray,
        label: np.ndarray,
        tiles: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        augment: AugFn | None = None,
        ignore_index: int = constants.IGNORE_INDEX,
    ):
        self.features = features
        self.label = label
        self.tiles = tiles.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.tiles.iloc[idx]
        r0, c0, h, w = int(t.row_off), int(t.col_off), int(t.height), int(t.width)
        img = self.features[:, r0 : r0 + h, c0 : c0 + w]
        msk = self.label[r0 : r0 + h, c0 : c0 + w].astype("int64")
        img = standardize(img, self.mean, self.std)
        if self.augment is not None:
            img, msk = self.augment(img, msk)
        return torch.from_numpy(img.copy()), torch.from_numpy(msk.copy())


__all__ = [
    "TileSegmentationDataset",
    "compute_feature_stats",
    "standardize",
]
