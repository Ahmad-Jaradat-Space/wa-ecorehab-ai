"""Windowed, overlapping tiled inference for segmentation models.

Reads the composite in windows (so arbitrarily large rasters never need to fit
in memory), builds the same feature stack used in training, predicts with
overlap, and feather-blends the tiles into a seamless probability mosaic. This
is the "not just a notebook" production path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from ecorehab import constants
from ecorehab.features.indices import build_feature_stack
from ecorehab.inference.stitch import ProbabilityAccumulator, tile_weights
from ecorehab.labels.tiling import generate_tiles
from ecorehab.models.unet import build_segmentation_model
from ecorehab.training.dataset import standardize
from ecorehab.utils.config import SegmentationConfig
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def load_segmentation_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild a segmentation model + its metadata from a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    seg_cfg = SegmentationConfig(
        arch=ckpt["arch"],
        encoder=ckpt.get("encoder", "resnet34"),
        base_channels=ckpt.get("base_channels", 32),
    )
    model = build_segmentation_model(seg_cfg, ckpt["in_channels"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {
        "mean": np.asarray(ckpt["mean"], dtype="float32"),
        "std": np.asarray(ckpt["std"], dtype="float32"),
        "bands": ckpt["bands"],
        "feature_indices": ckpt["feature_indices"],
        "include_bands": ckpt["include_bands"],
        "class_ids": ckpt["class_ids"],
        "class_names": ckpt["class_names"],
        "num_classes": ckpt["num_classes"],
        "ignore_index": ckpt.get("ignore_index", constants.IGNORE_INDEX),
    }
    return model, meta


def _window_features(src, window: Window, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read a window, map nodata->NaN, build the standardised feature stack.

    Returns ``(features, valid)`` where ``valid`` is a 2D mask of pixels that had
    at least one finite band BEFORE the nodata sentinel was zeroed out by
    standardisation. This must use the nodata-mapped array (nodata is the finite
    sentinel ``-9999.0``, so ``isfinite`` on the raw read would mark it valid).
    """
    arr = src.read(window=window).astype("float32")
    if src.nodata is not None and not np.isnan(src.nodata):
        arr = np.where(arr == src.nodata, np.nan, arr)
    valid = np.any(np.isfinite(arr), axis=0)
    feats, _ = build_feature_stack(
        arr, meta["bands"], meta["feature_indices"], include_bands=meta["include_bands"]
    )
    return standardize(feats, meta["mean"], meta["std"]), valid


@torch.no_grad()
def predict_raster_windowed(
    composite_path: str | Path,
    model: torch.nn.Module,
    meta: dict[str, Any],
    tile_size: int = 256,
    overlap: int = 32,
    device: torch.device | str = "cpu",
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, Any, str]:
    """Run overlapping windowed inference over a composite GeoTIFF.

    Returns:
        (probabilities (C, H, W), prediction (H, W), transform, crs).
    """
    model.eval()
    num_classes = meta["num_classes"]
    with rasterio.open(composite_path) as src:
        h, w = src.height, src.width
        transform, crs = src.transform, str(src.crs)
        stride = max(1, tile_size - overlap)
        tiles = generate_tiles((h, w), tile_size=min(tile_size, h, w), stride=min(stride, h, w))
        acc = ProbabilityAccumulator(num_classes, h, w)
        valid_any = np.zeros((h, w), dtype=bool)
        wts = tile_weights(min(tile_size, h, w), min(tile_size, h, w), overlap)
        for t in tiles:
            window = Window(t.col_off, t.row_off, t.width, t.height)
            feats, tile_valid = _window_features(src, window, meta)
            x = torch.from_numpy(feats[None]).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            if tta:
                prob = prob + torch.softmax(model(torch.flip(x, dims=[3])), dim=1).flip(dims=[3])
                prob = prob + torch.softmax(model(torch.flip(x, dims=[2])), dim=1).flip(dims=[2])
            p = prob[0].cpu().numpy()
            p = p / p.sum(axis=0, keepdims=True)
            acc.add(p, t.row_off, t.col_off, wts[: t.height, : t.width])
            valid_any[
                t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width
            ] |= tile_valid
    proba = acc.finalize()
    pred = proba.argmax(axis=0).astype("uint8")
    pred[~valid_any] = meta["ignore_index"]
    logger.info("Windowed inference over %dx%d with %d tiles done", h, w, len(tiles))
    return proba, pred, transform, crs


__all__ = ["load_segmentation_checkpoint", "predict_raster_windowed"]
