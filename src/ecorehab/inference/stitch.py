"""Seamless stitching of overlapping tile predictions.

Overlapping inference removes the blocky seams a hard tile grid produces at
class boundaries. Each tile's contribution is feathered with a weight that
tapers to (near) zero at the tile edge, so the overlap region is a smooth blend
of both tiles' probabilities.
"""

from __future__ import annotations

import numpy as np


def tile_weights(height: int, width: int, overlap: int) -> np.ndarray:
    """A (H, W) feather weight: ~1 in the centre, ramping down over ``overlap``.

    A small floor keeps weights strictly positive so no pixel is left unwritten.
    """
    if overlap <= 0:
        return np.ones((height, width), dtype="float32")
    ramp_h = _ramp(height, overlap)
    ramp_w = _ramp(width, overlap)
    w = np.outer(ramp_h, ramp_w).astype("float32")
    return np.clip(w, 0.05, 1.0)


def _ramp(n: int, overlap: int) -> np.ndarray:
    overlap = min(overlap, n // 2)
    r = np.ones(n, dtype="float32")
    if overlap > 0:
        edge = np.linspace(0.0, 1.0, overlap + 1)[1:]
        r[:overlap] = edge
        r[-overlap:] = edge[::-1]
    return r


class ProbabilityAccumulator:
    """Accumulate weighted tile probabilities into a full-raster mosaic."""

    def __init__(self, num_classes: int, height: int, width: int):
        self.prob_sum = np.zeros((num_classes, height, width), dtype="float32")
        self.weight_sum = np.zeros((height, width), dtype="float32")

    def add(self, prob: np.ndarray, row_off: int, col_off: int, weights: np.ndarray) -> None:
        c, h, w = prob.shape
        self.prob_sum[:, row_off : row_off + h, col_off : col_off + w] += prob * weights[None]
        self.weight_sum[row_off : row_off + h, col_off : col_off + w] += weights

    def finalize(self) -> np.ndarray:
        denom = np.maximum(self.weight_sum, 1e-6)
        return self.prob_sum / denom[None]


__all__ = ["ProbabilityAccumulator", "tile_weights"]
