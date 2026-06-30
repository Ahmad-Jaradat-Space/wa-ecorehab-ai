"""Per-pixel predictive uncertainty from class-probability rasters."""

from __future__ import annotations

import numpy as np


def predictive_entropy(probs: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Shannon entropy of the class-probability vector at each pixel.

    Args:
        probs: (C, H, W) probabilities (per-pixel they should sum ~1).
        normalise: divide by log(C) so the result is in [0, 1].

    Returns:
        (H, W) entropy map. High = the model is unsure between classes.
    """
    if probs.ndim != 3:
        raise ValueError(f"probs must be (C,H,W), got {probs.shape}")
    c = probs.shape[0]
    p = np.clip(probs, 1e-9, 1.0)
    ent = -np.sum(p * np.log(p), axis=0)
    if normalise and c > 1:
        ent = ent / np.log(c)
    return ent.astype("float32")


def confidence_margin(probs: np.ndarray) -> np.ndarray:
    """Top1 - top2 probability margin per pixel (low margin = uncertain)."""
    if probs.ndim != 3:
        raise ValueError(f"probs must be (C,H,W), got {probs.shape}")
    srt = np.sort(probs, axis=0)
    if probs.shape[0] < 2:
        return srt[-1].astype("float32")
    return (srt[-1] - srt[-2]).astype("float32")


def low_confidence_mask(probs: np.ndarray, entropy_threshold: float = 0.5) -> np.ndarray:
    """Boolean mask of pixels flagged for ecologist review (high entropy)."""
    return predictive_entropy(probs) >= entropy_threshold


__all__ = ["confidence_margin", "low_confidence_mask", "predictive_entropy"]
