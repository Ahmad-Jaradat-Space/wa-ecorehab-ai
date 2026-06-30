"""Balanced pixel sampling for the classical (pixel-wise) baselines."""

from __future__ import annotations

import numpy as np

from ecorehab import constants


def sample_pixels(
    features: np.ndarray,
    label: np.ndarray,
    class_ids: list[int],
    n_per_class: int,
    ignore_index: int = constants.IGNORE_INDEX,
    seed: int = constants.DEFAULT_SEED,
    restrict_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced pixel sample from a feature stack.

    Args:
        features: (C, H, W) feature stack.
        label: (H, W) integer labels (``ignore_index`` excluded).
        class_ids: class ids to sample.
        n_per_class: cap of pixels per class (rare classes take all available).
        restrict_mask: optional (H, W) bool; only sample where True (e.g. train area).

    Returns:
        (X, y) with X shape (N, C) and y shape (N,). Rows with non-finite
        features are dropped.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (C,H,W), got {features.shape}")
    c, h, w = features.shape
    if label.shape != (h, w):
        raise ValueError(f"label {label.shape} does not match features {(h, w)}")
    rng = np.random.default_rng(seed)
    feat_flat = features.reshape(c, -1).T  # (H*W, C)
    lab_flat = label.reshape(-1)
    finite = np.all(np.isfinite(feat_flat), axis=1)
    base = finite & (lab_flat != ignore_index)
    if restrict_mask is not None:
        base &= restrict_mask.reshape(-1).astype(bool)

    xs, ys = [], []
    for cid in class_ids:
        idx = np.flatnonzero(base & (lab_flat == cid))
        if idx.size == 0:
            continue
        if idx.size > n_per_class:
            idx = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(feat_flat[idx])
        ys.append(np.full(idx.size, cid, dtype="int64"))
    if not xs:
        raise ValueError("no valid pixels sampled; check labels/mask/features")
    X = np.concatenate(xs).astype("float32")
    y = np.concatenate(ys)
    # Shuffle so class order does not bias models that are order-sensitive.
    order = rng.permutation(len(y))
    return X[order], y[order]


__all__ = ["sample_pixels"]
