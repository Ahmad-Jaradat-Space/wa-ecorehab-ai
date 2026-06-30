"""Probability calibration diagnostics: reliability curve, ECE, Brier score.

Calibrated probabilities matter for ecological decisions: a "0.7 native" pixel
should be native ~70% of the time so an ecologist can threshold low-confidence
areas for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityCurve:
    bin_confidence: np.ndarray  # mean predicted confidence per bin
    bin_accuracy: np.ndarray  # empirical accuracy per bin
    bin_count: np.ndarray  # samples per bin
    ece: float  # expected calibration error


def _prepare(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype="float64")
    y_true = np.asarray(y_true).reshape(-1)
    if probs.ndim == 1:  # binary confidence of the predicted (positive) class
        conf = probs
        pred = (probs >= 0.5).astype(int)
    else:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    return conf, pred, y_true


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> ReliabilityCurve:
    """Top-label reliability curve + ECE.

    Args:
        probs: (N, C) class probabilities or (N,) positive-class confidence.
        y_true: (N,) integer labels aligned with ``probs``.
    """
    conf, pred, y_true = _prepare(probs, y_true)
    correct = (pred == y_true).astype("float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins, dtype="int64")
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(sel.sum())
        bin_cnt[b] = cnt
        if cnt:
            bin_conf[b] = conf[sel].mean()
            bin_acc[b] = correct[sel].mean()
            ece += (cnt / n) * abs(bin_acc[b] - bin_conf[b])
    return ReliabilityCurve(bin_conf, bin_acc, bin_cnt, float(ece))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (top-label)."""
    return reliability_curve(probs, y_true, n_bins).ece


def brier_score(probs: np.ndarray, y_true: np.ndarray, num_classes: int | None = None) -> float:
    """Multiclass Brier score: mean squared error of the probability vector."""
    probs = np.asarray(probs, dtype="float64")
    y_true = np.asarray(y_true).reshape(-1)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    k = num_classes or probs.shape[1]
    onehot = np.zeros((len(y_true), k), dtype="float64")
    valid = (y_true >= 0) & (y_true < k)
    onehot[np.arange(len(y_true))[valid], y_true[valid]] = 1.0
    return float(np.mean(np.sum((probs[:, :k] - onehot) ** 2, axis=1)))


__all__ = [
    "ReliabilityCurve",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]
