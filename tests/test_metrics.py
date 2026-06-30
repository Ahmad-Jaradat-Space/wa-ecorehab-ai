"""Metric correctness on hand-computable cases + calibration."""

from __future__ import annotations

import numpy as np
import pytest

from ecorehab.evaluation.calibration import brier_score, expected_calibration_error
from ecorehab.evaluation.metrics import (
    confusion_matrix,
    metrics_from_confusion,
    segmentation_metrics,
)


def test_confusion_matrix_ignores_index():
    y_true = np.array([0, 1, 2, 255, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    cm = confusion_matrix(y_true, y_pred, num_classes=3, ignore_index=255)
    assert cm.sum() == 4  # ignored pixel dropped
    assert np.array_equal(cm, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]]))


def test_perfect_prediction_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    cm = confusion_matrix(y, y, num_classes=3)
    m = metrics_from_confusion(cm)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["mean_iou"] == pytest.approx(1.0)
    assert m["overall_accuracy"] == pytest.approx(1.0)


def test_known_precision_recall():
    # class 1: TP=2, FP=1, FN=1 -> P=R=F1=2/3, IoU=2/4=0.5
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1])
    m = segmentation_metrics(y_true, y_pred, num_classes=2)
    c1 = m["per_class"][1]
    assert c1["precision"] == pytest.approx(2 / 3)
    assert c1["recall"] == pytest.approx(2 / 3)
    assert c1["f1"] == pytest.approx(2 / 3)
    assert c1["iou"] == pytest.approx(0.5)


def test_imbalance_accuracy_vs_macro_f1():
    # 98 correct class-0, rare class-1 entirely missed.
    y_true = np.array([0] * 98 + [1, 1])
    y_pred = np.array([0] * 100)
    m = segmentation_metrics(y_true, y_pred, num_classes=2)
    assert m["overall_accuracy"] == pytest.approx(0.98)
    # macro-F1 must expose the failure (class 1 F1 = 0).
    assert m["macro_f1"] < 0.6
    assert m["per_class"][1]["recall"] == 0.0


def test_confusion_matrix_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros(4), np.zeros(5), num_classes=2)


def test_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert brier_score(probs, y) == pytest.approx(0.0)


def test_ece_in_unit_range():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=500)
    y = rng.integers(0, 3, size=500)
    ece = expected_calibration_error(probs, y, n_bins=10)
    assert 0.0 <= ece <= 1.0
