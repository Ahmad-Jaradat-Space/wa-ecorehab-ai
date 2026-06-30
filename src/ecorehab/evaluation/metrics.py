"""Classification/segmentation metrics for imbalanced ecological classes.

Overall accuracy is deliberately *not* the headline: a model can score 90%
accuracy while missing a rare class entirely. We report per-class
precision/recall/F1/IoU plus macro/weighted aggregates and balanced accuracy.
``ignore_index`` pixels are excluded everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ecorehab import constants


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int = constants.IGNORE_INDEX,
) -> np.ndarray:
    """Row=true, col=pred confusion matrix (ignore pixels removed)."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    valid = (yt != ignore_index) & np.isfinite(yp.astype("float64"))
    yt, yp = yt[valid].astype(int), yp[valid].astype(int)
    cm = np.zeros((num_classes, num_classes), dtype="int64")
    inb = (yt >= 0) & (yt < num_classes) & (yp >= 0) & (yp < num_classes)
    np.add.at(cm, (yt[inb], yp[inb]), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    """Per-class precision/recall/F1/IoU + aggregates from a confusion matrix."""
    cm = cm.astype("float64")
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)
    true_sum = cm.sum(axis=1)
    fp = pred_sum - tp
    fn = true_sum - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = _safe(tp, tp + fp)
        recall = _safe(tp, tp + fn)
        f1 = _safe(2 * precision * recall, precision + recall)
        iou = _safe(tp, tp + fp + fn)

    support = true_sum
    total = cm.sum()
    overall_acc = float(tp.sum() / total) if total else 0.0
    # balanced accuracy = mean recall over classes that have support
    present = support > 0
    balanced_acc = float(recall[present].mean()) if present.any() else 0.0
    weights = support / support.sum() if support.sum() else np.zeros_like(support)

    return {
        "per_class": {
            int(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "iou": float(iou[i]),
                "support": int(support[i]),
            }
            for i in range(cm.shape[0])
        },
        "macro_f1": float(np.nanmean(_present(f1, present))),
        "weighted_f1": float(np.nansum(f1 * weights)),
        "mean_iou": float(np.nanmean(_present(iou, present))),
        "macro_precision": float(np.nanmean(_present(precision, present))),
        "macro_recall": float(np.nanmean(_present(recall, present))),
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
    }


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
    ignore_index: int = constants.IGNORE_INDEX,
) -> dict[str, Any]:
    """End-to-end metrics dict from label/prediction arrays."""
    cm = confusion_matrix(y_true, y_pred, num_classes, ignore_index)
    out = metrics_from_confusion(cm)
    out["confusion_matrix"] = cm.tolist()
    if class_names:
        out["class_names"] = list(class_names)
        named = {}
        for i, name in enumerate(class_names):
            if i in out["per_class"]:
                named[name] = out["per_class"][i]
        out["per_class_named"] = named
    return out


def _safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype="float64"), where=den > 0)


def _present(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Return values only for classes with support (NaN elsewhere) for averaging."""
    out = np.where(present, values, np.nan)
    return out


__all__ = [
    "confusion_matrix",
    "metrics_from_confusion",
    "segmentation_metrics",
]
