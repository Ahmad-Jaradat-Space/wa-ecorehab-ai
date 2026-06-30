"""Classical pixel-wise models (RandomForest / XGBoost / LightGBM).

These operate on per-pixel spectral + index features. RandomForest needs only
scikit-learn; XGBoost/LightGBM are optional ([boost] extra). The
:class:`ClassicalPixelModel` wraps an estimator and adds geospatial-aware raster
prediction that respects nodata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_classical_model(model_type: str, params: dict[str, Any], seed: int = 42) -> Any:
    """Construct a scikit-learn-compatible classifier with ``predict_proba``."""
    model_type = model_type.lower()
    params = dict(params)
    if model_type == "random_forest":
        params.setdefault("random_state", seed)
        return RandomForestClassifier(**params)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is required for model_type='xgboost'. Install: pip install '.[boost]'"
            ) from exc
        params.setdefault("random_state", seed)
        params.setdefault("eval_metric", "mlogloss")
        return XGBClassifier(**params)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "lightgbm is required for model_type='lightgbm'. Install: pip install '.[boost]'"
            ) from exc
        params.setdefault("random_state", seed)
        return LGBMClassifier(**params)
    raise ValueError(f"unknown model_type {model_type!r}")


@dataclass
class ClassicalPixelModel:
    """An estimator + its feature/class metadata, with raster prediction."""

    estimator: Any
    feature_names: list[str]
    class_ids: list[int]
    model_type: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassicalPixelModel:
        self.estimator.fit(X, y)
        return self

    def predict_proba_raster(
        self, features: np.ndarray, nodata_value: float = -1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict over a (C, H, W) feature stack.

        Returns:
            (probabilities (n_classes, H, W), prediction (H, W)). Pixels with any
            non-finite feature are set to nodata in the prediction and 0 in probs.
        """
        c, h, w = features.shape
        flat = features.reshape(c, -1).T  # (H*W, C)
        finite = np.all(np.isfinite(flat), axis=1)
        n_classes = len(self.class_ids)
        proba = np.zeros((flat.shape[0], n_classes), dtype="float32")
        pred = np.full(flat.shape[0], nodata_value, dtype="float32")
        if finite.any():
            p = self.estimator.predict_proba(flat[finite])
            # Map estimator class order onto our contiguous class id order.
            proba[finite] = _align_proba(p, self.estimator.classes_, self.class_ids)
            pred[finite] = self.class_ids_array[np.argmax(proba[finite], axis=1)]
        proba_img = proba.T.reshape(n_classes, h, w)
        pred_img = pred.reshape(h, w)
        return proba_img, pred_img

    @property
    def class_ids_array(self) -> np.ndarray:
        return np.asarray(self.class_ids)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_names": self.feature_names,
                "class_ids": self.class_ids,
                "model_type": self.model_type,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClassicalPixelModel:
        blob = joblib.load(path)
        return cls(
            estimator=blob["estimator"],
            feature_names=blob["feature_names"],
            class_ids=blob["class_ids"],
            model_type=blob["model_type"],
        )


def _align_proba(
    proba: np.ndarray, estimator_classes: np.ndarray, class_ids: list[int]
) -> np.ndarray:
    """Reorder estimator probability columns to match ``class_ids`` order."""
    out = np.zeros((proba.shape[0], len(class_ids)), dtype="float32")
    col = {int(c): i for i, c in enumerate(estimator_classes)}
    for j, cid in enumerate(class_ids):
        if cid in col:
            out[:, j] = proba[:, col[cid]]
    return out


__all__ = ["ClassicalPixelModel", "build_classical_model"]
