"""Imbalanced metrics, calibration, uncertainty, spatial CV, area estimation."""

from ecorehab.evaluation.area_estimation import (
    area_by_class_ha,
    area_by_polygon,
    bootstrap_area_ci,
)
from ecorehab.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from ecorehab.evaluation.metrics import (
    confusion_matrix,
    segmentation_metrics,
)
from ecorehab.evaluation.uncertainty import predictive_entropy

__all__ = [
    "area_by_class_ha",
    "area_by_polygon",
    "bootstrap_area_ci",
    "brier_score",
    "confusion_matrix",
    "expected_calibration_error",
    "predictive_entropy",
    "reliability_curve",
    "segmentation_metrics",
]
