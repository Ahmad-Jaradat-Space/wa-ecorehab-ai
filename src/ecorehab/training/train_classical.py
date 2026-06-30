"""Train a classical pixel baseline (RandomForest / XGBoost / LightGBM).

Pixels are sampled (class-balanced) from the *train* tiles only, so the spatial
split is respected. The model is evaluated on the *test* tiles against the
honest target (demo truth, or weak labels in real mode), and a full-AOI
probability + prediction map is written for the notebook/report.

CLI:
    python -m ecorehab.training.train_classical --config configs/random_forest.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from ecorehab import constants, pipeline
from ecorehab.evaluation.calibration import brier_score, expected_calibration_error
from ecorehab.evaluation.metrics import segmentation_metrics
from ecorehab.labels.sampling import sample_pixels
from ecorehab.models.classical import ClassicalPixelModel, build_classical_model
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import save_json, write_raster
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def train_classical(cfg: Config) -> dict:
    """Train, evaluate, and persist a classical pixel model. Returns metrics."""
    feats, fnames, comp = pipeline.load_feature_stack(cfg)
    label = pipeline.load_label(cfg)
    tiles = pipeline.load_tile_index(cfg)
    class_ids = [c.id for c in cfg.labels.classes]
    class_names = [c.name for c in cfg.labels.classes]
    shape = label.shape

    train_mask = pipeline.split_mask(tiles, "train", shape)
    test_mask = pipeline.split_mask(tiles, "test", shape)

    X, y = sample_pixels(
        feats,
        label,
        class_ids,
        cfg.classical.n_pixels_per_class,
        ignore_index=cfg.labels.ignore_index,
        seed=cfg.project.seed,
        restrict_mask=train_mask,
    )
    logger.info("Sampled %d training pixels (%s)", len(y), np.bincount(y).tolist())

    estimator = build_classical_model(
        cfg.classical.model_type, cfg.classical.params, cfg.project.seed
    )
    model = ClassicalPixelModel(estimator, fnames, class_ids, cfg.classical.model_type).fit(X, y)
    ckpt = model.save(cfg.classical.checkpoint)
    logger.info("Saved %s model -> %s", cfg.classical.model_type, ckpt)

    proba, pred = model.predict_proba_raster(feats, nodata_value=constants.PROB_NODATA)

    # Evaluate on the test split against the honest target.
    # nodata pixels carry pred == PROB_NODATA (a *finite* sentinel), so guard on
    # the sentinel explicitly rather than np.isfinite.
    target_kind, target = pipeline.evaluation_target(cfg)
    predicted = pred != constants.PROB_NODATA
    eval_pred = np.where(test_mask & predicted, pred, cfg.labels.ignore_index).astype("int64")
    metrics = segmentation_metrics(
        target, eval_pred, len(class_ids), class_names, cfg.labels.ignore_index
    )
    metrics["evaluation_target"] = target_kind

    # Calibration on the test pixels (exclude nodata predictions).
    valid = test_mask & (target != cfg.labels.ignore_index) & predicted
    if valid.any():
        P = proba.transpose(1, 2, 0)[valid]
        yt = target[valid]
        metrics["ece"] = expected_calibration_error(P, yt)
        metrics["brier"] = brier_score(P, yt, len(class_ids))

    # Persist outputs for the notebook / report.
    _write_outputs(cfg, proba, pred, comp, metrics)
    logger.info(
        "%s test: macroF1=%.3f meanIoU=%.3f balAcc=%.3f (target=%s)",
        cfg.classical.model_type,
        metrics["macro_f1"],
        metrics["mean_iou"],
        metrics["balanced_accuracy"],
        target_kind,
    )
    return metrics


def _write_outputs(cfg: Config, proba, pred, comp, metrics) -> None:
    out_pred = cfg.resolved_paths().outputs / "predictions"
    out_metrics = cfg.resolved_paths().outputs / "metrics"
    name = cfg.classical.model_type
    pred_u8 = np.where(pred != constants.PROB_NODATA, pred, constants.IGNORE_INDEX).astype("uint8")
    write_raster(
        out_pred / f"{name}_prediction.tif",
        pred_u8,
        comp.transform,
        comp.crs,
        nodata=constants.IGNORE_INDEX,
        band_names=("prediction",),
    )
    write_raster(
        out_pred / f"{name}_probability.tif",
        proba.astype("float32"),
        comp.transform,
        comp.crs,
        nodata=constants.PROB_NODATA,
        band_names=tuple(c.name for c in cfg.labels.classes),
    )
    save_json(metrics, out_metrics / f"{name}_metrics.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train classical pixel baseline.")
    parser.add_argument("--config", required=True, help="path to a classical config YAML")
    args = parser.parse_args(argv)
    train_classical(load_config(args.config))


if __name__ == "__main__":
    main()
