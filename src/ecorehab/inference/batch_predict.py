"""Batch inference CLI: composite -> prediction/probability/uncertainty rasters,
polygons, and uncertainty-aware area summaries.

CLI:
    python -m ecorehab.inference.batch_predict --config configs/inference.yaml

Outputs (under outputs/):
    predictions/prediction_mask.tif
    predictions/native_vegetation_probability.tif
    predictions/uncertainty.tif
    vector_outputs/predicted_polygons.gpkg
    vector_outputs/area_summary.csv                (by class, with bootstrap CI)
    vector_outputs/area_summary_by_polygon.csv     (by tenement, if available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecorehab import artifacts, constants, pipeline
from ecorehab.evaluation.area_estimation import area_by_polygon, bootstrap_area_ci
from ecorehab.evaluation.uncertainty import predictive_entropy
from ecorehab.inference.polygonize import polygonize_prediction
from ecorehab.inference.tile_inference import (
    load_segmentation_checkpoint,
    predict_raster_windowed,
)
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import read_vector, write_raster, write_vector
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def _native_class_id(class_names: list[str], class_ids: list[int]) -> int:
    for cid, name in zip(class_ids, class_names, strict=True):
        if "native" in name.lower():
            return cid
    return class_ids[min(1, len(class_ids) - 1)]


def batch_predict(cfg: Config) -> dict[str, Path]:
    """Run batch inference and write all rasters/vectors. Returns output paths."""
    inf = cfg.inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    composite_path = artifacts.composite_path(cfg)
    if not composite_path.exists():
        raise FileNotFoundError(f"composite not found: {composite_path}. Run make_demo first.")

    if inf.model_kind == "segmentation":
        proba, pred, transform, crs, class_ids, class_names = _predict_segmentation(
            cfg, composite_path, device
        )
    else:
        proba, pred, transform, crs, class_ids, class_names = _predict_classical(cfg)

    root = cfg.resolved_paths().root
    out_dir = inf.out_dir if inf.out_dir.is_absolute() else (root / inf.out_dir)
    vec_dir = (
        inf.vector_out_dir if inf.vector_out_dir.is_absolute() else (root / inf.vector_out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1) prediction mask
    paths["prediction"] = write_raster(
        out_dir / "prediction_mask.tif",
        pred.astype("uint8"),
        transform,
        crs,
        nodata=cfg.labels.ignore_index,
        band_names=("prediction",),
    )
    # 2) native-vegetation probability
    if inf.write_probabilities:
        nid = _native_class_id(class_names, class_ids)
        band = proba[class_ids.index(nid)].astype("float32")
        paths["probability"] = write_raster(
            out_dir / "native_vegetation_probability.tif",
            band,
            transform,
            crs,
            nodata=constants.PROB_NODATA,
            band_names=("p_native",),
        )
    # 3) uncertainty (predictive entropy)
    if inf.write_uncertainty:
        ent = predictive_entropy(proba)
        ent[pred == cfg.labels.ignore_index] = constants.PROB_NODATA
        paths["uncertainty"] = write_raster(
            out_dir / "uncertainty.tif",
            ent,
            transform,
            crs,
            nodata=constants.PROB_NODATA,
            band_names=("entropy",),
        )
    # 4) polygonise
    name_map = dict(zip(class_ids, class_names, strict=True))
    if inf.polygonize:
        polys = polygonize_prediction(
            pred, transform, crs, name_map, cfg.labels.min_polygon_area_ha, cfg.labels.ignore_index
        )
        paths["polygons"] = write_vector(polys, vec_dir / "predicted_polygons.gpkg")
        logger.info("Polygonised %d features", len(polys))
    # 5) area summaries
    paths["area_summary"] = _write_area_summary(cfg, pred, transform, class_ids, name_map, vec_dir)
    paths["area_by_polygon"] = _write_area_by_polygon(
        cfg, pred, transform, crs, class_ids, name_map, vec_dir
    )
    logger.info("Batch inference complete -> %s", out_dir)
    return paths


def _predict_segmentation(cfg, composite_path, device):
    model, meta = load_segmentation_checkpoint(cfg.inference.checkpoint, device)
    proba, pred, transform, crs = predict_raster_windowed(
        composite_path,
        model,
        meta,
        tile_size=cfg.inference.tile_size,
        overlap=cfg.inference.overlap,
        device=device,
        tta=cfg.inference.tta,
    )
    return proba, pred, transform, crs, meta["class_ids"], meta["class_names"]


def _predict_classical(cfg):
    from ecorehab.models.classical import ClassicalPixelModel

    feats, _, comp = pipeline.load_feature_stack(cfg)
    model = ClassicalPixelModel.load(cfg.inference.checkpoint)
    proba, pred = model.predict_proba_raster(feats, nodata_value=constants.PROB_NODATA)
    pred = np.where(pred != constants.PROB_NODATA, pred, cfg.labels.ignore_index).astype("uint8")
    names = [c.name for c in cfg.labels.classes]
    return proba, pred, comp.transform, comp.crs, model.class_ids, names


def _write_area_summary(cfg, pred, transform, class_ids, name_map, vec_dir) -> Path:
    rows = []
    for cid in class_ids:
        ci = bootstrap_area_ci(
            pred,
            transform,
            cid,
            n_boot=300,
            ignore_index=cfg.labels.ignore_index,
            seed=cfg.project.seed,
        )
        rows.append(
            {
                "class_id": cid,
                "class_name": name_map[cid],
                "area_ha": round(ci.point_ha, 2),
                "area_lower_ha": round(ci.lower_ha, 2),
                "area_upper_ha": round(ci.upper_ha, 2),
                "ci_level": ci.confidence_level,
            }
        )
    df = pd.DataFrame(rows)
    path = vec_dir / "area_summary.csv"
    df.to_csv(path, index=False)
    logger.info("Area summary:\n%s", df.to_string(index=False))
    return path


def _write_area_by_polygon(cfg, pred, transform, crs, class_ids, name_map, vec_dir) -> Path | None:
    ten_path = artifacts.tenements_path(cfg)
    if not ten_path.exists():
        return None
    tenements = read_vector(ten_path)
    id_field = "tenement_id" if "tenement_id" in tenements.columns else None
    df = area_by_polygon(
        pred, transform, crs, tenements, class_ids, name_map, id_field, cfg.labels.ignore_index
    )
    path = vec_dir / "area_summary_by_polygon.csv"
    df.round(2).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tiled batch inference + area reporting.")
    parser.add_argument("--config", required=True, help="path to inference config YAML")
    args = parser.parse_args(argv)
    batch_predict(load_config(args.config))


if __name__ == "__main__":
    main()
