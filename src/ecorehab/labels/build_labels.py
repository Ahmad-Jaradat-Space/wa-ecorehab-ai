"""Build weak labels + confidence raster from the composite and vegetation polygons.

CLI:
    python -m ecorehab.labels.build_labels --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecorehab import artifacts, constants
from ecorehab.labels.weak_labels import build_weak_labels
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import read_raster, read_vector, write_raster
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def build_labels(cfg: Config) -> Path:
    """Generate and write labels.tif + label_confidence.tif. Returns label path."""
    comp_path = artifacts.composite_path(cfg)
    veg_path = artifacts.vegetation_path(cfg)
    if not comp_path.exists():
        raise FileNotFoundError(
            f"composite not found at {comp_path}; run build_composites (or make_demo) first."
        )
    if not veg_path.exists():
        raise FileNotFoundError(
            f"vegetation vectors not found at {veg_path}; run download_vectors first."
        )
    composite = read_raster(comp_path)
    vegetation = read_vector(veg_path)
    logger.info("Building weak labels for '%s' (%d veg polygons)", cfg.aoi.name, len(vegetation))

    result = build_weak_labels(composite, vegetation, cfg)

    label_path = artifacts.label_path(cfg)
    write_raster(
        label_path,
        result.label,
        composite.transform,
        composite.crs,
        nodata=cfg.labels.ignore_index,
        band_names=("label",),
    )
    write_raster(
        artifacts.label_confidence_path(cfg),
        result.confidence.astype("float32"),
        composite.transform,
        composite.crs,
        nodata=constants.PROB_NODATA,
        band_names=("confidence",),
    )
    artifacts.write_manifest(
        cfg,
        {
            "stage": "build_labels",
            "label_raster": str(label_path),
            "class_pixel_counts": result.class_counts,
            "ignore_index": cfg.labels.ignore_index,
            "buffers_m": {
                "positive_inward": cfg.labels.positive_inward_buffer_m,
                "boundary_exclusion": cfg.labels.boundary_exclusion_buffer_m,
            },
        },
    )
    logger.info("Wrote labels -> %s | counts=%s", label_path, result.class_counts)
    return label_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build weak labels for an AOI.")
    parser.add_argument("--config", required=True, help="path to AOI config YAML")
    args = parser.parse_args(argv)
    build_labels(load_config(args.config))


if __name__ == "__main__":
    main()
