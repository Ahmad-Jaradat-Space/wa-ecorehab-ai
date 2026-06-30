"""Unified ``ecorehab`` command-line entry point.

Thin dispatcher over the per-stage modules so the whole pipeline is reachable
from one command (and the Docker image's entrypoint):

    ecorehab make-demo         --config configs/aoi_swan_coastal_plain.yaml
    ecorehab build-composites  --config configs/aoi_swan_coastal_plain.yaml
    ecorehab download-vectors  --config configs/aoi_swan_coastal_plain.yaml
    ecorehab build-labels      --config configs/aoi_swan_coastal_plain.yaml
    ecorehab build-tiles       --config configs/aoi_swan_coastal_plain.yaml
    ecorehab train-classical   --config configs/random_forest.yaml
    ecorehab train-segmentation --config configs/unet.yaml
    ecorehab predict           --config configs/inference.yaml
    ecorehab report            --config configs/report.yaml
    ecorehab pipeline          --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse

from ecorehab.utils.config import load_config
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def _run_pipeline(config: str) -> None:
    """Run the full demo pipeline end-to-end against the AOI in ``config``.

    The model/inference/report configs are resolved next to ``config`` (so this
    works from any CWD) and have their AOI-determining sections overridden by the
    chosen AOI, so ``ecorehab pipeline --config configs/aoi_wheatbelt.yaml`` runs
    every stage on the Wheatbelt AOI rather than the swan default.
    """
    from pathlib import Path

    from ecorehab.data.make_demo import make_demo
    from ecorehab.inference.batch_predict import batch_predict
    from ecorehab.labels.build_labels import build_labels
    from ecorehab.labels.build_tiles import build_tiles
    from ecorehab.reporting.build_site_report import build_site_report
    from ecorehab.training.train_classical import train_classical
    from ecorehab.training.train_segmentation import train_segmentation

    cfg = load_config(config)
    cfg_dir = Path(config).resolve().parent

    def _for_aoi(name: str):
        """Load a sibling config and bind it to the chosen AOI."""
        sub = load_config(cfg_dir / name)
        for section in ("project", "paths", "aoi", "data", "labels", "tiling", "features"):
            setattr(sub, section, getattr(cfg, section))
        return sub

    make_demo(cfg)
    build_labels(cfg)
    build_tiles(cfg)
    train_classical(_for_aoi("random_forest.yaml"))
    train_segmentation(_for_aoi("unet.yaml"))
    batch_predict(_for_aoi("inference.yaml"))
    build_site_report(_for_aoi("report.yaml"))
    logger.info("Pipeline complete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ecorehab", description="WA EcoRehab AI pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in [
        "make-demo",
        "build-composites",
        "download-vectors",
        "build-labels",
        "build-tiles",
        "train-classical",
        "train-segmentation",
        "predict",
        "report",
        "pipeline",
    ]:
        p = sub.add_parser(name, help=f"run the {name} stage")
        p.add_argument("--config", required=True, help="path to a YAML config")

    args = parser.parse_args(argv)
    cmd, config = args.command, args.config

    if cmd == "make-demo":
        from ecorehab.data.make_demo import make_demo

        make_demo(load_config(config))
    elif cmd == "build-composites":
        from ecorehab.data.build_composites import build_composite

        build_composite(load_config(config))
    elif cmd == "download-vectors":
        from ecorehab.data.download_vectors import download_vectors

        download_vectors(load_config(config))
    elif cmd == "build-labels":
        from ecorehab.labels.build_labels import build_labels

        build_labels(load_config(config))
    elif cmd == "build-tiles":
        from ecorehab.labels.build_tiles import build_tiles

        build_tiles(load_config(config))
    elif cmd == "train-classical":
        from ecorehab.training.train_classical import train_classical

        train_classical(load_config(config))
    elif cmd == "train-segmentation":
        from ecorehab.training.train_segmentation import train_segmentation

        train_segmentation(load_config(config))
    elif cmd == "predict":
        from ecorehab.inference.batch_predict import batch_predict

        batch_predict(load_config(config))
    elif cmd == "report":
        from ecorehab.reporting.build_site_report import build_site_report

        build_site_report(load_config(config))
    elif cmd == "pipeline":
        _run_pipeline(config)


if __name__ == "__main__":
    main()
