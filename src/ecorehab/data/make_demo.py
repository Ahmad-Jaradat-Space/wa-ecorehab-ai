"""One-shot generator for the synthetic demo AOI (composite + vectors + truth).

Convenience wrapper so a single command bootstraps everything the offline
pipeline needs. Equivalent to running ``build_composites`` + ``download_vectors``
in demo mode.

CLI:
    python -m ecorehab.data.make_demo --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse

from ecorehab.data.build_composites import build_composite
from ecorehab.data.download_vectors import download_vectors
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def make_demo(cfg: Config) -> None:
    """Generate the full synthetic AOI on disk."""
    if not cfg.aoi.demo:
        logger.warning("aoi.demo is False; forcing demo generation for make_demo.")
        cfg = cfg.model_copy(deep=True)
        cfg.aoi.demo = True
    download_vectors(cfg)
    build_composite(cfg)
    logger.info("Demo AOI ready under %s", cfg.resolved_paths().processed / cfg.aoi.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic demo AOI.")
    parser.add_argument("--config", required=True, help="path to AOI config YAML")
    args = parser.parse_args(argv)
    make_demo(load_config(args.config))


if __name__ == "__main__":
    main()
