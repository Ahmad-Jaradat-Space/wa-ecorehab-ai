"""Generate the tile index with a spatial-block split.

CLI:
    python -m ecorehab.labels.build_tiles --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecorehab import artifacts
from ecorehab.labels.tiling import build_tile_index
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import read_raster, write_vector
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def build_tiles(cfg: Config) -> Path:
    """Build and write the tile index GeoDataFrame. Returns its path."""
    label_path = artifacts.label_path(cfg)
    if not label_path.exists():
        raise FileNotFoundError(f"labels not found at {label_path}; run build_labels first.")
    label = read_raster(label_path)
    class_ids = [c.id for c in cfg.labels.classes]
    split = cfg.tiling.split
    index = build_tile_index(
        label.array[0],
        label.transform,
        label.crs,
        tile_size=cfg.tiling.tile_size,
        stride=cfg.tiling.stride,
        class_ids=class_ids,
        ignore_index=cfg.labels.ignore_index,
        split_method=split.method,
        block_size_px=split.block_size_px,
        val_fraction=split.val_fraction,
        test_fraction=split.test_fraction,
        min_valid_fraction=cfg.tiling.min_valid_fraction,
        seed=split.seed,
    )
    out = artifacts.tile_index_path(cfg)
    write_vector(index, out)
    counts = index["split"].value_counts().to_dict()
    artifacts.write_manifest(
        cfg,
        {
            "stage": "build_tiles",
            "tile_index": str(out),
            "n_tiles": int(len(index)),
            "tile_size": cfg.tiling.tile_size,
            "split_method": split.method,
            "split_counts": counts,
        },
    )
    logger.info("Wrote %d tiles -> %s | splits=%s", len(index), out, counts)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate tile index + spatial split.")
    parser.add_argument("--config", required=True, help="path to AOI config YAML")
    args = parser.parse_args(argv)
    build_tiles(load_config(args.config))


if __name__ == "__main__":
    main()
