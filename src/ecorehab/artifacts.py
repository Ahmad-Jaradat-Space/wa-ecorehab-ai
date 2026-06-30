"""Canonical artifact paths + data manifest.

Every pipeline stage agrees on where intermediate products live by going through
these helpers, so ``build_composites`` writes the composite where
``build_labels`` and ``build_tiles`` expect to read it. Processed artifacts live
under ``data/processed/<aoi>/`` (gitignored); only small example PNGs and the
optional ``data/sample/`` fixture are committed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ecorehab.utils.config import Config
from ecorehab.utils.io import save_json


def aoi_dir(cfg: Config) -> Path:
    p = cfg.resolved_paths().processed / cfg.aoi.name
    p.mkdir(parents=True, exist_ok=True)
    return p


def composite_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "composite.tif"


def vegetation_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "native_vegetation.gpkg"


def tenements_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "mining_tenements.gpkg"


def aoi_boundary_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "aoi.gpkg"


def label_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "labels.tif"


def label_confidence_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "label_confidence.tif"


def truth_path(cfg: Config) -> Path:
    """Hidden ground-truth label, only available in demo mode (for honest eval)."""
    return aoi_dir(cfg) / "truth_labels.tif"


def tiles_dir(cfg: Config) -> Path:
    p = aoi_dir(cfg) / "tiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tile_index_path(cfg: Config) -> Path:
    return tiles_dir(cfg) / "tile_index.geojson"


def manifest_path(cfg: Config) -> Path:
    return aoi_dir(cfg) / "data_manifest.json"


def write_manifest(cfg: Config, entries: dict[str, object]) -> Path:
    """Append/update the data manifest with provenance for reproducibility."""
    path = manifest_path(cfg)
    manifest = {
        "aoi": cfg.aoi.name,
        "demo_mode": cfg.aoi.demo,
        "crs": cfg.project.crs,
        "resolution_m": cfg.project.resolution_m,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **entries,
    }
    save_json(manifest, path)
    return path


__all__ = [
    "aoi_boundary_path",
    "aoi_dir",
    "composite_path",
    "label_confidence_path",
    "label_path",
    "manifest_path",
    "tenements_path",
    "tile_index_path",
    "tiles_dir",
    "truth_path",
    "vegetation_path",
    "write_manifest",
]
