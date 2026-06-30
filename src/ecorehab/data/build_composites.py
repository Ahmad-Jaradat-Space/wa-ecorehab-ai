"""Build the Sentinel-2-style surface-reflectance composite for an AOI.

Demo mode writes the synthetic composite (and the hidden truth label, used only
for honest demo evaluation). Real mode pulls DEA Sentinel-2 ARD via STAC.

CLI:
    python -m ecorehab.data.build_composites --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ecorehab import artifacts, constants
from ecorehab.data.demo import generate_demo_scene
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import write_raster, write_vector
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def build_composite(cfg: Config) -> Path:
    """Write the composite GeoTIFF (demo or real) and return its path."""
    if cfg.aoi.demo:
        return _build_demo_composite(cfg)
    return _build_real_composite(cfg)


def _build_demo_composite(cfg: Config) -> Path:
    logger.info("Generating synthetic demo composite for AOI '%s'", cfg.aoi.name)
    scene = generate_demo_scene(cfg)
    out = artifacts.composite_path(cfg)
    cube = np.where(np.isfinite(scene.composite), scene.composite, constants.REFLECTANCE_NODATA)
    write_raster(
        out,
        cube.astype("float32"),
        scene.transform,
        scene.crs,
        nodata=constants.REFLECTANCE_NODATA,
        band_names=scene.band_names,
    )
    # Hidden truth (demo only) for honest evaluation, and AOI boundary.
    truth = scene.truth.astype("uint8")
    write_raster(
        artifacts.truth_path(cfg),
        truth,
        scene.transform,
        scene.crs,
        nodata=constants.IGNORE_INDEX,
        band_names=("truth_class",),
    )
    write_vector(scene.aoi, artifacts.aoi_boundary_path(cfg))
    artifacts.write_manifest(
        cfg,
        {
            "stage": "build_composites",
            "mode": "demo",
            "composite": str(out),
            "bands": list(scene.band_names),
            "shape": list(scene.composite.shape),
            "note": "Synthetic data. Pixel values are fabricated; geospatial "
            "metadata (CRS, transform, polygons) is real.",
        },
    )
    logger.info("Wrote demo composite -> %s", out)
    return out


def _build_real_composite(cfg: Config) -> Path:
    from ecorehab.data.stac import load_composite

    bbox = _aoi_bbox_lonlat(cfg)
    logger.info("Loading real DEA Sentinel-2 composite for bbox %s", bbox)
    ds = load_composite(cfg, bbox)
    # Stack into (bands, H, W) in canonical order, scale reflectance to [0,1].
    from ecorehab.data.stac import DEA_S2_ASSETS

    arrays = [ds[DEA_S2_ASSETS[b]].values for b in cfg.data.bands]
    cube = np.stack(arrays).astype("float32") / constants.S2_REFLECTANCE_SCALE
    cube = np.where(np.isfinite(cube), cube, constants.REFLECTANCE_NODATA)
    transform = ds.odc.transform  # type: ignore[attr-defined]
    out = artifacts.composite_path(cfg)
    write_raster(
        out,
        cube,
        transform,
        cfg.project.crs,
        nodata=constants.REFLECTANCE_NODATA,
        band_names=tuple(cfg.data.bands),
    )
    artifacts.write_manifest(
        cfg,
        {"stage": "build_composites", "mode": "real", "composite": str(out), "bbox_lonlat": bbox},
    )
    logger.info("Wrote real composite -> %s", out)
    return out


def _aoi_bbox_lonlat(cfg: Config) -> tuple[float, float, float, float]:
    """Approximate WGS84 bbox for STAC search."""
    if cfg.aoi.bbox is not None:
        from pyproj import Transformer

        t = Transformer.from_crs(cfg.project.crs, constants.GEOGRAPHIC_CRS, always_xy=True)
        minx, miny, maxx, maxy = cfg.aoi.bbox
        xs, ys = t.transform([minx, maxx], [miny, maxy])
        return (min(xs), min(ys), max(xs), max(ys))
    # Build a small box around the configured centre (size from demo extent).
    lon, lat = cfg.aoi.lonlat
    half_km = (cfg.data.demo.size_px * cfg.project.resolution_m) / 2 / 1000.0
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build AOI composite (demo or DEA Sentinel-2).")
    parser.add_argument("--config", required=True, help="path to AOI config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    build_composite(cfg)


if __name__ == "__main__":
    main()
