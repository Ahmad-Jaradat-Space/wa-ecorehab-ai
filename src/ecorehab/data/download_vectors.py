"""Acquire vegetation + mining-tenement vectors for an AOI.

Demo mode writes the synthetic vectors generated alongside the composite. Real
mode downloads WA Government open data (CC BY 4.0):

  * Native Vegetation Extent: https://catalogue.data.wa.gov.au/dataset/native-vegetation-extent
  * Mining Tenements (DMIRS-003): https://catalogue.data.wa.gov.au/dataset/mining-tenements-dmirs-003

Real-mode acquisition uses the published ArcGIS/WFS endpoints; URLs and an AOI
bbox filter are configurable. If the network or [rs] extras are unavailable the
command fails with a clear message rather than corrupting the demo path.

CLI:
    python -m ecorehab.data.download_vectors --config configs/aoi_swan_coastal_plain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecorehab import artifacts
from ecorehab.data.demo import generate_demo_scene
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import write_vector
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)

# WA government open-data feature services (real mode). These are the public
# DMIRS/Landgate ArcGIS REST endpoints; query with an AOI bbox to keep volume low.
WA_NATIVE_VEG_URL = (
    "https://services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/"
    "Environment/MapServer"  # Native Vegetation Extent (see docs/data_sources.md)
)
WA_TENEMENTS_URL = (
    "https://services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services/"
    "Mining/MapServer"  # Mining Tenements DMIRS-003
)


def download_vectors(cfg: Config) -> dict[str, Path]:
    """Write vegetation + tenement + AOI vectors; return their paths."""
    if cfg.aoi.demo:
        return _demo_vectors(cfg)
    return _real_vectors(cfg)


def _demo_vectors(cfg: Config) -> dict[str, Path]:
    logger.info("Generating synthetic demo vectors for AOI '%s'", cfg.aoi.name)
    scene = generate_demo_scene(cfg)
    veg = write_vector(scene.vegetation, artifacts.vegetation_path(cfg))
    ten = write_vector(scene.tenements, artifacts.tenements_path(cfg))
    aoi = write_vector(scene.aoi, artifacts.aoi_boundary_path(cfg))
    artifacts.write_manifest(
        cfg,
        {
            "stage": "download_vectors",
            "mode": "demo",
            "native_vegetation": str(veg),
            "mining_tenements": str(ten),
            "n_vegetation_polygons": int(len(scene.vegetation)),
            "license": "synthetic (MIT); real layers are CC BY 4.0",
        },
    )
    logger.info("Wrote %d veg polygons, %d tenements", len(scene.vegetation), len(scene.tenements))
    return {"vegetation": veg, "tenements": ten, "aoi": aoi}


def _real_vectors(cfg: Config) -> dict[str, Path]:
    """Download WA open vectors clipped to the AOI (requires network + [rs])."""
    import geopandas as gpd

    from ecorehab.data.build_composites import _aoi_bbox_lonlat

    bbox = _aoi_bbox_lonlat(cfg)
    logger.info("Downloading WA Native Vegetation Extent + Mining Tenements for %s", bbox)
    # NOTE: the exact ArcGIS layer ids change over time; see docs/data_sources.md
    # for the current catalogue links. We read via the OGC/ArcGIS query interface.
    try:
        veg_gdf = gpd.read_file(WA_NATIVE_VEG_URL, bbox=bbox)
        ten_gdf = gpd.read_file(WA_TENEMENTS_URL, bbox=bbox)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Real vector download failed. Check network and the current WA "
            "open-data endpoints in docs/data_sources.md, or run in demo mode "
            "(aoi.demo: true). Underlying error: " + str(exc)
        ) from exc
    veg = write_vector(veg_gdf.to_crs(cfg.project.crs), artifacts.vegetation_path(cfg))
    ten = write_vector(ten_gdf.to_crs(cfg.project.crs), artifacts.tenements_path(cfg))
    artifacts.write_manifest(
        cfg,
        {
            "stage": "download_vectors",
            "mode": "real",
            "native_vegetation": str(veg),
            "mining_tenements": str(ten),
            "license": "CC BY 4.0 (WA Government)",
        },
    )
    return {"vegetation": veg, "tenements": ten}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Acquire WA vegetation + tenement vectors.")
    parser.add_argument("--config", required=True, help="path to AOI config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    download_vectors(cfg)


if __name__ == "__main__":
    main()
