"""DEA STAC access for Sentinel-2 ARD (real mode).

This is the real-data path. It is intentionally dependency-light and degrades
with a clear message if the optional [rs] extras (pystac-client + odc-stac) are
absent or there is no network, so the offline demo path is never blocked.

Reference:
  https://knowledge.dea.ga.gov.au/notebooks/How_to_guides/Downloading_data_with_STAC/
"""

from __future__ import annotations

from typing import Any

from ecorehab.utils.config import Config
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)

# DEA Sentinel-2 ARD asset names -> our canonical band order (constants.BAND_NAMES).
DEA_S2_ASSETS = {
    "blue": "nbart_blue",
    "green": "nbart_green",
    "red": "nbart_red",
    "rededge": "nbart_red_edge_1",
    "nir": "nbart_nir_1",
    "swir1": "nbart_swir_2",
    "swir2": "nbart_swir_3",
}


def search_items(cfg: Config, bbox_lonlat: tuple[float, float, float, float]) -> list[Any]:
    """Search DEA STAC for Sentinel-2 ARD items over a WGS84 bbox.

    Returns a list of pystac Items. Requires the [rs] extra.
    """
    try:
        from pystac_client import Client
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pystac-client is required for real-mode STAC search. "
            "Install with: pip install '.[rs]'"
        ) from exc

    client = Client.open(cfg.data.stac_url)
    search = client.search(
        collections=[cfg.data.collection],
        bbox=list(bbox_lonlat),
        datetime=f"{cfg.data.date_start}/{cfg.data.date_end}",
        query={"eo:cloud_cover": {"lt": cfg.data.max_cloud_cover}},
    )
    items = list(search.items())
    logger.info("DEA STAC returned %d items for %s", len(items), cfg.aoi.name)
    return items


def load_composite(cfg: Config, bbox_lonlat: tuple[float, float, float, float]):
    """Load + composite Sentinel-2 ARD via odc-stac into an xarray Dataset.

    Returns a band-named, reprojected, median-composited dataset in the project
    CRS. Requires the [rs] extra (odc-stac). See module docstring for references.
    """
    try:
        import odc.stac
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "odc-stac is required for real-mode imagery loading. "
            "Install with: pip install '.[rs]'"
        ) from exc

    items = search_items(cfg, bbox_lonlat)
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 items found for {cfg.aoi.name} in "
            f"{cfg.data.date_start}..{cfg.data.date_end} (cloud < {cfg.data.max_cloud_cover}%). "
            "Widen the date range or relax max_cloud_cover."
        )
    assets = [DEA_S2_ASSETS[b] for b in cfg.data.bands]
    ds = odc.stac.load(
        items,
        bands=assets,
        crs=cfg.project.crs,
        resolution=cfg.project.resolution_m,
        bbox=list(bbox_lonlat),
        chunks={},
    )
    # Median composite over time -> robust to clouds.
    composite = ds.median(dim="time", skipna=True)
    return composite


__all__ = ["DEA_S2_ASSETS", "load_composite", "search_items"]
