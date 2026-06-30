"""Lightweight context / overlay maps (no heavy basemap dependencies).

For a polished interactive map use leafmap/folium (the [viz] extra); these
matplotlib helpers keep the core dependency-light and CI-friendly.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from matplotlib import pyplot as plt

from ecorehab import constants

# Rough Western Australia extent in WGS84 for a schematic context map.
_WA_EXTENT = (112.5, -35.5, 129.5, -13.5)  # (minlon, minlat, maxlon, maxlat)


def aoi_location_map(aoi: gpd.GeoDataFrame, ax=None, label: str = "AOI") -> Any:
    """Schematic map of WA with the AOI marked (states CRS as WGS84)."""
    ax = ax or plt.gca()
    aoi_ll = aoi.to_crs(constants.GEOGRAPHIC_CRS)
    minlon, minlat, maxlon, maxlat = _WA_EXTENT
    ax.add_patch(
        plt.Rectangle(
            (minlon, minlat),
            maxlon - minlon,
            maxlat - minlat,
            fill=False,
            edgecolor="grey",
            linewidth=1.0,
        )
    )
    aoi_ll.boundary.plot(ax=ax, color="red", linewidth=1.5)
    c = aoi_ll.geometry.iloc[0].centroid
    ax.plot(c.x, c.y, "r*", markersize=14)
    ax.annotate(label, (c.x, c.y), textcoords="offset points", xytext=(8, 8), color="red")
    ax.set_xlim(minlon - 1, maxlon + 1)
    ax.set_ylim(minlat - 1, maxlat + 1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Western Australia - AOI location (EPSG:4326)")
    ax.set_aspect("equal")
    return ax


def overlay_polygons(ax, gdf: gpd.GeoDataFrame, color="yellow", linewidth=0.8, **kw) -> Any:
    """Overlay polygon boundaries on an existing axis (image or map)."""
    gdf.boundary.plot(ax=ax, color=color, linewidth=linewidth, **kw)
    return ax


def split_map(tile_index: gpd.GeoDataFrame, ax=None) -> Any:
    """Map of train/val/test tiles coloured by split (EPSG noted from CRS)."""
    ax = ax or plt.gca()
    colors = {"train": "#377eb8", "val": "#ff7f00", "test": "#4daf4a"}
    for split, color in colors.items():
        sub = tile_index[tile_index["split"] == split]
        if not sub.empty:
            sub.plot(ax=ax, facecolor=color, edgecolor="white", alpha=0.7, label=split)
    ax.set_title(f"Spatial split ({tile_index.crs})")
    ax.legend()
    ax.set_axis_off()
    return ax


__all__ = ["aoi_location_map", "overlay_polygons", "split_map"]
