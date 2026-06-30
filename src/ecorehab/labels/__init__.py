"""Weak-label generation, rasterization, tiling, and spatial splits."""

from ecorehab.labels.rasterize import polygons_to_mask, rasterize_geometries
from ecorehab.labels.sampling import sample_pixels
from ecorehab.labels.tiling import (
    Tile,
    build_tile_index,
    generate_tiles,
    spatial_block_split,
)
from ecorehab.labels.weak_labels import build_weak_labels

__all__ = [
    "Tile",
    "build_tile_index",
    "build_weak_labels",
    "generate_tiles",
    "polygons_to_mask",
    "rasterize_geometries",
    "sample_pixels",
    "spatial_block_split",
]
