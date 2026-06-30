"""Tile generation and spatial splitting.

Tiles are stored as an *index* (a GeoDataFrame of windows + split assignment),
not as thousands of small files: the training dataset reads pixel windows
directly from the composite/label GeoTIFFs. Each tile carries its CRS bounds, so
the split is reproducible and auditable on a map.

Spatial-block splitting is the headline feature: tiles are grouped into coarse
blocks and whole blocks are assigned to train/val/test. Adjacent tiles therefore
never straddle the split boundary, which prevents the spatial autocorrelation
leakage that makes a naive random split over-optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box

from ecorehab import constants


@dataclass(frozen=True)
class Tile:
    """A single tile window in pixel coordinates."""

    row_off: int
    col_off: int
    height: int
    width: int

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    def bounds(self, transform: Affine) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in CRS units for this tile."""
        minx, maxy = transform * (self.col_off, self.row_off)
        maxx, miny = transform * (self.col_off + self.width, self.row_off + self.height)
        return (minx, miny, maxx, maxy)


def generate_tiles(
    raster_shape: tuple[int, int], tile_size: int, stride: int | None = None
) -> list[Tile]:
    """Generate tiles covering ``raster_shape`` (rows, cols).

    The last row/column of tiles is shifted inward so every tile is exactly
    ``tile_size`` (no ragged edges), at the cost of a small overlap at the border.
    """
    h, w = raster_shape
    if tile_size > h or tile_size > w:
        raise ValueError(f"tile_size {tile_size} larger than raster {raster_shape}")
    stride = stride or tile_size
    rows = _offsets(h, tile_size, stride)
    cols = _offsets(w, tile_size, stride)
    return [Tile(r, c, tile_size, tile_size) for r in rows for c in cols]


def _offsets(extent: int, size: int, stride: int) -> list[int]:
    offs = list(range(0, extent - size + 1, stride))
    last = extent - size
    if not offs or offs[-1] != last:
        offs.append(last)
    return offs


def spatial_block_split(
    tiles: list[Tile],
    block_size_px: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = constants.DEFAULT_SEED,
) -> list[str]:
    """Assign each tile to train/val/test by the coarse block it belongs to.

    Returns a list of split labels aligned with ``tiles``. Tiles in the same
    block always share a split -> no spatial leakage across the split boundary.
    """
    rng = np.random.default_rng(seed)
    block_ids = [(t.row_off // block_size_px, t.col_off // block_size_px) for t in tiles]
    unique_blocks = sorted(set(block_ids))
    perm = rng.permutation(len(unique_blocks))
    n = len(unique_blocks)
    n_test = max(1, int(round(test_fraction * n))) if test_fraction > 0 else 0
    n_val = max(1, int(round(val_fraction * n))) if val_fraction > 0 else 0
    n_val = min(n_val, max(0, n - n_test - 1))  # keep at least one train block
    assign: dict[tuple[int, int], str] = {}
    for rank, block_idx in enumerate(perm):
        blk = unique_blocks[block_idx]
        if rank < n_test:
            assign[blk] = "test"
        elif rank < n_test + n_val:
            assign[blk] = "val"
        else:
            assign[blk] = "train"
    return [assign[b] for b in block_ids]


def random_split(
    n_tiles: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = constants.DEFAULT_SEED,
) -> list[str]:
    """Naive per-tile random split (for the leakage comparison, NOT primary use)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tiles)
    n_test = int(round(test_fraction * n_tiles))
    n_val = int(round(val_fraction * n_tiles))
    splits = ["train"] * n_tiles
    for rank, idx in enumerate(perm):
        if rank < n_test:
            splits[idx] = "test"
        elif rank < n_test + n_val:
            splits[idx] = "val"
    return splits


def tile_class_fractions(
    label_tile: np.ndarray, class_ids: list[int], ignore_index: int
) -> dict[str, float]:
    """Fraction of valid pixels in each class + the valid fraction of the tile."""
    total = label_tile.size
    valid = label_tile != ignore_index
    n_valid = int(valid.sum())
    out: dict[str, float] = {"valid_fraction": n_valid / total if total else 0.0}
    for c in class_ids:
        out[f"frac_{c}"] = (int(np.count_nonzero(label_tile == c)) / n_valid) if n_valid else 0.0
    return out


def build_tile_index(
    label: np.ndarray,
    transform: Affine,
    crs: str,
    tile_size: int,
    stride: int,
    class_ids: list[int],
    ignore_index: int = constants.IGNORE_INDEX,
    split_method: str = "spatial_block",
    block_size_px: int = 256,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    min_valid_fraction: float = 0.5,
    seed: int = constants.DEFAULT_SEED,
) -> gpd.GeoDataFrame:
    """Build a tile index GeoDataFrame with split assignment + class stats.

    Tiles whose valid fraction is below ``min_valid_fraction`` are dropped
    (mostly nodata/ignore). Geometry is the tile footprint in ``crs``.
    """
    tiles = generate_tiles(label.shape, tile_size, stride)
    if split_method == "spatial_block":
        splits = spatial_block_split(tiles, block_size_px, val_fraction, test_fraction, seed)
    elif split_method == "random":
        splits = random_split(len(tiles), val_fraction, test_fraction, seed)
    else:
        raise ValueError(f"unknown split_method {split_method!r}")

    records = []
    geoms = []
    for tile, split in zip(tiles, splits, strict=True):
        sub = label[
            tile.row_off : tile.row_off + tile.height,
            tile.col_off : tile.col_off + tile.width,
        ]
        stats = tile_class_fractions(sub, class_ids, ignore_index)
        if stats["valid_fraction"] < min_valid_fraction:
            continue
        rec = {
            "row_off": tile.row_off,
            "col_off": tile.col_off,
            "height": tile.height,
            "width": tile.width,
            "block_row": tile.row_off // block_size_px,
            "block_col": tile.col_off // block_size_px,
            "split": split,
            **stats,
        }
        records.append(rec)
        geoms.append(box(*tile.bounds(transform)))
    if not records:
        raise ValueError(
            "no tiles passed the min_valid_fraction filter; "
            "lower tiling.min_valid_fraction or tile_size"
        )
    return gpd.GeoDataFrame(records, geometry=geoms, crs=crs)


__all__ = [
    "Tile",
    "build_tile_index",
    "generate_tiles",
    "random_split",
    "spatial_block_split",
    "tile_class_fractions",
]
