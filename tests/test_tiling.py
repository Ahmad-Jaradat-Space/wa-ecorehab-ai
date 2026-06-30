"""Tile generation + spatial-split leakage prevention."""

from __future__ import annotations

from ecorehab.labels.tiling import (
    Tile,
    build_tile_index,
    generate_tiles,
    random_split,
    spatial_block_split,
)


def test_generate_tiles_cover_and_size():
    tiles = generate_tiles((256, 256), tile_size=128, stride=128)
    assert len(tiles) == 4
    for t in tiles:
        assert t.height == 128 and t.width == 128
        assert t.row_off + t.height <= 256
        assert t.col_off + t.width <= 256


def test_generate_tiles_handles_non_divisible_extent():
    # 300x300 with 128 tiles -> last tile shifted inward, still exactly 128.
    tiles = generate_tiles((300, 300), tile_size=128, stride=128)
    for t in tiles:
        assert t.height == 128 and t.width == 128
        assert t.row_off + 128 <= 300 and t.col_off + 128 <= 300
    # Bottom-right corner must be covered.
    assert any(t.row_off + 128 == 300 and t.col_off + 128 == 300 for t in tiles)


def test_tile_bounds_match_transform():
    from rasterio.transform import from_origin

    transform = from_origin(1000, 5000, 10.0, 10.0)
    t = Tile(row_off=0, col_off=0, height=10, width=10)
    minx, miny, maxx, maxy = t.bounds(transform)
    assert (minx, maxy) == (1000, 5000)
    assert (maxx, miny) == (1100, 4900)


def test_spatial_block_split_no_block_straddles_splits():
    tiles = generate_tiles((512, 512), tile_size=64, stride=64)
    splits = spatial_block_split(
        tiles, block_size_px=128, val_fraction=0.2, test_fraction=0.2, seed=1
    )
    # All tiles sharing a block id must share a split -> no leakage.
    block_to_split: dict[tuple[int, int], str] = {}
    for t, s in zip(tiles, splits, strict=True):
        key = (t.row_off // 128, t.col_off // 128)
        block_to_split.setdefault(key, s)
        assert block_to_split[key] == s
    assert set(splits) <= {"train", "val", "test"}
    assert "train" in splits


def test_random_split_proportions():
    splits = random_split(100, val_fraction=0.2, test_fraction=0.2, seed=0)
    assert splits.count("val") == 20
    assert splits.count("test") == 20
    assert splits.count("train") == 60


def test_build_tile_index_drops_low_valid_tiles(scene, small_config):
    from ecorehab.labels.weak_labels import build_weak_labels
    from ecorehab.utils.io import RasterData

    comp = RasterData(scene.composite, scene.transform, scene.crs, band_names=scene.band_names)
    label = build_weak_labels(comp, scene.vegetation, small_config).label
    class_ids = [c.id for c in small_config.labels.classes]
    idx = build_tile_index(
        label,
        scene.transform,
        scene.crs,
        tile_size=64,
        stride=64,
        class_ids=class_ids,
        block_size_px=128,
        min_valid_fraction=0.5,
    )
    assert len(idx) > 0
    assert (idx["valid_fraction"] >= 0.5).all()
    assert set(idx["split"]) <= {"train", "val", "test"}
    assert idx.crs is not None
