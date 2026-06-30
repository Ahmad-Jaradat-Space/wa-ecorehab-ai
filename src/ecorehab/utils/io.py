"""Raster and vector IO that preserves geospatial metadata.

All raster writes go through :func:`write_raster`, which keeps CRS, transform,
nodata, and (by default) writes tiled, compressed GeoTIFFs that are
Cloud-Optimized-GeoTIFF friendly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine


@dataclass
class RasterData:
    """A raster in memory together with the metadata needed to write it back.

    Attributes:
        array: (bands, height, width) array.
        transform: affine transform mapping pixel -> CRS coordinates.
        crs: coordinate reference system string (e.g. ``"EPSG:3577"``).
        nodata: nodata sentinel, or None.
        band_names: optional per-band names, length == bands.
    """

    array: np.ndarray
    transform: Affine
    crs: str
    nodata: float | None = None
    band_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.array.ndim == 2:
            self.array = self.array[np.newaxis, ...]
        if self.array.ndim != 3:
            raise ValueError(f"array must be 2D or 3D, got shape {self.array.shape}")

    @property
    def count(self) -> int:
        return self.array.shape[0]

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def width(self) -> int:
        return self.array.shape[2]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)

    def with_nodata_as_nan(self) -> RasterData:
        """Return a float32 copy where nodata pixels are NaN (for index maths)."""
        arr = self.array.astype("float32")
        if self.nodata is not None and not np.isnan(self.nodata):
            arr = np.where(arr == self.nodata, np.nan, arr)
        return RasterData(arr, self.transform, self.crs, nodata=None, band_names=self.band_names)


def read_raster(path: str | Path, bands: list[int] | None = None) -> RasterData:
    """Read a GeoTIFF into a :class:`RasterData` (band-first array)."""
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        arr = src.read(idx)
        names = None
        if src.descriptions and any(src.descriptions):
            names = tuple(src.descriptions[i - 1] or f"band{i}" for i in idx)
        return RasterData(
            array=arr,
            transform=src.transform,
            crs=str(src.crs) if src.crs else "",
            nodata=src.nodata,
            band_names=names,
        )


def write_raster(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: str | RioCRS,
    nodata: float | None = None,
    band_names: tuple[str, ...] | None = None,
    dtype: str | None = None,
    compress: str = "deflate",
    tiled: bool = True,
) -> Path:
    """Write a (bands, H, W) or (H, W) array to a tiled, compressed GeoTIFF.

    Keeps CRS, transform, and nodata. Tiling + overviews make the output
    Cloud-Optimized-GeoTIFF friendly for windowed reads in batch inference.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"array must be 2D or 3D, got {array.shape}")
    count, height, width = array.shape
    out_dtype = dtype or array.dtype.name
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": out_dtype,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "predictor": 2 if out_dtype.startswith(("int", "uint")) else 3,
        "BIGTIFF": "IF_SAFER",
    }
    if tiled:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(out_dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
        # Internal overviews -> COG-friendly.
        if tiled and max(height, width) >= 512:
            dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.nearest)
    return path


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector dataset (GeoPackage/GeoJSON/Shapefile) via pyogrio."""
    return gpd.read_file(path)


def write_vector(gdf: gpd.GeoDataFrame, path: str | Path, layer: str | None = None) -> Path:
    """Write a GeoDataFrame. Format inferred from extension (.gpkg/.geojson)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gpkg":
        gdf.to_file(path, layer=layer or path.stem, driver="GPKG")
    elif path.suffix.lower() in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
    else:
        gdf.to_file(path)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting (numpy-aware)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


__all__ = [
    "RasterData",
    "load_json",
    "read_raster",
    "read_vector",
    "save_json",
    "write_raster",
    "write_vector",
]
