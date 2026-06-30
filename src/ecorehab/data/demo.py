"""Deterministic synthetic-but-georeferenced demo AOI for Western Australia.

This is the project's offline backbone. It fabricates *pixel values*, but every
piece of geospatial machinery around them is real: a proper EPSG:3577 affine
transform centred on the Swan Coastal Plain, GeoPackage polygons, NaN nodata,
and class structure that is genuinely learnable. The synthetic native-vegetation
polygons are derived from the hidden truth map (then simplified + buffered) so
they imitate the real WA Native Vegetation Extent: spatially coherent, but not
pixel-aligned with the imagery, which is exactly why the weak-label boundary
handling in :mod:`ecorehab.labels.weak_labels` matters.

The scene deliberately includes confusers the docs discuss as failure modes:
managed-green agricultural fields (spectrally close to native vegetation), a
sinuous river and water bodies, and bare/disturbed ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from ecorehab import constants
from ecorehab.utils.config import Config
from ecorehab.utils.crs import transform_point

# Per-class mean surface reflectance (blue, green, red, rededge, nir, swir1, swir2).
# Tuned so NDVI/NDWI/BSI separate the classes the way real land covers do.
_CLASS_REFLECTANCE = {
    "native": np.array([0.035, 0.060, 0.045, 0.190, 0.340, 0.170, 0.090], dtype="float32"),
    "agriculture": np.array([0.045, 0.080, 0.060, 0.170, 0.280, 0.210, 0.130], dtype="float32"),
    "bare": np.array([0.130, 0.170, 0.220, 0.250, 0.270, 0.330, 0.280], dtype="float32"),
    "water": np.array([0.055, 0.050, 0.040, 0.030, 0.020, 0.012, 0.008], dtype="float32"),
}


@dataclass
class DemoScene:
    """A complete synthetic AOI."""

    composite: np.ndarray  # (bands, H, W) float32 reflectance, NaN = nodata
    transform: Affine
    crs: str
    band_names: tuple[str, ...]
    truth: np.ndarray  # (H, W) uint8 class map; IGNORE_INDEX where nodata
    vegetation: gpd.GeoDataFrame  # native-vegetation polygons (label source)
    tenements: gpd.GeoDataFrame  # mining-tenement polygons (AOI overlays)
    aoi: gpd.GeoDataFrame  # AOI boundary


def _normalise_field(rng: np.random.Generator, shape_: tuple[int, int], sigma: float) -> np.ndarray:
    """Low-frequency random field in [0, 1] (spatially coherent)."""
    white = rng.standard_normal(shape_).astype("float32")
    field = gaussian_filter(white, sigma=sigma, mode="reflect")
    lo, hi = field.min(), field.max()
    return (field - lo) / (hi - lo + 1e-9)


def _river_mask(shape_: tuple[int, int], amplitude: float, period: float, width: int) -> np.ndarray:
    """A sinuous river crossing the scene left-to-right."""
    h, w = shape_
    cols = np.arange(w)
    centre = h * 0.5 + amplitude * h * np.sin(2 * np.pi * cols / (period * w))
    rows = np.arange(h)[:, None]
    return np.abs(rows - centre[None, :]) <= width


def _build_truth(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_map, agriculture_mask). Classes: 0 disturbed, 1 native, 2 water."""
    n = cfg.data.demo.size_px
    veg_field = _normalise_field(rng, (n, n), sigma=n * 0.035)
    water_field = _normalise_field(rng, (n, n), sigma=n * 0.025)

    cls = np.zeros((n, n), dtype="uint8")  # default: disturbed_or_cleared (0)

    # Native vegetation = upper quantile of the vegetation propensity field.
    native = veg_field > np.quantile(veg_field, 0.76)
    cls[native] = 1

    # Water: a few blobs + a river. Water overrides native.
    water = water_field > np.quantile(water_field, 0.965)
    water |= _river_mask((n, n), amplitude=0.12, period=1.0, width=max(2, n // 256))
    cls[water] = 2

    # Managed-green agriculture: rectangular fields in the lower-left, carved from
    # the disturbed class. Spectrally green -> a deliberate confuser vs native.
    agri = np.zeros((n, n), dtype=bool)
    field_px = max(24, n // 12)
    for r0 in range(n // 2, n, field_px):
        for c0 in range(0, n // 2, field_px):
            if rng.random() < 0.55:
                r1, c1 = min(n, r0 + field_px - 2), min(n, c0 + field_px - 2)
                block = np.zeros((n, n), dtype=bool)
                block[r0:r1, c0:c1] = True
                block &= cls == 0  # only convert disturbed pixels
                agri |= block
    return cls, agri


def _synthesize_reflectance(
    cfg: Config, cls: np.ndarray, agri: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Build the (bands, H, W) reflectance cube from the class map."""
    n = cls.shape[0]
    nb = len(cfg.data.bands)
    cube = np.zeros((nb, n, n), dtype="float32")

    # Base signature per pixel.
    sig = np.empty((nb, n, n), dtype="float32")
    sig[:] = _CLASS_REFLECTANCE["bare"][:nb, None, None]  # disturbed defaults toward bare
    sig[:, cls == 1] = _CLASS_REFLECTANCE["native"][:nb, None]
    sig[:, cls == 2] = _CLASS_REFLECTANCE["water"][:nb, None]
    sig[:, agri] = _CLASS_REFLECTANCE["agriculture"][:nb, None]

    # Within-class brightness variation (coherent) + per-pixel noise.
    brightness = 1.0 + 0.18 * (_normalise_field(rng, (n, n), sigma=n * 0.02) - 0.5)
    for b in range(nb):
        band = sig[b] * brightness
        band = gaussian_filter(band, sigma=0.6, mode="reflect")
        band += rng.normal(0.0, 0.012, size=(n, n)).astype("float32")
        cube[b] = band
    np.clip(cube, 0.0, 1.0, out=cube)
    return cube


def _add_nodata_corner(cube: np.ndarray, cls: np.ndarray, frac: float = 0.06) -> None:
    """Blank a corner block as nodata to exercise nodata handling downstream."""
    n = cube.shape[1]
    k = int(n * frac)
    cube[:, :k, :k] = np.nan
    cls[:k, :k] = constants.IGNORE_INDEX


def _polygons_from_mask(
    mask: np.ndarray, transform: Affine, crs: str, min_pixels: int
) -> list[BaseGeometry]:
    geoms: list[BaseGeometry] = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        # Approx pixel count from area / pixel area.
        px_area = abs(transform.a * transform.e)
        if poly.area / px_area < min_pixels:
            continue
        # Simplify + tiny outward buffer -> imperfect, non-pixel-aligned boundary,
        # mimicking a generalised government vegetation polygon layer.
        poly = poly.simplify(abs(transform.a) * 1.5).buffer(abs(transform.a) * 0.5)
        if not poly.is_empty and poly.is_valid:
            geoms.append(poly)
    return geoms


def generate_demo_scene(cfg: Config) -> DemoScene:
    """Generate the full deterministic demo AOI from a config."""
    rng = np.random.default_rng(cfg.data.demo.seed)
    n = cfg.data.demo.size_px
    res = cfg.project.resolution_m

    # Centre the AOI on the configured lon/lat, projected to EPSG:3577.
    cx, cy = transform_point(cfg.aoi.lonlat[0], cfg.aoi.lonlat[1], cfg.project.crs)
    x0 = cx - (n / 2) * res
    y_top = cy + (n / 2) * res
    transform = Affine(res, 0.0, x0, 0.0, -res, y_top)

    cls, agri = _build_truth(cfg, rng)
    cube = _synthesize_reflectance(cfg, cls, agri, rng)
    _add_nodata_corner(cube, cls)

    crs = cfg.project.crs
    band_names = tuple(cfg.data.bands)

    # Vegetation polygons derived from the (valid) native mask.
    native_mask = cls == 1
    min_px = int((cfg.labels.min_polygon_area_ha * 10_000) / (res * res))
    veg_geoms = _polygons_from_mask(native_mask, transform, crs, min_pixels=max(4, min_px))
    vegetation = gpd.GeoDataFrame(
        {"veg_id": range(len(veg_geoms)), "source": "synthetic_native_veg"},
        geometry=veg_geoms,
        crs=crs,
    )

    # AOI boundary (full valid extent).
    minx, miny = x0, y_top - n * res
    maxx, maxy = x0 + n * res, y_top
    aoi = gpd.GeoDataFrame(
        {"name": [cfg.aoi.name]}, geometry=[box(minx, miny, maxx, maxy)], crs=crs
    )

    # Mining tenements: a few rectangles overlapping the AOI for area summaries.
    tenements = _make_tenements(cfg, rng, (minx, miny, maxx, maxy))

    return DemoScene(
        composite=cube,
        transform=transform,
        crs=crs,
        band_names=band_names,
        truth=cls,
        vegetation=vegetation,
        tenements=tenements,
        aoi=aoi,
    )


def _make_tenements(
    cfg: Config, rng: np.random.Generator, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    geoms, ids = [], []
    for i in range(cfg.data.demo.n_tenements):
        tw = w * rng.uniform(0.25, 0.4)
        th = h * rng.uniform(0.25, 0.4)
        tx = minx + rng.uniform(0.05, 0.55) * w
        ty = miny + rng.uniform(0.05, 0.55) * h
        geoms.append(box(tx, ty, min(tx + tw, maxx), min(ty + th, maxy)))
        ids.append(f"M70/{1000 + i}")
    return gpd.GeoDataFrame(
        {"tenement_id": ids, "status": "LIVE"}, geometry=geoms, crs=cfg.project.crs
    )


__all__ = ["DemoScene", "generate_demo_scene"]
