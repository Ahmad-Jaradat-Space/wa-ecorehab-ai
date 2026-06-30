"""Spectral indices computed from a band-first surface-reflectance array.

Input arrays are (bands, H, W) float reflectance in [0, 1]; nodata should be
encoded as NaN so it propagates through the index maths. All indices are
returned in their natural range (mostly [-1, 1]).
"""

from __future__ import annotations

import numpy as np

from ecorehab import constants

# Index -> formula expressed over named bands. Each callable takes a dict of
# {band_name: 2D array} and returns a 2D array.
_EPS = 1e-6


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Normalised-difference style ratio that is NaN-safe and zero-safe."""
    den = np.where(np.abs(den) < _EPS, np.nan, den)
    return num / den


def _bands_dict(
    array: np.ndarray, band_names: list[str] | tuple[str, ...]
) -> dict[str, np.ndarray]:
    if array.shape[0] != len(band_names):
        raise ValueError(f"array has {array.shape[0]} bands but {len(band_names)} names were given")
    return {name: array[i].astype("float32") for i, name in enumerate(band_names)}


def _require(b: dict[str, np.ndarray], *names: str) -> None:
    missing = [n for n in names if n not in b]
    if missing:
        raise ValueError(f"index requires bands {missing} which are not present")


def ndvi(b: dict[str, np.ndarray]) -> np.ndarray:
    _require(b, "nir", "red")
    return _safe_ratio(b["nir"] - b["red"], b["nir"] + b["red"])


def evi(b: dict[str, np.ndarray]) -> np.ndarray:
    _require(b, "nir", "red", "blue")
    den = b["nir"] + 6.0 * b["red"] - 7.5 * b["blue"] + 1.0
    return 2.5 * _safe_ratio(b["nir"] - b["red"], den)


def ndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    """McFeeters NDWI (water): (green - nir) / (green + nir)."""
    _require(b, "green", "nir")
    return _safe_ratio(b["green"] - b["nir"], b["green"] + b["nir"])


def mndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    _require(b, "green", "swir1")
    return _safe_ratio(b["green"] - b["swir1"], b["green"] + b["swir1"])


def nbr(b: dict[str, np.ndarray]) -> np.ndarray:
    _require(b, "nir", "swir2")
    return _safe_ratio(b["nir"] - b["swir2"], b["nir"] + b["swir2"])


def bsi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Bare Soil Index: ((swir1+red)-(nir+blue)) / ((swir1+red)+(nir+blue))."""
    _require(b, "swir1", "red", "nir", "blue")
    num = (b["swir1"] + b["red"]) - (b["nir"] + b["blue"])
    den = (b["swir1"] + b["red"]) + (b["nir"] + b["blue"])
    return _safe_ratio(num, den)


_INDEX_FUNCS = {
    "ndvi": ndvi,
    "evi": evi,
    "ndwi": ndwi,
    "mndwi": mndwi,
    "nbr": nbr,
    "bsi": bsi,
}


def compute_index(
    array: np.ndarray, band_names: list[str] | tuple[str, ...], name: str
) -> np.ndarray:
    """Compute a single named spectral index -> 2D array."""
    name = name.lower()
    if name not in _INDEX_FUNCS:
        raise ValueError(f"unknown index {name!r}; available: {sorted(_INDEX_FUNCS)}")
    return _INDEX_FUNCS[name](_bands_dict(array, band_names))


def compute_indices(
    array: np.ndarray,
    band_names: list[str] | tuple[str, ...],
    names: list[str] | tuple[str, ...] = constants.INDEX_NAMES,
) -> tuple[np.ndarray, list[str]]:
    """Compute several indices -> ((n_idx, H, W) array, names)."""
    b = _bands_dict(array, band_names)
    out = []
    for n in names:
        nl = n.lower()
        if nl not in _INDEX_FUNCS:
            raise ValueError(f"unknown index {n!r}")
        out.append(_INDEX_FUNCS[nl](b))
    if not out:
        h, w = array.shape[1], array.shape[2]
        return np.empty((0, h, w), dtype="float32"), []
    return np.stack(out).astype("float32"), [n.lower() for n in names]


def build_feature_stack(
    array: np.ndarray,
    band_names: list[str] | tuple[str, ...],
    indices: list[str] | tuple[str, ...] = constants.INDEX_NAMES,
    include_bands: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Stack selected reflectance bands and spectral indices.

    Returns:
        (features, feature_names) where features is (n_features, H, W).
    """
    layers: list[np.ndarray] = []
    names: list[str] = []
    if include_bands:
        for i, bn in enumerate(band_names):
            layers.append(array[i].astype("float32"))
            names.append(bn)
    idx_arr, idx_names = compute_indices(array, band_names, indices)
    for i, n in enumerate(idx_names):
        layers.append(idx_arr[i])
        names.append(n)
    if not layers:
        raise ValueError("feature stack is empty: enable include_bands and/or indices")
    return np.stack(layers).astype("float32"), names


__all__ = [
    "build_feature_stack",
    "bsi",
    "compute_index",
    "compute_indices",
    "evi",
    "mndwi",
    "nbr",
    "ndvi",
    "ndwi",
]
