"""Auxiliary masks (water / cloud / valid) derived from spectral indices.

In real mode these would be refined with DEA Water Observations and DEA Land
Cover; here they provide transparent spectral fallbacks used by weak labelling.
"""

from __future__ import annotations

import numpy as np

from ecorehab.features.indices import compute_index


def water_mask(
    array: np.ndarray, band_names: list[str] | tuple[str, ...], ndwi_threshold: float = 0.1
) -> np.ndarray:
    """Boolean water mask from NDWI (McFeeters)."""
    ndwi = compute_index(array, band_names, "ndwi")
    return np.nan_to_num(ndwi, nan=-1.0) > ndwi_threshold


def valid_mask(array: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with finite values in every band."""
    return np.all(np.isfinite(array), axis=0)


def bright_cloud_mask(
    array: np.ndarray, band_names: list[str] | tuple[str, ...], blue_threshold: float = 0.3
) -> np.ndarray:
    """Crude bright-surface (cloud) mask. Real mode should use the ARD fmask band."""
    if "blue" not in band_names:
        return np.zeros(array.shape[1:], dtype=bool)
    blue = array[list(band_names).index("blue")]
    return np.nan_to_num(blue, nan=0.0) > blue_threshold


__all__ = ["bright_cloud_mask", "valid_mask", "water_mask"]
