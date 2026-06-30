"""Optional texture features (local variance / GLCM-style).

Texture helps separate structurally rough native vegetation from smooth managed
fields. It is opt-in (``features.texture: true``) because it adds compute.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def local_mean(band: np.ndarray, size: int = 5) -> np.ndarray:
    """Local mean over a square window (NaN-aware)."""
    mask = np.isfinite(band).astype("float32")
    filled = np.where(mask > 0, band, 0.0).astype("float32")
    num = uniform_filter(filled, size=size, mode="nearest")
    den = uniform_filter(mask, size=size, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def local_variance(band: np.ndarray, size: int = 5) -> np.ndarray:
    """Local variance over a square window (NaN-aware). A simple texture proxy."""
    mean = local_mean(band, size)
    mean_sq = local_mean(band**2, size)
    var = mean_sq - mean**2
    return np.clip(var, 0.0, None)


def texture_stack(
    array: np.ndarray, band_names: list[str] | tuple[str, ...], band: str = "nir", size: int = 5
) -> tuple[np.ndarray, list[str]]:
    """Compute a small texture stack (local variance of one band)."""
    if band not in band_names:
        raise ValueError(f"texture band {band!r} not in {list(band_names)}")
    idx = list(band_names).index(band)
    var = local_variance(array[idx].astype("float32"), size=size)
    return var[np.newaxis, ...], [f"{band}_var{size}"]


__all__ = ["local_mean", "local_variance", "texture_stack"]
