"""Spectral-safe geometric augmentations for segmentation tiles.

We use flips/90-degree rotations and mild additive noise only. Aggressive colour
jitter would corrupt the spectral signal that distinguishes vegetation from bare
ground, so it is deliberately avoided (see notebook + docs/limitations.md).
Implemented in numpy to avoid a hard albumentations dependency; the [dl] extra's
albumentations can be swapped in for richer pipelines.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ecorehab import constants

AugFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def make_augmenter(
    enabled: bool = True, noise_std: float = 0.01, seed: int = constants.DEFAULT_SEED
) -> AugFn:
    """Return an augmenter ``(image CxHxW, mask HxW) -> (image, mask)``.

    If ``enabled`` is False the identity transform is returned.
    """
    rng = np.random.default_rng(seed)

    def identity(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return image, mask

    if not enabled:
        return identity

    def augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:  # horizontal flip
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if rng.random() < 0.5:  # vertical flip
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
        k = int(rng.integers(0, 4))  # 0/90/180/270 rotation
        if k:
            image = np.rot90(image, k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k)
        if noise_std > 0:
            image = image + rng.normal(0.0, noise_std, size=image.shape).astype(image.dtype)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    return augment


__all__ = ["AugFn", "make_augmenter"]
