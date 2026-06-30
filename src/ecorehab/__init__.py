"""WA EcoRehab AI.

Open-source geospatial machine-learning pipeline for native vegetation extent,
disturbance, and rehabilitation-proxy monitoring in Western Australia.

The package is organised into clear stages that mirror a production ecological
monitoring system:

    data        open-data acquisition + the synthetic demo AOI generator
    labels      weak-label generation, rasterization, tiling, spatial splits
    features    spectral indices and texture features
    models      classical (RandomForest/XGBoost) and segmentation (U-Net) models
    training    dataset/augmentation plumbing + train entry points
    evaluation  imbalanced metrics, calibration, uncertainty, spatial CV, area
    inference   tiled/windowed batch prediction, stitching, polygonisation
    reporting   maps, figures, and the decision-grade HTML site report
    utils       config (pydantic), CRS helpers, IO, logging

Equal-area analysis and all hectare reporting use EPSG:3577 (Australian Albers).
"""

from __future__ import annotations

__version__ = "0.1.0"

from ecorehab import constants
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.logging import get_logger

__all__ = ["Config", "constants", "get_logger", "load_config", "__version__"]
