"""Production-style tiled/windowed batch inference."""

from ecorehab.inference.polygonize import polygonize_prediction
from ecorehab.inference.stitch import tile_weights
from ecorehab.inference.tile_inference import (
    load_segmentation_checkpoint,
    predict_raster_windowed,
)

__all__ = [
    "load_segmentation_checkpoint",
    "polygonize_prediction",
    "predict_raster_windowed",
    "tile_weights",
]
