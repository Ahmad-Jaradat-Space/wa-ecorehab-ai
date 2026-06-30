"""Project-wide constants: CRS, band conventions, class schema, sentinels.

Keeping these in one place avoids magic numbers scattered across modules and
gives the tests and notebooks a single source of truth.
"""

from __future__ import annotations

# --- Coordinate reference systems -------------------------------------------------
# EPSG:3577 = GDA94 / Australian Albers. Equal-area, metre units. Used for ALL
# area (hectare) reporting and for label/tile generation so pixels are area-comparable.
EQUAL_AREA_CRS: str = "EPSG:3577"
# A representative UTM zone for SW Western Australia (zone 50S), used when imagery
# arrives in native UTM before reprojection to the equal-area working CRS.
WA_UTM_CRS: str = "EPSG:32750"
GEOGRAPHIC_CRS: str = "EPSG:4326"

# --- Imagery conventions ----------------------------------------------------------
# Sentinel-2 surface-reflectance band order used throughout the package. The demo
# composite is generated with exactly these bands; the real STAC loader maps DEA
# asset names onto this order.
BAND_NAMES: tuple[str, ...] = (
    "blue",  # ~B02 490 nm
    "green",  # ~B03 560 nm
    "red",  # ~B04 665 nm
    "rededge",  # ~B05 705 nm
    "nir",  # ~B08 842 nm
    "swir1",  # ~B11 1610 nm
    "swir2",  # ~B12 2190 nm
)
# Surface reflectance is stored as float32 in [0, 1]. DEA/Sentinel-2 ARD ships
# int16 scaled by 10000; the loader divides by this factor.
S2_REFLECTANCE_SCALE: float = 10000.0
DEFAULT_RESOLUTION_M: float = 10.0

# Spectral indices the feature stack can compute (see features/indices.py).
INDEX_NAMES: tuple[str, ...] = ("ndvi", "evi", "ndwi", "mndwi", "nbr", "bsi")

# --- Label schema -----------------------------------------------------------------
# Minimum-viable 3-class schema (config can override). Integer ids are contiguous
# from 0 so they map directly onto model logits / sklearn class indices.
DEFAULT_CLASSES: tuple[dict[str, object], ...] = (
    {"id": 0, "name": "disturbed_or_cleared", "color": "#d9a05b"},
    {"id": 1, "name": "native_remnant_vegetation", "color": "#1b7837"},
    {"id": 2, "name": "water_or_wetland", "color": "#2c7fb8"},
)

# Sentinel value for "ignore this pixel" in label rasters. Used as torch
# CrossEntropy ignore_index and filtered out before sklearn metrics. 255 keeps
# label rasters writable as uint8.
IGNORE_INDEX: int = 255

# Raster nodata sentinels.
REFLECTANCE_NODATA: float = -9999.0
PROB_NODATA: float = -1.0

# --- Reproducibility --------------------------------------------------------------
DEFAULT_SEED: int = 42

__all__ = [
    "BAND_NAMES",
    "DEFAULT_CLASSES",
    "DEFAULT_RESOLUTION_M",
    "DEFAULT_SEED",
    "EQUAL_AREA_CRS",
    "GEOGRAPHIC_CRS",
    "IGNORE_INDEX",
    "INDEX_NAMES",
    "PROB_NODATA",
    "REFLECTANCE_NODATA",
    "S2_REFLECTANCE_SCALE",
    "WA_UTM_CRS",
]
