"""Typed, validated, YAML-driven configuration.

Every CLI in the package takes ``--config path/to.yaml`` and loads it through
:func:`load_config`. A YAML file may ``extends`` a base config (single level or
chained) so the six shipped configs avoid duplicating the AOI block. All paths
are resolved relative to the repository root unless absolute.

Design choices:
  * pydantic v2 with ``extra="forbid"`` so a typo in YAML fails loudly.
  * Every section has sensible defaults; a config only overrides what it needs.
  * The equal-area working CRS defaults to EPSG:3577 (see constants).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecorehab import constants


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ClassDef(_Base):
    """One label class."""

    id: int
    name: str
    color: str = "#999999"


class ProjectConfig(_Base):
    name: str = "wa_ecorehab"
    seed: int = constants.DEFAULT_SEED
    crs: str = constants.EQUAL_AREA_CRS  # equal-area working CRS for labels + area
    resolution_m: float = constants.DEFAULT_RESOLUTION_M


class PathsConfig(_Base):
    """Filesystem layout. ``root`` anchors every other (relative) path."""

    root: Path = Path(".")
    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    sample: Path = Path("data/sample")
    outputs: Path = Path("outputs")

    def resolve(self) -> PathsConfig:
        """Return a copy with all sub-paths made absolute under ``root``."""
        root = self.root.resolve()
        return PathsConfig(
            root=root,
            raw=self._abs(root, self.raw),
            interim=self._abs(root, self.interim),
            processed=self._abs(root, self.processed),
            sample=self._abs(root, self.sample),
            outputs=self._abs(root, self.outputs),
        )

    @staticmethod
    def _abs(root: Path, p: Path) -> Path:
        return p if p.is_absolute() else (root / p)


class AOIConfig(_Base):
    """Area of interest. In demo mode the AOI is generated synthetically; in real
    mode ``bbox`` (in the project CRS) or ``geometry_path`` selects the extent."""

    name: str = "swan_coastal_plain"
    demo: bool = True
    # bbox in project CRS (EPSG:3577 metres): [minx, miny, maxx, maxy]
    bbox: tuple[float, float, float, float] | None = None
    geometry_path: Path | None = None
    # approximate WGS84 centre, used only for context maps / STAC search
    lonlat: tuple[float, float] = (115.86, -32.05)  # Perth / Swan Coastal Plain


class DemoConfig(_Base):
    """Synthetic AOI generator settings (deterministic)."""

    size_px: int = 768  # square AOI side in pixels (768 * 10 m = 7.68 km)
    seed: int = constants.DEFAULT_SEED
    n_veg_patches: int = 14
    n_water_bodies: int = 3
    n_tenements: int = 2


class DataConfig(_Base):
    """Imagery / composite settings."""

    bands: list[str] = Field(default_factory=lambda: list(constants.BAND_NAMES))
    # STAC (real mode)
    stac_url: str = "https://explorer.dea.ga.gov.au/stac"
    collection: str = "ga_s2am_ard_3"
    date_start: str = "2023-01-01"
    date_end: str = "2023-03-31"  # late summer composite for SW WA
    max_cloud_cover: float = 20.0
    composite: Literal["median", "geomedian"] = "median"
    demo: DemoConfig = Field(default_factory=DemoConfig)


class LabelsConfig(_Base):
    """Weak-label generation parameters."""

    classes: list[ClassDef] = Field(
        default_factory=lambda: [ClassDef(**c) for c in constants.DEFAULT_CLASSES]  # type: ignore[arg-type]
    )
    ignore_index: int = constants.IGNORE_INDEX
    # boundary-noise handling (metres)
    positive_inward_buffer_m: float = 20.0
    boundary_exclusion_buffer_m: float = 20.0
    min_polygon_area_ha: float = 0.5
    # spectral gates for deriving disturbed/water without external land cover
    water_ndwi_threshold: float = 0.10
    disturbed_ndvi_threshold: float = 0.25  # below -> likely cleared/bare
    native_ndvi_threshold: float = 0.45  # negatives must be clearly non-vegetated

    @property
    def num_classes(self) -> int:
        return len(self.classes)


class SplitConfig(_Base):
    method: Literal["spatial_block", "random"] = "spatial_block"
    block_size_px: int = 256  # spatial-block grid cell side
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = constants.DEFAULT_SEED


class TilingConfig(_Base):
    tile_size: int = 128
    stride: int = 128  # == tile_size -> no overlap for training
    min_valid_fraction: float = 0.5  # drop tiles that are mostly ignore/nodata
    split: SplitConfig = Field(default_factory=SplitConfig)


class FeaturesConfig(_Base):
    indices: list[str] = Field(default_factory=lambda: list(constants.INDEX_NAMES))
    include_bands: bool = True
    texture: bool = False  # GLCM-style texture is opt-in (slower)


class ClassicalConfig(_Base):
    """RandomForest / gradient-boosting pixel baseline."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_type: Literal["random_forest", "xgboost", "lightgbm"] = "random_forest"
    n_pixels_per_class: int = 20000  # balanced pixel sampling cap per class
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint: Path = Path("outputs/metrics/classical_model.joblib")


class SegmentationConfig(_Base):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    arch: Literal["unet", "smp_unet", "segformer"] = "unet"
    encoder: str = "resnet34"  # used by smp_unet / informational for unet
    in_channels: int | None = None  # None -> inferred from features at runtime
    base_channels: int = 32  # width of the from-scratch U-Net
    loss: Literal["ce", "dice", "focal", "ce_dice"] = "ce_dice"
    class_weights: list[float] | None = None  # None -> inverse-frequency
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 0
    patience: int = 6  # early-stopping patience on val macro-F1
    amp: bool = False
    checkpoint: Path = Path("outputs/metrics/segmentation_model.pt")


class InferenceConfig(_Base):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_kind: Literal["segmentation", "classical"] = "segmentation"
    checkpoint: Path = Path("outputs/metrics/segmentation_model.pt")
    tile_size: int = 256
    overlap: int = 32  # halo for seamless stitching
    batch_size: int = 8
    tta: bool = False  # flip-based test-time augmentation
    write_probabilities: bool = True
    write_uncertainty: bool = True
    polygonize: bool = True
    out_dir: Path = Path("outputs/predictions")
    vector_out_dir: Path = Path("outputs/vector_outputs")


class ReportConfig(_Base):
    """Decision-grade site report."""

    title: str = "WA EcoRehab AI - Site Report"
    aoi_name: str = "Swan Coastal Plain (demo)"
    prediction_raster: Path = Path("outputs/predictions/prediction_mask.tif")
    probability_raster: Path = Path("outputs/predictions/native_vegetation_probability.tif")
    uncertainty_raster: Path = Path("outputs/predictions/uncertainty.tif")
    polygons: Path = Path("outputs/vector_outputs/predicted_polygons.gpkg")
    metrics_json: Path = Path("outputs/metrics/segmentation_metrics.json")
    area_csv: Path = Path("outputs/vector_outputs/area_summary.csv")
    n_bootstrap: int = 500
    out_html: Path = Path("outputs/reports/site_report.html")


class Config(_Base):
    """Root configuration object."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    aoi: AOIConfig = Field(default_factory=AOIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    classical: ClassicalConfig = Field(default_factory=ClassicalConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="after")
    def _check_classes(self) -> Config:
        ids = [c.id for c in self.labels.classes]
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                f"class ids must be contiguous from 0; got {ids}. "
                "Contiguous ids map directly onto model logits."
            )
        if self.labels.ignore_index in ids:
            raise ValueError("ignore_index must not collide with a class id")
        return self

    def resolved_paths(self) -> PathsConfig:
        return self.paths.resolve()


# --------------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config extends detected at {path}")
    _seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    base = _load_yaml_with_extends(parent_path, _seen)
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> Config:
    """Load and validate a config YAML (supports ``extends``).

    Args:
        path: path to a YAML config file.

    Returns:
        A validated :class:`Config`. The ``paths.root`` defaults to the config
        file's parent's parent (repo root) when not set in YAML, so relative
        data/output paths resolve correctly regardless of working directory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = _load_yaml_with_extends(path)
    cfg = Config(**data)
    # Default root = repo root (configs/ is one level under it) unless YAML set it.
    if "paths" not in data or "root" not in data.get("paths", {}):
        cfg.paths.root = path.resolve().parent.parent
    return cfg


__all__ = [
    "AOIConfig",
    "ClassDef",
    "ClassicalConfig",
    "Config",
    "DataConfig",
    "DemoConfig",
    "FeaturesConfig",
    "InferenceConfig",
    "LabelsConfig",
    "PathsConfig",
    "ProjectConfig",
    "ReportConfig",
    "SegmentationConfig",
    "SplitConfig",
    "TilingConfig",
    "load_config",
]
