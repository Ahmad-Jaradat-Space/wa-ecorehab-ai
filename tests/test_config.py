"""Config loading, validation, and the `extends` inheritance mechanism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ecorehab import constants
from ecorehab.utils.config import Config, load_config


def test_load_base_config():
    cfg = load_config("configs/aoi_swan_coastal_plain.yaml")
    assert cfg.project.crs == constants.EQUAL_AREA_CRS
    assert cfg.labels.num_classes == 3
    assert cfg.aoi.demo is True
    # paths.root defaults to repo root so relative data paths resolve.
    assert cfg.resolved_paths().processed.is_absolute()


def test_extends_inherits_and_overrides():
    cfg = load_config("configs/random_forest.yaml")
    # inherited from base
    assert cfg.aoi.name == "swan_coastal_plain"
    assert cfg.project.crs == constants.EQUAL_AREA_CRS
    # overridden
    assert cfg.classical.model_type == "random_forest"
    assert cfg.classical.params["n_estimators"] == 300


def test_extends_chain_wheatbelt():
    cfg = load_config("configs/aoi_wheatbelt.yaml")
    assert cfg.aoi.name == "wheatbelt"
    assert cfg.aoi.lonlat[0] == pytest.approx(117.65)
    assert cfg.data.demo.seed == 7


def test_all_shipped_configs_load():
    for name in [
        "aoi_swan_coastal_plain",
        "aoi_wheatbelt",
        "aoi_mining_proxy",
        "random_forest",
        "xgboost",
        "unet",
        "segformer",
        "inference",
        "report",
    ]:
        cfg = load_config(f"configs/{name}.yaml")
        assert isinstance(cfg, Config)


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  nonsense_key: 1\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_non_contiguous_class_ids_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "labels:\n  classes:\n"
        "    - {id: 0, name: a}\n"
        "    - {id: 2, name: b}\n"  # gap -> invalid
    )
    with pytest.raises(ValidationError):
        load_config(bad)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
