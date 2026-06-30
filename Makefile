# WA EcoRehab AI - developer workflow
# Demo-mode targets run end-to-end with synthetic-but-georeferenced data (no network).

PY ?= python
CONFIG ?= configs/aoi_swan_coastal_plain.yaml
VENV ?= .venv

.PHONY: help install install-dev install-all venv lint fmt fmt-check test \
        demo-data download-vectors composites labels tiles \
        train-rf train-unet predict report pipeline clean clean-outputs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:  ## Create a local virtualenv in .venv
	$(PY) -m venv $(VENV)

install:  ## Install the core package (demo pipeline + tests)
	$(PY) -m pip install -e .

install-dev:  ## Install core + dev tooling
	$(PY) -m pip install -e ".[dev]"

install-all:  ## Install everything (DL, boosting, RS, viz, mlops, dev)
	$(PY) -m pip install -e ".[all]"

lint:  ## Run ruff lint
	ruff check src tests

fmt:  ## Auto-format with black + ruff --fix
	black src tests
	ruff check --fix src tests

fmt-check:  ## Check formatting + lint without modifying (CI gate)
	black --check src tests
	ruff check src tests

test:  ## Run the test suite
	pytest

# ---- End-to-end demo pipeline (synthetic data, fully offline) -------------------
demo-data:  ## Generate the synthetic-but-georeferenced demo AOI (raster + vectors)
	$(PY) -m ecorehab.data.make_demo --config $(CONFIG)

download-vectors:  ## Acquire vegetation/tenement vectors (demo fallback if no network)
	$(PY) -m ecorehab.data.download_vectors --config $(CONFIG)

composites:  ## Build the Sentinel-2-style surface-reflectance composite
	$(PY) -m ecorehab.data.build_composites --config $(CONFIG)

labels:  ## Build weak labels from vegetation polygons + masks
	$(PY) -m ecorehab.labels.build_labels --config $(CONFIG)

tiles:  ## Generate tiles with a spatial-block split
	$(PY) -m ecorehab.labels.build_tiles --config $(CONFIG)

train-rf:  ## Train the RandomForest pixel baseline
	$(PY) -m ecorehab.training.train_classical --config configs/random_forest.yaml

train-unet:  ## Train the U-Net segmentation baseline
	$(PY) -m ecorehab.training.train_segmentation --config configs/unet.yaml

predict:  ## Run tiled batch inference -> prediction/uncertainty GeoTIFFs + polygons
	$(PY) -m ecorehab.inference.batch_predict --config configs/inference.yaml

report:  ## Build the decision-grade HTML site report
	$(PY) -m ecorehab.reporting.build_site_report --config configs/report.yaml

pipeline: demo-data composites labels tiles train-rf train-unet predict report  ## Full demo pipeline

clean-outputs:  ## Remove generated outputs (keeps committed examples)
	find outputs -type f ! -path 'outputs/examples/*' ! -name '.gitkeep' -delete

clean: clean-outputs  ## Clean caches + outputs
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
