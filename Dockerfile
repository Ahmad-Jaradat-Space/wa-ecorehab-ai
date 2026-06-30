# WA EcoRehab AI - CPU image.
# GDAL/PROJ/GEOS ship inside the rasterio/geopandas/pyproj wheels, so a slim
# Python base is enough for the core demo pipeline. PyTorch is CPU-only here;
# build a CUDA variant from a devel base if GPU training is required.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # rasterio/pyproj bundle their own data; keep GDAL quiet about missing system PROJ
    PROJ_NETWORK=OFF

# Minimal system deps. Most geospatial native libs come from manylinux wheels;
# build-essential covers the few sdist-only packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
# Core + deep-learning + boosting (CPU). Drop extras to slim the image.
RUN pip install --upgrade pip && pip install ".[dl,boost,rs]"

# Copy the rest (configs, docs, tests).
COPY . .

# Default: show the CLI help. Override the command to run a stage, e.g.:
#   docker run --rm -v $PWD/outputs:/app/outputs wa-ecorehab-ai \
#       python -m ecorehab.inference.batch_predict --config configs/inference.yaml
ENTRYPOINT ["python", "-m", "ecorehab.cli"]
CMD ["--help"]
