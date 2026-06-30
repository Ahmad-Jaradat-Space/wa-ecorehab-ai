"""Open-data acquisition + the synthetic demo AOI generator.

Real mode pulls Sentinel-2 ARD via DEA STAC (:mod:`ecorehab.data.stac`) and WA
government vectors (:mod:`ecorehab.data.download_vectors`). Demo mode generates a
fully georeferenced synthetic AOI (:mod:`ecorehab.data.demo`) so the entire
pipeline runs offline with real CRS, transforms, polygons, tiling, and area maths.
"""
