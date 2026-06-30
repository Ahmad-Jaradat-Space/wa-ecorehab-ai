"""Inference output shapes, stitching, polygonisation, and model forward passes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecorehab.features.indices import build_feature_stack
from ecorehab.inference.polygonize import polygonize_prediction
from ecorehab.inference.stitch import ProbabilityAccumulator, tile_weights
from ecorehab.models.losses import build_loss
from ecorehab.models.unet import UNet


def test_unet_forward_shape_multispectral():
    model = UNet(in_channels=13, num_classes=3, base_channels=8)
    x = torch.randn(2, 13, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)


def test_unet_accepts_arbitrary_channel_count():
    model = UNet(in_channels=7, num_classes=4, base_channels=8)
    out = model(torch.randn(1, 7, 32, 32))
    assert out.shape == (1, 4, 32, 32)


def test_losses_run_with_ignore_index():
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (2, 16, 16))
    target[0, 0, 0] = 255  # ignore
    for name in ["ce", "dice", "focal", "ce_dice"]:
        loss = build_loss(name, ignore_index=255)(logits, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_focal_pt_from_unweighted_prob():
    # Pins the fix: the focal modulation (1-pt)^gamma must use pt = UNWEIGHTED
    # softmax prob of the true class, with the class weight applied separately.
    # The buggy version used pt = exp(-weighted_ce), which this reference catches.
    import torch.nn.functional as F

    from ecorehab.models.losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    w = torch.tensor([0.5, 2.0, 1.0])
    gamma = 2.0

    logp = F.log_softmax(logits, dim=1)
    ce_u = -logp.gather(1, target.unsqueeze(1)).squeeze(1)  # unweighted CE
    p_true = torch.exp(-ce_u)
    manual = (w[target] * (1 - p_true) ** gamma * ce_u).mean()

    focal = FocalLoss(gamma=gamma, weight=w, ignore_index=255)(logits, target)
    assert torch.allclose(focal, manual, atol=1e-5)


def test_tile_weights_taper_at_edges():
    w = tile_weights(64, 64, overlap=16)
    assert w.shape == (64, 64)
    # Centre weight should exceed corner weight.
    assert w[32, 32] > w[0, 0]
    assert (w > 0).all()


def test_probability_accumulator_blends():
    acc = ProbabilityAccumulator(num_classes=2, height=10, width=10)
    prob = np.ones((2, 5, 5), dtype="float32") * 0.5
    w = np.ones((5, 5), dtype="float32")
    acc.add(prob, 0, 0, w)
    acc.add(prob, 0, 0, w)  # same region twice
    out = acc.finalize()
    assert out.shape == (2, 10, 10)
    assert out[:, 0, 0] == pytest.approx([0.5, 0.5])  # weighted mean preserved


def test_classical_raster_prediction_shape(scene, small_config):
    from ecorehab.labels.sampling import sample_pixels
    from ecorehab.labels.weak_labels import build_weak_labels
    from ecorehab.models.classical import ClassicalPixelModel, build_classical_model
    from ecorehab.utils.io import RasterData

    comp = RasterData(scene.composite, scene.transform, scene.crs, band_names=scene.band_names)
    label = build_weak_labels(comp, scene.vegetation, small_config).label
    feats, names = build_feature_stack(
        scene.composite, scene.band_names, small_config.features.indices
    )
    class_ids = [c.id for c in small_config.labels.classes]
    X, y = sample_pixels(feats, label, class_ids, n_per_class=500)
    est = build_classical_model("random_forest", {"n_estimators": 20, "max_depth": 8})
    model = ClassicalPixelModel(est, names, class_ids, "random_forest").fit(X, y)
    proba, pred = model.predict_proba_raster(feats)
    assert proba.shape == (len(class_ids), feats.shape[1], feats.shape[2])
    assert pred.shape == (feats.shape[1], feats.shape[2])
    # probabilities sum ~1 on valid pixels
    valid = np.isfinite(feats).all(axis=0)
    assert np.allclose(proba[:, valid].sum(axis=0), 1.0, atol=1e-4)


def test_windowed_inference_masks_nodata(tmp_path):
    # The composite encodes nodata as the finite sentinel -9999.0; windowed
    # inference must map it to ignore (regression for the validity-mask bug).
    import numpy as np
    from rasterio.transform import from_origin

    from ecorehab.inference.tile_inference import predict_raster_windowed
    from ecorehab.models.unet import UNet
    from ecorehab.utils.io import write_raster

    bands = ("blue", "green", "red", "rededge", "nir", "swir1", "swir2")
    arr = np.random.default_rng(0).random((7, 64, 64)).astype("float32")
    arr[:, :16, :16] = -9999.0  # nodata corner
    path = tmp_path / "comp.tif"
    write_raster(
        path,
        arr,
        from_origin(-1_500_000, -3_500_000, 10.0, 10.0),
        "EPSG:3577",
        nodata=-9999.0,
        band_names=bands,
    )

    model = UNet(in_channels=13, num_classes=3, base_channels=8).eval()
    meta = {
        "mean": np.zeros(13, "float32"),
        "std": np.ones(13, "float32"),
        "bands": list(bands),
        "feature_indices": ["ndvi", "evi", "ndwi", "mndwi", "nbr", "bsi"],
        "include_bands": True,
        "class_ids": [0, 1, 2],
        "class_names": ["a", "b", "c"],
        "num_classes": 3,
        "ignore_index": 255,
    }
    _, pred, _, _ = predict_raster_windowed(path, model, meta, tile_size=32, overlap=8)
    assert (pred[:16, :16] == 255).all()  # nodata corner -> ignore
    assert (pred[32:, 32:] != 255).all()  # valid region -> a real class


def test_polygonize_areas_positive(scene):
    from rasterio.transform import from_origin

    pred = np.zeros((50, 50), dtype="uint8")
    pred[:25, :] = 1  # class 1 over half
    transform = from_origin(0, 500, 10.0, 10.0)
    gdf = polygonize_prediction(pred, transform, "EPSG:3577", {0: "a", 1: "b"}, min_area_ha=0.0)
    assert len(gdf) >= 2
    assert (gdf["area_ha"] > 0).all()
    # class 1 covers 25*50 px * 0.01 ha = 12.5 ha
    b_area = gdf[gdf.class_id == 1]["area_ha"].sum()
    assert b_area == pytest.approx(12.5, rel=1e-6)
