"""Generate the example figures committed under outputs/examples/ for the README.

Run after the demo pipeline (make-demo -> labels -> tiles -> train -> predict):
    python scripts/make_example_figures.py --config configs/aoi_swan_coastal_plain.yaml

These PNGs are the only outputs intentionally committed to git.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecorehab import artifacts, pipeline  # noqa: E402
from ecorehab.evaluation.calibration import brier_score, reliability_curve  # noqa: E402
from ecorehab.features.indices import compute_index  # noqa: E402
from ecorehab.reporting import figures as F  # noqa: E402
from ecorehab.reporting.maps import aoi_location_map, overlay_polygons, split_map  # noqa: E402
from ecorehab.utils.config import load_config  # noqa: E402
from ecorehab.utils.io import read_raster, read_vector  # noqa: E402


def _save(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_pipeline_diagram(out, res):
    """A clean horizontal pipeline diagram used as the README hero image."""
    stages = [
        ("Open data\nDEA S2 · WA vectors", "#2c7fb8"),
        ("Weak labels\n+ ignore mask", "#d9a05b"),
        ("Spatial-block\ntiling", "#7b6888"),
        ("RandomForest\n+ U-Net", "#1b7837"),
        ("Spatial eval\n+ calibration", "#c0504d"),
        ("Tiled inference\nCOG · polygons", "#4a7c59"),
        ("Hectares + CI\nsite report", "#34495e"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 2.6))
    n = len(stages)
    w, gap = 1.12, 0.38
    for i, (label, color) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(
            plt.Rectangle((x, 0), w, 1, facecolor=color, edgecolor="white", linewidth=2, alpha=0.92)
        )
        ax.text(
            x + w / 2,
            0.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + w + gap, 0.5),
                xytext=(x + w, 0.5),
                arrowprops={"arrowstyle": "-|>", "color": "#555", "lw": 2},
            )
    ax.set_xlim(-0.2, n * (w + gap) - gap + 0.2)
    ax.set_ylim(-0.25, 1.25)
    ax.axis("off")
    ax.set_title(
        "WA EcoRehab AI — end-to-end ecological monitoring pipeline (EPSG:3577)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    _save(fig, out / "00_pipeline.png")


def main(config: str) -> None:
    F.set_plot_style()
    cfg = load_config(config)
    out = cfg.resolved_paths().outputs / "examples"
    out.mkdir(parents=True, exist_ok=True)
    classes = cfg.labels.classes
    names = [c.name for c in classes]
    res = cfg.project.resolution_m

    comp = pipeline.load_composite(cfg)
    label = read_raster(artifacts.label_path(cfg)).array[0]
    veg = read_vector(artifacts.vegetation_path(cfg))
    aoi = read_vector(artifacts.aoi_boundary_path(cfg))
    tiles = read_vector(artifacts.tile_index_path(cfg))
    truth = pipeline.load_truth(cfg)
    odir = cfg.resolved_paths().outputs

    make_pipeline_diagram(out, res)

    # 1) Data audit.
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    aoi_location_map(aoi, ax=axes[0, 0], label=cfg.aoi.name)
    ext = F.raster_extent(comp.transform, comp.height, comp.width)
    F.show_rgb(
        axes[0, 1], comp.array, comp.band_names, "True colour + native veg polygons", extent=ext
    )
    overlay_polygons(axes[0, 1], veg.to_crs(comp.crs), color="yellow", linewidth=0.6)
    F.add_scale_bar(axes[0, 1], res, units="metres")
    F.show_false_color(axes[1, 0], comp.array, comp.band_names)
    F.add_scale_bar(axes[1, 0], res, units="pixels")
    ndvi = compute_index(comp.array, comp.band_names, "ndvi")
    F.show_index(axes[1, 1], ndvi, "NDVI")
    F.add_scale_bar(axes[1, 1], res, units="pixels", color="black")
    fig.suptitle(
        "Exploratory data audit — Swan Coastal Plain (synthetic demo AOI)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out / "01_data_audit.png")

    # 2) Labels + spatial split.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    F.show_labels(
        axes[0],
        label,
        classes,
        "Weak labels (ignore = transparent)",
        ignore_index=cfg.labels.ignore_index,
    )
    F.add_scale_bar(axes[0], res, units="pixels")
    split_map(tiles, ax=axes[1])
    fig.tight_layout()
    _save(fig, out / "02_labels_and_split.png")

    # 3) U-Net evaluation: confusion + per-class + area-with-CI.
    metrics = json.loads((odir / "metrics" / "unet_metrics.json").read_text())
    area_df = pd.read_csv(odir / "vector_outputs" / "area_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    F.plot_confusion_matrix(metrics["confusion_matrix"], names, ax=axes[0])
    F.plot_per_class_metrics(metrics, names, ax=axes[1])
    F.plot_area_bars(area_df, ax=axes[2])
    fig.suptitle(
        f"U-Net evaluation on the spatial hold-out vs truth — "
        f"macro-F1={metrics['macro_f1']:.3f}, mIoU={metrics['mean_iou']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out / "03_evaluation.png")

    # 4) Prediction + uncertainty.
    pred = read_raster(odir / "predictions" / "prediction_mask.tif").array[0]
    unc = read_raster(odir / "predictions" / "uncertainty.tif").with_nodata_as_nan().array[0]
    rgb = F._rgb(comp.array, comp.band_names, ("red", "green", "blue"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    F.show_labels(
        axes[0],
        pred,
        classes,
        "U-Net prediction over true colour",
        ignore_index=cfg.labels.ignore_index,
        background=rgb,
    )
    F.add_scale_bar(axes[0], res, units="pixels")
    im = axes[1].imshow(unc, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Predictive entropy (uncertainty)")
    axes[1].set_axis_off()
    F.add_scale_bar(axes[1], res, units="pixels")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="normalised entropy")
    fig.tight_layout()
    _save(fig, out / "04_prediction_and_uncertainty.png")

    # 5) Model comparison.
    rf = json.loads((odir / "metrics" / "random_forest_metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    keys = ["macro_f1", "mean_iou", "weighted_f1", "balanced_accuracy"]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [rf[k] for k in keys], 0.4, label="RandomForest", color="#7b6888")
    ax.bar(x + 0.2, [metrics[k] for k in keys], 0.4, label="U-Net", color="#1b7837")
    for i, k in enumerate(keys):
        ax.text(i - 0.2, rf[k] + 0.01, f"{rf[k]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, metrics[k] + 0.01, f"{metrics[k]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["macro-F1", "mean IoU", "weighted-F1", "balanced acc"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the spatial hold-out", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, out / "05_model_comparison.png")

    # 6) Calibration + uncertainty distribution + error map (deep dive).
    if truth is not None:
        proba = (
            read_raster(odir / "predictions" / "unet_probability.tif").with_nodata_as_nan().array
        )
        test_mask = pipeline.split_mask(tiles, "test", truth.shape)
        valid = test_mask & (truth != cfg.labels.ignore_index) & np.isfinite(proba).all(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        if valid.any():
            P = proba.transpose(1, 2, 0)[valid]
            rc = reliability_curve(P, truth[valid], 10)
            F.plot_reliability(rc, ax=axes[0])
            axes[0].set_title(
                f"Reliability — ECE={rc.ece:.3f}, Brier={brier_score(P, truth[valid], 3):.3f}"
            )
        axes[1].hist(unc[np.isfinite(unc)].ravel(), bins=30, color="#c0504d")
        axes[1].set_title("Uncertainty distribution")
        axes[1].set_xlabel("normalised entropy")
        axes[1].set_ylabel("pixels")
        axes[1].spines[["top", "right"]].set_visible(False)
        fp_native = (pred == 1) & (truth != 1) & (truth != cfg.labels.ignore_index)
        axes[2].imshow(rgb)
        axes[2].imshow(np.ma.masked_equal(fp_native, 0), cmap="cool", alpha=0.9)
        axes[2].set_title("False-positive native (agriculture confusion)")
        axes[2].set_axis_off()
        F.add_scale_bar(axes[2], res, units="pixels")
        fig.suptitle(
            "Calibration, uncertainty, and the over-prediction failure mode",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, out / "06_calibration_and_failure.png")

    print(f"Wrote example figures -> {out}")
    for p in sorted(out.glob("*.png")):
        print(" ", p.name, f"{p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aoi_swan_coastal_plain.yaml")
    main(ap.parse_args().config)
