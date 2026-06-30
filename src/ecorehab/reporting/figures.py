"""Reusable scientific figures with consistent class colours.

The notebook imports these so plotting logic lives in the package, not in
notebook cells. Every map states its CRS where relevant and uses the same class
colour scheme defined by the config.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from ecorehab.utils.config import ClassDef


def set_plot_style() -> None:
    """Apply a clean, consistent scientific style for all figures + the notebook."""
    try:
        import seaborn as sns

        sns.set_theme(context="notebook", style="white", font_scale=1.0)
    except Exception:  # pragma: no cover
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "image.interpolation": "nearest",
            "legend.frameon": True,
            "legend.framealpha": 0.85,
        }
    )


def _nice_length(span_m: float) -> float:
    """A round 1/2/5 x 10^k length, ~a quarter of the span, for a scale bar."""
    raw = max(span_m * 0.25, 1.0)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (5, 2, 1):
        if raw >= m * mag:
            return m * mag
    return mag


def add_scale_bar(ax, resolution_m: float, units: str = "pixels", color: str = "white") -> Any:
    """Add a geospatial scale bar (lower-left) to a map/image axis.

    Args:
        resolution_m: ground sample distance in metres.
        units: "pixels" if the axis is in pixel coordinates (imshow without
            extent), or "metres" if it is in CRS metres (imshow with extent).
    """
    from matplotlib import font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x0, x1 = ax.get_xlim()
    width = abs(x1 - x0)
    span_m = width * resolution_m if units == "pixels" else width
    bar_m = _nice_length(span_m)
    size = bar_m / resolution_m if units == "pixels" else bar_m
    label = f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    bar = AnchoredSizeBar(
        ax.transData,
        size,
        label,
        "lower left",
        pad=0.4,
        color=color,
        frameon=True,
        size_vertical=width * 0.012,
        fontproperties=fm.FontProperties(size=8),
    )
    bar.patch.set(alpha=0.45, facecolor="black", edgecolor="none")
    ax.add_artist(bar)
    return ax


def class_cmap(classes: list[ClassDef]) -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete colormap + norm matching the configured class ids/colours."""
    ids = [c.id for c in classes]
    colors = [c.color for c in classes]
    cmap = ListedColormap(colors)
    bounds = [min(ids) - 0.5] + [i + 0.5 for i in sorted(ids)]
    return cmap, BoundaryNorm(bounds, cmap.N)


def _stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] for display (NaN-safe)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, [p_low, p_high])
    out = (arr - lo) / (hi - lo + 1e-9)
    return np.clip(out, 0, 1)


def _rgb(composite: np.ndarray, band_names, bands) -> np.ndarray:
    idx = [list(band_names).index(b) for b in bands]
    rgb = np.dstack([_stretch(composite[i]) for i in idx])
    return np.nan_to_num(rgb, nan=0.0)


def raster_extent(transform, height: int, width: int) -> list[float]:
    """imshow ``extent`` [xmin, xmax, ymin, ymax] in CRS units for a north-up raster.

    Pass this (with ``origin='upper'``) so vector overlays in the same CRS align
    with the image instead of forcing matplotlib to autoscale to metre coords.
    """
    import rasterio

    minx, miny, maxx, maxy = rasterio.transform.array_bounds(height, width, transform)
    return [minx, maxx, miny, maxy]


def show_rgb(ax, composite, band_names, title="True colour (RGB)", extent=None) -> Any:
    """True-colour composite (red, green, blue)."""
    ax.imshow(_rgb(composite, band_names, ("red", "green", "blue")), extent=extent, origin="upper")
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def show_false_color(ax, composite, band_names, title="False colour (NIR-R-G)", extent=None) -> Any:
    """False-colour composite (NIR, red, green): vegetation appears red."""
    ax.imshow(_rgb(composite, band_names, ("nir", "red", "green")), extent=extent, origin="upper")
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def show_index(ax, index_array, name="NDVI", cmap="RdYlGn", vmin=-1, vmax=1, extent=None) -> Any:
    im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
    ax.set_title(name)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def show_labels(
    ax, label, classes, title="Labels", ignore_index=255, background=None, extent=None
) -> Any:
    """Class label map with the configured discrete colours; ignore -> transparent."""
    cmap, norm = class_cmap(classes)
    if background is not None:
        ax.imshow(background, extent=extent, origin="upper")
    masked = np.ma.masked_equal(label, ignore_index)
    ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        alpha=0.65 if background is not None else 1.0,
        extent=extent,
        origin="upper",
    )
    ax.set_title(title)
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c.color, label=c.name) for c in classes]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)
    return ax


def plot_class_distribution(counts: dict[int, int], classes: list[ClassDef], ax=None) -> Any:
    ax = ax or plt.gca()
    names = [c.name for c in classes]
    vals = [counts.get(c.id, 0) for c in classes]
    colors = [c.color for c in classes]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("pixels")
    ax.set_title("Class distribution (label pixels)")
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_confusion_matrix(cm, class_names, ax=None, normalize=True) -> Any:
    ax = ax or plt.gca()
    cm = np.asarray(cm, dtype="float64")
    disp = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            ax.text(
                j,
                i,
                f"{disp[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if disp[i, j] > 0.5 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_per_class_metrics(metrics: dict, class_names, ax=None) -> Any:
    ax = ax or plt.gca()
    per = metrics["per_class"]
    x = np.arange(len(class_names))
    width = 0.2

    def _cls(i: int) -> dict:
        # per_class keys are ints in-memory but strings after a JSON round-trip.
        return per.get(i, per.get(str(i), {}))

    for k, off in zip(["precision", "recall", "f1", "iou"], [-1.5, -0.5, 0.5, 1.5], strict=True):
        vals = [_cls(i).get(k, 0.0) for i in range(len(class_names))]
        ax.bar(x + off * width, vals, width, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Per-class metrics")
    ax.legend(fontsize=7, ncol=4)
    return ax


def plot_reliability(curve, ax=None) -> Any:
    """Reliability diagram from an evaluation.calibration.ReliabilityCurve."""
    ax = ax or plt.gca()
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    valid = curve.bin_count > 0
    ax.plot(curve.bin_confidence[valid], curve.bin_accuracy[valid], "o-", label="model")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability (ECE={curve.ece:.3f})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def plot_area_bars(area_df, ax=None) -> Any:
    """Bar chart of class area with bootstrap confidence-interval error bars."""
    ax = ax or plt.gca()
    names = area_df["class_name"].tolist()
    point = area_df["area_ha"].to_numpy()
    lower = area_df["area_lower_ha"].to_numpy()
    upper = area_df["area_upper_ha"].to_numpy()
    err = np.vstack([point - lower, upper - point])
    ax.bar(names, point, yerr=err, capsize=5, color="#4a7c59")
    ax.set_ylabel("hectares")
    ax.set_title("Predicted area by class (95% bootstrap CI)")
    ax.tick_params(axis="x", rotation=20)
    return ax


__all__ = [
    "add_scale_bar",
    "class_cmap",
    "plot_area_bars",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_reliability",
    "raster_extent",
    "set_plot_style",
    "show_false_color",
    "show_index",
    "show_labels",
    "show_rgb",
]
