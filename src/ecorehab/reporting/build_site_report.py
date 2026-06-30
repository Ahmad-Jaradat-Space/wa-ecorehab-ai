"""Decision-grade HTML site report.

Assembles the batch-inference outputs (prediction, probability, uncertainty,
metrics, area summary) into a single self-contained HTML file with embedded
figures and tables. This is the "decision-grade ecological summary" deliverable.

CLI:
    python -m ecorehab.reporting.build_site_report --config configs/report.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from ecorehab import artifacts  # noqa: E402
from ecorehab.reporting import figures as F  # noqa: E402
from ecorehab.utils.config import Config, load_config  # noqa: E402
from ecorehab.utils.io import load_json, read_raster  # noqa: E402
from ecorehab.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem auto;max-width:1000px;color:#222;line-height:1.5}
 h1{border-bottom:3px solid #1b7837;padding-bottom:.3rem}
 h2{color:#1b7837;margin-top:2rem}
 table{border-collapse:collapse;margin:1rem 0;width:100%}
 th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right;font-size:.9rem}
 th{background:#f0f5f1;text-align:center}
 td:first-child,th:first-child{text-align:left}
 .fig{margin:1rem 0;text-align:center}
 .fig img{max-width:100%;border:1px solid #eee}
 .caption{font-size:.85rem;color:#555;margin-top:.3rem}
 .note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:.6rem 1rem;font-size:.9rem}
 .meta{color:#666;font-size:.85rem}
</style></head><body>
<h1>{{ title }}</h1>
<p class="meta">AOI: <b>{{ aoi_name }}</b> &middot; Equal-area CRS: EPSG:3577 &middot; Generated: {{ generated }}
{% if demo %}&middot; <b>DEMO MODE (synthetic imagery)</b>{% endif %}</p>

<div class="note"><b>Scope &amp; caveats.</b> This is a public-data <i>proxy</i> for native-vegetation
extent and disturbance. Labels are weakly supervised from government vegetation polygons and spectral
rules; metrics are computed on a spatial-block hold-out. This is <b>not</b> a species-level map nor a
mine-rehabilitation compliance assessment. See docs/limitations.md.</div>

<h2>1. Area summary (with uncertainty)</h2>
{{ area_table }}
<div class="fig">{{ area_fig }}<div class="caption">Predicted area by class with 95% spatial block-bootstrap confidence intervals.</div></div>

<h2>2. Vegetation map</h2>
<div class="fig">{{ map_fig }}<div class="caption">Predicted classes over the true-colour composite.</div></div>
<div class="fig">{{ prob_fig }}<div class="caption">Native-vegetation probability (model confidence).</div></div>
<div class="fig">{{ uncert_fig }}<div class="caption">Predictive entropy: bright = low confidence, flag for ecologist review.</div></div>

<h2>3. Model performance (spatial hold-out, target: {{ target }})</h2>
<p class="meta">macro-F1 = {{ macro_f1 }} &middot; mean IoU = {{ mean_iou }} &middot; balanced accuracy = {{ bal_acc }}
{% if ece %}&middot; ECE = {{ ece }}{% endif %}</p>
{{ metrics_table }}
<div class="fig">{{ cm_fig }}<div class="caption">Row-normalised confusion matrix on the hold-out split.</div></div>

{% if per_tenement %}
<h2>4. Per-management-unit summary</h2>
{{ per_tenement }}
{% endif %}

<h2>Provenance</h2>
<p class="meta">Prediction: {{ pred_path }}<br>Metrics: {{ metrics_path }}</p>
</body></html>
"""


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}">'


def build_site_report(cfg: Config) -> Path:
    """Render the HTML site report. Returns the output path."""
    rep = cfg.report
    root = cfg.resolved_paths().root
    classes = cfg.labels.classes
    class_names = [c.name for c in classes]

    pred = read_raster(_resolve(root, rep.prediction_raster))
    composite = read_raster(artifacts.composite_path(cfg)).with_nodata_as_nan()
    metrics = (
        load_json(_resolve(root, rep.metrics_json))
        if _resolve(root, rep.metrics_json).exists()
        else {}
    )
    area_df = (
        pd.read_csv(_resolve(root, rep.area_csv)) if _resolve(root, rep.area_csv).exists() else None
    )

    # --- figures ---
    map_fig = _prediction_map_fig(pred.array[0], composite.array, composite.band_names, cfg)
    prob_fig = _raster_fig(
        _resolve(root, rep.probability_raster), "Native probability", "viridis", 0, 1
    )
    uncert_fig = _raster_fig(
        _resolve(root, rep.uncertainty_raster), "Uncertainty (entropy)", "magma", 0, 1
    )

    area_fig_html, area_table = "", "<p>(area summary not available)</p>"
    if area_df is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        F.plot_area_bars(area_df, ax=ax)
        area_fig_html = _fig_to_img(fig)
        area_table = area_df.to_html(index=False, float_format=lambda x: f"{x:,.1f}")

    cm_fig_html, metrics_table = "", "<p>(metrics not available)</p>"
    if metrics.get("confusion_matrix"):
        fig, ax = plt.subplots(figsize=(5, 4))
        F.plot_confusion_matrix(metrics["confusion_matrix"], class_names, ax=ax)
        cm_fig_html = _fig_to_img(fig)
        metrics_table = _metrics_table_html(metrics, class_names)

    per_tenement = None
    poly_csv = _resolve(root, rep.area_csv).parent / "area_summary_by_polygon.csv"
    if poly_csv.exists():
        per_tenement = pd.read_csv(poly_csv).to_html(
            index=False, float_format=lambda x: f"{x:,.1f}"
        )

    manifest = artifacts.manifest_path(cfg)
    demo = cfg.aoi.demo

    html = Template(_TEMPLATE).render(
        title=rep.title,
        aoi_name=rep.aoi_name,
        demo=demo,
        generated=_now(manifest),
        area_table=area_table,
        area_fig=area_fig_html,
        map_fig=map_fig,
        prob_fig=prob_fig,
        uncert_fig=uncert_fig,
        target=metrics.get("evaluation_target", "n/a"),
        macro_f1=_fmt(metrics.get("macro_f1")),
        mean_iou=_fmt(metrics.get("mean_iou")),
        bal_acc=_fmt(metrics.get("balanced_accuracy")),
        ece=_fmt(metrics.get("ece")),
        metrics_table=metrics_table,
        cm_fig=cm_fig_html,
        per_tenement=per_tenement,
        pred_path=str(rep.prediction_raster),
        metrics_path=str(rep.metrics_json),
    )
    out = _resolve(root, rep.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Wrote site report -> %s", out)
    return out


def _prediction_map_fig(pred, composite, band_names, cfg):
    from ecorehab.reporting.figures import _rgb

    fig, ax = plt.subplots(figsize=(7, 7))
    rgb = _rgb(composite, band_names or tuple(cfg.data.bands), ("red", "green", "blue"))
    F.show_labels(
        ax,
        pred,
        cfg.labels.classes,
        title="Predicted vegetation classes",
        ignore_index=cfg.labels.ignore_index,
        background=rgb,
    )
    return _fig_to_img(fig)


def _raster_fig(path: Path, title: str, cmap: str, vmin, vmax) -> str:
    if not Path(path).exists():
        return "<p>(not available)</p>"
    arr = read_raster(path).with_nodata_as_nan().array[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_img(fig)


def _metrics_table_html(metrics: dict, class_names: list[str]) -> str:
    per = metrics["per_class"]
    rows = []
    for i, name in enumerate(class_names):
        d = per.get(i, per.get(str(i), {}))
        rows.append(
            {
                "class": name,
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "f1": d.get("f1"),
                "iou": d.get("iou"),
                "support": d.get("support"),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_html(index=False, float_format=lambda x: f"{x:.3f}")


def _resolve(root: Path, p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _now(manifest: Path) -> str:
    try:
        return load_json(manifest).get("generated_utc", "")
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the HTML site report.")
    parser.add_argument("--config", required=True, help="path to report config YAML")
    args = parser.parse_args(argv)
    build_site_report(load_config(args.config))


if __name__ == "__main__":
    main()
