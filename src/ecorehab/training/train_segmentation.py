"""Train a PyTorch semantic-segmentation model (U-Net / smp / SegFormer).

Highlights:
  * multispectral input (bands + spectral indices), not just RGB;
  * imbalance handled via inverse-frequency class weights + CE/Dice loss;
  * train/val/test come from the spatial-block split (no leakage);
  * early stopping on validation macro-F1;
  * checkpoint stores everything inference needs (arch, channels, feature
    config, normalisation stats, class ids) so batch_predict is self-contained.

CLI:
    python -m ecorehab.training.train_segmentation --config configs/unet.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecorehab import constants, pipeline
from ecorehab.evaluation.metrics import confusion_matrix, metrics_from_confusion
from ecorehab.labels.tiling import generate_tiles
from ecorehab.models.losses import build_loss, inverse_frequency_weights
from ecorehab.models.unet import build_segmentation_model
from ecorehab.training.augmentations import make_augmenter
from ecorehab.training.dataset import (
    TileSegmentationDataset,
    compute_feature_stats,
    standardize,
)
from ecorehab.utils.config import Config, load_config
from ecorehab.utils.io import save_json, write_raster
from ecorehab.utils.logging import get_logger

logger = get_logger(__name__)


def train_segmentation(cfg: Config) -> dict:
    """Train + evaluate + persist a segmentation model. Returns metrics."""
    torch.manual_seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats, fnames, comp = pipeline.load_feature_stack(cfg)
    label = pipeline.load_label(cfg)
    tiles = pipeline.load_tile_index(cfg)
    class_ids = [c.id for c in cfg.labels.classes]
    class_names = [c.name for c in cfg.labels.classes]
    num_classes = len(class_ids)
    ignore = cfg.labels.ignore_index

    train_tiles = tiles[tiles.split == "train"]
    val_tiles = tiles[tiles.split == "val"]
    test_tiles = tiles[tiles.split == "test"]
    if len(train_tiles) == 0 or len(val_tiles) == 0:
        raise ValueError("need non-empty train and val splits; check tiling config")

    mean, std = compute_feature_stats(feats, train_tiles)
    augment = make_augmenter(enabled=True, seed=cfg.project.seed)
    train_ds = TileSegmentationDataset(feats, label, train_tiles, mean, std, augment, ignore)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.segmentation.batch_size,
        shuffle=True,
        num_workers=cfg.segmentation.num_workers,
        drop_last=False,
    )

    in_channels = feats.shape[0]
    model = build_segmentation_model(cfg.segmentation, in_channels, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model %s | in_ch=%d classes=%d params=%.2fM",
        cfg.segmentation.arch,
        in_channels,
        num_classes,
        n_params / 1e6,
    )

    # Inverse-frequency class weights from train label counts.
    train_label = label[pipeline.split_mask(tiles, "train", label.shape)]
    counts = {c: int(np.count_nonzero(train_label == c)) for c in class_ids}
    weights = (
        torch.tensor(cfg.segmentation.class_weights, dtype=torch.float32)
        if cfg.segmentation.class_weights
        else inverse_frequency_weights(counts, num_classes)
    ).to(device)
    criterion = build_loss(cfg.segmentation.loss, ignore_index=ignore, class_weights=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.segmentation.lr, weight_decay=cfg.segmentation.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.segmentation.epochs)

    history = []
    best_macro_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    target_kind, target = pipeline.evaluation_target(cfg)

    for epoch in range(1, cfg.segmentation.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for img, msk in train_dl:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, msk)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * img.size(0)
        scheduler.step()
        epoch_loss /= max(1, len(train_ds))

        val_metrics = _evaluate_tiles(
            model, feats, target, val_tiles, mean, std, device, num_classes, ignore
        )
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss, "val_macro_f1": val_metrics["macro_f1"]}
        )
        logger.info(
            "epoch %02d | loss %.4f | val macroF1 %.3f meanIoU %.3f",
            epoch,
            epoch_loss,
            val_metrics["macro_f1"],
            val_metrics["mean_iou"],
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.segmentation.patience:
                logger.info(
                    "early stopping at epoch %d (best val macroF1 %.3f)", epoch, best_macro_f1
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation + full-AOI prediction outputs.
    test_metrics = _evaluate_tiles(
        model, feats, target, test_tiles, mean, std, device, num_classes, ignore
    )
    test_metrics["evaluation_target"] = target_kind
    test_metrics["history"] = history
    test_metrics["best_val_macro_f1"] = best_macro_f1
    test_metrics["n_params"] = int(n_params)

    _save_checkpoint(
        cfg, model, mean, std, fnames, class_ids, class_names, in_channels, num_classes
    )
    proba, pred = predict_full(model, feats, mean, std, device, num_classes)
    _write_outputs(cfg, proba, pred, comp, test_metrics)
    logger.info(
        "TEST macroF1=%.3f meanIoU=%.3f balAcc=%.3f (target=%s)",
        test_metrics["macro_f1"],
        test_metrics["mean_iou"],
        test_metrics["balanced_accuracy"],
        target_kind,
    )
    return test_metrics


@torch.no_grad()
def _evaluate_tiles(model, feats, target, tiles, mean, std, device, num_classes, ignore) -> dict:
    """Accumulate a confusion matrix over the tiles of one split -> metrics."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype="int64")
    for _, t in tiles.iterrows():
        r0, c0, h, w = int(t.row_off), int(t.col_off), int(t.height), int(t.width)
        img = standardize(feats[:, r0 : r0 + h, c0 : c0 + w], mean, std)
        x = torch.from_numpy(img[None]).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        cm += confusion_matrix(target[r0 : r0 + h, c0 : c0 + w], pred, num_classes, ignore)
    out = metrics_from_confusion(cm)
    out["confusion_matrix"] = cm.tolist()
    return out


@torch.no_grad()
def predict_full(model, feats, mean, std, device, num_classes):
    """Tile the whole AOI and assemble averaged class probabilities + argmax."""
    model.eval()
    _, h, w = feats.shape
    tile = min(256, h, w)
    tiles = generate_tiles((h, w), tile_size=tile, stride=tile)
    prob_sum = np.zeros((num_classes, h, w), dtype="float32")
    counts = np.zeros((h, w), dtype="float32")
    for t in tiles:
        img = standardize(
            feats[:, t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width], mean, std
        )
        x = torch.from_numpy(img[None]).to(device)
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        prob_sum[:, t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] += p
        counts[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] += 1
    counts = np.maximum(counts, 1)
    proba = prob_sum / counts[None]
    pred = proba.argmax(axis=0).astype("uint8")
    # nodata where every input feature was NaN.
    nodata = np.all(~np.isfinite(feats), axis=0)
    pred[nodata] = constants.IGNORE_INDEX
    return proba, pred


def _save_checkpoint(
    cfg, model, mean, std, fnames, class_ids, class_names, in_channels, num_classes
):
    ckpt = {
        "state_dict": model.state_dict(),
        "arch": cfg.segmentation.arch,
        "encoder": cfg.segmentation.encoder,
        "base_channels": cfg.segmentation.base_channels,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": fnames,
        "feature_indices": list(cfg.features.indices),
        "include_bands": cfg.features.include_bands,
        "bands": list(cfg.data.bands),
        "class_ids": class_ids,
        "class_names": class_names,
        "ignore_index": cfg.labels.ignore_index,
    }
    path = Path(cfg.segmentation.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    logger.info("Saved checkpoint -> %s", path)


def _write_outputs(cfg, proba, pred, comp, metrics) -> None:
    out_pred = cfg.resolved_paths().outputs / "predictions"
    out_metrics = cfg.resolved_paths().outputs / "metrics"
    name = cfg.segmentation.arch
    write_raster(
        out_pred / f"{name}_prediction.tif",
        pred.astype("uint8"),
        comp.transform,
        comp.crs,
        nodata=constants.IGNORE_INDEX,
        band_names=("prediction",),
    )
    write_raster(
        out_pred / f"{name}_probability.tif",
        proba.astype("float32"),
        comp.transform,
        comp.crs,
        nodata=constants.PROB_NODATA,
        band_names=tuple(c.name for c in cfg.labels.classes),
    )
    save_json(metrics, out_metrics / f"{name}_metrics.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", required=True, help="path to a segmentation config YAML")
    args = parser.parse_args(argv)
    train_segmentation(load_config(args.config))


if __name__ == "__main__":
    main()
