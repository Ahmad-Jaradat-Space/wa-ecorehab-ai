"""Segmentation losses tuned for imbalanced ecological classes.

All losses respect ``ignore_index`` so boundary/nodata pixels never contribute
to the gradient. ``build_loss`` selects between cross-entropy, Dice, focal, and
the combined CE+Dice default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss with ignore handling."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        target_clamped = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        onehot = onehot * mask
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss (down-weights easy, abundant pixels)."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pt must come from the UNWEIGHTED CE (pt = softmax prob of the true class);
        # mixing class weights into pt corrupts the (1-pt)^gamma modulation. Apply
        # the class weight as a separate multiplicative factor instead.
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target.clamp(min=0)]  # per-pixel class weight
            loss = loss * w
        valid = target != self.ignore_index
        return loss[valid].mean() if valid.any() else loss.sum() * 0.0


class CEDiceLoss(nn.Module):
    """Combined weighted cross-entropy + Dice (robust default)."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_w * self.ce(logits, target) + self.dice_w * self.dice(logits, target)


def build_loss(
    name: str,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Construct a loss by name: ``ce`` | ``dice`` | ``focal`` | ``ce_dice``."""
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(weight=class_weights, ignore_index=ignore_index)
    if name == "ce_dice":
        return CEDiceLoss(weight=class_weights, ignore_index=ignore_index)
    raise ValueError(f"unknown loss {name!r}; choose ce|dice|focal|ce_dice")


def inverse_frequency_weights(class_counts: dict[int, int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    counts = torch.tensor(
        [max(1, class_counts.get(c, 0)) for c in range(num_classes)], dtype=torch.float32
    )
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


__all__ = [
    "CEDiceLoss",
    "DiceLoss",
    "FocalLoss",
    "build_loss",
    "inverse_frequency_weights",
]
