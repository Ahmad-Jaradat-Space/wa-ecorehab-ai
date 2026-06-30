"""A compact from-scratch U-Net + segmentation-model factory.

The hand-written U-Net depends only on torch (no smp/timm), so the segmentation
baseline always runs. ``build_segmentation_model`` dispatches to smp/SegFormer
variants when the optional [dl] extras are installed.

Crucially the input channel count is configurable: we feed multispectral bands +
spectral indices (e.g. 13 channels), not just RGB.
"""

from __future__ import annotations

import torch
from torch import nn

from ecorehab.utils.config import SegmentationConfig


def _norm(channels: int, max_groups: int = 8) -> nn.Module:
    """GroupNorm rather than BatchNorm.

    Segmentation here trains with few, small batches; BatchNorm's running
    statistics never converge in that regime, which squashes logit magnitudes at
    eval time and leaves softmax probabilities near-uniform (useless for
    calibration/uncertainty). GroupNorm is batch-size independent and behaves
    identically in train/eval, so probabilities stay informative.
    """
    groups = max(1, min(max_groups, channels))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """4-level U-Net. Works on any input size divisible by 16."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc = _DoubleConv(in_channels, c)
        self.down1 = self._down(c, c * 2)
        self.down2 = self._down(c * 2, c * 4)
        self.down3 = self._down(c * 4, c * 8)
        self.down4 = self._down(c * 8, c * 16)
        self.up1 = _Up(c * 16, c * 8)
        self.up2 = _Up(c * 8, c * 4)
        self.up3 = _Up(c * 4, c * 2)
        self.up4 = _Up(c * 2, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    @staticmethod
    def _down(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd-sized (defensive; tiles are multiples of 16 by convention).
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


def build_segmentation_model(
    cfg: SegmentationConfig, in_channels: int, num_classes: int
) -> nn.Module:
    """Construct a segmentation model from config.

    ``unet`` is always available. ``smp_unet`` / ``segformer`` require the [dl]
    extra (segmentation-models-pytorch + timm) and raise a clear error otherwise.
    """
    arch = cfg.arch.lower()
    if arch == "unet":
        return UNet(in_channels, num_classes, base_channels=cfg.base_channels)
    if arch in {"smp_unet", "segformer"}:
        from ecorehab.models.segformer import build_smp_model

        return build_smp_model(arch, cfg.encoder, in_channels, num_classes)
    raise ValueError(f"unknown segmentation arch {cfg.arch!r}")


__all__ = ["UNet", "build_segmentation_model"]
