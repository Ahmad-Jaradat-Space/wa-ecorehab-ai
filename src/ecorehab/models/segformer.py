"""Modern segmentation backbones via segmentation-models-pytorch (optional).

These require the [dl] extra (``pip install ".[dl]"``: segmentation-models-pytorch
+ timm). They are kept behind a lazy import so the core U-Net baseline never
depends on them. ``in_channels`` is passed through, so multispectral input works.
"""

from __future__ import annotations

from torch import nn


def build_smp_model(arch: str, encoder: str, in_channels: int, num_classes: int) -> nn.Module:
    """Build an smp model: ``smp_unet`` (U-Net + pretrained encoder) or ``segformer``.

    Raises a clear ImportError if segmentation-models-pytorch is not installed.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - exercised only without [dl]
        raise ImportError(
            "segmentation-models-pytorch is required for arch "
            f"{arch!r}. Install with: pip install '.[dl]'"
        ) from exc

    arch = arch.lower()
    # Pretrained encoder weights are ImageNet-RGB; smp adapts the first conv to
    # in_channels by repeating/averaging weights. encoder_weights=None avoids a
    # network download in offline/CI environments.
    if arch == "smp_unet":
        return smp.Unet(
            encoder_name=encoder or "resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    if arch == "segformer":
        return smp.Segformer(
            encoder_name=encoder or "mit_b0",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    raise ValueError(f"unsupported smp arch {arch!r}")


__all__ = ["build_smp_model"]
