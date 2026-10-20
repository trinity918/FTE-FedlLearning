"""LoRA-ViT: frozen timm ViT-Base/16 + LoRA adapters for CIFAR-100.

Reference:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).
  Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.

Architecture:
  Base:      timm vit_base_patch16_224 — 86M parameters, all frozen.
  LoRA:      r=8, alpha=16, applied to attn.qkv (fused QKV projection) and
             mlp.fc1 in every transformer block.
  Trainable: adapter parameters + classification head, < 2M total.

Input handling for CIFAR-100 (32×32 → 224×224):
  ViT-Base/16 was designed for 224×224 images (14×14 = 196 patches of 16×16).
  CIFAR-100 images are 32×32.  We bilinearly resize inside forward() to
  224×224 before passing to the ViT backbone.  This is the standard practice
  for fine-tuning ViT-Base on CIFAR (see Touvron et al., DeiT, 2021).

Initialization / pretrained-weight status:
  The constructor `pretrained` argument controls whether timm loads the
  official ViT-Base/16 weights from the HuggingFace hub (~330 MB).
  The default is `pretrained=False`, meaning the base model is **randomly
  initialized**.  The `pretrained_loaded` attribute is set from the
  `pretrained` argument; timm raises if a requested download fails, but the
  loaded weights are not independently checked.

  In the current project environment, all builds use `pretrained=False`
  (offline use).  No pretrained weights have been downloaded or verified.
  The `verify_models.py` output therefore shows `Init: random`.
  Passing `pretrained=True` requires internet access to the timm HuggingFace
  hub and is a one-time ~330 MB download.

  With `pretrained=False`, training updates LoRA adapters and the head on
  top of a frozen, randomly initialized ViT-Base.  Such runs are NOT
  pretrained fine-tuning and should not be described as such.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from fte_fl.models.base import FTEModel, count_parameters
from fte_fl.models.lora import inject_lora

_VIT_MODEL_NAME = "vit_base_patch16_224"
_LORA_TARGETS = {"qkv", "fc1"}  # attn QKV projection + MLP first layer


class LoRAViT(FTEModel):
    """Frozen ViT-Base/16 with LoRA adapters for CIFAR-100.

    Trainable parameters are the LoRA adapters (lora_A, lora_B in each
    replaced module) and the classification head.  All other base ViT
    weights are frozen.
    The trainable count is asserted < 2 000 000 at construction time.
    """

    MAX_TRAINABLE: int = 2_000_000

    def __init__(
        self,
        num_classes: int = 100,
        r: int = 8,
        alpha: float = 16.0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required for LoRAViT")

        # Load base ViT (all weights frozen after construction by inject_lora)
        self.base = timm.create_model(
            _VIT_MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.pretrained_loaded = pretrained
        # Record the original num_classes for the head replacement
        self._num_classes = num_classes

        # Freeze ALL base parameters first
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Inject LoRA into attn.qkv and mlp.fc1 in every transformer block
        inject_lora(self.base, target_module_names=_LORA_TARGETS, r=r, alpha=alpha)

        # Unfreeze the classification head (it was replaced by timm for num_classes)
        for p in self.base.head.parameters():
            p.requires_grad_(True)

        _, trainable, _ = count_parameters(self)
        assert trainable < self.MAX_TRAINABLE, (
            f"LoRAViT trainable params {trainable:,} >= {self.MAX_TRAINABLE:,}. "
            f"Reduce LoRA rank or target modules."
        )

        self.r = r
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) — resized to 224×224 internally if needed.

        CIFAR-100 images are 32×32.  This model bilinearly upscales them to
        224×224 inside forward() to match ViT-Base/16's expected resolution.
        """
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.base(x)
