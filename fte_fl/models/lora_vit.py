"""LoRA-ViT: frozen timm ViT-Base/16 + LoRA adapters for CIFAR-100.

Reference:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).
  Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.

Architecture:
  Base:  timm vit_base_patch16_224  — 86M parameters (all frozen).
  LoRA:  r=8, alpha=16, applied to attn.qkv (implicitly q and v via
         the fused QKV projection) and mlp.fc1 in every transformer block.
  Trainable: < 2 000 000 parameters (asserted at construction).

Input handling for CIFAR-100 (32×32 → 224×224):
  ViT-Base/16 was designed for 224×224 images (14×14 = 196 patches of 16×16).
  CIFAR-100 images are 32×32.  We use torchvision-style bilinear resize inside
  forward() to upscale to 224×224 before passing to the ViT backbone.
  This is the simplest, most compatible approach and matches common fine-tuning
  practice for CIFAR on ViT-Base (see Touvron et al., DeiT, 2021).
  Alternative (adjusting positional embeddings for 32×32 / patch=2) was
  considered but discarded because it requires re-initializing pos_embed and
  loses the pretrained spatial structure.

Pretrained-weight status:
  Weights are downloaded once from the timm HuggingFace hub
  (~330 MB, vit_base_patch16_224.augreg_in21k_ft_in1k or similar).
  When offline or if download fails, pass pretrained=False; the base model
  will be randomly initialized and the LoRA adapter structure is identical.
  The build argument `pretrained` is exposed to the caller.
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

    Only LoRA adapter parameters (lora_A, lora_B in each replaced module)
    receive gradients.  All base ViT weights are frozen.
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
