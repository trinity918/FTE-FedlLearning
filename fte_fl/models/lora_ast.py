"""LoRA-adapted spectrogram transformer for Speech Commands.

This module implements an AST-*style* spectrogram transformer with LoRA
adapters.  It is NOT a faithful reproduction of the Audio Spectrogram
Transformer (AST) published by Gong et al. 2021, and must not be described
as one.  See the "Deviations" section below.

Architecture:
  A Vision Transformer (ViT) adapted to accept 2-D log-mel spectrogram
  patches, following the same general approach as Gong et al. 2021.
  LoRA adapters (r=8, α=16) are applied to the top 6 transformer blocks'
  fused QKV projection and MLP fc1.  All other base parameters are frozen.
  The classification head is kept trainable.

  Input:  (B, 1, 64, 101) log-mel spectrogram.
  Patches: 16×16 non-overlapping → 4×7 = 28 patches (time padded to 112).
  embed_dim=192, depth=12, num_heads=3.
  Trainable < 2,000,000 parameters (asserted at construction).

Deviations from Gong et al. 2021 (arXiv:2104.01778) — research-critical:
  1. Random initialization: Gong et al. initialize from DeiT-Base pretrained
     on ImageNet-21k.  This implementation uses random weights throughout.
     This is a fundamental limitation affecting convergence speed and final
     accuracy; results are not comparable to the published AST numbers.
  2. Non-overlapping patches: The paper uses 16×16 patches with stride 10×10
     (overlapping), yielding ~248 tokens.  We use stride=16 (non-overlapping),
     yielding 28 tokens.  This significantly reduces temporal resolution.
  3. Smaller backbone: DeiT-Base uses embed_dim=768, heads=12.  We use
     embed_dim=192, heads=3 — a much smaller configuration.
  4. 64 mel bins: The paper's primary experiments use 128 mel bins.

Frozen random backbone:
  Because the base is random AND frozen, only the LoRA adapters in the top
  `lora_blocks` blocks and the head are trained; the patch embedding,
  positional embedding, CLS token, final norm and lower blocks stay at their
  random initial values.  This is LoRA on random features, not fine-tuning
  of a pretrained model, and its accuracy may be limited by that design.

Why timm was not used:
  timm 1.0.30 does not contain a native AST implementation.  Searching the
  model registry for 'ast', 'audio_spectrogram', 'wav2vec', and 'hubert'
  returned zero genuine matches.  The class LoRAAST was therefore constructed
  from the ViT building blocks in vit_tiny.py.

References:
  Gong et al., "AST: Audio Spectrogram Transformer",
  Interspeech 2021 (arXiv:2104.01778).
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).
"""
from __future__ import annotations

import math
from typing import Set

import torch
import torch.nn as nn

from fte_fl.models.base import FTEModel, count_parameters
from fte_fl.models.vit_tiny import TransformerBlock
from fte_fl.models.lora import inject_lora

_LORA_TARGETS: Set[str] = {"qkv", "fc1"}


class SpectrogramPatchEmbed(nn.Module):
    """Project log-mel spectrogram patches into an embedding space.

    Pads the time axis to the next multiple of patch_size, then applies a
    strided Conv2d as the patch projection (standard ViT PatchEmbed approach).

    Input:  (B, 1, F, T) — single-channel log-mel spectrogram.
    Output: (B, num_patches, embed_dim).
    """

    def __init__(
        self,
        freq_bins: int = 64,
        time_frames: int = 101,
        patch_size: int = 16,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pad_t = math.ceil(time_frames / patch_size) * patch_size - time_frames
        t_padded = time_frames + self.pad_t

        assert freq_bins % patch_size == 0, (
            f"freq_bins ({freq_bins}) must be divisible by patch_size ({patch_size})"
        )
        self.n_freq = freq_bins // patch_size
        self.n_time = t_padded // patch_size
        self.num_patches = self.n_freq * self.n_time

        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t > 0:
            x = nn.functional.pad(x, (0, self.pad_t))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SpectrogramTransformerBase(nn.Module):
    """Randomly-initialized ViT-style backbone over 2-D spectrogram patches.

    This is the base network used by LoRAAST.  It is architecturally similar
    to the AST of Gong et al. 2021, but initialized from random weights rather
    than pretrained ImageNet weights.  See the module docstring for all
    deviations from the published AST.
    """

    def __init__(
        self,
        freq_bins: int = 64,
        time_frames: int = 101,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int = 35,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = SpectrogramPatchEmbed(
            freq_bins, time_frames, patch_size, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class LoRAAST(FTEModel):
    """LoRA-adapted AST-style spectrogram transformer for Speech Commands.

    This is an AST-*style* model (ViT applied to 2-D mel patches), NOT a
    faithful implementation of Gong et al. 2021.  Key deviations: random
    initialization (no ImageNet pretraining), non-overlapping patches, and a
    smaller backbone (embed_dim=192 vs. 768).  See the module docstring for
    the full list of deviations.

    LoRA is applied to the top `lora_blocks` transformer blocks' QKV
    projection and MLP fc1.  All other base parameters are frozen.
    The classification head is trainable.
    Base initialization is always random (pretrained_loaded = False).
    Trainable parameter count is asserted < 2,000,000 at construction.
    """

    MAX_TRAINABLE: int = 2_000_000

    def __init__(
        self,
        num_classes: int = 35,
        freq_bins: int = 64,
        time_frames: int = 101,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        r: int = 8,
        alpha: float = 16.0,
        lora_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.base = SpectrogramTransformerBase(
            freq_bins=freq_bins,
            time_frames=time_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
        )
        # Base is always randomly initialized — no pretrained weights available.
        self.pretrained_loaded: bool = False

        # Freeze ALL base parameters
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Apply LoRA to the top `lora_blocks` transformer blocks only
        for blk in self.base.blocks[-lora_blocks:]:
            inject_lora(blk, target_module_names=_LORA_TARGETS, r=r, alpha=alpha)

        # Unfreeze the classification head
        for p in self.base.head.parameters():
            p.requires_grad_(True)

        _, trainable, _ = count_parameters(self)
        assert trainable < self.MAX_TRAINABLE, (
            f"LoRAAST trainable params {trainable:,} >= {self.MAX_TRAINABLE:,}. "
            f"Reduce lora_blocks, r, or target modules."
        )

        self.r = r
        self.alpha = alpha
        self.lora_blocks = lora_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 64, 101) log-mel spectrogram → logits (B, num_classes)."""
        return self.base(x)
