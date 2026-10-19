"""LoRA-AST: Audio Spectrogram Transformer with LoRA adapters for Speech Commands.

Architecture choice and rationale:
  timm 1.0.30 does NOT contain a native AST (Audio Spectrogram Transformer)
  implementation.  The model registry search confirms this: the substring 'ast'
  only matches FasteRNet/FastViT models.

  The AST (Gong et al., 2021) is architecturally identical to a Vision
  Transformer (ViT) applied to 2-D log-mel spectrogram patches.  Its key
  distinguishing characteristics are:
    - 2-D patch embedding over the (mel-bin × time-frame) spectrogram.
    - Overlapping patches (stride < patch_size) for fine-grained audio detail.
    - Positional embedding adapted for non-square input aspect ratios.
  We implement this from scratch using the ViT building blocks defined in
  vit_tiny.py, then apply LoRA adapters to the top transformer blocks.

Reference:
  Gong et al., "AST: Audio Spectrogram Transformer",
  Interspeech 2021 (arXiv:2104.01778).

  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).

Configuration (non-overlapping patches for simplicity):
  Input: (B, 1, 64, 101) log-mel spectrogram.
  Patch: 16×16 with stride 16 (non-overlapping).
  Frequency patches: 64/16 = 4; Time patches: ceil(101/16)=7 (zero-padded to 112).
  Total patches: 4×7 = 28.
  embed_dim=192, depth=12, num_heads=3.
  Base model is randomly initialised (no pretrained AST weights available
  offline; see PAPER_NOTES.md).

LoRA applied to:
  The top `lora_blocks` transformer block attention q/k/v projections and
  MLP fc1, following the same approach as lora_vit.py.
  Trainable < 2 000 000 (asserted at construction).

Deviation from Gong 2021:
  - Gong et al. use a DeiT-Base backbone pretrained on ImageNet, then
    fine-tuned.  We implement the transformer architecture identically but
    initialize from scratch (no pretrained weights).  This is documented as
    a limitation in PAPER_NOTES.md.
  - Gong et al. use 16×16 overlapping patches (stride 10×10).  We use
    non-overlapping 16×16 patches for simplicity; this reduces the sequence
    length from ~248 to 28 tokens.
"""
from __future__ import annotations

import math
from typing import Set

import torch
import torch.nn as nn

from fte_fl.models.base import FTEModel, count_parameters
from fte_fl.models.vit_tiny import TransformerBlock, MLP, Attention
from fte_fl.models.lora import LoRALinear, inject_lora

_LORA_TARGETS: Set[str] = {"qkv", "fc1"}


class SpectrogramPatchEmbed(nn.Module):
    """Project log-mel spectrogram patches into an embedding space.

    Pads the time axis to be divisible by patch_size, then applies a
    strided Conv2d as the patch projection (standard PatchEmbed).

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
        # Pad time to next multiple of patch_size
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
        # x: (B, 1, F, T)
        if self.pad_t > 0:
            x = nn.functional.pad(x, (0, self.pad_t))  # pad T dim
        x = self.proj(x)                 # (B, D, n_freq, n_time)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class ASTBase(nn.Module):
    """Audio Spectrogram Transformer backbone (from scratch).

    Implements the ViT-style architecture of Gong et al. (2021) for
    log-mel spectrogram inputs, with CLS token and learnable positional
    embeddings.
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
    """Audio Spectrogram Transformer with LoRA adapters for Speech Commands.

    The base AST is built from scratch (see module docstring for rationale).
    LoRA is applied to the top `lora_blocks` blocks' attention and MLP layers.
    All base parameters outside the LoRA targets are frozen after construction.
    The head is kept trainable.

    Trainable parameter count is asserted < 2 000 000 at construction.
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
        lora_blocks: int = 6,  # apply LoRA to the top lora_blocks blocks
    ) -> None:
        super().__init__()
        self.base = ASTBase(
            freq_bins=freq_bins,
            time_frames=time_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
        )
        self.pretrained_loaded = False  # No pretrained AST weights available

        # Freeze ALL base parameters
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Apply LoRA to the top `lora_blocks` transformer blocks only
        top_blocks = self.base.blocks[-lora_blocks:]
        for blk in top_blocks:
            inject_lora(blk, target_module_names=_LORA_TARGETS, r=r, alpha=alpha)

        # Unfreeze classification head
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
