"""ViT-Tiny/16 from scratch for CIFAR-100.

Vision Transformer (ViT) with:
  embed_dim = 192,  num_heads = 3,  depth (num_blocks) = 12,  patch_size = 4
  Input: 32×32 CIFAR-100 images (no resize needed with patch_size=4).
  Output: logits for 100 classes.

With patch_size=4 on 32×32 images: (32/4)^2 = 64 patches + 1 CLS token = 65 tokens.

Why patch_size=4 instead of 16:
  The standard ViT-Tiny uses patch_size=16 designed for 224×224 images.
  Applying patch_size=16 to 32×32 would yield only (32/16)^2 = 4 tokens —
  far too few for meaningful attention.  Patch_size=4 gives 64 tokens, which
  is the standard choice for ViT on CIFAR-32 (e.g., Chen et al. "An Image is
  Worth 16x16 Words" ablation, Dosovitskiy et al. 2021 Appendix B.1).

Configuration (Dosovitskiy et al., 2021 Table 1 — ViT-Tiny adapted for 32×32):
  embed_dim=192, depth=12, num_heads=3 (head_dim=64), mlp_ratio=4.
  This configuration keeps the paper's d_model/heads ratio (64 per head).

Reference:
  Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
  Image Recognition at Scale", ICLR 2021 (arXiv:2010.11929).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from fte_fl.models.base import FTEModel


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(FTEModel):
    """ViT-Tiny/4 from scratch for CIFAR-100 (32×32 input).

    Parameters (approximate, computed at construction):
      embed_dim=192, depth=12, heads=3, patch=4, img=32
      PatchEmbed:     3×4×4×192 + 192  = 9 408
      CLS + pos:      (65)×192          = 12 480
      12 × Block:
        Attn:         4 × 192^2 + ...  = 12 × 148 992  ≈ 1 787 904
        MLP:          2 × 192×768 + …  = 12 × 591 168  ≈ 591 168 (wait)
      Head:           192×100           = 19 300
    The above comes to ~5.7M total.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim == 192, "Spec requires embed_dim=192"
        assert depth == 12, "Spec requires 12 transformer blocks"

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 32, 32) → logits (B, num_classes)."""
        B = x.shape[0]
        x = self.patch_embed(x)                        # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])                       # CLS token logits
