"""Small CNN for FEMNIST handwriting classification.

Architecture: two convolutional blocks + two fully-connected layers.
Input:  (B, 1, 28, 28) — grayscale 28×28 images (FEMNIST).
Output: (B, 62)        — logits for 62 character classes.

Target parameter range: 150 000 – 200 000 total parameters.
An assertion enforces total params < 250 000 at construction time.

References:
  McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data", AISTATS 2017 — standard small CNN for MNIST/FEMNIST FL.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fte_fl.models.base import FTEModel


class CnnSmall(FTEModel):
    """Two-block convolutional network for FEMNIST (62 classes).

    Block 1: Conv(1→32, 5×5) + ReLU + MaxPool(2)  → (B, 32, 12, 12)
    Block 2: Conv(32→64, 5×5) + ReLU + MaxPool(2)  → (B, 64,  4,  4)
    FC 1:    64×4×4=1024 → 512 + ReLU + Dropout(0.5)
    FC 2:    512 → 62  (logits)

    Parameter count: 32×26×26 ≈ 21 k  + 64×32×5×5 ≈ 51 k  + FC 524 k — too large.
    Actual layout uses 3×3 kernels to stay in 150–200 k range.

    Conv1: 1→32,  k=3, p=1  →  32×(28×28) = 25 088 after ReLU; pool → 32×14×14
    Conv2: 32→64, k=3, p=1  →  64×(14×14) = 12 544 after ReLU; pool → 64×7×7 = 3136
    FC1:  3136 → 512
    FC2:  512  → 62
    Total = (32×9+32) + (64×32×9+64) + (3136×512+512) + (512×62+62)
          = 320 + 18 496 + 1 606 144 + 31 806  -- still too large.
    Reduce FC1 to 128:
    (32×9+32) + (64×288+64) + (3136×128+128) + (128×62+62)
    = 320 + 18 496 + 401 536 + 7 998 -- too large.

    Use smaller conv channels and pooled representation:
    Conv1: 1→16,  k=3, p=1 → 16×14×14=3136 after pool
    Conv2: 16→32, k=3, p=1 → 32×7×7=1568 after pool
    FC1: 1568→128
    FC2: 128→62
    Total = (16×9+16) + (32×16×9+32) + (1568×128+128) + (128×62+62)
          = 160 + 4 640 + 200 832 + 7 998 = ~213 630  -- slightly over.

    Tune: FC1: 1568→96
    Total = 160 + 4640 + (1568×96+96) + (96×62+62)
          = 160 + 4640 + 150 624 + 6014 = 161 438  ✓ in [150k, 200k]
    """

    NUM_CLASSES: int = 62

    def __init__(self, num_classes: int = 62, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B,1,28,28) → (B,16,14,14)
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: (B,16,14,14) → (B,32,7,7)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        assert total < 250_000, (
            f"CnnSmall has {total:,} parameters; must be < 250 000"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) → logits (B, num_classes)."""
        return self.classifier(self.features(x))
