"""Model registry for the simulator (Milestone 2 models plus a test-only MLP)."""
from __future__ import annotations

import math

import torch.nn as nn

from fte_fl.models.base import FTEModel


class TinyMLP(FTEModel):
    """Two-layer MLP used ONLY with the synthetic dataset in tests and smoke runs."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(input_shape), hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name: str, num_classes: int, input_shape: tuple[int, ...]) -> FTEModel:
    """Construct a model with the Milestone 2 defaults (caller seeds torch first)."""
    if name == "mlp":
        return TinyMLP(input_shape, num_classes)
    if name == "cnn_small":
        from fte_fl.models.cnn_small import CnnSmall
        return CnnSmall(num_classes=num_classes)
    if name == "bc_resnet1":
        from fte_fl.models.bc_resnet import BCResNet1
        return BCResNet1(num_classes=num_classes)
    if name == "lora_ast":
        from fte_fl.models.lora_ast import LoRAAST
        return LoRAAST(num_classes=num_classes)
    if name == "vit_tiny":
        from fte_fl.models.vit_tiny import ViTTiny
        return ViTTiny(num_classes=num_classes)
    if name == "lora_vit":
        from fte_fl.models.lora_vit import LoRAViT
        return LoRAViT(num_classes=num_classes, pretrained=False)
    raise ValueError(f"unknown model {name!r}")
