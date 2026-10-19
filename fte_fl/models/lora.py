"""Low-Rank Adaptation (LoRA) utilities.

Reference:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).

A LoRA adapter replaces a linear layer W (d_out × d_in) with
    W_new(x) = W(x) + (B @ A)(x) * (alpha / r)
where A ∈ R^{r × d_in}, B ∈ R^{d_out × r} are the trainable adapters,
r is the rank, and alpha/r is a fixed scaling factor.

Only A and B are updated during fine-tuning; W is frozen.
"""
from __future__ import annotations

import math
from typing import Set

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a LoRA adapter.

    The base weight W is stored as a *frozen* nn.Linear.
    The adapters A (r × in_features) and B (out_features × r) are trainable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        assert r > 0
        in_f = base_linear.in_features
        out_f = base_linear.out_features

        # Freeze the base weight
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

        # Kaiming uniform for A (same as default Linear init), zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scale


def inject_lora(
    model: nn.Module,
    target_module_names: Set[str],
    r: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    """Replace named Linear sub-modules with LoRALinear adapters in-place.

    Args:
        model: The model whose Linear layers will be replaced.
        target_module_names: Set of attribute names (relative to their parent)
            that should be wrapped.  E.g. {'q', 'v', 'fc1'}.
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.

    Returns:
        The same model, modified in-place, with frozen base + trainable adapters.
    """
    for name, module in list(model.named_modules()):
        # Get the leaf attribute name (last part after the final '.')
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in target_module_names:
            continue
        if not isinstance(module, nn.Linear):
            continue
        # Navigate to parent and replace
        parent_name, _, attr = name.rpartition(".")
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha))
    return model
