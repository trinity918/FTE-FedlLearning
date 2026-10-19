"""Shared model interface utilities for FTE-FL.

Every model in this project must satisfy:
  - forward(x) -> logits tensor of shape (B, num_classes)
  - trainable_vector() -> 1-D float32 tensor of all trainable parameters,
    flattened in a deterministic, consistent order (torch parameter iteration order).

This module provides:
  - FTEModel: base class mixin that provides trainable_vector() and helpers.
  - count_parameters(): standalone utility for (total, trainable, frozen) counts.
  - load_trainable_vector(): inverse of trainable_vector() for round-trip testing.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return (total, trainable, frozen) parameter counts.

    Counts individual scalar parameters (not tensors).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def trainable_vector(model: nn.Module) -> torch.Tensor:
    """Return a 1-D float32 tensor of all trainable parameters in
    deterministic iteration order (model.parameters() with requires_grad).

    The order is stable across calls as long as the model architecture
    and parameter names do not change between calls.
    """
    parts = [p.detach().cpu().float().view(-1)
             for p in model.parameters() if p.requires_grad]
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def load_trainable_vector(model: nn.Module, vec: torch.Tensor) -> None:
    """Write a flat trainable vector back into the model's trainable parameters.

    This is the inverse of trainable_vector().  Modifies parameters in-place.
    Raises ValueError if vec has the wrong total length.
    """
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if vec.numel() != expected:
        raise ValueError(
            f"Vector length {vec.numel()} does not match trainable params {expected}"
        )
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(vec[offset: offset + n].view(p.shape).to(p.dtype).to(p.device))
            offset += n


def estimated_ciphertext_chunks(trainable_params: int, chunk_size: int = 8192) -> int:
    """Ceiling of trainable_params / chunk_size — used for HE cost estimation."""
    return math.ceil(trainable_params / chunk_size)


class FTEModel(nn.Module):
    """Base mixin providing the standard FTE-FL model interface.

    Subclasses must implement forward().  They inherit trainable_vector(),
    load_trainable_vector(), and count_parameters() automatically.
    """

    def trainable_vector(self) -> torch.Tensor:
        """1-D float32 tensor of all trainable parameters (detached, on CPU)."""
        return trainable_vector(self)

    def load_trainable_vector(self, vec: torch.Tensor) -> None:
        """Restore trainable parameters from a flat vector (round-trip inverse)."""
        load_trainable_vector(self, vec)

    def count_parameters(self) -> tuple[int, int, int]:
        """Return (total, trainable, frozen) scalar parameter counts."""
        return count_parameters(self)
