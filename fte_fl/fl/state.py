"""Model state <-> flat vector convention used by the simulator.

The aggregated vector of a model is, in this order:
  1. every parameter with requires_grad=True, in `named_parameters()` order
     (identical to `FTEModel.trainable_vector()`), then
  2. every floating-point buffer, in `named_buffers()` order (BatchNorm
     running_mean / running_var; only BCResNet1 has any).
Excluded: frozen parameters (requires_grad=False: the LoRA backbones) and
integer buffers (BatchNorm num_batches_tracked).  Excluded tensors are never
aggregated; every client starts from the global model's copy and the global
model keeps its own.  Integer buffers are unused because every BatchNorm
layer must have a momentum (checked; `momentum=None` would make
num_batches_tracked part of the computation and is rejected).

Floating-point buffers are included so that the robust rules filter them
together with the weights: an excluded running statistic would have to be
averaged unfiltered, giving attackers an unfiltered channel.  Consequence:
an attacked running_var can become negative after aggregation; the
simulator then fails with a non-finite-evaluation error instead of
clamping it.

Client update = local vector - global vector, in float64.  The global
vector is always read back from the float32 model after loading
(`set_vector` then `get_vector`), so the vector the simulator aggregates is
exactly the model it evaluates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _entries(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.momentum is None:
            raise ValueError("BatchNorm with momentum=None (cumulative averaging) is not supported")
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    buffers = [(n, b) for n, b in model.named_buffers() if b.is_floating_point()]
    return params + buffers


def vector_names(model: nn.Module) -> list[str]:
    """Names of the tensors in the vector, in vector order."""
    return [n for n, _ in _entries(model)]


def vector_size(model: nn.Module) -> int:
    return sum(t.numel() for _, t in _entries(model))


def get_vector(model: nn.Module) -> np.ndarray:
    """The model's aggregated tensors flattened into one float64 vector."""
    parts = [t.detach().cpu().double().reshape(-1).numpy() for _, t in _entries(model)]
    return np.concatenate(parts) if parts else np.zeros(0)


def set_vector(model: nn.Module, vec: np.ndarray) -> None:
    """Write `vec` into the model's aggregated tensors (cast to each tensor's dtype)."""
    vec = np.asarray(vec)
    entries = _entries(model)
    expected = sum(t.numel() for _, t in entries)
    if vec.ndim != 1 or vec.shape[0] != expected:
        raise ValueError(f"vector has shape {vec.shape}, model expects ({expected},)")
    if not np.all(np.isfinite(vec)):
        raise ValueError("vector contains NaN or infinity")
    offset = 0
    with torch.no_grad():
        for _, t in entries:
            n = t.numel()
            t.copy_(torch.from_numpy(vec[offset:offset + n].copy()).view(t.shape).to(t.dtype))
            offset += n


def copy_excluded(src: nn.Module, dst: nn.Module) -> None:
    """Copy every tensor NOT in the vector (frozen params, integer buffers) from src to dst."""
    included = set(vector_names(src))
    dst_state = dst.state_dict()
    with torch.no_grad():
        for name, value in src.state_dict().items():
            if name not in included:
                dst_state[name].copy_(value)
