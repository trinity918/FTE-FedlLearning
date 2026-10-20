"""Label-poisoning attacks on a malicious client's local training labels.

These functions transform label arrays only.  They never touch features,
model updates, or evaluation data; the caller (the future simulator) is
responsible for applying them only to a malicious client's local training
set.  Inputs are not modified; a new array/tensor of the same type, dtype
and shape is returned.

Accepted labels: a 0-D or 1-D integer `torch.Tensor` or `numpy.ndarray`
(bool dtypes rejected), with every value in [0, num_classes).
"""
from __future__ import annotations

import numpy as np
import torch


def _check_class(value: int, num_classes: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= value < num_classes:
        raise ValueError(f"{name}={value} is outside [0, {num_classes})")


def _check_labels(labels, num_classes: int) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"num_classes must be an int >= 2, got {num_classes!r}")
    if isinstance(labels, torch.Tensor):
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"labels must have an integer dtype, got {labels.dtype}")
    elif isinstance(labels, np.ndarray):
        if labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(f"labels must have an integer dtype, got {labels.dtype}")
    else:
        raise TypeError(
            f"labels must be a torch.Tensor or numpy.ndarray, got {type(labels).__name__}"
        )
    if labels.ndim > 1:
        raise ValueError(f"labels must be 0-D or 1-D, got shape {tuple(labels.shape)}")
    size = labels.numel() if isinstance(labels, torch.Tensor) else labels.size
    if size == 0:
        raise ValueError("labels must not be empty")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError(f"labels must lie in [0, {num_classes})")


def _copy(labels):
    return labels.clone() if isinstance(labels, torch.Tensor) else labels.copy()


def flip_labels(labels, source_class: int, target_class: int, num_classes: int):
    """Targeted label flip: every `source_class` label becomes `target_class`.

    All other labels are returned unchanged.  This is a single
    source -> target mapping, not a permutation of all classes.
    """
    _check_labels(labels, num_classes)
    _check_class(source_class, num_classes, "source_class")
    _check_class(target_class, num_classes, "target_class")
    if source_class == target_class:
        raise ValueError("source_class and target_class must differ")
    out = _copy(labels)
    out[labels == source_class] = target_class
    return out


def backdoor_labels(labels, num_classes: int, target_class: int = 0):
    """Backdoor label poisoning: every label becomes `target_class`.

    Use this only for samples that also receive a trigger (see
    `fte_fl.attacks.triggers`); trigger insertion is deliberately separate.
    Which samples to poison is the caller's decision.
    """
    _check_labels(labels, num_classes)
    _check_class(target_class, num_classes, "target_class")
    out = _copy(labels)
    out[...] = target_class
    return out
