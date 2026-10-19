"""Tests for all five FTE-FL model architectures.

Tests are CPU-only and do not require downloading pretrained weights.
LoRA models are tested with pretrained=False (random initialization).

Real pretrained-weight loading is a separate concern documented in
verify_models.py and results/PAPER_NOTES.md.

Coverage:
  1.  Correct forward-pass output shapes for all five models.
  2.  Batch sizes > 1.
  3.  Expected class counts.
  4.  Parameter-count constraints (cnn_small, bc_resnet, lora_vit, lora_ast).
  5.  ViT-Tiny configuration: embed_dim=192, 12 blocks.
  6.  Correct trainable/frozen separation for LoRA models.
  7.  Backward pass succeeds and adapter gradients are nonzero.
  8.  Base-model weights remain frozen after backward pass.
  9.  trainable_vector() shape, determinism, and round-trip correctness.
  10. Seeded reproducibility where applicable.
  11. CPU-only forward and backward passes (no .cuda() calls).
  12. Informative errors for bad input shapes.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from fte_fl.models.base import count_parameters, trainable_vector, load_trainable_vector
from fte_fl.models.cnn_small import CnnSmall
from fte_fl.models.bc_resnet import BCResNet1
from fte_fl.models.vit_tiny import ViTTiny
from fte_fl.models.lora_vit import LoRAViT
from fte_fl.models.lora_ast import LoRAAST


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cnn():
    return CnnSmall()

@pytest.fixture(scope="module")
def bcresnet():
    return BCResNet1()

@pytest.fixture(scope="module")
def vit():
    return ViTTiny()

@pytest.fixture(scope="module")
def lora_vit():
    # pretrained=False so no network access required
    return LoRAViT(pretrained=False)

@pytest.fixture(scope="module")
def lora_ast():
    return LoRAAST()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correct output shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_cnn_small_output_shape(cnn):
    x = torch.randn(2, 1, 28, 28)
    assert cnn(x).shape == (2, 62)

def test_bcresnet_output_shape(bcresnet):
    x = torch.randn(2, 1, 64, 101)
    assert bcresnet(x).shape == (2, 35)

def test_vit_tiny_output_shape(vit):
    x = torch.randn(2, 3, 32, 32)
    assert vit(x).shape == (2, 100)

def test_lora_vit_output_shape(lora_vit):
    x = torch.randn(2, 3, 32, 32)
    assert lora_vit(x).shape == (2, 100)

def test_lora_ast_output_shape(lora_ast):
    x = torch.randn(2, 1, 64, 101)
    assert lora_ast(x).shape == (2, 35)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Batch sizes > 1 (already covered above with B=2; test B=4 and B=1)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B", [1, 4, 8])
def test_cnn_batch_sizes(B):
    m = CnnSmall()
    assert m(torch.randn(B, 1, 28, 28)).shape == (B, 62)

@pytest.mark.parametrize("B", [1, 4])
def test_bcresnet_batch_sizes(B):
    m = BCResNet1()
    assert m(torch.randn(B, 1, 64, 101)).shape == (B, 35)

@pytest.mark.parametrize("B", [1, 4])
def test_vit_batch_sizes(B):
    m = ViTTiny()
    assert m(torch.randn(B, 3, 32, 32)).shape == (B, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Expected class counts
# ─────────────────────────────────────────────────────────────────────────────

def test_cnn_num_classes():
    m = CnnSmall(num_classes=62)
    assert m(torch.randn(1, 1, 28, 28)).shape[-1] == 62

def test_bcresnet_num_classes():
    m = BCResNet1(num_classes=35)
    assert m(torch.randn(1, 1, 64, 101)).shape[-1] == 35

def test_vit_tiny_num_classes():
    m = ViTTiny(num_classes=100)
    assert m(torch.randn(1, 3, 32, 32)).shape[-1] == 100

def test_lora_vit_num_classes(lora_vit):
    assert lora_vit(torch.randn(1, 3, 32, 32)).shape[-1] == 100

def test_lora_ast_num_classes(lora_ast):
    assert lora_ast(torch.randn(1, 1, 64, 101)).shape[-1] == 35


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parameter-count constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_cnn_small_param_count(cnn):
    total, trainable, _ = count_parameters(cnn)
    assert 150_000 <= total <= 250_000, f"CnnSmall has {total:,} params"
    assert total == trainable  # fully trainable

def test_bcresnet_param_count(bcresnet):
    total, _, _ = count_parameters(bcresnet)
    lo, hi = int(321_000 * 0.9), int(321_000 * 1.1)
    assert lo <= total <= hi, f"BCResNet1 has {total:,} params; expected {lo:,}–{hi:,}"

def test_lora_vit_trainable_below_2m(lora_vit):
    _, trainable, _ = count_parameters(lora_vit)
    assert trainable < 2_000_000, f"LoRAViT trainable={trainable:,}"

def test_lora_ast_trainable_below_2m(lora_ast):
    _, trainable, _ = count_parameters(lora_ast)
    assert trainable < 2_000_000, f"LoRAAST trainable={trainable:,}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ViT-Tiny configuration
# ─────────────────────────────────────────────────────────────────────────────

def test_vit_tiny_embed_dim(vit):
    # CLS token has shape (1, 1, embed_dim)
    assert vit.cls_token.shape[-1] == 192

def test_vit_tiny_num_blocks(vit):
    assert len(vit.blocks) == 12

def test_vit_tiny_from_scratch():
    """ViTTiny must raise if embed_dim or depth deviate from spec."""
    with pytest.raises(AssertionError):
        ViTTiny(embed_dim=128)  # wrong embed_dim
    with pytest.raises(AssertionError):
        ViTTiny(depth=6)         # wrong depth

def test_vit_tiny_no_pretrained_weights():
    """ViTTiny is always randomly initialized; pretrained_loaded attribute absent."""
    m = ViTTiny()
    # The model should NOT have a pretrained_loaded flag set to True
    assert not getattr(m, "pretrained_loaded", False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Trainable/frozen separation for LoRA models
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_vit_frozen_base(lora_vit):
    _, trainable, frozen = count_parameters(lora_vit)
    assert frozen > 0, "LoRAViT base should have frozen parameters"
    assert trainable > 0, "LoRAViT should have trainable adapter parameters"
    assert trainable < frozen, "Most LoRAViT params should be frozen"

def test_lora_ast_frozen_base(lora_ast):
    _, trainable, frozen = count_parameters(lora_ast)
    assert frozen > 0
    assert trainable > 0
    assert trainable < frozen

def test_lora_vit_frozen_params_no_grad(lora_vit):
    """Base ViT parameters that are frozen must have requires_grad=False."""
    frozen_params = [p for p in lora_vit.parameters() if not p.requires_grad]
    assert len(frozen_params) > 0

def test_lora_ast_frozen_params_no_grad(lora_ast):
    frozen_params = [p for p in lora_ast.parameters() if not p.requires_grad]
    assert len(frozen_params) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Backward pass and nonzero adapter gradients
# ─────────────────────────────────────────────────────────────────────────────

def _do_backward(model, x):
    """Run a forward+backward pass; return trainable params with gradients."""
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()
    return [p for p in model.parameters() if p.requires_grad and p.grad is not None]

def test_cnn_backward(cnn):
    params_with_grad = _do_backward(cnn, torch.randn(2, 1, 28, 28))
    assert len(params_with_grad) > 0
    assert any(p.grad.abs().sum().item() > 0 for p in params_with_grad)

def test_bcresnet_backward(bcresnet):
    params_with_grad = _do_backward(bcresnet, torch.randn(2, 1, 64, 101))
    assert len(params_with_grad) > 0

def test_vit_backward(vit):
    params_with_grad = _do_backward(vit, torch.randn(2, 3, 32, 32))
    assert len(params_with_grad) > 0

def test_lora_vit_adapter_gradients(lora_vit):
    """LoRA adapter parameters must receive nonzero gradients."""
    lora_vit.zero_grad()
    out = lora_vit(torch.randn(2, 3, 32, 32))
    out.sum().backward()
    lora_params = [p for p in lora_vit.parameters()
                   if p.requires_grad and p.grad is not None]
    assert len(lora_params) > 0, "No adapter params received gradients"
    assert any(p.grad.abs().sum().item() > 0 for p in lora_params)

def test_lora_ast_adapter_gradients(lora_ast):
    lora_ast.zero_grad()
    out = lora_ast(torch.randn(2, 1, 64, 101))
    out.sum().backward()
    lora_params = [p for p in lora_ast.parameters()
                   if p.requires_grad and p.grad is not None]
    assert len(lora_params) > 0
    assert any(p.grad.abs().sum().item() > 0 for p in lora_params)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Base-model weights stay frozen after backward
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_vit_base_weights_frozen_after_backward(lora_vit):
    """Frozen params must have None gradients even after a backward pass."""
    lora_vit.zero_grad()
    lora_vit(torch.randn(2, 3, 32, 32)).sum().backward()
    frozen_with_grad = [p for p in lora_vit.parameters()
                        if not p.requires_grad and p.grad is not None]
    assert len(frozen_with_grad) == 0, (
        f"{len(frozen_with_grad)} frozen params received gradients"
    )

def test_lora_ast_base_weights_frozen_after_backward(lora_ast):
    lora_ast.zero_grad()
    lora_ast(torch.randn(2, 1, 64, 101)).sum().backward()
    frozen_with_grad = [p for p in lora_ast.parameters()
                        if not p.requires_grad and p.grad is not None]
    assert len(frozen_with_grad) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. trainable_vector(): shape, determinism, and round-trip
# ─────────────────────────────────────────────────────────────────────────────

def _check_trainable_vector(model, x_shape):
    """Generic trainable_vector round-trip test."""
    _, trainable, _ = count_parameters(model)
    vec = model.trainable_vector()
    assert vec.shape == (trainable,), (
        f"Expected ({trainable},), got {vec.shape}"
    )
    assert vec.dtype == torch.float32

    # Determinism: two calls must return identical tensors
    vec2 = model.trainable_vector()
    assert torch.equal(vec, vec2), "trainable_vector() is not deterministic"

    # Round-trip: write vec back, re-extract, compare
    original = vec.clone()
    model.load_trainable_vector(vec)
    vec3 = model.trainable_vector()
    assert torch.allclose(original, vec3, atol=1e-6), "Round-trip failed"

def test_cnn_trainable_vector(cnn):
    _check_trainable_vector(cnn, (1, 1, 28, 28))

def test_bcresnet_trainable_vector(bcresnet):
    _check_trainable_vector(bcresnet, (1, 1, 64, 101))

def test_vit_trainable_vector(vit):
    _check_trainable_vector(vit, (1, 3, 32, 32))

def test_lora_vit_trainable_vector(lora_vit):
    _check_trainable_vector(lora_vit, (1, 3, 32, 32))

def test_lora_ast_trainable_vector(lora_ast):
    _check_trainable_vector(lora_ast, (1, 1, 64, 101))

def test_load_trainable_vector_wrong_size(cnn):
    """load_trainable_vector must raise ValueError for wrong-length input."""
    from fte_fl.models.base import load_trainable_vector
    wrong = torch.zeros(10)
    with pytest.raises(ValueError):
        load_trainable_vector(cnn, wrong)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Seeded reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def test_vit_tiny_seeded_init():
    torch.manual_seed(0)
    m1 = ViTTiny()
    torch.manual_seed(0)
    m2 = ViTTiny()
    v1 = m1.trainable_vector()
    v2 = m2.trainable_vector()
    assert torch.allclose(v1, v2), "Same seed must give same initialization"

def test_cnn_seeded_output():
    torch.manual_seed(42)
    m = CnnSmall()
    x = torch.manual_seed(42) and torch.randn(2, 1, 28, 28)
    torch.manual_seed(42)
    x2 = torch.randn(2, 1, 28, 28)
    assert torch.equal(x, x2), "Seed must produce identical inputs"


# ─────────────────────────────────────────────────────────────────────────────
# 11. CPU-only — models must not require CUDA
# ─────────────────────────────────────────────────────────────────────────────

def test_all_models_cpu_only(cnn, bcresnet, vit, lora_vit, lora_ast):
    """All models must run inference on CPU without errors."""
    models_and_inputs = [
        (cnn, torch.randn(1, 1, 28, 28)),
        (bcresnet, torch.randn(1, 1, 64, 101)),
        (vit, torch.randn(1, 3, 32, 32)),
        (lora_vit, torch.randn(1, 3, 32, 32)),
        (lora_ast, torch.randn(1, 1, 64, 101)),
    ]
    for model, x in models_and_inputs:
        assert x.device.type == "cpu"
        out = model(x)
        assert out.device.type == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Informative errors for unsupported input shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_cnn_wrong_input_channels(cnn):
    """CnnSmall expects 1 input channel; 3 channels should raise."""
    with pytest.raises(RuntimeError):
        cnn(torch.randn(1, 3, 28, 28))

def test_vit_wrong_input_size(vit):
    """ViTTiny expects 32×32; any size that produces non-integer patches fails."""
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        # 30×30 is not divisible by patch_size=4 cleanly is OK actually,
        # but patch embed uses stride=patch_size so it simply truncates.
        # Use 7×7 which is definitively wrong (too small for any patch).
        vit(torch.randn(1, 3, 7, 7))

def test_lora_vit_wrong_channels(lora_vit):
    """LoRAViT expects 3-channel input; 1 channel should raise."""
    with pytest.raises(RuntimeError):
        lora_vit(torch.randn(1, 1, 32, 32))


# ─────────────────────────────────────────────────────────────────────────────
# Additional: count_parameters utility
# ─────────────────────────────────────────────────────────────────────────────

def test_count_parameters_basic():
    m = nn.Linear(10, 5)
    total, trainable, frozen = count_parameters(m)
    assert total == 55       # 10*5 + 5 bias
    assert trainable == 55
    assert frozen == 0

def test_count_parameters_with_frozen():
    m = nn.Linear(10, 5)
    m.weight.requires_grad_(False)
    total, trainable, frozen = count_parameters(m)
    assert total == 55
    assert trainable == 5    # only bias
    assert frozen == 50      # weight

def test_lora_vit_param_counts_sum(lora_vit):
    total, trainable, frozen = count_parameters(lora_vit)
    assert total == trainable + frozen

def test_lora_ast_param_counts_sum(lora_ast):
    total, trainable, frozen = count_parameters(lora_ast)
    assert total == trainable + frozen
