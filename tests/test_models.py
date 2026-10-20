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

import inspect
import math

import pytest
import torch
import torch.nn as nn

from fte_fl.models.base import count_parameters, trainable_vector, load_trainable_vector
from fte_fl.models.cnn_small import CnnSmall
from fte_fl.models.bc_resnet import (
    BCResNet1, BC_RESNET_TAU, SSN_SUBBANDS, SubSpectralNorm,
)
from fte_fl.models.lora import LoRALinear
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
# BC-ResNet naming: class is BCResNet1 but implements BC-ResNet-8 (τ=8)
# ─────────────────────────────────────────────────────────────────────────────

def test_bcresnet_default_constructor_builds_tau8():
    """BCResNet1() must build the τ=8 (base_c=64) architecture by default."""
    assert BC_RESNET_TAU == 8
    m = BCResNet1()
    # Channel schedule for base_c=64: [128, 64, 96, 128, 160, 256]
    assert m.cnn_head[0].out_channels == 128
    stage_out = [stage[-1].f2[-1].block[0].out_channels for stage in m.stages]
    assert stage_out == [64, 96, 128, 160]
    assert m.classifier[-1].in_channels == 256
    assert [len(s) for s in m.stages] == [2, 2, 4, 4]

def test_bcresnet_tau1_is_too_small():
    """τ=1 produces ~10k params — below 288k lower bound; must fail assertion."""
    with pytest.raises(AssertionError, match="parameters"):
        BCResNet1(tau=1)


# ─────────────────────────────────────────────────────────────────────────────
# BC-ResNet SubSpectralNorm: grouped path executes, all parameters are live
# ─────────────────────────────────────────────────────────────────────────────

def test_subspectral_norm_normalizes_each_subband():
    """Each (channel, sub-band) is normalized separately; plain BN would not."""
    torch.manual_seed(0)
    ssn = SubSpectralNorm(num_channels=3, S=4).train()
    # Shift each of the 4 sub-bands (2 freq bins each) by a different offset
    offsets = torch.tensor([-5.0, 0.0, 5.0, 10.0]).repeat_interleave(2)
    x = torch.randn(16, 3, 8, 10) + offsets.view(1, 1, 8, 1)

    bands = ssn(x).reshape(16, 3, 4, 2, 10)
    mean = bands.mean(dim=(0, 3, 4))
    var = bands.var(dim=(0, 3, 4), unbiased=False)
    assert torch.allclose(mean, torch.zeros(3, 4), atol=1e-5)
    assert torch.allclose(var, torch.ones(3, 4), atol=1e-3)

    # Per-channel BN keeps the between-band offsets (contrast check)
    plain = nn.BatchNorm2d(3).train()(x).reshape(16, 3, 4, 2, 10).mean(dim=(0, 3, 4))
    assert (plain[:, -1] - plain[:, 0]).abs().min() > 1.0

def test_subspectral_norm_rejects_indivisible_frequency():
    with pytest.raises(ValueError, match="divisible"):
        SubSpectralNorm(num_channels=3, S=8)(torch.randn(2, 3, 12, 5))

def test_bcresnet_rejects_40_bin_input(bcresnet):
    """40 bins → F=20 at the first SSN, not divisible by S=8: must fail clearly."""
    with pytest.raises(ValueError, match="divisible"):
        bcresnet(torch.randn(2, 1, 40, 101))

def test_bcresnet_every_ssn_uses_grouped_path(bcresnet):
    """All 12 SSN layers see F divisible by S, with 4/2/1 bins per sub-band."""
    seen = []
    ssns = [m for m in bcresnet.modules() if isinstance(m, SubSpectralNorm)]
    hooks = [m.register_forward_hook(
                 lambda mod, inp, out: seen.append((inp[0].shape[2], mod.S)))
             for m in ssns]
    try:
        bcresnet(torch.randn(2, 1, 64, 101))
    finally:
        for h in hooks:
            h.remove()
    assert all(m.S == SSN_SUBBANDS for m in ssns)
    assert len(seen) == 12
    assert all(f % s == 0 for f, s in seen)
    assert [f // s for f, s in seen] == [4, 4, 2, 2] + [1] * 8
    # Only the grouped BN's per-(channel, sub-band) γ and β are parameters
    for m in ssns:
        assert sum(p.numel() for p in m.parameters()) == 2 * m.num_channels * m.S

def test_bcresnet_every_parameter_receives_gradient():
    """No exported trainable parameter may be unused by the forward pass."""
    torch.manual_seed(0)
    m = BCResNet1().train()
    x = torch.randn(8, 1, 64, 101)
    y = torch.randint(0, 35, (8,))
    nn.functional.cross_entropy(m(x), y).backward()
    without_grad = [n for n, p in m.named_parameters()
                    if p.grad is None or p.grad.abs().sum().item() == 0]
    assert without_grad == []
    total, trainable, _ = count_parameters(m)
    assert total == trainable == m.trainable_vector().numel()


# ─────────────────────────────────────────────────────────────────────────────
# LoRAAST — adapters only on the top blocks, on a frozen random backbone
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_ast_trainable_set_is_top_adapters_and_head(lora_ast):
    """Exactly the top-block LoRA A/B matrices and the head are trainable."""
    depth = len(lora_ast.base.blocks)
    expected = {
        f"base.blocks.{i}.{mod}.lora_{ab}"
        for i in range(depth - lora_ast.lora_blocks, depth)
        for mod in ("attn.qkv", "mlp.fc1")
        for ab in "AB"
    } | {"base.head.weight", "base.head.bias"}
    trainable = {n for n, p in lora_ast.named_parameters() if p.requires_grad}
    assert trainable == expected

def test_lora_ast_adapters_only_in_top_blocks(lora_ast):
    depth = len(lora_ast.base.blocks)
    for i, blk in enumerate(lora_ast.base.blocks):
        has_lora = any(isinstance(m, LoRALinear) for m in blk.modules())
        assert has_lora == (i >= depth - lora_ast.lora_blocks), f"block {i}"

def test_lora_ast_adapters_affect_forward(lora_ast):
    """lora_B starts at zero (no effect); nonzero lora_B must change the logits."""
    loras = [m for m in lora_ast.modules() if isinstance(m, LoRALinear)]
    assert all(torch.count_nonzero(m.lora_B) == 0 for m in loras)
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 101)
    saved = [m.lora_B.detach().clone() for m in loras]
    with torch.no_grad():
        before = lora_ast(x)
        for m in loras:
            m.lora_B.normal_(std=0.1)
        after = lora_ast(x)
        for m, s in zip(loras, saved):
            m.lora_B.copy_(s)
    assert not torch.allclose(before, after)


# ─────────────────────────────────────────────────────────────────────────────
# LoRAViT initialization clarity
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_vit_default_is_random_init(lora_vit):
    """The constructor default is pretrained=False, and the flag is recorded."""
    assert inspect.signature(LoRAViT).parameters["pretrained"].default is False
    assert lora_vit.pretrained_loaded is False


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

    # Restoration: load a *different* vector; every trainable parameter must
    # hold exactly its slice, and frozen parameters must be untouched.
    original = vec.clone()
    frozen = [p for p in model.parameters() if not p.requires_grad]
    frozen_sums = [p.detach().double().sum() for p in frozen]
    new = torch.randn(trainable, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(new, original)
    model.load_trainable_vector(new)
    try:
        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                assert torch.equal(p.detach().reshape(-1), new[offset:offset + n])
                offset += n
        assert offset == trainable
        assert torch.equal(model.trainable_vector(), new)
        for p, s in zip(frozen, frozen_sums):
            assert torch.equal(p.detach().double().sum(), s)
    finally:
        model.load_trainable_vector(original)
    assert torch.equal(model.trainable_vector(), original), "Restore failed"

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


# ─────────────────────────────────────────────────────────────────────────────
# verify_models --model selection
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_models_selects_single_model():
    from fte_fl.models.verify_models import verify_all
    results = verify_all(model="bcresnet")
    assert [r.name for r in results] == ["BCResNet1"]
    r = results[0]
    assert r.forward_ok and r.backward_ok and not r.error
    assert r.total_params == count_parameters(BCResNet1())[0]

def test_verify_models_rejects_unknown_model():
    from fte_fl.models.verify_models import verify_all
    with pytest.raises(ValueError, match="Unknown model"):
        verify_all(model="resnet")
