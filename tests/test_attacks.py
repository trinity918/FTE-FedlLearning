"""Tests for Milestone 3 attacks: update attacks, ALIE, label flipping, DBA-style triggers.

All tests use synthetic tensors; no dataset, download, or training is needed.
Passing tests show the attacks follow their stated contracts.  They say
nothing about attack effectiveness.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

import fte_fl.data.speech_commands as sc
from fte_fl.attacks.labels import backdoor_labels, flip_labels
from fte_fl.attacks.triggers import (
    IMAGE_SHAPES,
    IMAGE_TRIGGER_VALUES,
    LOG_MEL_EPS,
    NUM_TRIGGER_PARTS,
    SPEECH_SHAPE,
    apply_image_trigger,
    apply_speech_trigger,
    image_trigger_mask,
    speech_trigger_magnitude,
)
from fte_fl.attacks.updates import (
    alie,
    alie_z_max,
    estimate_mean_std,
    gaussian_noise,
    scaling,
    sign_flip,
    zero_update,
)

UPDATE_ATTACKS = {
    "sign_flip": sign_flip,
    "scaling": scaling,
    "zero_update": zero_update,
    "gaussian_noise": lambda u: gaussian_noise(u, std=1.0, seed=0),
}


def _vec(dtype=torch.float32, n=17, seed=0):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Update attacks: shared contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", UPDATE_ATTACKS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_update_attack_preserves_shape_dtype_device_and_input(name, dtype):
    u = _vec(dtype)
    before = u.clone()
    out = UPDATE_ATTACKS[name](u)
    assert out.shape == u.shape and out.dtype == dtype and out.device == u.device
    assert out.data_ptr() != u.data_ptr()
    assert torch.equal(u, before)

@pytest.mark.parametrize("name", UPDATE_ATTACKS)
@pytest.mark.parametrize("bad", [
    torch.zeros(0),                                  # empty
    torch.zeros(2, 3),                               # not 1-D
    torch.tensor(1.0),                               # 0-D
    torch.tensor([1.0, float("nan")]),
    torch.tensor([1.0, float("inf")]),
    torch.tensor([1.0, -float("inf")]),
])
def test_update_attack_rejects_invalid_values_and_shapes(name, bad):
    with pytest.raises(ValueError):
        UPDATE_ATTACKS[name](bad)

@pytest.mark.parametrize("name", UPDATE_ATTACKS)
@pytest.mark.parametrize("bad", [
    torch.tensor([1, 2, 3]),                         # integer
    torch.tensor([1.0, 2.0], dtype=torch.float16),
    np.ones(3, dtype=np.float32),                    # not a tensor
])
def test_update_attack_rejects_invalid_types(name, bad):
    with pytest.raises(TypeError):
        UPDATE_ATTACKS[name](bad)


def test_sign_flip_is_exact_negation():
    u = _vec(torch.float64)
    assert torch.equal(sign_flip(u), -u)
    assert torch.equal(sign_flip(u) + u, torch.zeros_like(u))

def test_scaling_default_gamma_is_10():
    assert inspect.signature(scaling).parameters["gamma"].default == 10.0
    u = _vec(torch.float64)
    assert torch.equal(scaling(u), u * 10.0)

@pytest.mark.parametrize("gamma", [0.5, 3, 25.0])
def test_scaling_uses_configured_gamma(gamma):
    u = _vec(torch.float64)
    assert torch.equal(scaling(u, gamma), u * float(gamma))

@pytest.mark.parametrize("gamma", [0.0, -1.0, float("nan"), float("inf")])
def test_scaling_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        scaling(_vec(), gamma)

@pytest.mark.parametrize("gamma", [True, "10", None])
def test_scaling_rejects_non_numeric_gamma(gamma):
    with pytest.raises(TypeError):
        scaling(_vec(), gamma)

def test_scaling_rejects_overflow():
    with pytest.raises(ValueError, match="overflow"):
        scaling(torch.tensor([3e38], dtype=torch.float32), 10.0)

def test_zero_update_returns_zeros():
    u = _vec(torch.float64)
    out = zero_update(u)
    assert torch.count_nonzero(out) == 0 and out.dtype == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian-noise attack (replacement semantics)
# ─────────────────────────────────────────────────────────────────────────────

def test_gaussian_noise_replaces_update():
    """Output depends only on (length, dtype, seed, std), not on update values."""
    a = gaussian_noise(_vec(n=50, seed=1), std=0.5, seed=7)
    b = gaussian_noise(_vec(n=50, seed=2) * 1000, std=0.5, seed=7)
    assert torch.equal(a, b)

def test_gaussian_noise_distribution_contract():
    n, std = 200_000, 2.5
    out = gaussian_noise(torch.zeros(n, dtype=torch.float64), std=std, seed=3)
    assert abs(out.mean().item()) < 5 * std / math.sqrt(n)
    assert abs(out.std().item() / std - 1) < 0.01
    # Gaussian, not e.g. uniform: ~68.3% of mass within one std
    assert abs((out.abs() < std).double().mean().item() - 0.6827) < 0.005

def test_gaussian_noise_reproducible_and_seed_dependent():
    u = _vec(n=64)
    assert torch.equal(gaussian_noise(u, 1.0, seed=11), gaussian_noise(u, 1.0, seed=11))
    assert not torch.equal(gaussian_noise(u, 1.0, seed=11), gaussian_noise(u, 1.0, seed=12))

def test_gaussian_noise_does_not_use_or_advance_global_rng():
    torch.manual_seed(0)
    state = torch.get_rng_state()
    a = gaussian_noise(_vec(n=32), 1.0, seed=5)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(999)
    assert torch.equal(gaussian_noise(_vec(n=32), 1.0, seed=5), a)

def test_gaussian_noise_has_no_default_std():
    assert inspect.signature(gaussian_noise).parameters["std"].default is inspect.Parameter.empty

@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_noise_rejects_invalid_std(std):
    with pytest.raises(ValueError):
        gaussian_noise(_vec(), std=std, seed=0)

@pytest.mark.parametrize("seed", [1.5, None, True, "0", np.int64(3)])
def test_gaussian_noise_rejects_non_int_seed(seed):
    with pytest.raises(TypeError):
        gaussian_noise(_vec(), std=1.0, seed=seed)

@pytest.mark.parametrize("seed", [-1, 2**64])
def test_gaussian_noise_rejects_out_of_range_seed(seed):
    with pytest.raises(ValueError):
        gaussian_noise(_vec(), std=1.0, seed=seed)

def test_gaussian_noise_accepts_seed_range_endpoints():
    for seed in (0, 2**64 - 1):
        out = gaussian_noise(_vec(), std=1.0, seed=seed)
        assert torch.equal(out, gaussian_noise(_vec(), std=1.0, seed=seed))


# ─────────────────────────────────────────────────────────────────────────────
# ALIE
# ─────────────────────────────────────────────────────────────────────────────

def test_alie_exact_output():
    mu = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    sigma = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    assert torch.equal(alie(mu, sigma, z=1.5), torch.tensor([0.25, -3.5, -2.5], dtype=torch.float64))

def test_alie_default_z_is_one():
    assert inspect.signature(alie).parameters["z"].default == 1.0
    mu, sigma = torch.tensor([3.0, 1.0]), torch.tensor([1.0, 0.25])
    assert torch.equal(alie(mu, sigma), torch.tensor([2.0, 0.75]))

def test_alie_positive_z_subtracts_and_negative_z_reverses_direction():
    """Positive z gives mu - z*sigma (NeurIPS Algorithm 3 with z = z_max);
    negative z moves the output the same distance the other way."""
    mu, sigma = torch.tensor([3.0, 1.0]), torch.tensor([1.0, 0.25])
    below = alie(mu, sigma, z=2.0)
    above = alie(mu, sigma, z=-2.0)
    assert torch.equal(below, mu - 2.0 * sigma)
    assert torch.equal(above, mu + 2.0 * sigma)
    assert (below < mu).all() and (above > mu).all()
    assert torch.equal(above - mu, mu - below)

def test_alie_zero_sigma_returns_mu_exactly():
    mu = _vec(torch.float64)
    out = alie(mu, torch.zeros_like(mu), z=3.0)
    assert torch.equal(out, mu)

def test_alie_deterministic_and_non_mutating():
    mu, sigma = _vec(seed=1), _vec(seed=2).abs()
    mu0, s0 = mu.clone(), sigma.clone()
    assert torch.equal(alie(mu, sigma, 0.7), alie(mu, sigma, 0.7))
    assert torch.equal(mu, mu0) and torch.equal(sigma, s0)

def test_alie_signature_takes_only_attacker_statistics():
    """No access to client updates: the only inputs are mu, sigma and z."""
    assert list(inspect.signature(alie).parameters) == ["mu", "sigma", "z"]

@pytest.mark.parametrize("mu, sigma", [
    (torch.zeros(3), torch.ones(4)),                              # shape mismatch
    (torch.zeros(3), torch.ones(3, dtype=torch.float64)),         # dtype mismatch
    (torch.zeros(3), torch.tensor([1.0, -0.1, 1.0])),             # negative sigma
    (torch.tensor([0.0, float("nan"), 0.0]), torch.ones(3)),
    (torch.zeros(3), torch.tensor([1.0, float("inf"), 1.0])),
    (torch.zeros(0), torch.zeros(0)),
])
def test_alie_rejects_invalid_statistics(mu, sigma):
    with pytest.raises(ValueError):
        alie(mu, sigma)

@pytest.mark.parametrize("z, exc", [
    (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError), ("1", TypeError),
])
def test_alie_rejects_invalid_z(z, exc):
    with pytest.raises(exc):
        alie(torch.zeros(3), torch.ones(3), z=z)

def test_estimate_mean_std_uses_only_supplied_rows_with_sample_std():
    gen = torch.Generator().manual_seed(4)
    supplied = torch.randn(5, 8, generator=gen, dtype=torch.float64)
    hidden_honest = torch.randn(20, 8, generator=gen, dtype=torch.float64)
    supplied0 = supplied.clone()

    mu, sigma = estimate_mean_std(supplied)
    ref = supplied.numpy()
    np.testing.assert_allclose(mu.numpy(), ref.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(sigma.numpy(), ref.std(axis=0, ddof=1), rtol=1e-12)
    assert torch.equal(supplied, supplied0)

    # The result is a function of the supplied rows alone: it differs from the
    # statistics of the full population that includes the hidden updates...
    everyone = torch.cat([supplied, hidden_honest])
    assert not torch.allclose(mu, everyone.mean(dim=0))
    # ...and changing any supplied row changes it.
    for i in range(supplied.shape[0]):
        changed = supplied.clone()
        changed[i] += 1.0
        assert not torch.equal(estimate_mean_std(changed)[0], mu)

def test_estimate_mean_std_constant_coordinate_gives_zero_sigma():
    rows = torch.tensor([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    mu, sigma = estimate_mean_std(rows)
    assert sigma[0] == 0
    assert torch.equal(alie(mu, sigma)[0:1], mu[0:1])

@pytest.mark.parametrize("bad", [
    torch.ones(1, 4),                                    # k < 2
    torch.ones(4),                                       # not 2-D
    torch.tensor([[1.0, float("nan")], [0.0, 0.0]]),
    torch.ones(3, 4, dtype=torch.int64),
])
def test_estimate_mean_std_rejects_invalid(bad):
    with pytest.raises((ValueError, TypeError)):
        estimate_mean_std(bad)

def test_alie_z_max_matches_neurips_worked_example():
    """NeurIPS proceedings, Sec. 3.3: n=50, m=24 gives s=2,
    (n-m-s)/(n-m) = 24/26 ~ 0.923 and z_max ~ 1.43 (not the arXiv 1.75)."""
    z = alie_z_max(50, 24)
    assert abs(z - 1.4261) < 1e-3
    assert abs(z - 1.7507) > 0.3

@pytest.mark.parametrize("n, m", [(10, 3), (10, 0), (3, 0), (51, 20), (100, 1)])
def test_alie_z_max_uses_neurips_threshold(n, m):
    s = math.floor(n / 2 + 1) - m
    p = (n - m - s) / (n - m)
    z = alie_z_max(n, m)
    assert abs(0.5 * (1 + math.erf(z / math.sqrt(2))) - p) < 1e-12

def test_alie_z_max_can_be_non_positive():
    # n=10, m=4: s=2, (10-4-2)/6 = 2/3 > 0.5; n=4, m=0: s=3, 1/4 < 0.5.
    assert alie_z_max(10, 4) > 0
    assert alie_z_max(4, 0) < 0

@pytest.mark.parametrize("n, m", [
    (1, 0),          # n < 2
    (10, 10),        # m >= n
    (10, -1),        # m < 0
    (4, 3),          # s = 0: attackers already hold a majority; probability 1
    (2, 0),          # s = 2 > n - m - 1 = 1: probability 0, quantile undefined
    (2, 1),          # s = 1 > n - m - 1 = 0: probability 0, quantile undefined
])
def test_alie_z_max_rejects_degenerate(n, m):
    with pytest.raises(ValueError):
        alie_z_max(n, m)


# ─────────────────────────────────────────────────────────────────────────────
# Label flipping and backdoor label poisoning
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [
    lambda v: torch.tensor(v, dtype=torch.long),
    lambda v: torch.tensor(v, dtype=torch.int32),
    lambda v: np.array(v, dtype=np.int64),
])
def test_flip_labels_changes_only_source_class(make):
    labels = make([0, 3, 5, 3, 9, 1, 3])
    original = make([0, 3, 5, 3, 9, 1, 3])
    out = flip_labels(labels, source_class=3, target_class=7, num_classes=10)
    expected = make([0, 7, 5, 7, 9, 1, 7])
    assert type(out) is type(labels) and out.dtype == labels.dtype
    assert (out == expected).all()
    assert (labels == original).all()       # caller's labels untouched

def test_flip_labels_no_source_present_is_unchanged_copy():
    labels = torch.tensor([1, 2, 4])
    out = flip_labels(labels, 3, 0, num_classes=5)
    assert torch.equal(out, labels) and out.data_ptr() != labels.data_ptr()

def test_flip_labels_scalar_label():
    assert int(flip_labels(torch.tensor(4), 4, 1, num_classes=62)) == 1
    assert int(flip_labels(torch.tensor(2), 4, 1, num_classes=62)) == 2

@pytest.mark.parametrize("src, tgt", [(10, 0), (0, 10), (-1, 0), (2, 2)])
def test_flip_labels_rejects_invalid_classes(src, tgt):
    with pytest.raises(ValueError):
        flip_labels(torch.tensor([0, 1, 2]), src, tgt, num_classes=10)

@pytest.mark.parametrize("bad, exc", [
    (torch.tensor([0.0, 1.0]), TypeError),
    (torch.tensor([True, False]), TypeError),
    (np.array([0.0, 1.0]), TypeError),
    ([0, 1, 2], TypeError),
    (torch.tensor([0, 10]), ValueError),            # label out of range
    (torch.tensor([-1, 0]), ValueError),
    (torch.zeros(2, 2, dtype=torch.long), ValueError),
    (torch.zeros(0, dtype=torch.long), ValueError),
])
def test_flip_labels_rejects_invalid_labels(bad, exc):
    with pytest.raises(exc):
        flip_labels(bad, 0, 1, num_classes=10)

def test_flip_labels_rejects_bool_class():
    with pytest.raises(TypeError):
        flip_labels(torch.tensor([0, 1]), True, 0, num_classes=10)

def test_backdoor_labels_default_target_is_zero():
    assert inspect.signature(backdoor_labels).parameters["target_class"].default == 0
    labels = torch.tensor([5, 9, 0, 34])
    out = backdoor_labels(labels, num_classes=35)
    assert torch.equal(out, torch.zeros(4, dtype=torch.long))
    assert torch.equal(labels, torch.tensor([5, 9, 0, 34]))

def test_backdoor_labels_custom_target_and_validation():
    out = backdoor_labels(np.array([1, 2, 3]), num_classes=100, target_class=42)
    assert out.tolist() == [42, 42, 42]
    with pytest.raises(ValueError):
        backdoor_labels(torch.tensor([1]), num_classes=10, target_class=10)


# ─────────────────────────────────────────────────────────────────────────────
# DBA-style image triggers
# ─────────────────────────────────────────────────────────────────────────────

# Official DBA configs (github.com/AI-secure/DBA, utils/*_params.yaml).
DBA_CIFAR = [[(0, c) for c in range(0, 6)], [(0, c) for c in range(9, 15)],
             [(4, c) for c in range(0, 6)], [(4, c) for c in range(9, 15)]]
DBA_MNIST = [[(0, c) for c in range(0, 4)], [(0, c) for c in range(6, 10)],
             [(3, c) for c in range(0, 4)], [(3, c) for c in range(6, 10)]]

@pytest.mark.parametrize("family, pattern", [("cifar100", DBA_CIFAR), ("femnist", DBA_MNIST)])
def test_image_trigger_parts_match_dba_coordinates(family, pattern):
    for i, pixels in enumerate(pattern):
        mask = image_trigger_mask(family, [i])
        assert sorted(map(tuple, torch.nonzero(mask).tolist())) == sorted(pixels)

@pytest.mark.parametrize("family", ["cifar100", "femnist"])
def test_image_trigger_parts_disjoint_and_union_is_global(family):
    masks = [image_trigger_mask(family, [i]) for i in range(NUM_TRIGGER_PARTS)]
    assert int(sum(m.int() for m in masks).max()) == 1
    assert torch.equal(torch.stack(masks).any(dim=0), image_trigger_mask(family))

@pytest.mark.parametrize("family", ["cifar100", "femnist"])
@pytest.mark.parametrize("batch", [False, True])
def test_image_trigger_localized_shape_bounds_and_non_mutating(family, batch):
    shape = IMAGE_SHAPES[family]
    x = torch.rand((5, *shape) if batch else shape, generator=torch.Generator().manual_seed(0))
    x0 = x.clone()
    out = apply_image_trigger(x, family)
    mask = image_trigger_mask(family)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(x, x0)
    assert (out[..., mask] == IMAGE_TRIGGER_VALUES[family]).all()    # every channel
    assert torch.equal(out[..., ~mask], x[..., ~mask])               # nothing else
    assert out.min() >= 0 and out.max() <= 1

@pytest.mark.parametrize("family, background", [("cifar100", 0.0), ("femnist", 1.0)])
def test_image_trigger_contrasts_with_background(family, background):
    """FEMNIST background is white (1.0), so its trigger is black; CIFAR's is white."""
    x = torch.full(IMAGE_SHAPES[family], background)
    out = apply_image_trigger(x, family)
    mask = image_trigger_mask(family)
    assert (out[..., mask] != background).all()

@pytest.mark.parametrize("family", ["cifar100", "femnist"])
def test_image_trigger_repeatable_and_parts_compose(family):
    x = torch.rand(IMAGE_SHAPES[family], generator=torch.Generator().manual_seed(1))
    assert torch.equal(apply_image_trigger(x, family, [2]), apply_image_trigger(x.clone(), family, [2]))
    seq = x
    for i in range(NUM_TRIGGER_PARTS):
        seq = apply_image_trigger(seq, family, [i])
    assert torch.equal(seq, apply_image_trigger(x, family))
    assert torch.equal(apply_image_trigger(apply_image_trigger(x, family, [0, 1]), family, [2, 3]),
                       apply_image_trigger(x, family))

@pytest.mark.parametrize("x, family, exc", [
    (torch.rand(3, 32, 32) * 2 - 1, "cifar100", ValueError),           # normalized-looking
    (torch.rand(1, 28, 28) + 0.5, "femnist", ValueError),
    (torch.rand(3, 28, 28), "cifar100", ValueError),                   # wrong shape
    (torch.rand(1, 3, 32, 32)[0, 0], "cifar100", ValueError),          # 2-D
    (torch.full((1, 28, 28), float("nan")), "femnist", ValueError),
    (torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8), "cifar100", TypeError),
    (torch.rand(3, 32, 32), "mnist", ValueError),                      # unknown family
])
def test_image_trigger_rejects_invalid_input(x, family, exc):
    with pytest.raises(exc):
        apply_image_trigger(x, family)

@pytest.mark.parametrize("parts", [[], [4], [-1], [0, 0], [True], [1.0]])
def test_trigger_rejects_invalid_parts(parts):
    with pytest.raises(ValueError):
        image_trigger_mask("cifar100", parts)
    with pytest.raises(ValueError):
        speech_trigger_magnitude(parts)

def test_triggers_do_not_take_or_assign_labels():
    """Target labels come only from the separate backdoor_labels operation."""
    for fn in (apply_image_trigger, apply_speech_trigger):
        assert not any("label" in p or "target" in p for p in inspect.signature(fn).parameters)
    assert isinstance(apply_image_trigger(torch.rand(3, 32, 32), "cifar100"), torch.Tensor)


# ─────────────────────────────────────────────────────────────────────────────
# DBA-inspired speech trigger: feature-space 1 kHz tone
# ─────────────────────────────────────────────────────────────────────────────

LOG_FLOOR = math.log(LOG_MEL_EPS)
TONE_MEL_ROWS = [21, 22, 23]     # mel filters covering FFT bins 24-26 (960-1040 Hz)


def _silence():
    return torch.full(SPEECH_SHAPE, LOG_FLOOR, dtype=torch.float32)


def test_speech_trigger_localized_to_1khz_mel_bins_and_part_frames():
    segments = torch.tensor_split(torch.arange(SPEECH_SHAPE[2]), NUM_TRIGGER_PARTS)
    assert [len(s) for s in segments] == [26, 25, 25, 25]
    for i in range(NUM_TRIGGER_PARTS):
        t = speech_trigger_magnitude([i])
        rows, frames = torch.nonzero(t, as_tuple=True)
        assert sorted(set(rows.tolist())) == TONE_MEL_ROWS
        assert sorted(set(frames.tolist())) == segments[i].tolist()
    assert torch.equal((speech_trigger_magnitude() > 0).sum(dim=0),
                       torch.full((SPEECH_SHAPE[2],), 3))

def test_speech_trigger_matches_loader_pipeline_for_a_real_tone(monkeypatch):
    """Applied to silence, the trigger reproduces the loader's log-mel of a 1 kHz tone.

    Compared in the linear mel domain on frames whose STFT window lies fully
    inside the clip (2..98); edge frames are idealized by design.
    """
    amp = 0.1
    n = torch.arange(16000, dtype=torch.float64)
    tone = (amp * torch.cos(2 * math.pi * 1000.0 * n / 16000)).float().numpy()
    monkeypatch.setattr(sc, "load_wav", lambda path: tone)
    loader_tone = sc.wav_to_logmel("synthetic-tone")

    monkeypatch.setattr(sc, "load_wav", lambda path: np.zeros(16000, dtype=np.float32))
    loader_silence = sc.wav_to_logmel("synthetic-silence")
    assert torch.allclose(loader_silence, _silence())

    triggered = apply_speech_trigger(loader_silence, amplitude=amp)
    inner = slice(2, 99)
    lin_loader = torch.exp(loader_tone[..., inner])
    lin_trigger = torch.exp(triggered[..., inner])
    assert torch.allclose(lin_trigger, lin_loader, atol=1e-3)
    # Energy above the floor appears only in the tone rows, for both.
    for lin in (lin_loader, lin_trigger):
        assert sorted(set(torch.nonzero(lin[0] > 1e-3, as_tuple=True)[0].tolist())) == TONE_MEL_ROWS

def test_speech_trigger_is_magnitude_addition_in_log_domain():
    gen = torch.Generator().manual_seed(2)
    x = LOG_FLOOR + 15 * torch.rand(SPEECH_SHAPE, generator=gen, dtype=torch.float64)
    t = speech_trigger_magnitude([1])
    out = apply_speech_trigger(x, [1])
    on = t > 0
    assert torch.allclose(out[0][on], torch.log(torch.exp(x[0][on]) + t[on]), rtol=0, atol=1e-12)
    assert torch.equal(out[0][~on], x[0][~on])          # untouched cells bit-exact
    assert (out[0][on] > x[0][on]).all()                # adding energy only increases

def test_speech_trigger_non_mutating_repeatable_and_batched():
    gen = torch.Generator().manual_seed(3)
    x = LOG_FLOOR + 14 * torch.rand((4, *SPEECH_SHAPE), generator=gen)
    x0 = x.clone()
    a = apply_speech_trigger(x, [0, 2])
    assert torch.equal(x, x0)
    assert torch.equal(a, apply_speech_trigger(x.clone(), [0, 2]))
    assert a.shape == x.shape and a.dtype == x.dtype
    assert torch.equal(a[1], apply_speech_trigger(x[1], [0, 2]))

def test_speech_trigger_parts_compose_to_global():
    gen = torch.Generator().manual_seed(5)
    x = LOG_FLOOR + 14 * torch.rand(SPEECH_SHAPE, generator=gen, dtype=torch.float64)
    seq = x
    for i in range(NUM_TRIGGER_PARTS):
        seq = apply_speech_trigger(seq, [i])
    assert torch.allclose(seq, apply_speech_trigger(x), rtol=0, atol=1e-12)
    total = sum(speech_trigger_magnitude([i]) for i in range(NUM_TRIGGER_PARTS))
    assert torch.equal(total, speech_trigger_magnitude())

def test_speech_trigger_amplitude_scales_magnitude():
    assert torch.allclose(speech_trigger_magnitude(amplitude=0.2), 2 * speech_trigger_magnitude(amplitude=0.1))

@pytest.mark.parametrize("x, exc", [
    (torch.zeros(SPEECH_SHAPE) - 20.0, ValueError),                   # below the log(1e-6) floor
    (torch.zeros(1, 40, 101), ValueError),                            # 40 mel bins
    (torch.zeros(1, 64, 98), ValueError),
    (torch.full(SPEECH_SHAPE, float("inf")), ValueError),
    (torch.zeros(SPEECH_SHAPE, dtype=torch.int64), TypeError),
])
def test_speech_trigger_rejects_invalid_input(x, exc):
    with pytest.raises(exc):
        apply_speech_trigger(x)

@pytest.mark.parametrize("kwargs", [
    {"freq_hz": 1010.0}, {"freq_hz": 40.0}, {"freq_hz": 8000.0}, {"freq_hz": float("nan")},
    {"amplitude": 0.0}, {"amplitude": 1.5}, {"amplitude": float("nan")},
])
def test_speech_trigger_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        speech_trigger_magnitude(**kwargs)
