"""Model-update (Byzantine) attacks.

Every function here operates on flat model-update vectors: 1-D floating
torch tensors such as the output of `FTEModel.trainable_vector()`.  Inputs are
never modified; each function returns a new tensor with the input's dtype and
device.  Empty vectors, non-floating dtypes, non-1-D shapes and non-finite
values raise.

None of these functions selects malicious clients, schedules attacks, or sees
other clients' updates; that is the job of the future simulator (Milestone 5).

References:
  ALIE: Baruch, Baruch and Goldberg, "A Little Is Enough: Circumventing
  Defenses For Distributed Learning", NeurIPS 2019 (arXiv:1902.06156).
"""
from __future__ import annotations

import math
import numbers

import torch
from scipy.stats import norm

_FLOAT_DTYPES = (torch.float32, torch.float64)
_MAX_SEED = 2**64 - 1


def _check_update(update: torch.Tensor, name: str = "update") -> None:
    """Raise unless `update` is a non-empty, finite, 1-D float32/float64 tensor."""
    if not isinstance(update, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(update).__name__}")
    if update.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must be float32 or float64, got {update.dtype}")
    if update.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(update.shape)}")
    if update.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(update).all():
        raise ValueError(f"{name} contains NaN or infinity")


def _check_real(value, name: str) -> float:
    """Return `value` as a float, rejecting bools and non-finite numbers."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _check_output(out: torch.Tensor, attack: str) -> torch.Tensor:
    if not torch.isfinite(out).all():
        raise ValueError(f"{attack} produced non-finite values (overflow)")
    return out


def sign_flip(update: torch.Tensor) -> torch.Tensor:
    """Sign-flip attack: return `-update`."""
    _check_update(update)
    return torch.neg(update)


def scaling(update: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Scaling attack: return `gamma * update`.

    `gamma` must be finite and > 0 (gamma = 0 is the zero attack and gamma < 0
    combines scaling with a sign flip; use those attacks explicitly instead).
    Raises if the scaled vector overflows the dtype.
    """
    _check_update(update)
    gamma = _check_real(gamma, "gamma")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return _check_output(update * gamma, "scaling")


def zero_update(update: torch.Tensor) -> torch.Tensor:
    """Zero-gradient attack: an all-zero vector with the input's shape, dtype and device."""
    _check_update(update)
    return torch.zeros_like(update)


def gaussian_noise(update: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    """Random-noise attack: REPLACE the update with i.i.d. N(0, std^2) noise.

    The honest update's values are not used (only its length, dtype and device);
    the output is not `update + noise`.  Noise is drawn on the CPU from a
    `torch.Generator` seeded with `seed`, so it does not read or advance the
    global RNG and is identical across devices for the same seed and dtype.
    There is no default `std`: no project-wide noise scale has been justified.
    `seed` must be an int (TypeError otherwise) in [0, 2**64 - 1], the range
    `torch.Generator.manual_seed` accepts for non-negative seeds (ValueError
    otherwise).
    """
    _check_update(update)
    std = _check_real(std, "std")
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must lie in [0, 2**64 - 1], got {seed}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(update.numel(), generator=gen, dtype=update.dtype) * std
    return _check_output(noise.to(update.device), "gaussian_noise")


def alie(mu: torch.Tensor, sigma: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """ALIE attack: return `mu - z * sigma` coordinate-wise.

    `mu` and `sigma` are the attacker's own coordinate-wise estimates of the
    mean and standard deviation of the update distribution; this function
    sees nothing else.  Where `sigma` is 0 the output equals `mu` exactly.

    This is line 5 of Algorithm 3 in the NeurIPS 2019 proceedings,
    (p_mal)_j = mu_j - z_max * sigma_j, with `z` in place of z_max.  A
    negative `z` moves the output in the opposite direction (mu + |z| sigma).
    `z = 1.0` is a configurable project default, not the paper's setting: the
    paper computes z_max from n and m (see `alie_z_max`).
    """
    _check_update(mu, "mu")
    _check_update(sigma, "sigma")
    if sigma.shape != mu.shape or sigma.dtype != mu.dtype or sigma.device != mu.device:
        raise ValueError(
            "mu and sigma must share shape, dtype and device; got "
            f"{tuple(mu.shape)}/{mu.dtype}/{mu.device} and "
            f"{tuple(sigma.shape)}/{sigma.dtype}/{sigma.device}"
        )
    if (sigma < 0).any():
        raise ValueError("sigma must be non-negative")
    z = _check_real(z, "z")
    return _check_output(mu - z * sigma, "alie")


def estimate_mean_std(updates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinate-wise mean and SAMPLE standard deviation (Bessel, ddof=1).

    `updates` has shape (k, d) and must contain only the updates the attacker
    legitimately holds (e.g. the colluding malicious clients' own updates for
    the current round).  Nothing outside `updates` is read.  k >= 2 is
    required for the sample standard deviation.
    """
    if not isinstance(updates, torch.Tensor):
        raise TypeError(f"updates must be a torch.Tensor, got {type(updates).__name__}")
    if updates.ndim != 2:
        raise ValueError(f"updates must have shape (k, d), got {tuple(updates.shape)}")
    if updates.shape[0] < 2:
        raise ValueError("sample standard deviation needs at least 2 updates")
    for i in range(updates.shape[0]):
        _check_update(updates[i], f"updates[{i}]")
    return updates.mean(dim=0), updates.std(dim=0, correction=1)


def alie_z_max(n: int, m: int) -> float:
    """z_max from ALIE Algorithm 3 (NeurIPS 2019 proceedings), lines 1-2.

    For n workers of which m are corrupted, s = floor(n/2 + 1) - m is the
    number of non-corrupted "supporters" needed for a majority, and
    z_max = max z such that Phi(z) < (n - m - s)/(n - m).  The returned value
    is the standard-normal quantile Phi^{-1}((n - m - s)/(n - m)), the
    supremum of that set.  Paper example: n=50, m=24 gives s=2, 24/26 and
    z_max ~ 1.43.  (The arXiv rendering uses the different threshold
    (n - s)/n; this helper follows the proceedings.)

    Valid range: n >= 2, 0 <= m < n and 1 <= s <= n - m - 1, so that the
    probability lies strictly in (0, 1) and the quantile is finite.  The
    result is <= 0 when the probability is <= 0.5.
    """
    for name, v in (("n", n), ("m", m)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{name} must be an int, got {type(v).__name__}")
    if n < 2 or not 0 <= m < n:
        raise ValueError(f"need n >= 2 and 0 <= m < n, got n={n}, m={m}")
    s = math.floor(n / 2 + 1) - m
    if not 1 <= s <= n - m - 1:
        raise ValueError(
            f"s = floor(n/2 + 1) - m = {s} must lie in [1, n - m - 1] = [1, {n - m - 1}]"
        )
    return float(norm.ppf((n - m - s) / (n - m)))
