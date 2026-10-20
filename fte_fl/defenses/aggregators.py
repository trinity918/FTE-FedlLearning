"""Plaintext aggregation rules: FedAvg, Krum, Multi-Krum, trimmed mean, Bulyan.

Every rule maps a stack of n client updates U of shape (n, d) to a
`Selection`: per-client weights a (n,) and an optional boolean keep-mask
M (n, d), such that the aggregate is

    aggregate = sum_i a_i * (M_i * U_i)          (M = None: all coordinates)

`combine` evaluates this.  The same Selection drives the FTE protocol, in
which client i encrypts a_i * (M_i * U_i) after plaintext filtering.

References (checked 2026-10-20 unless noted):
  FedAvg:       McMahan et al., AISTATS 2017 (weights n_k / n; not re-checked).
  Krum/Multi-Krum: Blanchard et al., NeurIPS 2017 (see `krum.py`).
  Trimmed mean: Yin, Chen, Ramchandran and Bartlett, "Byzantine-Robust
                Distributed Learning: Towards Optimal Statistical Rates",
                ICML 2018, Definition 2.
  Bulyan:       El Mhamdi, Guerraoui and Rouault, "The Hidden Vulnerability of
                Distributed Learning in Byzantium", ICML 2018, Sec. 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fte_fl.defenses.krum import (
    krum,
    multi_krum,
    pairwise_squared_distances,
    scores_from_distances,
)


@dataclass(frozen=True)
class Selection:
    """Output of a filtering rule: aggregate = sum_i weights[i] * (mask[i] * U[i])."""

    weights: np.ndarray
    mask: np.ndarray | None = None

    @property
    def contributors(self) -> np.ndarray:
        """Indices of clients whose contribution can be non-zero."""
        active = self.weights > 0
        if self.mask is not None:
            active &= self.mask.any(axis=1)
        return np.flatnonzero(active)

    def contribution(self, i: int, update: np.ndarray) -> np.ndarray:
        """Client i's filtered contribution weights[i] * (mask[i] * update)."""
        update = np.asarray(update, dtype=np.float64)
        if self.mask is not None:
            update = np.where(self.mask[i], update, 0.0)
        return self.weights[i] * update


def check_updates(updates) -> np.ndarray:
    """Return `updates` as a float64 (n, d) array, n >= 1 and d >= 1, all finite.

    Accepts a real-valued array-like (e.g. a numpy array or CPU tensor);
    bool, complex and object dtypes are rejected.  The input is not modified.
    """
    arr = np.asarray(updates)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
        raise TypeError(f"updates must be a real numeric array, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"updates must have shape (n_clients >= 1, d >= 1), got {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("updates must be finite")
    return arr


def _check_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def combine(updates, selection: Selection) -> np.ndarray:
    """Evaluate sum_i a_i * (M_i * U_i) in plaintext."""
    u = check_updates(updates)
    if selection.weights.shape != (u.shape[0],):
        raise ValueError("selection weights do not match the number of updates")
    if selection.mask is not None:
        if selection.mask.shape != u.shape:
            raise ValueError("selection mask does not match the updates' shape")
        u = np.where(selection.mask, u, 0.0)
    return selection.weights @ u


# ─── FedAvg ──────────────────────────────────────────────────────────────────

def fedavg_selection(updates, weights=None) -> Selection:
    """FedAvg: weights proportional to `weights` (e.g. sample counts n_k).

    `weights` must be finite, non-negative and sum to > 0; default is equal
    weights.  No filtering: every client with a positive weight contributes.
    """
    n = check_updates(updates).shape[0]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights)
        if w.dtype == np.bool_ or not (
            np.issubdtype(w.dtype, np.floating) or np.issubdtype(w.dtype, np.integer)
        ):
            raise TypeError(f"weights must be real numbers, got dtype {w.dtype}")
        w = w.astype(np.float64)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    return Selection(weights=w)


def fedavg(updates, weights=None) -> np.ndarray:
    return combine(updates, fedavg_selection(updates, weights))


# ─── Krum / Multi-Krum (reuse krum.py) ──────────────────────────────────────

def krum_selection(updates, f: int) -> Selection:
    """Krum: the single minimum-score update (weight 1)."""
    u = check_updates(updates)
    idx, _ = krum(u, f)
    w = np.zeros(u.shape[0])
    w[idx] = 1.0
    return Selection(weights=w)


def multi_krum_selection(updates, f: int, m: int | None = None) -> Selection:
    """Multi-Krum: average of the m best-scoring updates (default m = n - f)."""
    u = check_updates(updates)
    selected, _ = multi_krum(u, f, m)
    w = np.zeros(u.shape[0])
    w[selected] = 1.0 / len(selected)
    return Selection(weights=w)


# ─── Coordinate-wise trimmed mean ────────────────────────────────────────────

def trimmed_mean_selection(updates, b: int) -> Selection:
    """Coordinate-wise trimmed mean (Yin et al. 2018, Definition 2).

    For every coordinate independently, sort the n client values, remove the
    b smallest and the b largest (2b values in total), and return the
    arithmetic mean of the remaining n - 2b values (their sum divided by
    n - 2b, the paper's (1 - 2 beta) m).  `b` is a count per side (the
    paper's beta * n); requires an int b >= 0 and n - 2b >= 1 (so n >= 1;
    b = 0 is the plain mean; n - 2b = 1 is the coordinate-wise median for
    odd n).  Robustness needs b >= the number of Byzantine clients (paper:
    beta >= alpha); that is the caller's choice.  Ties are broken by client
    index, which does not change the result.
    """
    u = check_updates(updates)
    n, d = u.shape
    b = _check_int(b, "b", 0)
    if n - 2 * b < 1:
        raise ValueError(f"trimmed mean needs n - 2b >= 1, got n={n}, b={b}")
    order = np.argsort(u, axis=0, kind="stable")
    mask = np.zeros((n, d), dtype=bool)
    np.put_along_axis(mask, order[b : n - b], True, axis=0)
    return Selection(weights=np.full(n, 1.0 / (n - 2 * b)), mask=mask)


def trimmed_mean(updates, b: int) -> np.ndarray:
    return combine(updates, trimmed_mean_selection(updates, b))


# ─── Bulyan ──────────────────────────────────────────────────────────────────

def bulyan_selection(updates, f: int) -> Selection:
    """Bulyan over Krum (El Mhamdi et al. 2018, Sec. 4).

    Supported configuration: f >= 1 and n >= 4f + 3 (otherwise ValueError).
      1. Selection: theta = n - 2f updates are chosen by applying the Krum
         score repeatedly; each chosen update is removed from the remaining
         set R and added to S.  Each step scores R with the same f and
         k = max(|R| - f - 2, 1) nearest neighbours, i.e. Krum's n - f - 2
         with n replaced by |R|.  |R| runs from n down to 2f + 1.  While
         |R| >= 2f + 3 this is exactly Krum.  The paper states a quorum of
         2f + 3 for every use of Krum, but its last two iterations see
         |R| = 2f + 2 and 2f + 1, where Krum's own condition |R| > 2f + 2
         fails; there we keep the same score formula (project choice; Krum's
         guarantee does not apply to those two steps).  The max(., 1) clamp
         is active only for f = 1 at |R| = 3 (k = 0 -> 1); otherwise
         k = |R| - f - 2 >= 1, and k <= |R| - 1 always holds.  Ties go to the
         smallest client index.
      2. Aggregation: beta = theta - 2f = n - 4f >= 3.  Per coordinate, the
         median is a value of S minimising the sum of absolute deviations
         (the lower middle value when theta is even), and the result is the
         mean of the beta values of S closest to it (ties: smaller client
         index).
    """
    u = check_updates(updates)
    n, d = u.shape
    f = _check_int(f, "f", 1)
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan requires n >= 4f + 3, got n={n}, f={f}")
    theta = n - 2 * f
    beta = theta - 2 * f

    distances = pairwise_squared_distances(u)
    remaining = list(range(n))
    selected: list[int] = []
    while len(selected) < theta:
        sub = distances[np.ix_(remaining, remaining)]
        scores = scores_from_distances(sub, max(len(remaining) - f - 2, 1))
        selected.append(remaining.pop(int(np.argmin(scores))))
    selected.sort()

    chosen = u[selected]
    median = np.sort(chosen, axis=0)[(theta - 1) // 2]
    closest = np.argsort(np.abs(chosen - median), axis=0, kind="stable")[:beta]
    sub_mask = np.zeros((theta, d), dtype=bool)
    np.put_along_axis(sub_mask, closest, True, axis=0)

    mask = np.zeros((n, d), dtype=bool)
    mask[selected] = sub_mask
    weights = np.zeros(n)
    weights[selected] = 1.0 / beta
    return Selection(weights=weights, mask=mask)


def bulyan(updates, f: int) -> np.ndarray:
    return combine(updates, bulyan_selection(updates, f))


RULES = {
    "fedavg": fedavg_selection,
    "krum": krum_selection,
    "multi_krum": multi_krum_selection,
    "trimmed_mean": trimmed_mean_selection,
    "bulyan": bulyan_selection,
}
