"""Krum and Multi-Krum gradient selection.

Krum and Multi-Krum: Blanchard, El Mhamdi, Guerraoui and Stainer, "Machine
Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.
Checked against the proceedings PDF (Sec. 4 and Sec. 6):
  - score s(i) = sum of the SQUARED Euclidean distances ||V_i - V_j||^2 to
    the n - f - 2 closest other vectors; Krum selects the minimum-score
    vector, ties broken by the smallest index (paper footnote 8);
  - requires 2f + 2 < n;
  - Multi-Krum selects the m in {1, ..., n} best-scoring vectors in one pass
    and outputs their average (m = 1 is Krum, m = n is averaging); the
    paper's experiments use m = n - f, which is the default here.
"""
from __future__ import annotations

import numpy as np


def _check_f(f: int) -> None:
    if isinstance(f, bool) or not isinstance(f, (int, np.integer)):
        raise TypeError(f"f must be an int, got {type(f).__name__}")
    if f < 0:
        raise ValueError(f"f must be >= 0, got {f}")


def _check_n_f(n: int, f: int) -> None:
    _check_f(f)
    if n <= 2 * f + 2:
        raise ValueError(f"Krum requires n > 2*f + 2, got n={n}, f={f}")


def _validate(grads: np.ndarray, f: int) -> np.ndarray:
    grads = np.asarray(grads, dtype=np.float64)
    if grads.ndim != 2:
        raise ValueError("grads must have shape (n_clients, n_parameters)")
    _check_n_f(grads.shape[0], f)
    if not np.all(np.isfinite(grads)):
        raise ValueError("Gradients must be finite")
    return grads


def pairwise_squared_distances(grads: np.ndarray) -> np.ndarray:
    """(n, n) matrix of squared Euclidean distances between rows of `grads`.

    Computed as |a|^2 + |b|^2 - 2 a.b and clipped at 0; the diagonal is 0.
    `grads` must already be a validated float64 (n, d) array.
    """
    norms = np.sum(grads * grads, axis=1)
    distances = norms[:, None] + norms[None, :] - 2 * grads @ grads.T
    np.maximum(distances, 0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def scores_from_distances(distances: np.ndarray, neighbor_count: int) -> np.ndarray:
    """Sum of each row's `neighbor_count` smallest off-diagonal distances."""
    distances = np.array(distances, dtype=np.float64)
    n = distances.shape[0]
    if distances.shape != (n, n) or not np.all(np.isfinite(distances)):
        raise ValueError("distances must be a finite square matrix")
    if not 1 <= neighbor_count <= n - 1:
        raise ValueError(f"neighbor_count must lie in [1, {n - 1}], got {neighbor_count}")
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
    return np.sum(nearest, axis=1)


def krum_scores_from_distances(distances: np.ndarray, f: int) -> np.ndarray:
    """Krum scores from an (n, n) squared-distance matrix (n > 2f + 2)."""
    n = np.shape(distances)[0]
    _check_n_f(n, f)
    return scores_from_distances(distances, n - f - 2)


def krum_from_distances(distances: np.ndarray, f: int) -> tuple[int, np.ndarray]:
    """Return (selected index, all scores); ties go to the smallest index."""
    scores = krum_scores_from_distances(distances, f)
    return int(np.argmin(scores)), scores


def multi_krum_from_distances(
    distances: np.ndarray, f: int, m: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the m best-scoring indices (ties: smaller index) and all scores.

    By default m = n - f, the setting used in the NeurIPS 2017 experiments.
    """
    scores = krum_scores_from_distances(distances, f)
    n = len(scores)
    if m is None:
        m = n - f
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise TypeError(f"m must be an int, got {type(m).__name__}")
    if not 1 <= m <= n:
        raise ValueError(f"m must be between 1 and {n}, got {m}")
    return np.argsort(scores, kind="stable")[:m], scores


def krum_scores(grads: np.ndarray, f: int) -> np.ndarray:
    """Return each client's sum of squared distances to its n-f-2 nearest neighbours."""
    grads = _validate(grads, f)
    return krum_scores_from_distances(pairwise_squared_distances(grads), f)


def krum(grads: np.ndarray, f: int) -> tuple[int, np.ndarray]:
    """Return (selected client index, all client scores)."""
    grads = _validate(grads, f)
    return krum_from_distances(pairwise_squared_distances(grads), f)


def multi_krum(
    grads: np.ndarray, f: int, m: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the m best-scoring client indices and all scores (default m = n - f)."""
    grads = _validate(grads, f)
    return multi_krum_from_distances(pairwise_squared_distances(grads), f, m)
