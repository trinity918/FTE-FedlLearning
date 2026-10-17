"""Krum and Multi-Krum gradient selection.

Krum: Blanchard et al., "Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent", NeurIPS 2017.
Multi-Krum: Mhamdi et al., "The Hidden Vulnerability of Distributed
Learning in Byzantium", ICDCS 2018 (also the basis for Bulyan).
"""
from __future__ import annotations

import numpy as np


def _validate(grads: np.ndarray, f: int) -> np.ndarray:
    grads = np.asarray(grads, dtype=np.float64)
    if grads.ndim != 2:
        raise ValueError("grads must have shape (n_clients, n_parameters)")
    n = grads.shape[0]
    if not 0 <= f or n <= 2 * f + 2:
        raise ValueError("Krum requires n > 2*f + 2")
    if not np.all(np.isfinite(grads)):
        raise ValueError("Gradients must be finite")
    return grads


def krum_scores(grads: np.ndarray, f: int) -> np.ndarray:
    """Return each client's sum of its n-f-2 nearest squared distances."""
    grads = _validate(grads, f)
    n = len(grads)
    neighbor_count = n - f - 2

    # Squared Euclidean distance matrix.
    norms = np.sum(grads * grads, axis=1)
    distances = norms[:, None] + norms[None, :] - 2 * grads @ grads.T
    np.maximum(distances, 0, out=distances)
    np.fill_diagonal(distances, np.inf)

    nearest = np.partition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
    return np.sum(nearest, axis=1)


def krum(grads: np.ndarray, f: int) -> tuple[int, np.ndarray]:
    """Return (selected client index, all client scores)."""
    scores = krum_scores(grads, f)
    return int(np.argmin(scores)), scores


def multi_krum(
    grads: np.ndarray, f: int, m: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected client indices and all client scores.

    Multi-Krum selects the m clients with the lowest Krum scores
    (Mhamdi et al., ICDCS 2018). By default, selects n-f-2 clients.
    """
    grads = _validate(grads, f)
    scores = krum_scores(grads, f)
    max_selected = len(grads) - f - 2
    if m is None:
        m = max_selected
    if not 1 <= m <= max_selected:
        raise ValueError(f"m must be between 1 and {max_selected}")

    selected = np.argsort(scores, kind="stable")[:m]
    return selected, scores
