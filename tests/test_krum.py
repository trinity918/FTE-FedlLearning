import numpy as np
import pytest

from fte_fl.defenses.krum import krum, krum_scores, multi_krum


def make_gradients():
    rng = np.random.default_rng(42)
    honest = rng.normal(0, 0.01, size=(5, 32))
    sign_flip = -10 * np.ones((1, 32))
    noise = rng.normal(100, 10, size=(1, 32))
    return np.vstack([honest, sign_flip, noise])


def test_krum_prefers_honest_gradient():
    grads = make_gradients()
    selected, scores = krum(grads, f=2)
    assert selected < 5
    assert scores[selected] < scores[5]
    assert scores[selected] < scores[6]


def test_multi_krum_selects_honest_gradients():
    grads = make_gradients()
    selected, scores = multi_krum(grads, f=2, m=3)
    assert len(selected) == 3
    assert all(index < 5 for index in selected)


def test_scores_are_finite_and_nonnegative():
    scores = krum_scores(make_gradients(), f=2)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_rejects_invalid_f():
    with pytest.raises(ValueError):
        krum(np.zeros((5, 4)), f=2)
