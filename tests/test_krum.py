import numpy as np
import pytest

from fte_fl.defenses.krum import (
    krum,
    krum_from_distances,
    krum_scores,
    multi_krum,
    multi_krum_from_distances,
    pairwise_squared_distances,
    scores_from_distances,
)


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


def test_attacker_score_much_greater_than_honest():
    """Spec requires 'attackers score >> honest' — verify the magnitude gap."""
    grads = make_gradients()
    _, scores = krum(grads, f=2)
    honest_min = np.min(scores[:5])
    # Both attackers should have scores at least 100x the best honest score.
    assert scores[5] > 100 * honest_min
    assert scores[6] > 100 * honest_min


def test_multi_krum_default_m_is_n_minus_f():
    """When m is None, multi_krum selects n - f clients (NeurIPS 2017, Sec. 6)."""
    grads = make_gradients()
    n = len(grads)
    f = 2
    expected_m = n - f  # 7 - 2 = 5
    selected, _ = multi_krum(grads, f=f, m=None)
    assert len(selected) == expected_m
    assert sorted(selected.tolist()) == [0, 1, 2, 3, 4]


def _reference_scores(grads, f):
    """Paper definition, computed independently: sum of the n-f-2 smallest
    squared Euclidean distances to the other vectors."""
    n = len(grads)
    scores = []
    for i in range(n):
        d = sorted(float(np.sum((grads[i] - grads[j]) ** 2)) for j in range(n) if j != i)
        scores.append(sum(d[: n - f - 2]))
    return np.array(scores)


@pytest.mark.parametrize("f", [0, 1, 2])
def test_scores_match_paper_definition_with_squared_distances(f):
    grads = np.random.default_rng(f).normal(size=(7, 5))
    np.testing.assert_allclose(krum_scores(grads, f), _reference_scores(grads, f), rtol=1e-10)


def test_scores_hand_computed():
    # 1-D points 0, 1, 3, 10, 20 with f=1: n-f-2 = 2 nearest squared distances.
    grads = np.array([[0.0], [1.0], [3.0], [10.0], [20.0]])
    expected = [1 + 9, 1 + 4, 4 + 9, 49 + 81, 100 + 289]
    np.testing.assert_allclose(krum_scores(grads, 1), expected)
    assert krum(grads, 1)[0] == 1


def test_krum_ties_go_to_smallest_index():
    grads = np.array([[1.0], [1.0], [1.0], [1.0], [9.0]])
    assert krum(grads, 1)[0] == 0
    selected, _ = multi_krum(grads, 1, m=2)
    assert selected.tolist() == [0, 1]


def test_from_distances_matches_plaintext():
    grads = make_gradients()
    dist = pairwise_squared_distances(np.asarray(grads, dtype=np.float64))
    assert krum_from_distances(dist, 2)[0] == krum(grads, 2)[0]
    assert multi_krum_from_distances(dist, 2, 4)[0].tolist() == multi_krum(grads, 2, 4)[0].tolist()


def test_inputs_not_modified():
    grads = make_gradients()
    before = grads.copy()
    krum(grads, 2)
    multi_krum(grads, 2)
    assert np.array_equal(grads, before)


@pytest.mark.parametrize("m", [0, 8, -1])
def test_multi_krum_rejects_m_out_of_range(m):
    with pytest.raises(ValueError):
        multi_krum(make_gradients(), f=2, m=m)


def test_multi_krum_m_equal_n_is_all_clients():
    selected, _ = multi_krum(make_gradients(), f=2, m=7)
    assert sorted(selected.tolist()) == list(range(7))


@pytest.mark.parametrize("f, exc", [(-1, ValueError), (1.5, TypeError), (True, TypeError), (None, TypeError)])
def test_rejects_invalid_f_types(f, exc):
    with pytest.raises(exc):
        krum(make_gradients(), f=f)


@pytest.mark.parametrize("grads", [np.zeros(5), np.full((7, 3), np.nan), np.zeros((7, 3, 1))])
def test_rejects_malformed_gradients(grads):
    with pytest.raises(ValueError):
        krum(grads, f=1)


def test_condition_boundary_n_equals_2f_plus_3():
    grads = np.random.default_rng(0).normal(size=(5, 3))
    krum(grads, f=1)                      # n = 5 > 2*1 + 2
    with pytest.raises(ValueError):
        krum(grads[:4], f=1)              # n = 4 = 2f + 2


def test_scores_from_distances_validation():
    with pytest.raises(ValueError):
        scores_from_distances(np.zeros((3, 4)), 1)
    with pytest.raises(ValueError):
        scores_from_distances(np.zeros((3, 3)), 3)
