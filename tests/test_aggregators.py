"""Tests for plaintext aggregation rules (FedAvg, Multi-Krum, trimmed mean, Bulyan).

Expected values come from hand computations or independent reference
implementations written directly from the papers' definitions.
"""
import numpy as np
import pytest

from fte_fl.defenses import aggregators as agg
from fte_fl.defenses.aggregators import (
    RULES,
    Selection,
    bulyan,
    bulyan_selection,
    check_updates,
    combine,
    fedavg,
    multi_krum_selection,
    krum_selection,
    trimmed_mean,
    trimmed_mean_selection,
)


def _updates(n=9, d=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


# ─── input validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad, exc", [
    (np.zeros(4), ValueError),                      # 1-D
    (np.zeros((0, 3)), ValueError),                 # no clients
    (np.zeros((3, 0)), ValueError),                 # no coordinates
    (np.array([[1.0, np.nan]]), ValueError),
    (np.array([[1.0, np.inf]]), ValueError),
    (np.array([[True, False]]), TypeError),
    (np.array([[1 + 2j]]), TypeError),
    (np.array([["a"]]), TypeError),
])
def test_check_updates_rejects_malformed(bad, exc):
    with pytest.raises(exc):
        check_updates(bad)


@pytest.mark.parametrize("rule, params", [
    ("fedavg", {}), ("krum", {"f": 1}), ("multi_krum", {"f": 1}),
    ("trimmed_mean", {"b": 2}), ("bulyan", {"f": 1}),
])
def test_rules_do_not_modify_input_and_return_float64(rule, params):
    u = _updates(n=8).astype(np.float32)
    before = u.copy()
    sel = RULES[rule](u, **params)
    out = combine(u, sel)
    assert np.array_equal(u, before)
    assert out.dtype == np.float64 and out.shape == (u.shape[1],)
    assert sel.weights.shape == (8,) and np.all(sel.weights >= 0)


def test_integer_updates_accepted():
    assert np.allclose(fedavg(np.array([[1, 2], [3, 4]])), [2.0, 3.0])


# ─── FedAvg ──────────────────────────────────────────────────────────────────

def test_fedavg_equal_weights_is_mean():
    u = _updates()
    np.testing.assert_allclose(fedavg(u), u.mean(axis=0), rtol=1e-12)


def test_fedavg_sample_count_weights():
    u = np.array([[1.0, 0.0], [0.0, 1.0], [4.0, 4.0]])
    # n_k = 1, 3, 0  ->  (1*[1,0] + 3*[0,1]) / 4
    np.testing.assert_allclose(fedavg(u, [1, 3, 0]), [0.25, 0.75])


def test_fedavg_weights_are_scale_invariant():
    u = _updates()
    w = np.arange(1, 10)
    np.testing.assert_allclose(fedavg(u, w), fedavg(u, 7.5 * w), rtol=1e-12)


@pytest.mark.parametrize("w, exc", [
    ([1, 1], ValueError),                  # wrong length
    ([1, -1, 1], ValueError),
    ([0, 0, 0], ValueError),
    ([1, np.nan, 1], ValueError),
    ([1, np.inf, 1], ValueError),
    ([True, True, True], TypeError),
    (["a", "b", "c"], TypeError),
])
def test_fedavg_rejects_invalid_weights(w, exc):
    with pytest.raises(exc):
        fedavg(np.zeros((3, 2)), w)


# ─── Krum / Multi-Krum selections ────────────────────────────────────────────

def test_krum_selection_is_one_hot_and_combines_to_selected_update():
    u = np.array([[0.0], [1.0], [3.0], [10.0], [20.0]])
    sel = krum_selection(u, f=1)
    assert sel.weights.tolist() == [0, 1, 0, 0, 0]
    assert combine(u, sel).tolist() == [1.0]


def test_multi_krum_selection_averages_selected():
    u = np.array([[0.0], [1.0], [3.0], [10.0], [20.0]])
    sel = multi_krum_selection(u, f=1, m=3)     # scores 10, 5, 13, 130, 389
    assert sel.contributors.tolist() == [0, 1, 2]
    np.testing.assert_allclose(combine(u, sel), [4.0 / 3])


# ─── Trimmed mean ────────────────────────────────────────────────────────────

def _reference_trimmed_mean(u, b):
    s = np.sort(u, axis=0)
    return s[b : len(u) - b].mean(axis=0)


def test_trimmed_mean_hand_computed():
    u = np.array([[1.0, 10.0], [2.0, -5.0], [100.0, 0.0], [3.0, 1.0], [-50.0, 2.0]])
    # coord 0 sorted -50,1,2,3,100 -> b=1 keeps 1,2,3 -> 2
    # coord 1 sorted -5,0,1,2,10  -> keeps 0,1,2 -> 1
    np.testing.assert_allclose(trimmed_mean(u, 1), [2.0, 1.0])


@pytest.mark.parametrize("n, b", [(5, 0), (5, 2), (8, 3), (9, 1), (10, 4)])
def test_trimmed_mean_matches_reference(n, b):
    u = _updates(n=n, d=11, seed=n + b)
    np.testing.assert_allclose(trimmed_mean(u, b), _reference_trimmed_mean(u, b), rtol=1e-12)


def test_trimmed_mean_b0_is_mean_and_ties_handled():
    u = np.array([[1.0], [1.0], [1.0], [5.0], [5.0]])
    np.testing.assert_allclose(trimmed_mean(u, 0), [13.0 / 5])
    np.testing.assert_allclose(trimmed_mean(u, 1), [7.0 / 3])
    sel = trimmed_mean_selection(u, 1)
    assert sel.mask.sum(axis=0).tolist() == [3]          # exactly n - 2b kept


def test_trimmed_mean_boundary_single_value_left_is_median():
    # n=5, b=2 (n - 2b = 1, the largest valid b): per coordinate only the
    # middle value survives.  coord 0: 7,-1,3,9,0 -> sorted -1,0,3,7,9 -> 3
    # coord 1: 2,2,8,-4,5 -> sorted -4,2,2,5,8 -> 2
    u = np.array([[7.0, 2.0], [-1.0, 2.0], [3.0, 8.0], [9.0, -4.0], [0.0, 5.0]])
    np.testing.assert_allclose(trimmed_mean(u, 2), [3.0, 2.0])
    with pytest.raises(ValueError):
        trimmed_mean(u, 3)                       # n - 2b = -1


def test_trimmed_mean_even_n_boundary():
    # n=4, b=1: drop min and max, mean of the 2 middle values.
    # coord 0: 10,-2,4,6 -> keep 4,6 -> 5 ; coord 1: 1,1,1,100 -> keep 1,1 -> 1
    u = np.array([[10.0, 1.0], [-2.0, 1.0], [4.0, 1.0], [6.0, 100.0]])
    np.testing.assert_allclose(trimmed_mean(u, 1), [5.0, 1.0])
    with pytest.raises(ValueError):
        trimmed_mean(u, 2)                       # n - 2b = 0: nothing left


def test_trimmed_mean_single_client_b0():
    np.testing.assert_allclose(trimmed_mean(np.array([[2.5, -1.0]]), 0), [2.5, -1.0])


def test_trimmed_mean_resists_b_extreme_clients():
    u = np.vstack([np.ones((6, 4)), np.full((2, 4), 1e6)])
    np.testing.assert_allclose(trimmed_mean(u, 2), np.ones(4))


@pytest.mark.parametrize("b, exc", [(-1, ValueError), (3, ValueError), (10, ValueError),
                                    (1.0, TypeError), (True, TypeError)])
def test_trimmed_mean_rejects_invalid_b(b, exc):
    with pytest.raises(exc):
        trimmed_mean(np.zeros((6, 2)), b)          # n=6 needs n - 2b >= 1, so b <= 2


# ─── Bulyan ──────────────────────────────────────────────────────────────────

def _reference_bulyan(u, f):
    """Written from El Mhamdi et al. 2018, Sec. 4, independently of the module."""
    n = len(u)
    theta, beta = n - 2 * f, n - 4 * f
    remaining, chosen = list(range(n)), []
    while len(chosen) < theta:
        k = max(len(remaining) - f - 2, 1)
        best, best_score = None, None
        for i in remaining:
            d = sorted(float(np.sum((u[i] - u[j]) ** 2)) for j in remaining if j != i)
            score = sum(d[:k])
            if best_score is None or score < best_score:       # ties: smaller index
                best, best_score = i, score
        remaining.remove(best)
        chosen.append(best)
    chosen.sort()
    out = np.empty(u.shape[1])
    for c in range(u.shape[1]):
        vals = [u[i, c] for i in chosen]
        # median: a value of S minimising the sum of absolute deviations;
        # among minimisers (both middle values when theta is even, equal in
        # exact arithmetic but not always after float summation) take the
        # smallest.
        costs = {m: sum(abs(z - m) for z in vals) for m in vals}
        best = min(costs.values())
        med = min(m for m, c in costs.items() if c <= best * (1 + 1e-12))
        closest = sorted(range(theta), key=lambda k: (abs(vals[k] - med), k))[:beta]
        out[c] = np.mean([vals[k] for k in closest])
    return out, chosen


@pytest.mark.parametrize("n, f, seed", [(7, 1, 0), (8, 1, 1), (11, 2, 2), (12, 2, 3), (15, 3, 4)])
def test_bulyan_matches_reference(n, f, seed):
    u = _updates(n=n, d=9, seed=seed)
    expected, _ = _reference_bulyan(u, f)
    np.testing.assert_allclose(bulyan(u, f), expected, rtol=1e-12)


def test_bulyan_single_outlier_never_selected():
    # n=7, f=1: theta=5, beta=3; the outlier 100 (client 5) must not be in S.
    u = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [100.0], [5.0]])
    expected, chosen = _reference_bulyan(u, 1)
    assert 5 not in chosen
    sel = bulyan_selection(u, 1)
    assert sel.weights[5] == 0
    np.testing.assert_allclose(combine(u, sel), expected)


def test_bulyan_coordinate_phase_even_theta_uses_lower_median():
    # n=8, f=1: theta=6, beta=4.  Identical rows except the last coordinate
    # values 0..5 plus two far outliers; selection keeps the 6 near rows.
    near = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    u = np.vstack([near, [[1e3], [-1e3]]])
    sel = bulyan_selection(u, 1)
    assert sorted(np.flatnonzero(sel.weights).tolist()) == [0, 1, 2, 3, 4, 5]
    # lower median = 2; closest 4 to 2: 2, 1, 3 (tie 1/3 both kept), then 0 vs 4
    # tie at distance 2 -> smaller client index 0.  Mean of {0,1,2,3} = 1.5.
    np.testing.assert_allclose(combine(u, sel), [1.5])


def test_bulyan_excludes_f_byzantine_outliers():
    rng = np.random.default_rng(7)
    honest = rng.normal(0, 0.01, size=(9, 5))
    byz = np.full((2, 5), 50.0)
    u = np.vstack([honest, byz])                  # n=11, f=2: 11 >= 4*2+3
    sel = bulyan_selection(u, 2)
    assert set(sel.contributors.tolist()) <= set(range(9))
    assert np.all(np.abs(combine(u, sel)) < 0.05)


@pytest.mark.parametrize("n, f, exc", [
    (6, 1, ValueError),     # n < 4f + 3
    (10, 2, ValueError),
    (7, 0, ValueError),     # f >= 1 required
    (7, -1, ValueError),
    (7, 1.0, TypeError),
])
def test_bulyan_rejects_unsupported_configurations(n, f, exc):
    with pytest.raises(exc):
        bulyan(_updates(n=n), f)


@pytest.mark.parametrize("f", [1, 2, 3])
def test_bulyan_minimum_participants(f):
    n = 4 * f + 3
    sel = bulyan_selection(_updates(n=n, d=3, seed=f), f)       # minimum n accepted
    theta, beta = n - 2 * f, n - 4 * f
    assert (sel.weights > 0).sum() == theta                      # |S| = n - 2f
    assert beta == 3 and np.allclose(sel.weights[sel.weights > 0], 1 / beta)
    assert sel.mask.sum(axis=0).tolist() == [beta] * 3           # beta values per coordinate
    with pytest.raises(ValueError):
        bulyan_selection(_updates(n=n - 1, d=3, seed=f), f)      # n = 4f + 2 rejected


@pytest.mark.parametrize("n, valid_f", [(7, [1]), (11, [1, 2]), (15, [1, 2, 3])])
def test_bulyan_byzantine_budget(n, valid_f):
    u = _updates(n=n, d=2)
    for f in range(0, n):
        if f in valid_f:
            bulyan_selection(u, f)
        else:
            with pytest.raises(ValueError):
                bulyan_selection(u, f)


@pytest.mark.parametrize("n, f", [(7, 1), (8, 1), (11, 2), (15, 3)])
def test_bulyan_krum_steps_and_neighbour_counts(monkeypatch, n, f):
    """Record every Krum scoring step: |R| goes n, n-1, ..., 2f+1 (theta steps);
    k = |R| - f - 2 (exact Krum) except the clamp k = 1 for f = 1, |R| = 3;
    only the last two steps have |R| <= 2f + 2."""
    calls = []
    real = agg.scores_from_distances

    def recording(dist, k):
        calls.append((dist.shape[0], k))
        return real(dist, k)

    monkeypatch.setattr(agg, "scores_from_distances", recording)
    bulyan_selection(_updates(n=n, d=4), f)
    sizes = [r for r, _ in calls]
    assert sizes == list(range(n, 2 * f, -1))                    # theta = n - 2f steps
    assert len(calls) == n - 2 * f
    for r, k in calls:
        assert 1 <= k <= r - 1
        if f == 1 and r == 3:
            assert k == 1                                        # clamped from 0
        else:
            assert k == r - f - 2
    assert [r for r in sizes if r <= 2 * f + 2] == [2 * f + 2, 2 * f + 1]


def test_bulyan_selection_shape_and_weights():
    u = _updates(n=11, d=4)
    sel = bulyan_selection(u, 2)                  # theta = 7, beta = 3
    assert (sel.weights > 0).sum() == 7 and np.allclose(sel.weights[sel.weights > 0], 1 / 3)
    assert sel.mask.sum(axis=0).tolist() == [3] * 4
    assert not sel.mask[sel.weights == 0].any()


# ─── Selection / combine ─────────────────────────────────────────────────────

def test_contribution_matches_combine():
    u = _updates(n=8, d=5)
    sel = trimmed_mean_selection(u, 2)
    total = sum(sel.contribution(i, u[i]) for i in range(8))
    np.testing.assert_allclose(total, combine(u, sel), rtol=1e-12)


def test_combine_rejects_mismatched_selection():
    u = _updates(n=4, d=3)
    with pytest.raises(ValueError):
        combine(u, Selection(weights=np.ones(3)))
    with pytest.raises(ValueError):
        combine(u, Selection(weights=np.ones(4), mask=np.ones((4, 2), dtype=bool)))


def test_rules_registry_names():
    assert sorted(RULES) == ["bulyan", "fedavg", "krum", "multi_krum", "trimmed_mean"]
    assert agg.RULES["bulyan"] is bulyan_selection
