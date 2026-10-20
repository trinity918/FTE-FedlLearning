"""Tests for the EtF, FTE and FTE + attestation protocols.

Real CKKS encryption (TenSEAL) and real commitments/proofs are used
throughout; nothing is mocked.  All parties run in one process: "data flow"
tests check (a) the structural boundary (the aggregator holds no secret key
and accepts only serialized ciphertexts) and (b) the message transcript the
orchestration records.  (b) is produced by the protocol code itself, so it
documents the implemented flow rather than enforcing it independently.
"""
import numpy as np
import pytest
import tenseal as ts

from fte_fl.attest.pedersen import Q
from fte_fl.crypto.ckks_manager import CKKSManager
from fte_fl.defenses.aggregators import RULES, combine
from fte_fl.defenses.krum import krum, krum_scores, multi_krum, pairwise_squared_distances
from fte_fl.defenses.protocols import (
    ETF_RULES,
    Aggregator,
    KeyHolder,
    attest_update,
    attestation_context,
    run_etf,
    run_fte,
    run_fte_attested,
)

# CKKS approximation tolerance for aggregates of values ~1e-2 .. 5 (observed
# errors are ~1e-6 or smaller).
ATOL = 1e-4


@pytest.fixture(scope="module")
def key_holder():
    # Small ring and chunk size keep tests fast and exercise multi-chunk vectors
    # (d = 40 -> chunks of 16, 16, 8).
    return KeyHolder(CKKSManager(poly_modulus_degree=8192, chunk_size=16))


@pytest.fixture(scope="module")
def updates():
    rng = np.random.default_rng(0)
    honest = rng.normal(0, 0.01, size=(7, 40))
    byzantine = np.vstack([np.full((1, 40), 5.0), rng.normal(3, 1, size=(1, 40))])
    return np.vstack([honest, byzantine])            # n = 9; clients 7, 8 Byzantine


def _kinds(result, sender=None, receiver=None):
    return [m.kind for m in result.transcript
            if (sender is None or m.sender == sender) and (receiver is None or m.receiver == receiver)]


# ─── structural boundary ─────────────────────────────────────────────────────

def test_aggregator_cannot_decrypt(key_holder):
    agg = Aggregator(key_holder.public_context())
    assert not agg.context.is_private()
    ct = ts.ckks_vector(agg.context, [1.0, 2.0])
    with pytest.raises(ValueError, match="secret"):
        ct.decrypt()


def test_aggregator_rejects_secret_key_and_plaintext(key_holder):
    secret_blob = key_holder._manager.context.serialize(save_secret_key=True)
    with pytest.raises(ValueError):
        Aggregator(secret_blob)
    agg = Aggregator(key_holder.public_context(galois_keys=False))
    with pytest.raises(TypeError):
        agg.sum([[np.ones(3)]])
    with pytest.raises(TypeError):
        agg.sum([np.ones(3)])
    with pytest.raises(TypeError):
        Aggregator(key_holder._manager.context)


def test_encrypted_distances_match_plaintext(key_holder, updates):
    agg = Aggregator(key_holder.public_context())
    client_ctx = ts.context_from(key_holder.public_context(galois_keys=False))
    enc = [[ts.ckks_vector(client_ctx, row[s:s + 16].tolist()).serialize() for s in range(0, 40, 16)]
           for row in updates]
    dist = key_holder.decrypt_distances(len(updates), agg.squared_distances(enc))
    # Observed CKKS error on these squared distances: up to ~4e-6 absolute
    # (ring degree 8192), i.e. ~5e-4 relative for distances ~8e-3.
    np.testing.assert_allclose(dist, pairwise_squared_distances(updates), rtol=1e-5, atol=1e-5)


# ─── EtF ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule, params", [
    ("fedavg", {}), ("fedavg", {"weights": np.arange(1, 10)}),
    ("krum", {"f": 2}), ("multi_krum", {"f": 2}), ("multi_krum", {"f": 2, "m": 3}),
])
def test_etf_matches_plaintext_rule(key_holder, updates, rule, params):
    if rule != "fedavg":
        # Premise: plaintext Krum scores are separated by far more than the
        # CKKS score error (<= (n-f-2) * ~4e-6 per distance, ~2e-5 here), so
        # the approximate selection must equal the plaintext one.
        assert np.diff(np.sort(krum_scores(updates, params["f"]))).min() > 1e-4
    res = run_etf(updates, rule, key_holder, **params)
    plain_sel = RULES[rule](updates, **params)
    np.testing.assert_allclose(res.aggregate, combine(updates, plain_sel), atol=ATOL)
    assert res.contributors.tolist() == plain_sel.contributors.tolist()


def test_ckks_distances_are_approximate_not_exact(key_holder, updates):
    """EtF filters on approximate distances: they differ from plaintext."""
    agg = Aggregator(key_holder.public_context())
    client_ctx = ts.context_from(key_holder.public_context(galois_keys=False))
    enc = [[ts.ckks_vector(client_ctx, row[s:s + 16].tolist()).serialize() for s in range(0, 40, 16)]
           for row in updates[:4]]
    dist = key_holder.decrypt_distances(4, agg.squared_distances(enc))
    exact = pairwise_squared_distances(updates[:4])
    off = ~np.eye(4, dtype=bool)
    assert not np.array_equal(dist[off], exact[off])
    assert np.abs(dist[off] - exact[off]).max() < 1e-5


def test_etf_near_tie_selection_is_not_guaranteed_to_match_plaintext(key_holder):
    """Limitation: with an exact plaintext tie, plaintext Krum picks the
    smaller index (0) but EtF may pick either tied client, because CKKS noise
    breaks the tie arbitrarily.  Only membership in the tied set is asserted."""
    d = 40
    a = np.zeros(d); a[:20] = 0.01                  # a and b have disjoint support,
    b = np.zeros(d); b[20:] = 0.03                  # so a.b = 0 exactly
    c = np.full(d, 5.0)
    u = np.vstack([a, -a, b, -b, c, -c])            # n = 6, f = 1
    scores = krum_scores(u, 1)
    assert scores[0] == scores[1] and scores[0] < scores[2:].min()   # exact tie
    assert krum(u, 1)[0] == 0
    res = run_etf(u, "krum", key_holder, f=1)
    assert res.contributors.tolist() in ([0], [1])
    np.testing.assert_allclose(res.aggregate, u[res.contributors[0]], atol=ATOL)


def test_etf_krum_selects_an_honest_client(key_holder, updates):
    assert np.diff(np.sort(krum_scores(updates, 2))).min() > 1e-4    # no near tie
    res = run_etf(updates, "krum", key_holder, f=2)
    assert res.contributors.tolist() == [krum(updates, 2)[0]]
    assert res.contributors[0] < 7


def test_etf_data_flow(key_holder, updates):
    n = len(updates)
    res = run_etf(updates, "multi_krum", key_holder, f=2)
    # Encryption happens first: the first n messages are client ciphertexts.
    assert res.transcript[:n] == [(f"client:{i}", "aggregator", "ciphertext_update") for i in range(n)]
    # Nobody ever sends a plaintext update; the filter role does not exist.
    assert all("plaintext" not in m.kind for m in res.transcript)
    assert all("filter" not in (m.sender, m.receiver) for m in res.transcript)
    # The key holder decrypts only distances and the aggregate.
    assert _kinds(res, receiver="key_holder") == ["encrypted_squared_distances", "encrypted_aggregate"]
    assert _kinds(res, sender="key_holder") == ["selection"]


def test_etf_fedavg_has_no_filtering_step(key_holder, updates):
    res = run_etf(updates, "fedavg", key_holder)
    assert _kinds(res, receiver="key_holder") == ["encrypted_aggregate"]
    assert res.contributors.tolist() == list(range(len(updates)))


@pytest.mark.parametrize("rule, params, exc", [
    ("trimmed_mean", {}, ValueError),
    ("bulyan", {}, ValueError),
    ("median", {}, ValueError),
    ("krum", {}, ValueError),                      # f missing
    ("krum", {"f": 2, "m": 3}, ValueError),
    ("fedavg", {"f": 1}, ValueError),
    ("krum", {"f": 2, "weights": [1] * 9}, ValueError),
    ("krum", {"f": 4}, ValueError),                # n = 9 <= 2f + 2
    ("multi_krum", {"f": 2, "m": 10}, ValueError),
])
def test_etf_rejects_unsupported(key_holder, updates, rule, params, exc):
    with pytest.raises(exc):
        run_etf(updates, rule, key_holder, **params)


def test_etf_rules_constant():
    assert ETF_RULES == ("fedavg", "krum", "multi_krum")


# ─── FTE ─────────────────────────────────────────────────────────────────────

FTE_CASES = [
    ("fedavg", {}), ("krum", {"f": 2}), ("multi_krum", {"f": 2}),
    ("trimmed_mean", {"b": 2}), ("bulyan", {"f": 1}),
]


@pytest.mark.parametrize("rule, params", FTE_CASES)
def test_fte_matches_plaintext_rule(key_holder, updates, rule, params):
    res = run_fte(updates, rule, key_holder, **params)
    expected = combine(updates, RULES[rule](updates, **params))
    np.testing.assert_allclose(res.aggregate, expected, atol=ATOL)


@pytest.mark.parametrize("rule, params", FTE_CASES)
def test_fte_data_flow(key_holder, updates, rule, params):
    n = len(updates)
    res = run_fte(updates, rule, key_holder, **params)
    kinds = [m.kind for m in res.transcript]
    # Filtering on plaintext happens before any encryption.
    assert res.transcript[:n] == [(f"client:{i}", "filter", "plaintext_update") for i in range(n)]
    first_ct = kinds.index("ciphertext_contribution")
    assert "plaintext_update" not in kinds[first_ct:]
    # Only contributors encrypt; the aggregator never receives plaintext.
    senders = [m.sender for m in res.transcript if m.kind == "ciphertext_contribution"]
    assert senders == [f"client:{i}" for i in res.contributors]
    assert all(m.kind != "plaintext_update" for m in res.transcript if m.receiver == "aggregator")
    assert _kinds(res, receiver="key_holder") == ["encrypted_aggregate"]


def test_fte_krum_only_selected_client_encrypts(key_holder, updates):
    res = run_fte(updates, "krum", key_holder, f=2)
    assert res.contributors.tolist() == [krum(updates, 2)[0]]
    assert sum(m.kind == "ciphertext_contribution" for m in res.transcript) == 1


def test_fte_multi_krum_excludes_byzantine(key_holder, updates):
    res = run_fte(updates, "multi_krum", key_holder, f=2)
    assert sorted(res.contributors.tolist()) == sorted(multi_krum(updates, 2)[0].tolist())
    assert 7 not in res.contributors and 8 not in res.contributors


@pytest.mark.parametrize("rule, params, exc", [
    ("median", {}, ValueError),
    ("bulyan", {"f": 2}, ValueError),            # n = 9 < 4*2 + 3
    ("trimmed_mean", {"b": 5}, ValueError),
    ("krum", {}, TypeError),                     # f missing
])
def test_fte_rejects_invalid(key_holder, updates, rule, params, exc):
    with pytest.raises(exc):
        run_fte(updates, rule, key_holder, **params)


def test_protocols_reject_malformed_updates(key_holder):
    bad = np.array([[1.0, np.nan], [0.0, 0.0]])
    with pytest.raises(ValueError):
        run_fte(bad, "fedavg", key_holder)
    with pytest.raises(ValueError):
        run_etf(bad, "fedavg", key_holder)


# ─── FTE + attestation ───────────────────────────────────────────────────────

ROUND = 4


def _submissions(updates, round_id=ROUND):
    return [attest_update(u, round_id, i) for i, u in enumerate(updates)]


def test_attested_all_valid_matches_fte(key_holder, updates):
    res = run_fte_attested(_submissions(updates), "multi_krum", key_holder, ROUND, f=2)
    plain = combine(updates, RULES["multi_krum"](updates, f=2))
    np.testing.assert_allclose(res.aggregate, plain, atol=ATOL)
    assert res.rejected.tolist() == []


def test_attested_data_flow(key_holder, updates):
    n = len(updates)
    res = run_fte_attested(_submissions(updates), "trimmed_mean", key_holder, ROUND, b=2)
    kinds = [m.kind for m in res.transcript]
    last_commit = max(i for i, k in enumerate(kinds) if k == "commitment_and_proof")
    first_plain = kinds.index("plaintext_update_and_opening")
    first_ct = kinds.index("ciphertext_contribution")
    assert last_commit < first_plain < first_ct          # commit -> filter -> encrypt
    assert kinds.count("commitment_and_proof") == 2 * n   # to filter and aggregator
    assert all(m.receiver == "filter" for m in res.transcript if m.kind == "plaintext_update_and_opening")


def test_attested_rejects_update_changed_after_commitment(key_holder, updates):
    subs = _submissions(updates)
    s = subs[3]
    subs[3] = type(s)(s.update + 0.5, s.commitment, s.opening, s.proof)   # different plaintext
    res = run_fte_attested(subs, "fedavg", key_holder, ROUND)
    assert res.rejected.tolist() == [3]
    keep = [i for i in range(len(updates)) if i != 3]
    np.testing.assert_allclose(res.aggregate, updates[keep].mean(axis=0), atol=ATOL)
    assert 3 not in res.contributors


def test_attested_rejects_wrong_round_and_replayed_proof(key_holder, updates):
    subs = _submissions(updates)
    subs[1] = attest_update(updates[1], ROUND + 1, 1)     # proof for another round
    subs[2] = subs[0]                                     # client 2 replays client 0's submission
    res = run_fte_attested(subs, "fedavg", key_holder, ROUND)
    assert res.rejected.tolist() == [1, 2]


def test_attested_rejects_forged_opening(key_holder, updates):
    subs = _submissions(updates)
    s = subs[5]
    forged = type(s.opening)(s.opening.m, (s.opening.r + 1) % Q)     # valid range, wrong r
    subs[5] = type(s)(s.update, s.commitment, forged, s.proof)
    res = run_fte_attested(subs, "fedavg", key_holder, ROUND)
    assert res.rejected.tolist() == [5]


def test_attested_contributor_indices_refer_to_original_clients(key_holder, updates):
    subs = _submissions(updates)
    subs[0] = attest_update(updates[0], ROUND + 1, 0)     # client 0 rejected
    res = run_fte_attested(subs, "krum", key_holder, ROUND, f=2)
    expected = 1 + krum(updates[1:], 2)[0]
    assert res.contributors.tolist() == [expected]


def test_attested_failure_modes(key_holder, updates):
    with pytest.raises(ValueError):                       # nothing passes
        run_fte_attested(_submissions(updates, ROUND + 1), "fedavg", key_holder, ROUND)
    subs = _submissions(updates[:7])
    subs[0] = attest_update(updates[0], ROUND + 1, 0)     # 6 left: krum f=2 needs n > 6
    with pytest.raises(ValueError):
        run_fte_attested(subs, "krum", key_holder, ROUND, f=2)
    with pytest.raises(ValueError):
        run_fte_attested([], "fedavg", key_holder, ROUND)
    with pytest.raises(TypeError):
        run_fte_attested([updates[0]], "fedavg", key_holder, ROUND)


@pytest.mark.parametrize("round_id, client_id", [(-1, 0), (0, -1), (1.0, 0), (True, 0)])
def test_attestation_context_validation(round_id, client_id):
    with pytest.raises(ValueError):
        attestation_context(round_id, client_id)


def test_attestation_context_distinct():
    assert attestation_context(1, 2) != attestation_context(2, 1)


# ─── payload sizes, timings, ciphertext counts ──────────────────────────────

def test_etf_accounting(key_holder, updates):
    n, d = updates.shape
    res = run_etf(updates, "krum", key_holder, f=2)
    chunks = -(-d // key_holder.chunk_size)                       # 3 chunks of <= 16
    assert res.ciphertexts == {"ciphertext_update": n * chunks,
                               "encrypted_squared_distances": n * (n - 1) // 2,
                               "encrypted_aggregate": chunks}
    for kind in ("ciphertext_update", "encrypted_squared_distances", "encrypted_aggregate",
                 "public_context_client_copy", "public_context_aggregator"):
        assert res.traffic[kind] > 0
    # the aggregator's context carries Galois keys for the dot products
    assert res.traffic["public_context_aggregator"] > res.traffic["public_context_client_copy"]
    assert "plaintext_update" not in res.traffic
    assert set(res.timings) == {"client_encrypt_s", "aggregator_distances_s",
                                "key_holder_decrypt_s", "filter_s", "aggregator_sum_s"}
    assert all(t > 0 for t in res.timings.values())


def test_fte_accounting(key_holder, updates):
    n, d = updates.shape
    res = run_fte(updates, "multi_krum", key_holder, f=2, m=4)
    chunks = -(-d // key_holder.chunk_size)
    assert res.traffic["plaintext_update"] == n * d * 8
    assert res.ciphertexts == {"ciphertext_contribution": 4 * chunks, "encrypted_aggregate": chunks}
    # same content (no Galois keys); serialized lengths differ slightly between
    # serializations (observed e.g. 1,876,181 vs 1,876,628 bytes), so compare loosely
    assert res.traffic["public_context_aggregator"] == pytest.approx(
        res.traffic["public_context_client_copy"], rel=0.01)
    assert set(res.timings) == {"filter_s", "client_encrypt_s", "aggregator_sum_s",
                                "key_holder_decrypt_s"}


def test_attested_accounting_and_weight_subsetting(key_holder, updates):
    subs = _submissions(updates[:5])
    subs[1] = attest_update(updates[1], ROUND + 1, 1)             # rejected
    weights = np.array([1.0, 100.0, 2.0, 3.0, 4.0])
    res = run_fte_attested(subs, "fedavg", key_holder, ROUND, weights=weights)
    assert res.rejected.tolist() == [1]
    keep = [0, 2, 3, 4]
    np.testing.assert_allclose(res.selection.weights, weights[keep] / weights[keep].sum())
    np.testing.assert_allclose(res.aggregate, weights[keep] / weights[keep].sum() @ updates[keep],
                               atol=ATOL)
    assert res.traffic["commitment_and_proof"] == 2 * 5 * 4 * 256
    assert res.traffic["opening"] == 5 * 2 * 256
    assert res.timings["filter_verify_s"] > 0
    with pytest.raises(ValueError, match="one entry per submission"):
        run_fte_attested(subs, "fedavg", key_holder, ROUND, weights=weights[keep])
