"""Tests for Pedersen commitments and the Schnorr-style (Okamoto) proof of opening.

These are tests of REAL cryptographic arithmetic (no mocks).  They check
correctness and rejection behaviour; they cannot establish computational
binding, hiding, or Fiat-Shamir soundness / zero knowledge (see the module
docstring of fte_fl.attest.pedersen).
"""
import random
import secrets

import numpy as np
import pytest

from fte_fl.attest import pedersen as pd
from fte_fl.attest.pedersen import (
    G, H, P, Q, Opening, Proof, commit, commit_update, in_group,
    prove_opening, update_digest, verify_opening, verify_proof,
)

CTX = b"fte-fl/round=1/client=0"


def _is_probable_prime(n: int, rounds: int = 32) -> bool:
    """Miller-Rabin with a fixed-seed witness generator (test helper)."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = random.Random(0)
    for _ in range(rounds):
        x = pow(rng.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ─── group parameters ────────────────────────────────────────────────────────

def test_group_is_rfc3526_safe_prime():
    assert P.bit_length() == 2048
    assert P >> 1984 == 2**64 - 1 and P & (2**64 - 1) == 2**64 - 1   # RFC: top/bottom 64 bits set
    assert _is_probable_prime(P) and _is_probable_prime(Q) and P == 2 * Q + 1


def test_generators_have_order_q():
    for x in (G, H):
        assert x not in (0, 1, P - 1)
        assert pow(x, Q, P) == 1
    assert G != H


def test_in_group():
    assert in_group(G) and in_group(H) and in_group(commit(5, 7))
    assert not in_group(0) and not in_group(P) and not in_group(-4)
    assert not in_group(P - 1)              # order 2, not in the order-q subgroup
    assert not in_group(True) and not in_group(4.0)


# ─── digest and commitment ───────────────────────────────────────────────────

def test_digest_is_deterministic_and_sensitive():
    u = np.array([0.1, -0.2, 0.3])
    assert update_digest(u) == update_digest(u.copy())
    assert update_digest(u) == update_digest(u.astype(np.float64).tolist())
    assert update_digest(u) != update_digest(np.array([0.1, -0.2, 0.30000000000000004]))
    assert update_digest(u) != update_digest(u[:2])
    assert 0 <= update_digest(u) < 2**256 < Q


@pytest.mark.parametrize("bad, exc", [
    (np.zeros((2, 2)), ValueError), (np.zeros(0), ValueError),
    (np.array([np.nan]), ValueError), (np.array([True]), TypeError),
])
def test_digest_rejects_malformed(bad, exc):
    with pytest.raises(exc):
        update_digest(bad)


def test_commitment_formula_and_homomorphism():
    m1, r1, m2, r2 = 11, 22, 33, 44
    assert commit(m1, r1) == pow(G, m1, P) * pow(H, r1, P) % P
    assert commit(m1, r1) * commit(m2, r2) % P == commit(m1 + m2, r1 + r2)


@pytest.mark.parametrize("m, r, exc", [(-1, 0, ValueError), (0, Q, ValueError),
                                       (1.0, 0, TypeError), (True, 0, TypeError)])
def test_commit_rejects_bad_exponents(m, r, exc):
    with pytest.raises(exc):
        commit(m, r)


def test_commit_update_opens_and_wrong_openings_fail():
    u = np.array([0.5, -1.5, 2.0])
    c, op = commit_update(u)
    assert in_group(c)
    assert verify_opening(c, u, op)
    assert not verify_opening(c, u + 1e-9, op)                   # altered update
    assert not verify_opening(c, u, Opening(op.m, (op.r + 1) % Q))
    assert not verify_opening(c, u, Opening(op.m + 1, op.r))
    assert not verify_opening(commit(op.m, 0), u, op)            # other commitment
    assert not verify_opening(c, u, (op.m, op.r))                # not an Opening
    assert not verify_opening(c, u, Opening(op.m, -1))


def test_commitments_are_randomized():
    u = np.ones(4)
    c1, o1 = commit_update(u)
    c2, o2 = commit_update(u)
    assert o1.m == o2.m and c1 != c2


# ─── proof of knowledge of an opening ────────────────────────────────────────

@pytest.fixture(scope="module")
def committed():
    u = np.linspace(-1, 1, 7)
    c, op = commit_update(u)
    return u, c, op, prove_opening(c, op, CTX)


def test_honest_proof_verifies(committed):
    _, c, _, proof = committed
    assert verify_proof(c, proof, CTX)


def test_proof_bound_to_context_and_commitment(committed):
    _, c, op, proof = committed
    assert not verify_proof(c, proof, b"fte-fl/round=2/client=0")
    assert not verify_proof(c, proof, b"")
    other_c, _ = commit_update(np.zeros(3))
    assert not verify_proof(other_c, proof, CTX)


@pytest.mark.parametrize("field", ["t", "s1", "s2"])
def test_tampered_proof_rejected(committed, field):
    _, c, _, proof = committed
    values = {"t": proof.t, "s1": proof.s1, "s2": proof.s2}
    values[field] = (values[field] * G % P) if field == "t" else (values[field] + 1) % Q
    assert not verify_proof(c, Proof(**values), CTX)


@pytest.mark.parametrize("bad", [
    Proof(t=0, s1=1, s2=1), Proof(t=P - 1, s1=1, s2=1),
    Proof(t=G, s1=Q, s2=1), Proof(t=G, s1=-1, s2=1), Proof(t=G, s1=1.0, s2=1),
])
def test_malformed_proof_rejected(committed, bad):
    _, c, _, _ = committed
    assert not verify_proof(c, bad, CTX)


def test_random_proofs_rejected(committed):
    _, c, _, _ = committed
    for _ in range(5):
        forged = Proof(t=pow(G, secrets.randbelow(Q), P), s1=secrets.randbelow(Q), s2=secrets.randbelow(Q))
        assert not verify_proof(c, forged, CTX)


def test_prove_rejects_wrong_opening_and_context_type(committed):
    _, c, op, _ = committed
    with pytest.raises(ValueError):
        prove_opening(c, Opening(op.m, (op.r + 1) % Q), CTX)
    with pytest.raises(TypeError):
        prove_opening(c, op, "not-bytes")
    with pytest.raises(TypeError):
        verify_proof(c, committed[3], "not-bytes")


def test_special_soundness_extraction(committed):
    """Two accepting responses to one first message t reveal (m, r).

    Demonstrates the sigma protocol's special-soundness structure (via the
    internal responder); it does not prove Fiat-Shamir soundness.
    """
    _, c, op, _ = committed
    k1, k2 = secrets.randbelow(Q), secrets.randbelow(Q)
    t = commit(k1, k2)
    c1, c2 = 12345, 67890
    a1, b1 = pd._respond(op, k1, k2, c1)
    a2, b2 = pd._respond(op, k1, k2, c2)
    for ch, (s1, s2) in ((c1, (a1, b1)), (c2, (a2, b2))):
        assert pow(G, s1, P) * pow(H, s2, P) % P == t * pow(c, ch, P) % P
    inv = pow(c1 - c2, -1, Q)
    assert ((a1 - a2) * inv % Q, (b1 - b2) * inv % Q) == (op.m, op.r)


def test_simulated_transcripts_satisfy_verification_equation(committed):
    """Honest-verifier ZK structure: choose (c, s1, s2) first and solve for t.

    Such transcripts satisfy the interactive verification equation without
    the opening.  (Under Fiat-Shamir the challenge is a hash, so this does
    not produce a verifying non-interactive proof; see the next check.)
    """
    _, c, _, _ = committed
    ch, s1, s2 = secrets.randbelow(Q), secrets.randbelow(Q), secrets.randbelow(Q)
    t = pow(G, s1, P) * pow(H, s2, P) * pow(pow(c, ch, P), -1, P) % P
    assert pow(G, s1, P) * pow(H, s2, P) % P == t * pow(c, ch, P) % P
    assert not verify_proof(c, Proof(t, s1, s2), CTX)
