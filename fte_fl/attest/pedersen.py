"""Pedersen commitments to update digests and a Schnorr-style proof of opening.

Implemented with real modular arithmetic (Python ints, stdlib only):

  Group       the order-q subgroup of quadratic residues modulo the RFC 3526
              2048-bit MODP prime p (group 14), q = (p - 1) / 2 prime.
  Generators  g = 2 (a quadratic residue since p = 7 mod 8, hence of order q);
              h = (SHA-256 expansion of a fixed public label mod p)^2.  h is
              derived from a hash rather than chosen as g^x, so log_g h is not
              known by construction; that it is infeasible to compute is an
              assumption (discrete-log hardness), not something tested.
  Digest      m = SHA-256(domain || length || float64 little-endian bytes of
              the update), an integer < 2^256 < q.
  Commitment  C = g^m h^r mod p, r uniform in [0, q) from `secrets`.
  Proof       non-interactive proof of knowledge of (m, r) with C = g^m h^r:
              the Okamoto generalisation of Schnorr's sigma protocol made
              non-interactive with the Fiat-Shamir transform.  The challenge
              hashes g, h, C, the first message t and a caller-supplied
              `context` (e.g. round and client id), so a proof does not
              verify under another context.

What the tests establish: honest commitments open and honest proofs verify;
wrong openings, altered updates, altered proofs, wrong contexts and
non-group elements are rejected; the group parameters are a safe prime with
generators of order q; two accepting transcripts with one first message
yield (m, r) (special-soundness extraction) and simulated transcripts satisfy
the verification equation (interactive honest-verifier ZK structure).

What the tests do NOT establish (standard properties under assumptions):
computational binding (needs discrete-log hardness and an unknown log_g h),
perfect hiding (needs uniform r), and soundness / zero knowledge of the
Fiat-Shamir proof (random-oracle model).  A commitment covers the SHA-256
digest of the exact float64 bytes, so binding to the update also relies on
SHA-256 collision resistance.  Nothing here proves anything about a CKKS
ciphertext, a filtering decision, or an update's validity.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import numpy as np

# RFC 3526, Section 3: 2048-bit MODP group, id 14 (copied from the RFC text).
P = int(
    "FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1"
    "29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD"
    "EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245"
    "E485B576 625E7EC6 F44C42E9 A637ED6B 0BFF5CB6 F406B7ED"
    "EE386BFB 5A899FA5 AE9F2411 7C4B1FE6 49286651 ECE45B3D"
    "C2007CB8 A163BF05 98DA4836 1C55D39A 69163FA8 FD24CF5F"
    "83655D23 DCA3AD96 1C62F356 208552BB 9ED52907 7096966D"
    "670C354E 4ABC9804 F1746C08 CA18217C 32905E46 2E36CE3B"
    "E39E772C 180E8603 9B2783A2 EC07A28F B5C55DF0 6F4C52C9"
    "DE2BCBF6 95581718 3995497C EA956AE5 15D22618 98FA0510"
    "15728E5A 8AACAA68 FFFFFFFF FFFFFFFF".replace(" ", ""),
    16,
)
Q = (P - 1) // 2
G = 2
_ELEMENT_BYTES = (P.bit_length() + 7) // 8

_H_LABEL = b"fte-fl/attest/pedersen/h/v1"
_DIGEST_DOMAIN = b"fte-fl/attest/update-digest/v1"
_CHALLENGE_DOMAIN = b"fte-fl/attest/okamoto-fiat-shamir/v1"


def _hash_to_group(label: bytes) -> int:
    """Square of a 2304-bit SHA-256 expansion of `label`, reduced mod p."""
    stream = b"".join(hashlib.sha256(label + i.to_bytes(4, "big")).digest() for i in range(9))
    h = pow(int.from_bytes(stream, "big") % P, 2, P)
    if h in (0, 1):
        raise RuntimeError("hash-to-group produced a degenerate element")
    return h


H = _hash_to_group(_H_LABEL)


def _check_exponent(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= value < Q:
        raise ValueError(f"{name} must lie in [0, q)")
    return value


def in_group(x) -> bool:
    """True iff x is an int in the order-q subgroup of Z_p^*."""
    return (
        isinstance(x, int) and not isinstance(x, bool)
        and 1 <= x < P and pow(x, Q, P) == 1
    )


def update_digest(update) -> int:
    """SHA-256 digest of a finite, real, 1-D update's exact float64 bytes."""
    arr = np.asarray(update)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
        raise TypeError(f"update must be real numeric, got dtype {arr.dtype}")
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"update must be a non-empty 1-D vector, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype="<f8")
    if not np.all(np.isfinite(arr)):
        raise ValueError("update must be finite")
    data = _DIGEST_DOMAIN + arr.size.to_bytes(8, "big") + arr.tobytes()
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def commit(m: int, r: int) -> int:
    """Pedersen commitment g^m h^r mod p."""
    _check_exponent(m, "m")
    _check_exponent(r, "r")
    return pow(G, m, P) * pow(H, r, P) % P


@dataclass(frozen=True)
class Opening:
    m: int
    r: int


@dataclass(frozen=True)
class Proof:
    t: int
    s1: int
    s2: int


def commit_update(update) -> tuple[int, Opening]:
    """Commit to `update_digest(update)` with fresh randomness."""
    opening = Opening(m=update_digest(update), r=secrets.randbelow(Q))
    return commit(opening.m, opening.r), opening


def _is_exponent(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < Q


def verify_opening(commitment: int, update, opening: Opening) -> bool:
    """True iff `opening` opens `commitment` and its m is the digest of `update`.

    A malformed `update` (see `update_digest`) raises; a malformed opening or
    commitment simply fails verification.
    """
    digest = update_digest(update)
    if not isinstance(opening, Opening) or not in_group(commitment):
        return False
    if not (_is_exponent(opening.m) and _is_exponent(opening.r)):
        return False
    return opening.m == digest and commit(opening.m, opening.r) == commitment


def _challenge(commitment: int, t: int, context: bytes) -> int:
    data = _CHALLENGE_DOMAIN + b"".join(
        x.to_bytes(_ELEMENT_BYTES, "big") for x in (G, H, commitment, t)
    ) + len(context).to_bytes(8, "big") + context
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % Q


def _respond(opening: Opening, k1: int, k2: int, c: int) -> tuple[int, int]:
    return (k1 + c * opening.m) % Q, (k2 + c * opening.r) % Q


def _check_context(context) -> bytes:
    if not isinstance(context, bytes):
        raise TypeError(f"context must be bytes, got {type(context).__name__}")
    return context


def prove_opening(commitment: int, opening: Opening, context: bytes) -> Proof:
    """Fiat-Shamir proof of knowledge of `opening` for `commitment`, bound to `context`."""
    _check_context(context)
    if not isinstance(opening, Opening) or commit(opening.m, opening.r) != commitment:
        raise ValueError("opening does not open the commitment")
    k1, k2 = secrets.randbelow(Q), secrets.randbelow(Q)
    t = commit(k1, k2)
    s1, s2 = _respond(opening, k1, k2, _challenge(commitment, t, context))
    return Proof(t=t, s1=s1, s2=s2)


def verify_proof(commitment: int, proof: Proof, context: bytes) -> bool:
    """Check g^s1 h^s2 == t * C^c (mod p) with c = H(g, h, C, t, context)."""
    _check_context(context)
    if not isinstance(proof, Proof) or not in_group(commitment) or not in_group(proof.t):
        return False
    if not (_is_exponent(proof.s1) and _is_exponent(proof.s2)):
        return False
    c = _challenge(commitment, proof.t, context)
    lhs = pow(G, proof.s1, P) * pow(H, proof.s2, P) % P
    return lhs == proof.t * pow(commitment, c, P) % P
