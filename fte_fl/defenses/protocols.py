"""EtF, FTE and FTE + attestation: single-process simulations with explicit roles.

Roles (all in one Python process; no network, no separate machines):
  client      holds its plaintext update.
  KeyHolder   holds the CKKS secret key and is the only role that decrypts.
              One trusted key holder; threshold decryption is not implemented.
  Aggregator  the server.  Built from a serialized PUBLIC context (no secret
              key), so TenSEAL refuses to decrypt there.  Accepts only
              serialized ciphertexts (bytes).
  filter      FTE only: receives PLAINTEXT updates and applies a rule from
              `aggregators.RULES`.  It sees every update in the clear; FTE
              offers no confidentiality against it.

EtF (encrypt, then filter) -- `run_etf`
  1. client i -> aggregator: Enc(u_i).
  2. aggregator: Enc(||u_i - u_j||^2) for all i < j, homomorphically
     (ciphertext subtraction and dot product, summed over chunks).
  3. aggregator -> key holder: the encrypted distances.  The key holder
     decrypts ONLY these and runs Krum / Multi-Krum on the CKKS-approximate
     distance matrix, returning the selection.  (In the tests, ring degree
     8192, the absolute error on squared distances was up to ~4e-6, which is
     ~5e-4 relative for nearly identical updates.)
  4. aggregator: sum_i w_i * Enc(u_i) with plaintext weights w_i.
  5. aggregator -> key holder: Enc(aggregate); the key holder decrypts it.
  Rules: fedavg (no filtering), krum, multi_krum.  Trimmed mean and Bulyan
  need coordinate-wise comparisons, which CKKS does not offer here; they are
  rejected.  Leakage: the key holder learns every pairwise squared distance
  and the aggregate; the aggregator learns the selection.  The decision is
  made on decrypted (approximate) distances: this is NOT a fully homomorphic
  robust aggregation, and the selection can differ from plaintext Krum when
  score gaps are below the CKKS error.

FTE (filter, then encrypt) -- `run_fte`
  1. client i -> filter: plaintext u_i.
  2. filter: Selection = rule(U) in plaintext; -> each contributing client:
     its weight a_i and mask M_i.
  3. contributing client i -> aggregator: Enc(a_i * (M_i * u_i)).
  4. aggregator: homomorphic sum; -> key holder: Enc(aggregate), decrypted.
  Non-contributing clients send no ciphertext.  Clients are trusted to
  encrypt exactly the filtered contribution; nothing checks this.

FTE + attestation -- `run_fte_attested`: FTE preceded by Pedersen
commitments and Schnorr-style proofs (`fte_fl.attest.pedersen`); see that
function's docstring for exactly what is verified.

`ProtocolResult.transcript` lists the messages the orchestration code sends,
in order.  It documents the data flow; it is produced by the same code, so
it does not independently enforce it.  The enforced boundary is that the
Aggregator holds no secret key and accepts only bytes.

`ProtocolResult.traffic` sums payload sizes in bytes by message kind:
`len()` of the serialized TenSEAL objects for ciphertexts and public
contexts; for plaintext updates, the float64 array size (8 bytes per
coordinate); for commitments, proofs and openings, a fixed-width encoding
of each integer (`pedersen.ELEMENT_BYTES`).  The two public-context
entries are the size of ONE copy (key distribution, not repeated per
round).  No message is sent over a network; these are payload sizes, not
measured traffic, and exclude framing and the plaintext selection
messages.  `ProtocolResult.timings` holds wall-clock seconds
(time.perf_counter) per step, all in this one process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import tenseal as ts

from fte_fl.attest.pedersen import (
    ELEMENT_BYTES,
    Opening,
    Proof,
    commit_update,
    prove_opening,
    verify_opening,
    verify_proof,
)
from fte_fl.crypto.ckks_manager import CKKSManager
from fte_fl.defenses.aggregators import RULES, Selection, check_updates
from fte_fl.defenses.krum import krum_from_distances, multi_krum_from_distances

ETF_RULES = ("fedavg", "krum", "multi_krum")


class Message(NamedTuple):
    sender: str
    receiver: str
    kind: str


@dataclass
class ProtocolResult:
    aggregate: np.ndarray            # decrypted by the key holder
    selection: Selection             # over the clients the rule ran on
    contributors: np.ndarray         # original client indices that sent ciphertexts
    transcript: list[Message]
    rejected: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    traffic: dict[str, int] = field(default_factory=dict)     # bytes by message kind
    timings: dict[str, float] = field(default_factory=dict)   # seconds by step
    ciphertexts: dict[str, int] = field(default_factory=dict)  # serialized CKKS vectors by kind


def _client(i: int) -> str:
    return f"client:{i}"


def _add(d: dict, key: str, value) -> None:
    d[key] = d.get(key, 0) + value


class _Timer:
    """Context manager adding the elapsed perf_counter seconds to timings[key]."""

    def __init__(self, timings: dict, key: str):
        self.timings, self.key = timings, key

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        _add(self.timings, self.key, time.perf_counter() - self.start)
        return False


class KeyHolder:
    """Holds the CKKS secret key; the only role that decrypts."""

    def __init__(self, manager: CKKSManager):
        if not manager.context.is_private():
            raise ValueError("the key holder needs a context with the secret key")
        self._manager = manager

    @property
    def chunk_size(self) -> int:
        return self._manager.chunk_size

    def public_context(self, galois_keys: bool = True) -> bytes:
        return self._manager.public_context_bytes(galois_keys)

    def decrypt_vector(self, chunks: list[bytes]) -> np.ndarray:
        ctx = self._manager.context
        return self._manager.decrypt([ts.ckks_vector_from(ctx, c) for c in chunks])

    def decrypt_distances(self, n: int, encrypted: dict[tuple[int, int], bytes]) -> np.ndarray:
        """Decrypt Enc(||u_i - u_j||^2), i < j, into a symmetric (n, n) matrix."""
        dist = np.zeros((n, n))
        for (i, j), blob in encrypted.items():
            value = ts.ckks_vector_from(self._manager.context, blob).decrypt()[0]
            dist[i, j] = dist[j, i] = max(value, 0.0)   # CKKS noise can dip below 0
        return dist


def _encrypt(public_ctx: ts.Context, vector: np.ndarray, chunk_size: int) -> list[bytes]:
    """Client-side encryption under the public key; returns serialized chunks."""
    return [
        ts.ckks_vector(public_ctx, vector[s : s + chunk_size].tolist()).serialize()
        for s in range(0, len(vector), chunk_size)
    ]


class Aggregator:
    """Server holding only a public CKKS context; operates on serialized ciphertexts."""

    def __init__(self, public_context: bytes):
        if not isinstance(public_context, bytes):
            raise TypeError("the aggregator must be given a serialized public context")
        self.context = ts.context_from(public_context)
        if self.context.is_private():
            raise ValueError("the aggregator must not receive the secret key")

    def _load(self, chunks: list[bytes]) -> list:
        if not isinstance(chunks, list) or not chunks or not all(isinstance(c, bytes) for c in chunks):
            raise TypeError("the aggregator accepts only non-empty lists of serialized ciphertexts")
        return [ts.ckks_vector_from(self.context, c) for c in chunks]

    def squared_distances(self, encrypted: list[list[bytes]]) -> dict[tuple[int, int], bytes]:
        """Enc(||u_i - u_j||^2) for all i < j: chunk-wise dot products, summed."""
        loaded = [self._load(e) for e in encrypted]
        if len({len(e) for e in loaded}) != 1:
            raise ValueError("encrypted updates have different chunk counts")
        out = {}
        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                total = None
                for a, b in zip(loaded[i], loaded[j]):
                    diff = a - b
                    sq = diff.dot(diff)
                    total = sq if total is None else total + sq
                out[(i, j)] = total.serialize()
        return out

    def weighted_sum(self, encrypted: list[list[bytes]], weights: list[float]) -> list[bytes]:
        """sum_k weights[k] * Enc(u_k) chunk-wise, with plaintext weights."""
        loaded = [self._load(e) for e in encrypted]
        if not loaded or len(loaded) != len(weights) or len({len(e) for e in loaded}) != 1:
            raise ValueError("need one weight per encrypted update and equal chunk counts")
        out = []
        for c in range(len(loaded[0])):
            total = None
            for vec, w in zip(loaded, weights):
                term = vec[c] if w == 1.0 else vec[c] * float(w)
                total = term.copy() if total is None else total + term
            out.append(total.serialize())
        return out

    def sum(self, encrypted: list[list[bytes]]) -> list[bytes]:
        return self.weighted_sum(encrypted, [1.0] * len(encrypted))


def _public_parties(key_holder: KeyHolder, galois_keys: bool,
                    traffic: dict) -> tuple[ts.Context, Aggregator]:
    """Clients' encryption context (no Galois keys) and the aggregator."""
    client_bytes = key_holder.public_context(galois_keys=False)
    aggregator_bytes = key_holder.public_context(galois_keys=galois_keys)
    traffic["public_context_client_copy"] = len(client_bytes)
    traffic["public_context_aggregator"] = len(aggregator_bytes)
    return ts.context_from(client_bytes), Aggregator(aggregator_bytes)


def _send(traffic: dict, ciphertexts: dict, kind: str, chunks: list[bytes]) -> None:
    """Account for serialized ciphertexts of one message kind (bytes and count)."""
    _add(traffic, kind, sum(len(c) for c in chunks))
    _add(ciphertexts, kind, len(chunks))


# ─── EtF ─────────────────────────────────────────────────────────────────────

def run_etf(updates, rule: str, key_holder: KeyHolder, *, f: int | None = None,
            m: int | None = None, weights=None) -> ProtocolResult:
    """Encrypt-then-filter round.  `rule` in ETF_RULES; see the module docstring."""
    if rule not in ETF_RULES:
        raise ValueError(
            f"rule {rule!r} is not supported under EtF (supported: {ETF_RULES}); "
            "coordinate-wise rules need comparisons that CKKS does not provide"
        )
    if rule == "fedavg" and (f is not None or m is not None):
        raise ValueError("fedavg takes no f or m")
    if rule != "fedavg" and weights is not None:
        raise ValueError(f"{rule} takes no weights")
    if rule == "krum" and m is not None:
        raise ValueError("krum takes no m")
    u = check_updates(updates)
    n = u.shape[0]
    traffic: dict[str, int] = {}
    timings: dict[str, float] = {}
    cts: dict[str, int] = {}
    client_ctx, aggregator = _public_parties(key_holder, rule != "fedavg", traffic)
    transcript: list[Message] = []

    encrypted = []
    for i in range(n):
        with _Timer(timings, "client_encrypt_s"):
            encrypted.append(_encrypt(client_ctx, u[i], key_holder.chunk_size))
        _send(traffic, cts, "ciphertext_update", encrypted[-1])
        transcript.append(Message(_client(i), "aggregator", "ciphertext_update"))

    if rule == "fedavg":
        with _Timer(timings, "filter_s"):
            selection = RULES["fedavg"](u, weights)
    else:
        if f is None:
            raise ValueError(f"{rule} requires f")
        with _Timer(timings, "aggregator_distances_s"):
            enc_dist = aggregator.squared_distances(encrypted)
        _send(traffic, cts, "encrypted_squared_distances", list(enc_dist.values()))
        transcript.append(Message("aggregator", "key_holder", "encrypted_squared_distances"))
        with _Timer(timings, "key_holder_decrypt_s"):
            dist = key_holder.decrypt_distances(n, enc_dist)
        with _Timer(timings, "filter_s"):
            w = np.zeros(n)
            if rule == "krum":
                w[krum_from_distances(dist, f)[0]] = 1.0
            else:
                chosen, _ = multi_krum_from_distances(dist, f, m)
                w[chosen] = 1.0 / len(chosen)
            selection = Selection(weights=w)
        transcript.append(Message("key_holder", "aggregator", "selection"))

    contributors = selection.contributors
    with _Timer(timings, "aggregator_sum_s"):
        enc_agg = aggregator.weighted_sum(
            [encrypted[i] for i in contributors], [selection.weights[i] for i in contributors]
        )
    _send(traffic, cts, "encrypted_aggregate", enc_agg)
    transcript.append(Message("aggregator", "key_holder", "encrypted_aggregate"))
    with _Timer(timings, "key_holder_decrypt_s"):
        aggregate = key_holder.decrypt_vector(enc_agg)
    return ProtocolResult(aggregate, selection, contributors, transcript,
                          traffic=traffic, timings=timings, ciphertexts=cts)


# ─── FTE ─────────────────────────────────────────────────────────────────────

def _fte_encrypt_and_sum(u, selection, ids, key_holder, transcript,
                         traffic, timings, cts) -> tuple[np.ndarray, np.ndarray]:
    """Steps after plaintext filtering: contributors encrypt, aggregator sums."""
    client_ctx, aggregator = _public_parties(key_holder, False, traffic)
    contributors = selection.contributors
    if len(contributors) == 0:
        raise ValueError("the filter selected no contributions")
    for k in contributors:
        transcript.append(Message("filter", _client(ids[k]), "selection"))
    encrypted = []
    for k in contributors:
        with _Timer(timings, "client_encrypt_s"):
            encrypted.append(_encrypt(client_ctx, selection.contribution(k, u[k]), key_holder.chunk_size))
        _send(traffic, cts, "ciphertext_contribution", encrypted[-1])
        transcript.append(Message(_client(ids[k]), "aggregator", "ciphertext_contribution"))
    with _Timer(timings, "aggregator_sum_s"):
        enc_agg = aggregator.sum(encrypted)
    _send(traffic, cts, "encrypted_aggregate", enc_agg)
    transcript.append(Message("aggregator", "key_holder", "encrypted_aggregate"))
    with _Timer(timings, "key_holder_decrypt_s"):
        aggregate = key_holder.decrypt_vector(enc_agg)
    return aggregate, np.asarray(ids)[contributors]


def run_fte(updates, rule: str, key_holder: KeyHolder, **rule_params) -> ProtocolResult:
    """Filter-then-encrypt round with any rule in `aggregators.RULES`."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {sorted(RULES)}")
    u = check_updates(updates)
    traffic: dict[str, int] = {"plaintext_update": int(u.size * 8)}
    timings: dict[str, float] = {}
    cts: dict[str, int] = {}
    transcript = [Message(_client(i), "filter", "plaintext_update") for i in range(u.shape[0])]
    with _Timer(timings, "filter_s"):
        selection = RULES[rule](u, **rule_params)
    aggregate, contributors = _fte_encrypt_and_sum(
        u, selection, list(range(u.shape[0])), key_holder, transcript, traffic, timings, cts
    )
    return ProtocolResult(aggregate, selection, contributors, transcript,
                          traffic=traffic, timings=timings, ciphertexts=cts)


# ─── FTE + attestation ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttestedSubmission:
    """What one client submits in an attested round.

    `commitment` and `proof` are public (sent to the filter and aggregator);
    `update` and `opening` go to the filter only.
    """

    update: np.ndarray
    commitment: int
    opening: Opening
    proof: Proof


def attestation_context(round_id: int, client_id: int) -> bytes:
    """Fiat-Shamir context binding a proof to one round and one client."""
    for name, v in (("round_id", round_id), ("client_id", client_id)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{name} must be a non-negative int, got {v!r}")
    return f"fte-fl/round={round_id}/client={client_id}".encode()


def attest_update(update, round_id: int, client_id: int) -> AttestedSubmission:
    """Client side: commit to the update digest and prove knowledge of the opening."""
    u = np.array(update, dtype=np.float64)
    commitment, opening = commit_update(u)
    proof = prove_opening(commitment, opening, attestation_context(round_id, client_id))
    return AttestedSubmission(u, commitment, opening, proof)


def run_fte_attested(submissions: list[AttestedSubmission], rule: str, key_holder: KeyHolder,
                     round_id: int, **rule_params) -> ProtocolResult:
    """FTE round in which every client first commits to its update.

    Client i (its index in `submissions`) publishes (C_i, proof_i) and sends
    (u_i, opening_i) to the filter.  The filter accepts client i only if
    proof_i verifies under context (round_id, i) AND the opening opens C_i
    with m = digest(u_i); other clients are excluded (`rejected`) before the
    rule runs on the accepted ones.  The aggregator re-verifies every
    contributor's public proof before using its ciphertext.  A `weights`
    rule parameter (FedAvg) is given per submission; the rule receives the
    accepted clients' entries.

    Verified: the plaintext the filter used equals the committed update
    (up to SHA-256 collisions), and each accepted proof is bound to this
    round and client id.  NOT verified: that a ciphertext encrypts the
    committed or filtered contribution, that the filter applied the rule
    correctly, or anything about update validity.  The filter sees every
    plaintext update and opening.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {sorted(RULES)}")
    if not isinstance(submissions, list) or not submissions:
        raise ValueError("submissions must be a non-empty list")
    if not all(isinstance(s, AttestedSubmission) for s in submissions):
        raise TypeError("submissions must be AttestedSubmission objects")
    u_all = check_updates(np.stack([np.asarray(s.update) for s in submissions]))
    n = len(submissions)
    # commitment + proof (t, s1, s2): 4 integers; opening (m, r): 2 integers
    traffic: dict[str, int] = {
        "commitment_and_proof": 2 * n * 4 * ELEMENT_BYTES,   # to filter and aggregator
        "plaintext_update": int(u_all.size * 8),
        "opening": n * 2 * ELEMENT_BYTES,
    }
    timings: dict[str, float] = {}

    transcript: list[Message] = []
    for i in range(n):
        transcript.append(Message(_client(i), "filter", "commitment_and_proof"))
        transcript.append(Message(_client(i), "aggregator", "commitment_and_proof"))
    for i in range(n):
        transcript.append(Message(_client(i), "filter", "plaintext_update_and_opening"))

    accepted, rejected = [], []
    with _Timer(timings, "filter_verify_s"):
        for i, s in enumerate(submissions):
            ctx = attestation_context(round_id, i)
            ok = verify_proof(s.commitment, s.proof, ctx) and verify_opening(s.commitment, u_all[i], s.opening)
            (accepted if ok else rejected).append(i)
    if not accepted:
        raise ValueError("no submission passed attestation")

    u = u_all[accepted]
    if rule_params.get("weights") is not None:
        w = np.asarray(rule_params["weights"])
        if w.shape != (n,):
            raise ValueError(f"weights must have one entry per submission ({n}), got {w.shape}")
        rule_params = {**rule_params, "weights": w[accepted]}
    with _Timer(timings, "filter_s"):
        selection = RULES[rule](u, **rule_params)
    with _Timer(timings, "aggregator_verify_s"):
        for k in selection.contributors:
            i = accepted[k]
            if not verify_proof(submissions[i].commitment, submissions[i].proof,
                                attestation_context(round_id, i)):
                raise RuntimeError(f"aggregator rejected client {i}'s attestation")
    cts: dict[str, int] = {}
    aggregate, contributors = _fte_encrypt_and_sum(u, selection, accepted, key_holder, transcript,
                                                   traffic, timings, cts)
    return ProtocolResult(aggregate, selection, contributors, transcript,
                          rejected=np.array(rejected, dtype=int), traffic=traffic,
                          timings=timings, ciphertexts=cts)
