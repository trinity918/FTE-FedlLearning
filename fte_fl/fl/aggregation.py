"""One aggregation step of the simulator under a configured protocol.

Weights are applied exactly once, inside the rule's Selection
(aggregate = sum_i a_i * (M_i * u_i)); the simulator passes raw updates.
  fedavg        a_i = n_i / sum_j n_j over the participants
                (fedavg_weighting="samples", McMahan et al.) or 1/n
                ("uniform").
  krum          a = 1 for the selected update, 0 otherwise.
  multi_krum    a = 1/m for the m selected updates (unweighted mean).
  trimmed_mean  a = 1/(n - 2b) with per-coordinate masks.
  bulyan        a = 1/beta on the selected set with per-coordinate masks.
The robust rules ignore sample counts: their aggregate is NOT a
sample-weighted FedAvg of the selected clients.

Protocols:
  plaintext     the rule on plaintext updates; no cryptography (baseline).
  etf / fte / fte_attested
                `fte_fl.defenses.protocols` (see that module for what each
                party sees and what is and is not verified).
For the encrypted protocols the simulator also computes, as a diagnostic it
can only compute because it holds every plaintext update, the maximum
absolute difference between the decrypted aggregate and the plaintext
evaluation of the same selection (`ckks_max_abs_error`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fte_fl.defenses import protocols
from fte_fl.defenses.aggregators import RULES, combine
from fte_fl.fl.config import SimConfig


@dataclass
class AggregationResult:
    aggregate: np.ndarray            # what is added to the global vector
    weights: np.ndarray              # a_i per participant (0 if rejected / not selected)
    kept_fraction: np.ndarray        # fraction of coordinates of u_i that enter the aggregate
    attestation_rejected: np.ndarray  # participant positions rejected by attestation
    traffic: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    ciphertexts: dict = field(default_factory=dict)
    ckks_max_abs_error: float | None = None


def _rule_kwargs(cfg: SimConfig, sample_counts) -> dict:
    kw = cfg.defense.rule_kwargs()
    if cfg.defense.name == "fedavg" and cfg.defense.fedavg_weighting == "samples":
        kw["weights"] = np.asarray(sample_counts, dtype=np.float64)
    return kw


def aggregate(deltas: np.ndarray, sample_counts, cfg: SimConfig, round_id: int,
              key_holder: protocols.KeyHolder | None = None,
              submissions: list | None = None) -> AggregationResult:
    """Aggregate the (n, d) participant updates.

    `submissions` (fte_attested only) replaces the honest attestation step;
    tests use it to inject tampered submissions.
    """
    n = deltas.shape[0]
    name = cfg.defense.name
    kw = _rule_kwargs(cfg, sample_counts)
    rejected = np.zeros(0, dtype=int)
    timings: dict[str, float] = {}
    ciphertexts: dict[str, int] = {}
    if cfg.protocol == "plaintext":
        start = time.perf_counter()
        sel = RULES[name](deltas, **kw)
        timings["filter_s"] = time.perf_counter() - start
        agg = combine(deltas, sel)
        traffic = {"plaintext_update": int(deltas.size * 8)}
        accepted, err = np.arange(n), None
    else:
        if key_holder is None:
            raise ValueError(f"protocol {cfg.protocol!r} needs a key holder")
        filtered = deltas
        if cfg.protocol == "etf":
            res = protocols.run_etf(deltas, name, key_holder, **kw)
        elif cfg.protocol == "fte":
            res = protocols.run_fte(deltas, name, key_holder, **kw)
        elif cfg.protocol == "fte_attested":
            if submissions is None:
                start = time.perf_counter()
                submissions = [protocols.attest_update(deltas[i], round_id, i) for i in range(n)]
                timings["client_attest_s"] = time.perf_counter() - start
            res = protocols.run_fte_attested(submissions, name, key_holder, round_id, **kw)
            rejected = res.rejected
            # the filter used the submitted (committed) updates
            filtered = np.stack([np.asarray(s.update, dtype=np.float64) for s in submissions])
        else:
            raise ValueError(f"unknown protocol {cfg.protocol!r}")
        accepted = np.setdiff1d(np.arange(n), rejected)
        sel, agg = res.selection, np.asarray(res.aggregate, dtype=np.float64)
        traffic = dict(res.traffic)
        timings.update(res.timings)
        ciphertexts = dict(res.ciphertexts)
        err = float(np.max(np.abs(agg - combine(filtered[accepted], sel))))

    weights = np.zeros(n)
    kept = np.zeros(n)
    weights[accepted] = sel.weights
    if sel.mask is None:
        kept[accepted] = (sel.weights > 0).astype(float)
    else:
        kept[accepted] = np.where(sel.weights > 0, sel.mask.mean(axis=1), 0.0)
    return AggregationResult(agg, weights, kept, rejected, traffic, timings, ciphertexts, err)
