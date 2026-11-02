"""Compare protocols on IDENTICAL client updates.

    python -m fte_fl.experiments.protocol_bench --config CONFIG --out DIR --budget F \
        [--checkpoint RUN_DIR/checkpoint.pt] [--rounds 101 102 103] \
        [--rules fedavg krum multi_krum trimmed_mean bulyan] \
        [--protocols plaintext etf fte fte_attested] [--set KEY=VALUE ...]

For each listed round r the simulator computes the round's participants and
(attacked) updates once, from the global vector in --checkpoint (or the
initial model), exactly as a training run would.  Every (protocol, rule)
pair then aggregates the same update matrix.  Unsupported pairs are
recorded with the validation error (e.g. EtF with trimmed_mean / Bulyan)
and not run.  The global model is NOT updated between rounds (each round
is an independent sample of participants at the same model state).

Per (round, protocol, rule) record in <out>/bench.jsonl: selection weights,
whether the set of selected clients equals the plaintext rule's, max
|aggregate - plaintext aggregate|, CKKS diagnostic error, protocol step
timings, total aggregation wall time, payload sizes, ciphertext counts,
process peak RSS so far (monotonic; not attributable to one protocol), and
for Krum-family rules the plaintext score gap at the selection boundary.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from fte_fl.defenses.krum import krum_scores
from fte_fl.experiments.run import apply_override
from fte_fl.fl.aggregation import aggregate
from fte_fl.fl.config import parse_sim_config
from fte_fl.fl.simulator import Simulation, _peak_rss_mb
from fte_fl.fl.state import get_vector, set_vector

PROTOCOLS = ("plaintext", "etf", "fte", "fte_attested")
RULE_PARAMS = {"fedavg": {}, "krum": {"f": None}, "multi_krum": {"f": None},
               "trimmed_mean": {"b": None}, "bulyan": {"f": None}}


def _variant(raw: dict, protocol: str, rule: str, budget: int) -> dict:
    raw = copy.deepcopy(raw)
    raw["experiment"]["protocol"] = protocol
    defense = {"name": rule}
    for key in RULE_PARAMS[rule]:
        defense[key] = budget
    raw["defense"] = defense
    return raw


def boundary_gap(deltas: np.ndarray, f: int, selected: np.ndarray) -> float:
    """Plaintext Krum score gap between the worst selected and best unselected update."""
    scores = krum_scores(deltas, f)
    sel = np.zeros(len(deltas), bool)
    sel[selected] = True
    if sel.all():
        return float("inf")
    return float(scores[~sel].min() - scores[sel].max())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="protocol comparison on identical updates")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rounds", type=int, nargs="+", default=[1])
    p.add_argument("--rules", nargs="+", default=list(RULE_PARAMS))
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    p.add_argument("--budget", type=int, required=True,
                   help="f for krum/multi_krum/bulyan and b for trimmed_mean")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text())
    for a in args.set:
        apply_override(raw, a)
    out = Path(args.out)
    # the base simulation must be able to hold a key holder: use FTE + fedavg
    base_raw = _variant(raw, "fte", "fedavg", 0)
    sim = Simulation(base_raw, out / "base")
    sim.setup(sim._prepare_dir())
    if args.checkpoint:
        state = torch.load(args.checkpoint, weights_only=False)
        set_vector(sim.model, state["vector"].numpy())
        sim.global_vec = get_vector(sim.model)
    (out / "bench_meta.json").write_text(json.dumps({
        "argv": argv, "config": args.config, "overrides": args.set, "checkpoint": args.checkpoint,
        "checkpoint_round": None if not args.checkpoint else int(state["round"]),
        "rounds": args.rounds, "rules": args.rules, "budget": args.budget}, indent=2))

    with open(out / "bench.jsonl", "a") as fh:
        for r in args.rounds:
            part, is_mal, active, deltas, counts, _, _ = sim.client_updates(r)
            plain = {}
            for protocol in args.protocols:
                for rule in args.rules:
                    rec = {"round": r, "protocol": protocol, "rule": rule, "participants": part,
                           "malicious_mask": is_mal.tolist(), "attack_active": active}
                    try:
                        cfg = parse_sim_config(_variant(base_raw, protocol, rule, args.budget))
                    except ValueError as exc:
                        rec["unsupported"] = str(exc)
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        continue
                    t = time.perf_counter()
                    res = aggregate(deltas, counts, cfg, r, sim.key_holder)
                    rec["aggregation_wall_s"] = time.perf_counter() - t
                    selected = np.flatnonzero(res.kept_fraction > 0)
                    if protocol == "plaintext":
                        plain[rule] = (res, selected)
                    if rule in plain:
                        p_res, p_sel = plain[rule]
                        rec["selection_matches_plaintext"] = bool(np.array_equal(selected, p_sel))
                        rec["max_abs_diff_vs_plaintext"] = float(np.max(np.abs(res.aggregate - p_res.aggregate)))
                    if rule in ("krum", "multi_krum"):
                        rec["plaintext_boundary_score_gap"] = boundary_gap(deltas, args.budget,
                                                                           plain[rule][1] if rule in plain else selected)
                    rec.update(weights=res.weights.tolist(), selected=selected.tolist(),
                               malicious_selected=int(is_mal[selected].sum()),
                               ckks_max_abs_error=res.ckks_max_abs_error, timings=res.timings,
                               payload_bytes=res.traffic, ciphertexts=res.ciphertexts,
                               peak_rss_mb_so_far=_peak_rss_mb())
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"round {r} {protocol:12s} {rule:12s} {rec['aggregation_wall_s']:8.2f}s "
                          f"match={rec.get('selection_matches_plaintext')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
