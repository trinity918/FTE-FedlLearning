"""Compare a run with its replay on another platform, round by round.

    python -m fte_fl.eval.platform_check REFERENCE_RUN_DIR REPLAY_RUN_DIR

Compares the config hash, client manifest, participants, test accuracy and
loss, local training loss and aggregate norm, and prints the first round at
which any of them differs plus the size of the differences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _rounds(run: Path) -> list[dict]:
    return [json.loads(ln) for ln in (run / "metrics.jsonl").read_text().splitlines()]


def compare(reference: Path, replay: Path) -> dict:
    ref_cfg = json.loads((reference / "config.json").read_text())
    new_cfg = json.loads((replay / "config.json").read_text())
    ref_man = json.loads((reference / "manifest.json").read_text())
    new_man = json.loads((replay / "manifest.json").read_text())
    ref, new = _rounds(reference), _rounds(replay)
    paired = list(zip(ref, new))
    first_diff = next((a["round"] for a, b in paired if (a["test"], a["aggregate_norm"]) != (b["test"], b["aggregate_norm"])), None)
    return {
        "same_config_hash": ref_cfg["config_hash"] == new_cfg["config_hash"],
        "same_partition": ref_man["data_info"]["partition_id"] == new_man["data_info"]["partition_id"],
        "platforms": [ref_cfg["environment"]["platform"], new_cfg["environment"]["platform"]],
        "rounds_compared": len(paired),
        "same_participants_all_rounds": all(a["participants"] == b["participants"] for a, b in paired),
        "first_round_with_a_numeric_difference": first_diff,
        "max_abs_accuracy_difference": max(abs(a["test"]["accuracy"] - b["test"]["accuracy"]) for a, b in paired),
        "max_abs_loss_difference": max(abs(a["test"]["loss"] - b["test"]["loss"]) for a, b in paired),
        "max_abs_aggregate_norm_difference": max(abs(a["aggregate_norm"] - b["aggregate_norm"]) for a, b in paired),
        "last_round_accuracy": [paired[-1][0]["test"]["accuracy"], paired[-1][1]["test"]["accuracy"]],
    }


if __name__ == "__main__":
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
