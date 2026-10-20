"""Run one simulation from a YAML config.

    python -m fte_fl.experiments.run --config fte_fl/configs/default.yaml \
        --run-dir results/runs/<name> [--set experiment.rounds=2 ...] [--resume]

--set KEY=VALUE overrides a dotted key; VALUE is parsed as YAML (so 2 is an
int, 0.1 a float, null None).  Overrides are part of the recorded raw
config and its hash.  Exit status: 0 completed, 1 failed (summary.json
holds the error).
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from fte_fl.fl.simulator import run_simulation


def apply_override(raw: dict, assignment: str) -> None:
    key, sep, value = assignment.partition("=")
    if not sep or not key:
        raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
    *parents, leaf = key.split(".")
    node = raw
    for p in parents:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {key}: {p!r} is not a mapping")
    node[leaf] = yaml.safe_load(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    with open(args.config) as fh:
        raw = yaml.safe_load(fh)
    for assignment in args.set:
        apply_override(raw, assignment)
    summary = run_simulation(raw, args.run_dir, resume=args.resume)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
