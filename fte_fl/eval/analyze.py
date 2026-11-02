"""Read completed run directories, check them, and summarise across seeds.

    python -m fte_fl.eval.analyze <grid output_dir> [--out DIR] [--last-k 10]

A run group is <output_dir>/<run name>/seed*/.  Checks (a failing check is
reported and the group is excluded from the tables, never silently used):
  * summary.json status is "completed";
  * metrics.jsonl holds exactly rounds 1..R with R = configured rounds;
  * every summarised metric is finite (a NaN or Inf is a problem, not a value);
  * checkpoint.pt exists, carries the run's config hash and round R, and
    its vector is finite;
  * all seeds of a group share one config apart from project.seed, and
    the seeds are distinct.
Statistics across seeds: mean and sample standard deviation (ddof=1; n/a
for a single seed) of
  final_*      the value in the last round, and
  lastk_*      each seed's mean over its last k rounds (default 10), a less
               noisy summary of the end of training.
Outputs in --out (default <output_dir>/_analysis): summary.csv,
summary.md, problems.txt and per-group plots (test accuracy and loss over
rounds, mean ± std band; ASR with clean accuracy for backdoor groups).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

SERIES = {
    "test_accuracy": lambda r: r["test"]["accuracy"],
    "test_loss": lambda r: r["test"]["loss"],
    "local_train_loss": lambda r: r["local_train_loss_mean_all"],
    "backdoor_asr": lambda r: r.get("backdoor_asr"),
    "flip_rate": lambda r: (r.get("label_flip") or {}).get("flip_rate"),
    "source_accuracy": lambda r: (r.get("label_flip") or {}).get("source_accuracy"),
    "malicious_rejected_frac": lambda r: r["filtering"]["tpr"],
    "honest_rejected_frac": lambda r: r["filtering"]["fpr"],
    "malicious_effective_weight": lambda r: r["filtering"]["malicious_effective_weight"],
    "round_time_s": lambda r: r["time_s"]["round"],
    "aggregation_time_s": lambda r: r["time_s"]["aggregation_total"],
}


class RunProblem(Exception):
    pass


def check_finite(run_dir: Path, metrics: list[dict]) -> None:
    for m in metrics:
        for key, read in SERIES.items():
            value = read(m)
            if value is not None and not math.isfinite(value):
                raise RunProblem(f"{run_dir}: {key} is {value} in round {m['round']}")


def check_checkpoint(run_dir: Path, config: dict, rounds: int) -> None:
    path = run_dir / "checkpoint.pt"
    if not path.exists():
        raise RunProblem(f"{run_dir}: no checkpoint.pt")
    state = torch.load(path, weights_only=False)
    if state["config_hash"] != config["config_hash"]:
        raise RunProblem(f"{run_dir}: checkpoint belongs to another config")
    if state["round"] != rounds:
        raise RunProblem(f"{run_dir}: checkpoint is at round {state['round']}, expected {rounds}")
    if not torch.isfinite(state["vector"]).all():
        raise RunProblem(f"{run_dir}: checkpoint vector is not finite")


def load_run(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RunProblem(f"{run_dir}: no summary.json")
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed":
        raise RunProblem(f"{run_dir}: status {summary['status']} ({summary.get('error', '')[:200]})")
    config = json.loads((run_dir / "config.json").read_text())
    metrics = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    rounds = config["parsed"]["rounds"]
    if [m["round"] for m in metrics] != list(range(1, rounds + 1)):
        raise RunProblem(f"{run_dir}: metrics do not hold exactly rounds 1..{rounds}")
    check_finite(run_dir, metrics)
    check_checkpoint(run_dir, config, rounds)
    return {"dir": run_dir, "config": config, "summary": summary, "metrics": metrics}


def _without_seed(raw: dict) -> dict:
    raw = copy.deepcopy(raw)
    raw.get("project", {}).pop("seed", None)
    return raw


def load_group(group_dir: Path) -> list[dict]:
    # seed<N> only: grid.py renames early failures to seed<N>.failed-<time>
    runs = [load_run(d) for d in sorted(group_dir.glob("seed*"))
            if d.is_dir() and d.name[4:].isdigit()]
    if not runs:
        raise RunProblem(f"{group_dir}: no seed directories")
    ref = _without_seed(runs[0]["config"]["raw"])
    for run in runs[1:]:
        if _without_seed(run["config"]["raw"]) != ref:
            raise RunProblem(f"{group_dir}: {run['dir'].name} differs in config beyond the seed")
    seeds = [run["config"]["parsed"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise RunProblem(f"{group_dir}: duplicate seeds {seeds}")
    return runs


def series(runs: list[dict], key: str) -> np.ndarray | None:
    """(seeds, rounds) array, or None if the metric is absent in these runs."""
    values = [[SERIES[key](m) for m in run["metrics"]] for run in runs]
    if all(v is None for row in values for v in row):
        return None
    return np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)


def _stats(x: np.ndarray) -> tuple[float, float | None]:
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else None)


def summarize(runs: list[dict], last_k: int) -> dict:
    row = {"seeds": len(runs), "rounds": runs[0]["config"]["parsed"]["rounds"]}
    for key in SERIES:
        s = series(runs, key)
        if s is None:
            continue
        row[f"final_{key}"] = _stats(s[:, -1])
        row[f"last{last_k}_{key}"] = _stats(np.nanmean(s[:, -last_k:], axis=1))
    p = runs[0]["config"]["parsed"]
    row.update(dataset=p["dataset"], model=p["model"], protocol=p["protocol"],
               defense=p["defense"]["name"], attack=p["attack"]["type"],
               num_malicious=p["attack"]["num_malicious"], sampling=p["sampling"],
               malicious_per_round=p["malicious_per_round"],
               clients_per_round=p["clients_per_round"],
               platforms="+".join(sorted({r["summary"]["environment_at_completion"]["platform"].split("-")[0]
                                          for r in runs})))
    return row


def _fmt(v) -> str:
    if isinstance(v, tuple):
        mean, std = v
        return f"{mean:.4f}" + ("" if std is None else f" ± {std:.4f}")
    return str(v)


def plot_group(name: str, runs: list[dict], out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    x = np.arange(1, runs[0]["config"]["parsed"]["rounds"] + 1)
    panels = [("test_accuracy", "clean test accuracy"), ("test_loss", "test loss")]
    if series(runs, "backdoor_asr") is not None:
        panels.append(("backdoor_asr", "backdoor ASR (triggered held-out, label != target)"))
    if series(runs, "flip_rate") is not None:
        panels.append(("flip_rate", "source->target flip rate (held-out source class)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.4))
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        s = series(runs, key)
        mean = np.nanmean(s, axis=0)
        ax.plot(x, mean, lw=1.2)
        if len(runs) > 1:
            std = np.nanstd(s, axis=0, ddof=1)
            ax.fill_between(x, mean - std, mean + std, alpha=0.25)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("round")
    fig.suptitle(f"{name} ({len(runs)} seeds, mean ± std)", fontsize=10)
    fig.tight_layout()
    path = out / f"{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths


def analyze(output_dir: Path, out: Path, last_k: int = 10, plots: bool = True) -> tuple[list, list]:
    out.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for group_dir in sorted(p for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        try:
            runs = load_group(group_dir)
        except RunProblem as exc:
            problems.append(str(exc))
            continue
        row = {"group": group_dir.name, **summarize(runs, last_k)}
        rows.append(row)
        if plots:
            plot_group(group_dir.name, runs, out)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("group",), k))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([_fmt(r.get(k, "")) for k in keys])
    md_keys = ["group", "seeds", "rounds", f"final_test_accuracy", f"last{last_k}_test_accuracy",
               f"final_test_loss", f"last{last_k}_backdoor_asr", f"last{last_k}_flip_rate",
               f"last{last_k}_malicious_rejected_frac", f"last{last_k}_honest_rejected_frac",
               f"last{last_k}_round_time_s"]
    with open(out / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(md_keys) + " |\n|" + "---|" * len(md_keys) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(_fmt(r.get(k, "")) for k in md_keys) + " |\n")
    (out / "problems.txt").write_text("\n".join(problems) + ("\n" if problems else ""), encoding="utf-8")
    return rows, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="summarise completed run groups")
    parser.add_argument("output_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--last-k", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    rows, problems = analyze(output_dir, Path(args.out) if args.out else output_dir / "_analysis",
                             args.last_k, not args.no_plots)
    print(f"{len(rows)} groups summarised, {len(problems)} problem(s)")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
