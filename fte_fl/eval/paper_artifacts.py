"""Build the paper's tables and figures from run directories.

    python -m fte_fl.eval.paper_artifacts --femnist DIR [--control DIR ...] [--speech MODEL=DIR ...]
        [--bench DIR] --out paper

Every value comes from run files via `analyze.load_group`, so a group that
fails a check is missing from the output and listed in <out>/tables/missing.txt.
A missing cell prints as "--"; nothing is imputed.  Seed statistics are the
mean and sample standard deviation (ddof=1) of each seed's mean over the last
10 rounds.  Comparisons with FedAvg pair runs by seed and give the mean
difference with a two-sided 95% t interval; with three seeds that interval has
2 degrees of freedom, so it is wide, and it is reported instead of a p-value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from fte_fl.eval import analyze

ATTACKS = ("honest", "signflip", "scaling", "zero", "gaussian", "alie", "labelflip", "backdoor")
DEFENSES = ("fedavg", "krum", "multikrum", "trimmed", "bulyan")
NAMES = {"honest": "No attack", "signflip": "Sign flip", "scaling": r"Scaling ($\gamma=10$)", "zero": "Zero",
         "gaussian": "Gaussian", "alie": "ALIE", "labelflip": r"Label flip", "backdoor": "Backdoor",
         "fedavg": "FedAvg", "krum": "Krum", "multikrum": "Multi-Krum", "trimmed": "Trimmed mean", "bulyan": "Bulyan"}
BENCH_NAMES = {"plaintext": "Plaintext", "etf": "EtF", "fte": "FTE", "fte_attested": "FTE+attestation"}
BENCH_RULES = {"fedavg": "fedavg", "krum": "krum", "multi_krum": "multikrum", "trimmed_mean": "trimmed", "bulyan": "bulyan"}
LAST_K = 10


def load_grid(root: Path, missing: list[str]) -> dict:
    groups = {}
    for name in sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        try:
            groups[name] = analyze.load_group(root / name)
        except analyze.RunProblem as exc:
            missing.append(str(exc))
    return groups


def per_seed(runs: list[dict], key: str) -> dict[int, float]:
    """{seed: mean over the last 10 rounds}, or {} when the metric does not apply."""
    s = analyze.series(runs, key)
    if s is None:
        return {}
    return {run["config"]["parsed"]["seed"]: float(np.nanmean(v[-LAST_K:])) for run, v in zip(runs, s)}


def cell(values: dict[int, float], digits: int = 3) -> str:
    if not values:
        return "--"
    v = np.array(list(values.values()))
    return f"{v.mean():.{digits}f}" if len(v) == 1 else f"{v.mean():.{digits}f} $\\pm$ {v.std(ddof=1):.{digits}f}"


def paired_ci(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float, int] | None:
    """Mean of a - b over shared seeds with its 95% t interval (None with fewer than 2 shared seeds)."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([a[s] - b[s] for s in seeds])
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half), len(d)


def matrix_tex(groups: dict, key: str, attacks=ATTACKS, prefix: str = "") -> str:
    lines = [r"\begin{tabular}{l" + "c" * len(DEFENSES) + "}", r"\toprule",
             "Attack & " + " & ".join(NAMES[d] for d in DEFENSES) + r" \\", r"\midrule"]
    for a in attacks:
        cells = [cell(per_seed(groups[f"{prefix}{a}__{d}"], key)) if f"{prefix}{a}__{d}" in groups else "--"
                 for d in DEFENSES]
        lines.append(f"{NAMES[a]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def paired_tex(groups: dict, key: str, baseline: str = "fedavg") -> tuple[str, dict]:
    rows, record = [], {}
    for a in ATTACKS:
        base = f"{a}__{baseline}"
        if base not in groups:
            continue
        for d in DEFENSES:
            name = f"{a}__{d}"
            if d == baseline or name not in groups:
                continue
            ci = paired_ci(per_seed(groups[name], key), per_seed(groups[base], key))
            if ci is None:
                continue
            record[name] = ci
            rows.append(f"{NAMES[a]} & {NAMES[d]} & {ci[0]:+.3f} & [{ci[1]:+.3f}, {ci[2]:+.3f}] & {ci[3]}" + r" \\")
    tex = "\n".join([r"\begin{tabular}{llccc}", r"\toprule",
                     r"Attack & Defense & $\Delta$ accuracy vs.\ FedAvg & 95\% CI & Seeds \\", r"\midrule",
                     *rows, r"\bottomrule", r"\end{tabular}", ""])
    return tex, record


def heatmap(groups: dict, key: str, title: str, path: Path, attacks=ATTACKS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((len(attacks), len(DEFENSES)), np.nan)
    for i, a in enumerate(attacks):
        for j, d in enumerate(DEFENSES):
            if f"{a}__{d}" in groups:
                values = per_seed(groups[f"{a}__{d}"], key)
                if values:
                    grid[i, j] = np.mean(list(values.values()))
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(attacks) + 1.2))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DEFENSES)), [NAMES[d] for d in DEFENSES], fontsize=8, rotation=20)
    ax.set_yticks(range(len(attacks)), [NAMES[a].replace("$", "").replace(r"\gamma", "γ") for a in attacks], fontsize=8)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if np.isnan(v) or v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curves(groups: dict, attacks: tuple, key: str, ylabel: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(attacks), figsize=(3.1 * len(attacks), 2.6), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], attacks):
        for d in DEFENSES:
            if f"{a}__{d}" not in groups:
                continue
            s = analyze.series(groups[f"{a}__{d}"], key)
            if s is None:
                continue
            x = np.arange(1, s.shape[1] + 1)
            mean = s.mean(axis=0)
            ax.plot(x, mean, lw=1.1, label=NAMES[d])
            if len(s) > 1:
                std = s.std(axis=0, ddof=1)
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set_title(NAMES[a].replace("$", "").replace(r"\gamma", "γ"), fontsize=9)
        ax.set_xlabel("round", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel(ylabel, fontsize=8)
    axes[0][-1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def femnist(root: Path, out: Path, missing: list[str]) -> dict:
    tables, figures = out / "tables", out / "figures"
    groups = load_grid(root, missing)
    (tables / "femnist_accuracy.tex").write_text(matrix_tex(groups, "test_accuracy"), encoding="utf-8")
    (tables / "femnist_tpr.tex").write_text(matrix_tex(groups, "malicious_rejected_frac", ATTACKS[1:]), encoding="utf-8")
    (tables / "femnist_fpr.tex").write_text(matrix_tex(groups, "honest_rejected_frac"), encoding="utf-8")
    (tables / "femnist_asr.tex").write_text(matrix_tex(groups, "backdoor_asr", ("backdoor",)), encoding="utf-8")
    (tables / "femnist_flip.tex").write_text(matrix_tex(groups, "flip_rate", ("labelflip",)), encoding="utf-8")
    tex, record = paired_tex(groups, "test_accuracy")
    (tables / "femnist_paired.tex").write_text(tex, encoding="utf-8")
    heatmap(groups, "test_accuracy", "FEMNIST clean test accuracy (last-10 mean)", figures / "femnist_heatmap.pdf")
    curves(groups, ("honest", "signflip", "scaling", "alie"), "test_accuracy", "test accuracy", figures / "femnist_curves.pdf")
    if any(k.startswith("backdoor__") for k in groups):
        curves(groups, ("backdoor",), "backdoor_asr", "backdoor ASR", figures / "femnist_asr.pdf")
    return {"groups": sorted(groups), "paired_vs_fedavg": record}


def control(roots: list[Path], main: Path, out: Path, missing: list[str]) -> dict:
    """Stratified no-op control against the uniform-sampling honest runs; later roots add to earlier ones."""
    ctl, base = {}, load_grid(main, missing)
    for root in roots:
        ctl.update(load_grid(root, missing))
    rows, record = [], {}
    for d in DEFENSES:
        a, b = ctl.get(f"noop__{d}"), base.get(f"honest__{d}")
        if a is None or b is None:
            continue
        va, vb = per_seed(a, "test_accuracy"), per_seed(b, "test_accuracy")
        ci = paired_ci(va, vb)
        record[d] = {"stratified": va, "uniform": vb, "difference": ci}
        rows.append(f"{NAMES[d]} & {cell(vb)} & {cell(va)} & {ci[0]:+.3f} & [{ci[1]:+.3f}, {ci[2]:+.3f}]" + r" \\")
    tex = "\n".join([r"\begin{tabular}{lcccc}", r"\toprule",
                     r"Defense & Uniform (Linux) & Stratified no-op (Windows) & $\Delta$ & 95\% CI \\", r"\midrule",
                     *rows, r"\bottomrule", r"\end{tabular}", ""])
    (out / "tables" / "control.tex").write_text(tex, encoding="utf-8")
    return record


def speech(model: str, root: Path, out: Path, missing: list[str]) -> dict:
    """Matrix of full-test accuracy from `full_eval` results of one model's grid."""
    results: dict = {}
    for path in sorted((root / "_full_eval").glob("*.json")):
        group, seed = path.stem.rsplit("__seed", 1)
        results.setdefault(group, {})[int(seed)] = json.loads(path.read_text())
    # {group: [round of divergence]}: a run that diverged is an outcome, not a missing cell
    diverged: dict[str, list[int]] = {}
    for path in sorted(root.glob("*/seed*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["status"] == "failed" and "FloatingPointError" in summary.get("error", ""):
            diverged.setdefault(path.parent.parent.name, []).append(summary["completed_rounds"] + 1)
    names = set(results) | set(diverged)
    attacks = tuple(a for a in ATTACKS if any(g.startswith(f"{a}__") for g in names))
    used = [d for d in DEFENSES if any(g.endswith(f"__{d}") for g in names)]

    def table(read, rows=attacks) -> str:
        lines = [r"\begin{tabular}{l" + "c" * len(used) + "}", r"\toprule",
                 "Attack & " + " & ".join(NAMES[d] for d in used) + r" \\", r"\midrule"]
        for a in rows:
            cells = []
            for d in used:
                group = f"{a}__{d}"
                text = cell({s: read(r) for s, r in results.get(group, {}).items() if read(r) is not None})
                if group in diverged:
                    note = "div.\\ round " + ", ".join(map(str, diverged[group]))
                    text = note if text == "--" else f"{text}; {note}"
                cells.append(text)
            lines.append(f"{NAMES[a]} & " + " & ".join(cells) + r" \\")
        return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])

    (out / "tables" / f"speech_{model}_accuracy.tex").write_text(table(lambda r: r["testing"]["accuracy"]), encoding="utf-8")
    if "backdoor" in attacks:
        (out / "tables" / f"speech_{model}_asr.tex").write_text(table(lambda r: r.get("backdoor_asr"), ("backdoor",)), encoding="utf-8")
    return {"accuracy": {g: {s: r["testing"]["accuracy"] for s, r in v.items()} for g, v in results.items()},
            "diverged_at_round": diverged}


def bench(root: Path, out: Path) -> dict:
    """One table per <root>/<dataset>/bench.jsonl; the validation_* directories are not read."""
    record: dict = {}
    for path in sorted(root.glob("*/bench.jsonl")):
        name = path.parent.name
        if name.startswith("validation"):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        lines = [r"\begin{tabular}{llccccc}", r"\toprule",
                 r"Protocol & Rule & Time (s) & Traffic (MB) & Selection = plaintext & Max error & Malicious kept \\",
                 r"\midrule"]
        record[name] = {}
        for protocol in dict.fromkeys(r["protocol"] for r in rows):
            for rule in dict.fromkeys(r["rule"] for r in rows):
                pair = [r for r in rows if r["protocol"] == protocol and r["rule"] == rule]
                label = f"{BENCH_NAMES[protocol]} & {NAMES[BENCH_RULES[rule]]}"
                if not pair:
                    continue
                if "unsupported" in pair[0]:
                    lines.append(f"{label} & \\multicolumn{{5}}{{c}}{{unsupported}}" + r" \\")
                    record[name][f"{protocol}/{rule}"] = "unsupported"
                    continue
                wall = np.array([r["aggregation_wall_s"] for r in pair])
                traffic = np.mean([sum(r["payload_bytes"].values()) for r in pair]) / 1e6
                matches = sum(bool(r["selection_matches_plaintext"]) for r in pair)
                error = max(r["max_abs_diff_vs_plaintext"] for r in pair)
                kept = sorted({r["malicious_selected"] for r in pair})
                spread = f" $\\pm$ {wall.std(ddof=1):.1f}" if len(wall) > 1 else ""
                lines.append(f"{label} & {wall.mean():.1f}{spread} & {traffic:.1f} & {matches}/{len(pair)} & "
                             f"{error:.1e} & {', '.join(map(str, kept))}" + r" \\")
                record[name][f"{protocol}/{rule}"] = {"wall_s": wall.tolist(), "traffic_mb": float(traffic),
                                                      "selection_matches": matches, "rounds": len(pair),
                                                      "max_abs_diff": error, "malicious_selected": kept}
        text = "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])
        (out / "tables" / f"bench_{name}.tex").write_text(text, encoding="utf-8")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--femnist", type=Path)
    p.add_argument("--control", type=Path, nargs="+", default=[])
    p.add_argument("--speech", action="append", default=[], help="MODEL=DIR")
    p.add_argument("--bench", type=Path)
    p.add_argument("--out", type=Path, default=Path("paper"))
    args = p.parse_args(argv)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    record: dict = {}
    if args.femnist:
        record["femnist"] = femnist(args.femnist, args.out, missing)
    if args.control and args.femnist:
        record["control"] = control(args.control, args.femnist, args.out, missing)
    for item in args.speech:
        model, root = item.split("=", 1)
        record[f"speech_{model}"] = speech(model, Path(root), args.out, missing)
    if args.bench:
        record["bench"] = bench(args.bench, args.out)
    (args.out / "tables" / "values.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (args.out / "tables" / "missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"{len(missing)} group(s) unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
