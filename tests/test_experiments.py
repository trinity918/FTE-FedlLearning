"""Tests for the grid runner (fte_fl.experiments.grid) and the analysis (fte_fl.eval.analyze)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fte_fl.eval import analyze
from fte_fl.experiments import grid

REAL_MEMORY_MB = grid.memory_mb

BASE = {
    "project": {"seed": 0},
    "experiment": {"dataset": "synthetic", "model": "mlp", "rounds": 3, "clients_per_round": 4,
                   "local_epochs": 1, "batch_size": 8, "optimizer": "sgd", "lr": 0.1},
    "defense": {"name": "fedavg"},
    "attack": {"type": "none"},
    "datasets": {"synthetic": {"num_clients": 6, "num_classes": 3, "samples_per_client": 12,
                               "test_samples": 60, "dim": 4, "separation": 2.0}},
}


PLENTY = {"guest": 64000.0, "host": 64000.0}
FAST = ["--settle-s", "0", "--poll-s", "0.01", "--sample-s", "0"]


@pytest.fixture(autouse=True)
def plenty_of_memory(monkeypatch):
    """Grid tests must not depend on this machine's free memory (tested separately)."""
    monkeypatch.setattr(grid, "memory_mb", lambda: dict(PLENTY))


def run_grid(argv):
    return grid.main(argv + FAST)


def finished(out):
    """Grid-log entries for finished or refused jobs (launch / memory events excluded)."""
    return [e for e in map(json.loads, (out / "_grid_log.jsonl").read_text().splitlines()) if "exit" in e]


def _spec(tmp_path, runs=None, seeds=(0, 1)):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(BASE))
    spec = {"base_config": str(tmp_path / "base.yaml"), "output_dir": str(tmp_path / "out"),
            "seeds": list(seeds), "common": {"experiment.rounds": 3},
            "runs": runs or {"fedavg": {}, "krum": {"defense.name": "krum", "defense.f": 0}}}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path, spec


def test_expand_jobs_and_overrides(tmp_path):
    _, spec = _spec(tmp_path)
    jobs = grid.expand(spec)
    assert [(j["name"], j["seed"]) for j in jobs] == [("fedavg", 0), ("fedavg", 1), ("krum", 0), ("krum", 1)]
    assert jobs[3]["overrides"] == {"experiment.rounds": 3, "defense.name": "krum", "defense.f": 0,
                                    "project.seed": 1}
    assert Path(jobs[3]["run_dir"]).parts[-3:] == ("out", "krum", "seed1")
    cmd = grid.command(spec, jobs[3], resume=False)
    assert "defense.name=\"krum\"" in cmd and "defense.f=0" in cmd and "--resume" not in cmd
    with pytest.raises(ValueError, match="unknown grid spec"):
        grid.expand({**spec, "extra": 1})
    with pytest.raises(ValueError, match="missing 'seeds'"):
        grid.expand({k: v for k, v in spec.items() if k != "seeds"})


def test_grid_runs_skips_completed_and_analysis_matches(tmp_path):
    path, spec = _spec(tmp_path)
    assert run_grid([str(path), "--jobs", "2"]) == 0
    out = tmp_path / "out"
    log = finished(out)
    assert sorted((e["name"], e["seed"], e["exit"]) for e in log) == [
        ("fedavg", 0, 0), ("fedavg", 1, 0), ("krum", 0, 0), ("krum", 1, 0)]
    assert run_grid([str(path)]) == 0                    # everything skipped
    assert len(finished(out)) == 4

    rows, problems = analyze.analyze(out, out / "_analysis", last_k=2, plots=True)
    assert problems == [] and [r["group"] for r in rows] == ["fedavg", "krum"]
    accs = [[json.loads(x)["test"]["accuracy"]
             for x in (out / "fedavg" / f"seed{s}" / "metrics.jsonl").read_text().splitlines()]
            for s in (0, 1)]
    final = np.array([a[-1] for a in accs])
    mean, std = rows[0]["final_test_accuracy"]
    assert mean == pytest.approx(final.mean()) and std == pytest.approx(final.std(ddof=1))
    lastk = np.array([np.mean(a[-2:]) for a in accs])
    assert rows[0]["last2_test_accuracy"][0] == pytest.approx(lastk.mean())
    assert rows[0]["seeds"] == 2 and rows[1]["defense"] == "krum"
    assert "final_backdoor_asr" not in rows[0]
    assert (out / "_analysis" / "fedavg.png").exists() and (out / "_analysis" / "summary.md").exists()


def test_analysis_reports_incomplete_failed_and_inconsistent_groups(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}, "b": {}, "c": {}})
    assert run_grid([str(path), "--jobs", "3"]) == 0
    out = tmp_path / "out"
    # a: truncated metrics
    m = out / "a" / "seed1" / "metrics.jsonl"
    m.write_text("\n".join(m.read_text().splitlines()[:2]) + "\n")
    # b: a failed run
    s = out / "b" / "seed0" / "summary.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "status": "failed", "error": "boom"}))
    # c: seeds whose configs differ beyond the seed
    c = out / "c" / "seed1" / "config.json"
    cfg = json.loads(c.read_text())
    cfg["raw"]["experiment"]["lr"] = 0.5
    c.write_text(json.dumps(cfg))
    rows, problems = analyze.analyze(out, out / "_analysis", plots=False)
    assert rows == []
    assert len(problems) == 3
    assert "rounds 1..3" in problems[0] and "status failed" in problems[1] and "differs" in problems[2]
    assert analyze.main([str(out), "--no-plots"]) == 1


def test_analysis_rejects_non_finite_metrics_and_bad_checkpoints(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}, "b": {}, "c": {}, "d": {}}, seeds=(0,))
    assert run_grid([str(path), "--jobs", "4"]) == 0
    out = tmp_path / "out"
    # a: NaN test accuracy in one round
    m = out / "a" / "seed0" / "metrics.jsonl"
    rows = [json.loads(x) for x in m.read_text().splitlines()]
    rows[1]["test"]["accuracy"] = float("nan")
    m.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # b: no checkpoint; c: checkpoint of an earlier round; d: non-finite vector
    (out / "b" / "seed0" / "checkpoint.pt").unlink()
    ckpt = out / "c" / "seed0" / "checkpoint.pt"
    state = torch.load(ckpt, weights_only=False)
    torch.save({**state, "round": 2}, ckpt)
    ckpt = out / "d" / "seed0" / "checkpoint.pt"
    state = torch.load(ckpt, weights_only=False)
    state["vector"][0] = float("inf")
    torch.save(state, ckpt)
    found, problems = analyze.analyze(out, out / "_analysis", plots=False)
    assert found == [] and len(problems) == 4
    assert "test_accuracy is nan in round 2" in problems[0]
    assert "no checkpoint" in problems[1]
    assert "at round 2" in problems[2]
    assert "not finite" in problems[3]


def test_grid_sets_aside_early_failures_and_analysis_ignores_them(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0,))
    run = tmp_path / "out" / "a" / "seed0"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"status": "failed", "error": "OSError: memory"}))
    assert grid.state(run) == "failed_early"
    assert run_grid([str(path)]) == 0
    kept = [p for p in (tmp_path / "out" / "a").iterdir() if ".failed-" in p.name]
    assert len(kept) == 1 and "memory" in (kept[0] / "summary.json").read_text()
    assert grid.state(run) == "completed"
    rows, problems = analyze.analyze(tmp_path / "out", tmp_path / "an", plots=False)
    assert problems == [] and rows[0]["seeds"] == 1


def test_grid_does_not_rerun_a_recorded_failure(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0,))
    assert run_grid([str(path)]) == 0
    run = tmp_path / "out" / "a" / "seed0"
    s = run / "summary.json"
    failed = {**json.loads(s.read_text()), "status": "failed", "error": "FloatingPointError: diverged"}
    s.write_text(json.dumps(failed))
    before = (run / "metrics.jsonl").read_text()
    assert grid.state(run) == "failed"
    assert run_grid([str(path)]) == 1
    assert json.loads(s.read_text()) == failed and (run / "metrics.jsonl").read_text() == before
    assert len(finished(tmp_path / "out")) == 1


def test_grid_launches_nothing_when_memory_is_low(tmp_path, monkeypatch):
    path, spec = _spec(tmp_path)
    # host short, guest plentiful: the host figure must decide
    monkeypatch.setattr(grid, "memory_mb", lambda: {"guest": 64000.0, "host": 1500.0})
    assert run_grid([str(path), "--jobs", "2", "--job-mb", "1000", "--min-available-mb", "1000"]) == 1
    log = finished(tmp_path / "out")
    assert len(log) == 4 and {e["start_state"] for e in log} == {"not_started_low_memory"}
    assert not any((tmp_path / "out").glob("*/seed*"))


def test_grid_launches_nothing_when_memory_cannot_be_read(tmp_path, monkeypatch):
    path, spec = _spec(tmp_path)

    def broken():
        raise RuntimeError("cannot query Windows host memory: exit 1")
    monkeypatch.setattr(grid, "memory_mb", broken)
    assert run_grid([str(path), "--jobs", "2"]) == 1
    entries = [json.loads(x) for x in (tmp_path / "out" / "_grid_log.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "memory_unknown" and "exit 1" in e["error"] for e in entries)
    assert {e["start_state"] for e in finished(tmp_path / "out")} == {"not_started_memory_unknown"}
    assert not any((tmp_path / "out").glob("*/seed*"))


def test_grid_waits_for_a_running_job_when_memory_is_short(tmp_path, monkeypatch):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0, 1))
    # first launch fits; then 3 short readings while job 1 runs (wait, don't give up); then it fits
    readings = [{"guest": 64000.0, "host": 3000.0}] + [{"guest": 64000.0, "host": 1500.0}] * 3
    calls = []

    def reading():
        calls.append(1)
        return readings[len(calls) - 1] if len(calls) <= len(readings) else {"guest": 64000.0, "host": 3000.0}
    monkeypatch.setattr(grid, "memory_mb", reading)
    assert run_grid([str(path), "--jobs", "2", "--job-mb", "1000", "--min-available-mb", "1000"]) == 0
    entries = [json.loads(x) for x in (tmp_path / "out" / "_grid_log.jsonl").read_text().splitlines()]
    launches = [e for e in entries if e.get("event") == "launch"]
    assert [e["memory_mb"]["host"] for e in launches] == [3000.0, 3000.0] and len(calls) == 5
    assert sorted(e["exit"] for e in finished(tmp_path / "out")) == [0, 0]


@pytest.mark.parametrize("damage", ["missing_checkpoint", "stale_checkpoint", "corrupt_checkpoint"])
def test_grid_refuses_to_resume_from_a_damaged_checkpoint_without_touching_files(tmp_path, damage):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0,))
    assert run_grid([str(path)]) == 0
    run = tmp_path / "out" / "a" / "seed0"
    s = run / "summary.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "status": "running"}))   # as if killed
    if damage == "missing_checkpoint":
        (run / "checkpoint.pt").unlink()
    elif damage == "stale_checkpoint":           # checkpoint of round 1, metrics of rounds 1-3
        import torch
        state = torch.load(run / "checkpoint.pt", weights_only=False)
        torch.save({**state, "round": 1}, run / "checkpoint.pt")
    else:
        (run / "checkpoint.pt").write_bytes(b"not a checkpoint")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    assert grid.state(run) == "resume"
    assert run_grid([str(path)]) == 1
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before          # nothing rewritten
    assert finished(tmp_path / "out")[-1]["exit"] == 1
    err = (tmp_path / "out" / "a" / "seed0.out").read_text()
    assert ("cannot resume" in err) if damage != "corrupt_checkpoint" else ("Error" in err)


def test_grid_writes_a_memory_trace(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0,))
    assert grid.main([str(path), "--settle-s", "0", "--sample-s", "0.05"]) == 0
    trace = [json.loads(x) for x in (tmp_path / "out" / "_memory_log.jsonl").read_text().splitlines()]
    assert len(trace) >= 2 and all(e["memory_mb"] == PLENTY for e in trace)
    assert max(e["running"] for e in trace) == 1


def test_memory_reading_on_this_machine():
    mem = REAL_MEMORY_MB()                      # the real readings (the autouse fixture fakes grid.memory_mb)
    if sys.platform == "win32":
        assert mem["guest"] is None and mem["host"] > 0
        return
    assert mem["guest"] > 0
    if grid.under_wsl():
        assert mem["host"] > 0                  # Windows host figure via powershell.exe
    else:
        assert mem["host"] is None


def test_protocol_bench_on_identical_updates(tmp_path):
    from fte_fl.experiments import protocol_bench
    base = {**BASE, "crypto": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60],
                               "global_scale": 2**40, "chunk_size": 4096}}
    base["experiment"] = {**base["experiment"], "clients_per_round": 7}
    base["datasets"] = {"synthetic": {**base["datasets"]["synthetic"], "num_clients": 9}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    assert protocol_bench.main(["--config", str(tmp_path / "base.yaml"), "--out", str(tmp_path / "b"),
                                "--rounds", "1", "2", "--budget", "1"]) == 0
    recs = [json.loads(x) for x in (tmp_path / "b" / "bench.jsonl").read_text().splitlines()]
    assert len(recs) == 2 * 4 * 5
    unsupported = {(r["protocol"], r["rule"]) for r in recs if "unsupported" in r}
    assert unsupported == {("etf", "trimmed_mean"), ("etf", "bulyan")}
    for r in recs:
        if "unsupported" in r:
            assert "supports only" in r["unsupported"]
            continue
        assert r["selection_matches_plaintext"] is True
        assert r["max_abs_diff_vs_plaintext"] < 1e-5
        if r["protocol"] != "plaintext":
            assert r["ckks_max_abs_error"] < 1e-5 and r["payload_bytes"]["encrypted_aggregate"] > 0
    by_round = {}
    for r in recs:
        by_round.setdefault(r["round"], set()).add(tuple(r["participants"]))
    assert all(len(v) == 1 for v in by_round.values())          # identical inputs per round
    assert by_round[1] != by_round[2]


def test_grid_resumes_a_partial_run(tmp_path):
    path, spec = _spec(tmp_path, runs={"a": {}}, seeds=(0,))
    assert run_grid([str(path)]) == 0
    run = tmp_path / "out" / "a" / "seed0"
    full = (run / "metrics.jsonl").read_text()
    # simulate a crash before the first checkpoint, after round 1's metrics line
    (run / "checkpoint.pt").unlink()
    (run / "metrics.jsonl").write_text(full.splitlines(keepends=True)[0])
    summary = json.loads((run / "summary.json").read_text())
    (run / "summary.json").write_text(json.dumps({**summary, "status": "running"}))
    assert grid.state(run) == "resume"
    assert run_grid([str(path)]) == 0
    entry = finished(tmp_path / "out")[-1]
    assert entry["start_state"] == "resume" and entry["exit"] == 0
    strip = lambda t: [{k: v for k, v in json.loads(x).items() if k not in ("time_s", "peak_rss_mb")}
                       for x in t.splitlines()]
    assert strip((run / "metrics.jsonl").read_text()) == strip(full)
