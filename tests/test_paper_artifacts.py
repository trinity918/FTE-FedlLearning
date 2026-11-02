"""Paper tables must show what happened: unsupported pairs and diverged runs are not missing cells."""
import json

from fte_fl.eval import paper_artifacts


def _bench_row(protocol, rule, wall, **extra):
    return {"protocol": protocol, "rule": rule, "aggregation_wall_s": wall, "payload_bytes": {"a": 1_000_000, "b": 2_000_000},
            "selection_matches_plaintext": True, "max_abs_diff_vs_plaintext": 1e-7, "malicious_selected": 0, **extra}


def test_bench_table_reports_every_pair_and_skips_validation_runs(tmp_path):
    root, out = tmp_path / "bench", tmp_path / "paper"
    (out / "tables").mkdir(parents=True)
    rows = [_bench_row("fte", "krum", 10.0), _bench_row("fte", "krum", 14.0),
            {"protocol": "etf", "rule": "krum", "unsupported": "needs comparison"}]
    for name in ("femnist", "validation_femnist"):
        (root / name).mkdir(parents=True)
        (root / name / "bench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = paper_artifacts.bench(root, out)
    assert set(record) == {"femnist"}
    assert record["femnist"]["etf/krum"] == "unsupported"
    assert record["femnist"]["fte/krum"]["selection_matches"] == 2
    assert record["femnist"]["fte/krum"]["traffic_mb"] == 3.0
    text = (out / "tables" / "bench_femnist.tex").read_text()
    assert "12.0 $\\pm$ 2.8" in text and "unsupported" in text and "2/2" in text
    assert not (out / "tables" / "bench_validation_femnist.tex").exists()


def test_speech_table_marks_diverged_runs(tmp_path):
    root, out = tmp_path / "grid", tmp_path / "paper"
    (out / "tables").mkdir(parents=True)
    (root / "_full_eval").mkdir(parents=True)
    (root / "_full_eval" / "honest__fedavg__seed0.json").write_text(json.dumps({"testing": {"accuracy": 0.5}}))
    failed = root / "signflip__multikrum" / "seed0"
    failed.mkdir(parents=True)
    (failed / "summary.json").write_text(json.dumps({"status": "failed", "completed_rounds": 0,
                                                     "error": "FloatingPointError: evaluation loss is not finite"}))
    record = paper_artifacts.speech("bcres", root, out, [])
    assert record["diverged_at_round"] == {"signflip__multikrum": [1]}
    assert "div.\\ round 1" in (out / "tables" / "speech_bcres_accuracy.tex").read_text()
