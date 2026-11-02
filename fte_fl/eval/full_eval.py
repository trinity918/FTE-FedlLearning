"""Score the final checkpoints of Speech Commands runs on the full test split.

    python -m fte_fl.eval.full_eval <grid output_dir> [--only SUBSTRING]

Speech runs evaluate each round on a seeded subset (`max_eval_samples`).  This
reloads a completed run's final global vector and evaluates clean accuracy on
all testing clips, plus backdoor ASR or label-flip rate when the run has that
attack.  Validation is not rescored: it only selects hyperparameters.  Results go to <output_dir>/_full_eval/<group>__seed<N>.json;
a result that already exists is left alone.

The frozen weights of a LoRA backbone are not in the checkpoint, so the model is
rebuilt from the run's seed as the simulator does.  Before anything is written,
the run's last in-loop subset accuracy must be reproduced exactly: a mismatch
means this is not the model that was trained, and the run is reported instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fte_fl.data.speech_cache import SpeechFeatureCache
from fte_fl.data.speech_commands import build_speech_eval
from fte_fl.fl import attacks as fl_attacks
from fte_fl.fl import metrics
from fte_fl.fl.config import parse_sim_config
from fte_fl.fl.data import MODEL_INIT, _speech_tensors, _subset, derive_seed, native_path
from fte_fl.fl.models import build_model
from fte_fl.fl.state import set_vector

def load_model(cfg, run_dir: Path, input_shape: tuple):
    torch.manual_seed(derive_seed(cfg.seed, MODEL_INIT))
    model = build_model(cfg.model, 35, input_shape)
    state = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    set_vector(model, state["vector"].numpy())
    return model, state["round"]


def attack_metrics(model, x: torch.Tensor, y: torch.Tensor, cfg) -> dict:
    atk, bs = cfg.attack, cfg.eval_batch_size
    if atk.type == "backdoor":
        triggered = fl_attacks.apply_trigger(x[y != atk.target_class], cfg.dataset, atk, None)
        pred = metrics.predictions(model, triggered, bs)
        return {"backdoor_asr": float((pred == atk.target_class).float().mean()), "backdoor_samples": len(pred)}
    if atk.type == "label_flip":
        pred = metrics.predictions(model, x[y == atk.source_class], bs)
        return {"flip_rate": float((pred == atk.target_class).float().mean()),
                "source_accuracy": float((pred == atk.source_class).float().mean()), "source_samples": len(pred)}
    return {}


def evaluate_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary["status"] != "completed":
        raise ValueError(f"{run_dir}: status {summary['status']}")
    cfg = parse_sim_config(config["raw"])
    if cfg.dataset != "speech_commands":
        raise ValueError(f"{run_dir}: dataset {cfg.dataset}")
    torch.set_num_threads(cfg.torch_threads)
    params = cfg.dataset_params
    cache = SpeechFeatureCache(params["feature_cache"])
    last = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    result: dict = {"run": str(run_dir), "rounds": last["round"]}
    ds = build_speech_eval(native_path(params["data_dir"]), "testing")
    x, y = _speech_tensors(ds, range(len(ds)), cache)
    model, ckpt_round = load_model(cfg, run_dir, tuple(x.shape[1:]))
    if ckpt_round != last["round"]:
        raise ValueError(f"{run_dir}: checkpoint at round {ckpt_round}, metrics end at {last['round']}")
    index = torch.as_tensor(_subset(len(ds), cfg.max_eval_samples, cfg.seed, 0), dtype=torch.long)
    replay = metrics.evaluate(model, x[index], y[index], cfg.eval_batch_size)["accuracy"]
    if replay != last["test"]["accuracy"]:
        raise ValueError(f"{run_dir}: subset accuracy {replay} != recorded {last['test']['accuracy']}")
    result["testing"] = {"clips": len(y), **metrics.evaluate(model, x, y, cfg.eval_batch_size)}
    result.update(attack_metrics(model, x, y, cfg))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    root = Path(args.output_dir)
    out = root / "_full_eval"
    out.mkdir(exist_ok=True)
    problems = 0
    for run_dir in sorted(p for p in root.glob("*/seed*") if p.is_dir() and p.name[4:].isdigit()):
        name = f"{run_dir.parent.name}__{run_dir.name}"
        target = out / f"{name}.json"
        if target.exists() or (args.only and args.only not in name):
            continue
        try:
            result = evaluate_run(run_dir)
        except ValueError as exc:
            print("PROBLEM:", exc, flush=True)
            problems += 1
            continue
        target.write_text(json.dumps(result, indent=2))
        print(name, f"test {result['testing']['accuracy']:.4f}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
