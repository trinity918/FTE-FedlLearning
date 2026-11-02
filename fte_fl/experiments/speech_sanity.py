"""Centralized check that a Speech model can learn from the cached features.

    python -m fte_fl.experiments.speech_sanity --model bc_resnet1 --out FILE
        [--clips 4000] [--batch 16] [--lr 0.1] [--steps 400]

Not a federated result.  Trains on pooled clips of the official training split
and reports validation accuracy and seconds per training sample, to tell a
broken pipeline from too few SGD steps, and to size federated runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fte_fl.data.speech_cache import SpeechFeatureCache
from fte_fl.data.speech_commands import SpeechCommandsDataset, build_speech_eval, official_split_files
from fte_fl.fl import metrics
from fte_fl.fl.data import _speech_tensors, _subset, native_path
from fte_fl.fl.models import build_model

DATA = "/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clips", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cache = SpeechFeatureCache("data_cache/speech_logmel")
    root = native_path(DATA)
    train = SpeechCommandsDataset([f"{root}/{rel}" for rel in official_split_files(root)["training"]])
    x, y = _speech_tensors(train, _subset(len(train), args.clips, 0, 0), cache)
    val = build_speech_eval(root, "validation")
    vx, vy = _speech_tensors(val, _subset(len(val), 1000, 0, 1), cache)
    model = build_model(args.model, 35, tuple(x.shape[1:]))
    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    log, t0, seen = [], time.perf_counter(), 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(y), (args.batch,))
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
        seen += args.batch
        if step % 100 == 0 or step == args.steps:
            spent = time.perf_counter() - t0
            acc = metrics.evaluate(model, vx, vy, 256)["accuracy"]
            log.append({"step": step, "train_loss": loss.item(), "val_acc": acc, "s_per_sample": spent / seen})
            print(args.model, log[-1], flush=True)
            args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log}, indent=1))
            t0 = time.perf_counter()
            seen = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
