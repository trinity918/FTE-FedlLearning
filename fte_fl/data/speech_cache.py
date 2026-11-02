"""Precomputed log-mel features for Speech Commands.

    python -m fte_fl.data.speech_cache --data-dir DIR --out DIR [--workers 3]

`wav_to_logmel` costs ~28 ms per clip, which dominates a Speech Commands run
(~25 min of data loading per run at 800 speakers).  This module stores its
exact output, one .npy shard per command word, so simulations read the same
values from disk.  It changes no feature: `--verify` and the tests compare
cached values with `wav_to_logmel` bit for bit.

A run opts in with `datasets.speech_commands.feature_cache: <dir>`.  Shards
already on disk are skipped, so an interrupted build continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from pathlib import Path

import numpy as np
import torch

from fte_fl.data.speech_commands import SPEECH_LABELS, _list_clips, wav_to_logmel

_INDEX = "index.json"


def _build_word(job: tuple[str, str, list[str]]) -> tuple[str, int]:
    data_dir, out, names = job
    torch.set_num_threads(1)
    word = names[0].split("/")[0]
    target = Path(out) / f"{word}.npy"
    if target.exists():
        return word, 0
    features = np.stack([wav_to_logmel(str(Path(data_dir) / rel)).squeeze(0).numpy() for rel in names])
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, features)
    tmp.replace(target)
    return word, len(names)


def build_cache(data_dir: str, out: str, workers: int = 1) -> dict:
    """Write one shard per word plus index.json; returns the index."""
    Path(out).mkdir(parents=True, exist_ok=True)
    clips = _list_clips(Path(data_dir))
    by_word: dict[str, list[str]] = {}
    for rel in clips:
        by_word.setdefault(rel.split("/")[0], []).append(rel)
    missing = set(SPEECH_LABELS) - set(by_word)
    if missing:
        raise ValueError(f"no clips for {sorted(missing)}")
    jobs = [(data_dir, out, by_word[w]) for w in sorted(by_word)]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            for word, n in pool.imap_unordered(_build_word, jobs):
                print(f"{word}: {n} clips", flush=True)
    else:
        for job in jobs:
            print(_build_word(job), flush=True)
    index = {"shape": [64, 101], "dtype": "float32", "clips": by_word}
    (Path(out) / _INDEX).write_text(json.dumps(index))
    return index


class SpeechFeatureCache:
    """Read-only lookup of cached log-mel features by clip path."""

    def __init__(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        index = json.loads((cache_dir / _INDEX).read_text())
        self._where = {rel: (word, i) for word, rels in index["clips"].items() for i, rel in enumerate(rels)}
        self._shards = {word: np.load(cache_dir / f"{word}.npy", mmap_mode="r") for word in index["clips"]}

    def __len__(self) -> int:
        return len(self._where)

    def get(self, filepath: str) -> torch.Tensor:
        """Features of a clip path, shape (1, 64, 101), as `wav_to_logmel` returns them."""
        path = Path(filepath)
        word, i = self._where[f"{path.parent.name}/{path.name}"]
        return torch.from_numpy(np.array(self._shards[word][i])).unsqueeze(0)


def verify_cache(data_dir: str, out: str, samples: int, seed: int = 0) -> int:
    """Compare `samples` random clips with `wav_to_logmel`; raises on any difference.

    Shards are built with one torch thread and the FFT's rounding depends on
    the thread count (measured: differences up to 1e-6 at 4 or 8 threads), so
    the comparison also uses one thread."""
    torch.set_num_threads(1)
    cache = SpeechFeatureCache(out)
    clips = _list_clips(Path(data_dir))
    if len(cache) != len(clips):
        raise ValueError(f"cache holds {len(cache)} clips, the dataset {len(clips)}")
    for rel in random.Random(seed).sample(clips, samples):
        path = str(Path(data_dir) / rel)
        if not torch.equal(cache.get(path), wav_to_logmel(path)):
            raise ValueError(f"cached features of {rel} differ from wav_to_logmel")
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify", type=int, default=500, help="random clips to compare after building")
    args = parser.parse_args(argv)
    build_cache(args.data_dir, args.out, args.workers)
    print(f"verified {verify_cache(args.data_dir, args.out, args.verify)} clips bit-identical", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
