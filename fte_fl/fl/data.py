"""In-memory federated datasets for the simulator.

Client data comes only from each dataset's training partition; evaluation
data only from its held-out partition:
  synthetic        Gaussian class clusters generated from the seed; IID
                   clients; separate generated test set; no validation set.
  femnist          LEAF per-writer train JSONs -> clients (one writer each,
                   `build_femnist_clients`); test = all writers' LEAF test
                   samples (LEAF splits per sample, so writers overlap with
                   the training clients; samples do not).  No validation set.
  speech_commands  official training split, one speaker per client
                   (`build_speech_clients`); validation / test = the official
                   validation / testing lists (speaker-disjoint).
  cifar100         official train set partitioned iid / dirichlet /
                   superclass; test = official test set.  No validation set.
Inputs are unnormalized: images in [0, 1] (CIFAR-100 uint8 / 255, HWC ->
CHW), log-mel features as produced by the loader.  Triggers therefore apply
directly (see fte_fl.attacks.triggers).

`max_samples_per_client` and `experiment.max_eval_samples` take seeded
random subsets (recorded in `info`) for smoke tests; they change the data
and must be reported with any result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fte_fl.fl.config import SimConfig

# Fixed by fte_fl.data.speech_commands.wav_to_logmel (SpeechCommandsDataset uses its defaults).
SPEECH_FEATURES = {"n_mels": 64, "n_fft": 400, "hop_length": 160, "sample_rate": 16000,
                   "clip_duration": 1.0}

_DATASET_KEYS = {
    "synthetic": {"num_clients", "num_classes", "samples_per_client", "test_samples", "dim",
                  "separation"},
    "femnist": {"train_dir", "test_dir", "num_clients", "max_samples_per_client"},
    "speech_commands": {"data_dir", "num_clients", "max_samples_per_client", *SPEECH_FEATURES},
    "cifar100": {"cache_dir", "num_clients", "partition", "alpha", "max_samples_per_client"},
}

# seed-derivation purposes (see derive_seed)
SAMPLING, TRAINING, MODEL_INIT, GAUSSIAN, POISON, MALICIOUS, SUBSET, SYNTHETIC = range(8)


def derive_seed(*keys: int) -> int:
    """A 64-bit seed determined by the integer keys (numpy SeedSequence)."""
    return int(np.random.SeedSequence(list(keys)).generate_state(1, dtype=np.uint64)[0])


@dataclass
class ClientData:
    client_id: str
    x: torch.Tensor          # float32 (n, *input_shape)
    y: torch.Tensor          # int64 (n,)

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass
class FederatedData:
    name: str
    clients: list[ClientData]
    test: tuple[torch.Tensor, torch.Tensor]
    validation: tuple[torch.Tensor, torch.Tensor] | None
    num_classes: int
    input_shape: tuple[int, ...]
    info: dict = field(default_factory=dict)


def _subset(n: int, k: int | None, seed: int, purpose_key: int) -> np.ndarray:
    """Sorted seeded subset of range(n) of size min(k, n) (all indices if k is None)."""
    if k is None or k >= n:
        return np.arange(n)
    rng = np.random.default_rng(derive_seed(seed, SUBSET, purpose_key))
    return np.sort(rng.choice(n, size=k, replace=False))


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _finish(name, clients, test, validation, num_classes, cfg: SimConfig, info) -> FederatedData:
    empty = [c.client_id for c in clients if len(c) == 0]
    if empty:
        raise ValueError(f"{len(empty)} clients have no training samples (e.g. {empty[0]!r}); "
                         "choose another partition seed/alpha or fewer clients")
    for split, xy in (("test", test), ("validation", validation)):
        if xy is not None and xy[1].numel() == 0:
            raise ValueError(f"the {split} set is empty")
    counts = [[c.client_id, len(c)] for c in clients]
    info = dict(info)
    info.update(
        num_clients=len(clients),
        client_samples=counts,
        train_samples=int(sum(len(c) for c in clients)),
        test_samples=int(test[1].numel()),
        validation_samples=None if validation is None else int(validation[1].numel()),
        partition_id=_digest(counts),
    )
    return FederatedData(name, clients, test, validation, num_classes,
                         tuple(test[0].shape[1:]), info)


def _limit(cid, x, y, k, seed, idx) -> ClientData:
    keep = _subset(len(y), k, seed, 1000 + idx)
    return ClientData(cid, x[keep], y[keep])


def _eval_subset(x, y, cfg: SimConfig, key: int):
    keep = torch.from_numpy(_subset(len(y), cfg.max_eval_samples, cfg.seed, key))
    return x[keep], y[keep]


def _synthetic(cfg: SimConfig, p: dict) -> FederatedData:
    rng = np.random.default_rng(derive_seed(cfg.seed, SYNTHETIC))
    c, dim = cfg.num_classes, int(p["dim"])
    means = rng.normal(0.0, float(p["separation"]), size=(c, dim))

    def draw(n):
        y = rng.integers(0, c, size=n)
        x = means[y] + rng.normal(size=(n, dim))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    clients = [ClientData(f"syn{i:04d}", *draw(int(p["samples_per_client"])))
               for i in range(cfg.num_clients)]
    test = _eval_subset(*draw(int(p["test_samples"])), cfg, 0)
    return _finish("synthetic", clients, test, None, c, cfg, {"generator": "gaussian_clusters"})


def _femnist(cfg: SimConfig, p: dict) -> FederatedData:
    from fte_fl.data.femnist import _load_all_users, build_femnist_clients

    raw = build_femnist_clients(p["train_dir"], num_clients=cfg.num_clients, seed=cfg.seed)
    if len(raw) < cfg.num_clients:
        raise ValueError(f"FEMNIST has only {len(raw)} writers, num_clients={cfg.num_clients}")
    clients = []
    for i, (wid, ds) in enumerate(raw.items()):
        x = torch.from_numpy(ds.x).reshape(-1, 1, 28, 28)
        clients.append(_limit(wid, x, torch.from_numpy(ds.y), p.get("max_samples_per_client"),
                              cfg.seed, i))
    users = _load_all_users(p["test_dir"])
    xs = [np.asarray(users[u]["x"], dtype=np.float32) for u in sorted(users)]
    ys = [np.asarray(users[u]["y"], dtype=np.int64) for u in sorted(users)]
    test = _eval_subset(torch.from_numpy(np.concatenate(xs)).reshape(-1, 1, 28, 28),
                        torch.from_numpy(np.concatenate(ys)), cfg, 0)
    return _finish("femnist", clients, test, None, 62, cfg,
                   {"train_dir": p["train_dir"], "test_dir": p["test_dir"],
                    "test_writers": len(users)})


def _speech_tensors(ds, indices) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [ds[int(i)][0] for i in indices]
    ys = [ds.labels[int(i)] for i in indices]
    return torch.stack(xs), torch.tensor(ys, dtype=torch.int64)


def _speech(cfg: SimConfig, p: dict) -> FederatedData:
    from fte_fl.data.speech_commands import build_speech_clients, build_speech_eval

    for key, value in SPEECH_FEATURES.items():
        if key in p and p[key] != value:
            raise ValueError(f"datasets.speech_commands.{key}={p[key]!r} is not supported; "
                             f"the loader computes features with {key}={value!r}")
    raw = build_speech_clients(p["data_dir"], num_clients=cfg.num_clients, seed=cfg.seed)
    if len(raw) < cfg.num_clients:
        raise ValueError(f"only {len(raw)} training speakers, num_clients={cfg.num_clients}")
    clients = []
    for i, (sid, ds) in enumerate(raw.items()):
        keep = _subset(len(ds), p.get("max_samples_per_client"), cfg.seed, 1000 + i)
        clients.append(ClientData(sid, *_speech_tensors(ds, keep)))
    evals = {}
    for key, split in ((0, "testing"), (1, "validation")):
        ds = build_speech_eval(p["data_dir"], split)
        evals[split] = _speech_tensors(ds, _subset(len(ds), cfg.max_eval_samples, cfg.seed, key))
    return _finish("speech_commands", clients, evals["testing"], evals["validation"], 35, cfg,
                   {"data_dir": p["data_dir"], "split": "official validation/testing lists"})


def _cifar100(cfg: SimConfig, p: dict) -> FederatedData:
    from torchvision.datasets import CIFAR100

    from fte_fl.data.cifar100 import dirichlet_partition, iid_partition, superclass_partition

    def load(train):
        ds = CIFAR100(root=p["cache_dir"], train=train, download=False)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
        return x, torch.tensor(ds.targets, dtype=torch.int64)

    x, y = load(True)
    partition = p.get("partition")
    if partition == "iid":
        if "alpha" in p:
            raise ValueError("alpha applies only to the dirichlet partition")
        parts = iid_partition(len(y), cfg.num_clients, seed=cfg.seed)
    elif partition == "dirichlet":
        parts = dirichlet_partition(y.numpy(), cfg.num_clients, float(p["alpha"]), 100, seed=cfg.seed)
    elif partition == "superclass":
        if "alpha" in p:
            raise ValueError("alpha applies only to the dirichlet partition")
        parts = superclass_partition(y.numpy(), cfg.num_clients, seed=cfg.seed)
    else:
        raise ValueError(f"datasets.cifar100.partition must be iid, dirichlet or superclass, "
                         f"got {partition!r}")
    clients = []
    for i, idx in enumerate(parts):
        t = torch.from_numpy(idx)
        clients.append(_limit(f"c{i:04d}", x[t], y[t], p.get("max_samples_per_client"), cfg.seed, i))
    test = _eval_subset(*load(False), cfg, 0)
    return _finish("cifar100", clients, test, None, 100, cfg,
                   {"cache_dir": p["cache_dir"], "partition": partition, "alpha": p.get("alpha")})


def build_federated_data(cfg: SimConfig) -> FederatedData:
    p = cfg.dataset_params
    unknown = set(p) - _DATASET_KEYS[cfg.dataset]
    if unknown:
        raise ValueError(f"unknown key(s) in datasets.{cfg.dataset}: {sorted(unknown)}")
    if p.get("max_samples_per_client") is not None:
        k = p["max_samples_per_client"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"max_samples_per_client must be a positive int, got {k!r}")
    builder = {"synthetic": _synthetic, "femnist": _femnist, "speech_commands": _speech,
               "cifar100": _cifar100}[cfg.dataset]
    data = builder(cfg, p)
    data.info["max_samples_per_client"] = p.get("max_samples_per_client")
    data.info["max_eval_samples"] = cfg.max_eval_samples
    return data
