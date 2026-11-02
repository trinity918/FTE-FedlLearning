"""Single-process federated training simulator.

One round r (1-based):
  1. sample `clients_per_round` participants uniformly without replacement
     (numpy Generator seeded by (seed, SAMPLING, r)); sorted by client index.
  2. each participant starts from the global model, trains locally with SGD
     (fresh optimizer every round, so no momentum carries over; torch seeded
     by (seed, TRAINING, r, client) for shuffling and dropout) on its clean
     data, or on its poisoned data if malicious and r >= attack.start_round;
     update u_i = local vector - global vector (fte_fl.fl.state).
  3. update attacks replace malicious rows (r >= start_round).
  4. aggregation under the configured protocol/defense (fte_fl.fl.aggregation);
     global vector += aggregate (server learning rate 1), then read back
     from the float32 model.
  5. evaluation on the held-out sets, attack metrics, filtering diagnostics.
Every seed is derived from (project seed, purpose, round, client), so a
resumed run recomputes exactly what an uninterrupted run computes.

Outputs in `run_dir` (never overwritten; an existing non-empty run_dir is an
error unless resuming):
  config.json     raw config, its hash, parsed settings, environment,
                  git commit, threat model, CKKS parameters
  manifest.json   client ids and sample counts, malicious clients, poisoning
                  record, evaluation-set sizes
  metrics.jsonl   one JSON object per completed round
  checkpoint.pt   global vector after the last completed round
  summary.json    status ("running", "completed" or "failed"), error text
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fte_fl.config import _git_commit, config_hash
from fte_fl.fl import attacks as fl_attacks
from fte_fl.fl import metrics
from fte_fl.fl.aggregation import aggregate
from fte_fl.fl.config import DATA_ATTACKS, UPDATE_ATTACKS, SimConfig, parse_sim_config
from fte_fl.fl.data import MODEL_INIT, SAMPLING, TRAINING, ClientData, build_federated_data, derive_seed
from fte_fl.fl.models import build_model
from fte_fl.fl.state import copy_excluded, get_vector, set_vector, vector_size

_REPO = Path(__file__).resolve().parents[2]
_PACKAGES = ("numpy", "scipy", "torch", "torchvision", "tenseal", "timm", "PyYAML")


def environment() -> dict:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=_REPO, capture_output=True, text=True)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "torch_threads": torch.get_num_threads(),
        "git_commit": _git_commit(str(_REPO)),
        "git_tracked_changes": None if status.returncode else bool(status.stdout.strip()),
    }


def sample_participants(cfg: SimConfig, round_id: int, malicious=()) -> list[int]:
    """Round participants, sorted.  uniform: clients_per_round of all clients
    without replacement.  stratified: exactly malicious_per_round of the
    malicious clients plus the rest from the honest clients, each without
    replacement, so every round has the same number of malicious participants."""
    rng = np.random.default_rng(derive_seed(cfg.seed, SAMPLING, round_id))
    if cfg.sampling == "uniform":
        return sorted(rng.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False).tolist())
    mal = sorted(malicious)
    honest = sorted(set(range(cfg.num_clients)) - set(mal))
    if not mal:
        raise ValueError("stratified sampling needs the malicious client set")
    picked = rng.choice(mal, size=cfg.malicious_per_round, replace=False).tolist()
    picked += rng.choice(honest, size=cfg.clients_per_round - cfg.malicious_per_round,
                         replace=False).tolist()
    return sorted(picked)


def local_train(model, client: ClientData, cfg: SimConfig, seed: int) -> float:
    """SGD for `local_epochs` epochs; returns the mean minibatch loss (local)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    model.train()
    losses = []
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(len(client), generator=gen)
        for s in range(0, len(client), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(client.x[idx]), client.y[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
    mean = float(np.mean(losses))
    if not math.isfinite(mean):
        raise FloatingPointError(f"client {client.client_id}: local training loss is not finite")
    return mean


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _peak_rss_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):      # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(n, ctypes.c_size_t) for n in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]

        c = Counters(cb=ctypes.sizeof(Counters))
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.PeakWorkingSetSize / (1024.0 * 1024.0)
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: KiB


class Simulation:
    def __init__(self, raw: dict, run_dir: str | Path, resume: bool = False):
        self.raw = raw
        self.cfg = parse_sim_config(raw)
        self.hash = config_hash(raw)
        self.run_dir = Path(run_dir)
        self.resume = resume

    # ── setup ───────────────────────────────────────────────────────────────
    def _prepare_dir(self) -> int:
        """Create or validate run_dir; return the last completed round (0 if new)."""
        d = self.run_dir
        if not self.resume:
            if d.exists() and any(d.iterdir()):
                raise FileExistsError(f"{d} exists and is not empty; refusing to overwrite a run")
            d.mkdir(parents=True, exist_ok=True)
            return 0
        saved = json.loads((d / "config.json").read_text())
        if saved["config_hash"] != self.hash:
            raise ValueError(f"cannot resume {d}: config hash {self.hash} != saved {saved['config_hash']}")
        ckpt = d / "checkpoint.pt"
        done = 0
        if ckpt.exists():
            state = torch.load(ckpt, weights_only=False)
            if state["config_hash"] != self.hash:
                raise ValueError("checkpoint belongs to a different config")
            self._resume_vector = state["vector"].numpy()
            done = int(state["round"])
        # drop the metrics line of the round after the checkpoint (crash between the two
        # writes); anything more means a lost or stale checkpoint: refuse before writing
        lines = (d / "metrics.jsonl").read_text().splitlines() if (d / "metrics.jsonl").exists() else []
        kept = [ln for ln in lines if json.loads(ln)["round"] <= done]
        if [json.loads(ln)["round"] for ln in kept] != list(range(1, done + 1)):
            raise ValueError(f"metrics.jsonl does not hold rounds 1..{done}")
        if len(lines) - len(kept) > 1:
            raise ValueError(f"cannot resume {d}: metrics.jsonl has {len(lines)} rounds but the checkpoint "
                             f"is {'missing' if not ckpt.exists() else f'at round {done}'}")
        (d / "metrics.jsonl").write_text("".join(ln + "\n" for ln in kept))
        return done

    def setup(self, done: int) -> int:
        cfg = self.cfg
        t0 = time.perf_counter()
        self.data = build_federated_data(cfg)
        data_s = time.perf_counter() - t0

        torch.manual_seed(derive_seed(cfg.seed, MODEL_INIT))
        self.model = build_model(cfg.model, self.data.num_classes, self.data.input_shape)
        self.worker = copy.deepcopy(self.model)
        self.global_vec = get_vector(self.model)
        self.dim = vector_size(self.model)
        total, trainable, frozen = self.model.count_parameters()

        atk = cfg.attack
        self.malicious = fl_attacks.choose_malicious(cfg.num_clients, atk)
        self.poisoned: dict[int, ClientData] = {}
        poison_log = []
        if atk.type in DATA_ATTACKS:
            for k, ci in enumerate(self.malicious):
                self.poisoned[ci], rec = fl_attacks.poison_client(
                    self.data.clients[ci], k, ci, cfg.dataset, self.data.num_classes, atk)
                poison_log.append(rec)
        self._build_attack_eval()

        self.key_holder = None
        crypto = None
        if cfg.protocol != "plaintext":
            from fte_fl.crypto.ckks_manager import CKKSManager
            from fte_fl.defenses.protocols import KeyHolder
            t1 = time.perf_counter()
            c = cfg.crypto
            mgr = CKKSManager(c["poly_modulus_degree"], list(c["coeff_mod_bit_sizes"]),
                              float(c["global_scale"]), int(c["chunk_size"]))
            self.key_holder = KeyHolder(mgr)
            crypto = {
                "poly_modulus_degree": c["poly_modulus_degree"],
                "coeff_mod_bit_sizes": list(c["coeff_mod_bit_sizes"]),
                "global_scale": c["global_scale"],
                "chunk_size": c["chunk_size"],
                "ciphertexts_per_update": math.ceil(self.dim / int(c["chunk_size"])),
                "keygen_s": time.perf_counter() - t1,
            }

        if done:
            set_vector(self.model, self._resume_vector)
            self.global_vec = get_vector(self.model)
        else:
            _write_json(self.run_dir / "config.json", {
                "config_hash": self.hash,
                "raw": self.raw,
                "parsed": _asdict(cfg),
                "environment": environment(),
                "model": {"name": cfg.model, "total_params": total, "trainable_params": trainable,
                          "frozen_params": frozen, "aggregated_vector_size": self.dim},
                "crypto": crypto,
                "threat_model": fl_attacks.__doc__,
            })
            _write_json(self.run_dir / "manifest.json", {
                "dataset": self.data.name,
                "data_info": self.data.info,
                "malicious_clients": [[ci, self.data.clients[ci].client_id] for ci in self.malicious],
                "poisoning": poison_log,
                "attack_eval": self.attack_eval_info,
                "data_build_s": data_s,
            })
        _write_json(self.run_dir / "summary.json", {"status": "running", "completed_rounds": done,
                                                    "rounds": cfg.rounds})
        return done

    def _build_attack_eval(self) -> None:
        atk, (x, y) = self.cfg.attack, self.data.test
        self.backdoor_eval = self.flip_eval = None
        info: dict = {}
        if atk.type == "backdoor":
            keep = y != atk.target_class
            xt = fl_attacks.apply_trigger(x[keep], self.cfg.dataset, atk, None)
            self.backdoor_eval = xt
            info = {"backdoor_samples": int(keep.sum()), "trigger": "global (all parts)"}
        elif atk.type == "label_flip":
            keep = y == atk.source_class
            self.flip_eval = x[keep]
            info = {"source_samples": int(keep.sum())}
            if not keep.any():
                raise ValueError("the evaluation set has no source-class samples")
        self.attack_eval_info = info

    # ── one round ───────────────────────────────────────────────────────────
    def client_updates(self, r: int):
        """Steps 1-3 of round r from the current global vector: participants,
        malicious mask, whether the attack is active, the (attacked) update
        matrix, sample counts, local losses and local training times."""
        cfg, atk = self.cfg, self.cfg.attack
        part = sample_participants(cfg, r, self.malicious)
        active = atk.type != "none" and r >= atk.start_round
        is_mal = np.array([ci in self.malicious for ci in part])
        deltas, losses, train_times, counts = [], [], [], []
        for ci in part:
            client = self.data.clients[ci]
            if active and atk.type in DATA_ATTACKS and ci in self.poisoned:
                client = self.poisoned[ci]
            set_vector(self.worker, self.global_vec)
            copy_excluded(self.model, self.worker)
            t = time.perf_counter()
            losses.append(local_train(self.worker, client, cfg, derive_seed(cfg.seed, TRAINING, r, ci)))
            train_times.append(time.perf_counter() - t)
            deltas.append(get_vector(self.worker) - self.global_vec)
            counts.append(len(client))
        deltas = np.stack(deltas)
        if active and atk.type in UPDATE_ATTACKS:
            deltas = fl_attacks.attack_updates(deltas, is_mal, part, r, atk)
        return part, is_mal, active, deltas, np.array(counts), losses, train_times

    def run_round(self, r: int) -> dict:
        cfg, atk = self.cfg, self.cfg.attack
        t_round = time.perf_counter()
        part, is_mal, active, deltas, counts, losses, train_times = self.client_updates(r)

        t = time.perf_counter()
        agg = aggregate(deltas, counts, cfg, r, self.key_holder)
        agg_s = time.perf_counter() - t
        set_vector(self.model, self.global_vec + agg.aggregate)
        self.global_vec = get_vector(self.model)

        t = time.perf_counter()
        bs = cfg.eval_batch_size
        rec: dict = {"round": r}
        rec["test"] = metrics.evaluate(self.model, *self.data.test, bs)
        rec["validation"] = (None if self.data.validation is None
                             else metrics.evaluate(self.model, *self.data.validation, bs))
        if self.backdoor_eval is not None:
            pred = metrics.predictions(self.model, self.backdoor_eval, bs)
            rec["backdoor_asr"] = float((pred == atk.target_class).float().mean())
        if self.flip_eval is not None:
            pred = metrics.predictions(self.model, self.flip_eval, bs)
            rec["label_flip"] = {"flip_rate": float((pred == atk.target_class).float().mean()),
                                 "source_accuracy": float((pred == atk.source_class).float().mean())}
        eval_s = time.perf_counter() - t

        honest = ~is_mal
        rec.update({
            "participants": [self.data.clients[ci].client_id for ci in part],
            "participant_indices": part,
            "malicious_participant_indices": [ci for ci, m in zip(part, is_mal) if m],
            "attack_active": bool(active),
            "local_train_loss_mean_all": float(np.mean(losses)),
            "local_train_loss_mean_honest": float(np.mean(np.array(losses)[honest])) if honest.any() else None,
            "aggregation_weights": agg.weights.tolist(),
            "kept_fraction": agg.kept_fraction.tolist(),
            "attestation_rejected_indices": [part[i] for i in agg.attestation_rejected],
            # assumed Byzantine budget: f (Krum family, Bulyan) or b (trimmed mean)
            "filtering": metrics.filtering_metrics(
                is_mal, agg.weights, agg.kept_fraction,
                cfg.defense.b if cfg.defense.name == "trimmed_mean" else cfg.defense.f),
            "ckks_max_abs_error": agg.ckks_max_abs_error,
            "update_norm_mean": float(np.linalg.norm(deltas, axis=1).mean()),
            "aggregate_norm": float(np.linalg.norm(agg.aggregate)),
            "time_s": {
                "local_train_total": float(sum(train_times)),
                "local_train_max": float(max(train_times)),
                "aggregation_total": agg_s,
                "protocol_steps": agg.timings,
                "evaluation": eval_s,
                "round": time.perf_counter() - t_round,
            },
            "payload_bytes": {
                **agg.traffic,
                "global_model_broadcast_float32": int(len(part) * self.dim * 4),
            },
            "ciphertexts": agg.ciphertexts,
            "peak_rss_mb": _peak_rss_mb(),
        })
        return rec

    # ── driver ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        start = time.perf_counter()
        done = 0
        owns_dir = False     # never write into a run_dir that failed validation
        try:
            done = self._prepare_dir()
            owns_dir = True
            torch.set_num_threads(self.cfg.torch_threads)
            done = self.setup(done)
            for r in range(done + 1, self.cfg.rounds + 1):
                rec = self.run_round(r)
                with open(self.run_dir / "metrics.jsonl", "a") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                tmp = self.run_dir / "checkpoint.pt.tmp"
                torch.save({"round": r, "vector": torch.from_numpy(self.global_vec.copy()),
                            "config_hash": self.hash}, tmp)
                os.replace(tmp, self.run_dir / "checkpoint.pt")
                done = r
                _write_json(self.run_dir / "summary.json",
                            {"status": "running", "completed_rounds": done, "rounds": self.cfg.rounds})
        except BaseException as exc:
            if owns_dir:
                _write_json(self.run_dir / "summary.json", {
                    "status": "failed", "completed_rounds": done, "rounds": self.cfg.rounds,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                    "elapsed_s_this_invocation": time.perf_counter() - start,
                })
            raise
        summary = {"status": "completed", "completed_rounds": done, "rounds": self.cfg.rounds,
                   "elapsed_s_this_invocation": time.perf_counter() - start,
                   "resumed": self.resume, "environment_at_completion": environment()}
        _write_json(self.run_dir / "summary.json", summary)
        return summary


def _asdict(cfg: SimConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def run_simulation(raw: dict, run_dir, resume: bool = False) -> dict:
    return Simulation(raw, run_dir, resume).run()
