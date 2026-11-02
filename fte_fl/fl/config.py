"""Simulator configuration: parse the YAML dict and reject invalid settings.

`parse_sim_config(raw)` reads the sections `project`, `experiment`,
`defense`, `attack`, `crypto` and `datasets.<experiment.dataset>` and
returns a frozen `SimConfig`.  Unknown keys in `experiment`, `defense` and
`attack` are errors (typos must not silently fall back to defaults).
Nothing is switched automatically: an unsupported rule/protocol pair, a
Byzantine budget the participant count cannot satisfy, or an attack that
the dataset cannot support raises ValueError.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from fte_fl.attacks.updates import alie_z_max
from fte_fl.defenses.aggregators import RULES
from fte_fl.defenses.protocols import ETF_RULES

# dataset -> models trained on it (Milestone 2 pairings; "mlp" is test-only)
DATASET_MODELS = {
    "synthetic": ("mlp",),
    "femnist": ("cnn_small",),
    "speech_commands": ("bc_resnet1", "lora_ast"),
    "cifar100": ("vit_tiny", "lora_vit"),
}
NUM_CLASSES = {"femnist": 62, "speech_commands": 35, "cifar100": 100}
PROTOCOLS = ("plaintext", "etf", "fte", "fte_attested")
UPDATE_ATTACKS = ("sign_flip", "scaling", "zero", "gaussian", "alie")
DATA_ATTACKS = ("label_flip", "backdoor")
ATTACKS = ("none",) + UPDATE_ATTACKS + DATA_ATTACKS
# datasets with a trigger family in fte_fl.attacks.triggers
BACKDOOR_DATASETS = ("femnist", "cifar100", "speech_commands")

_EXPERIMENT_KEYS = {
    "dataset", "model", "rounds", "clients_per_round", "local_epochs", "batch_size",
    "optimizer", "lr", "momentum", "weight_decay", "protocol", "eval_batch_size",
    "max_eval_samples", "sampling", "malicious_per_round",
}
_DEFENSE_KEYS = {"name", "f", "b", "m", "fedavg_weighting"}
_ATTACK_KEYS = {
    "type", "num_malicious", "start_round", "seed", "gamma", "std", "z",
    "source_class", "target_class", "poison_fraction", "trigger_mode",
    "speech_amplitude", "speech_freq_hz",
}


def _int(value, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _opt_int(value, name: str, minimum: int | None = None) -> int | None:
    return None if value is None else _int(value, name, minimum)


def _float(value, name: str, positive: bool = False, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def _check_keys(section: dict, allowed: set, name: str) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in {name}: {sorted(unknown)}")


@dataclass(frozen=True)
class DefenseConfig:
    name: str
    f: int | None = None
    b: int | None = None
    m: int | None = None
    fedavg_weighting: str = "samples"   # "samples" (n_k) or "uniform"

    def rule_kwargs(self) -> dict:
        """Keyword arguments for RULES[name], excluding FedAvg's weights."""
        return {k: getattr(self, k) for k in ("f", "b", "m") if getattr(self, k) is not None}


def check_rule_params(defense: DefenseConfig, n: int) -> None:
    """Raise unless `defense` is valid for n participating (or accepted) updates.

    Mirrors the constraints the rules enforce (aggregators.py / krum.py) so an
    invalid configuration fails before any training.
    """
    name, f, b, m = defense.name, defense.f, defense.b, defense.m
    allowed = {"fedavg": set(), "krum": {"f"}, "multi_krum": {"f", "m"},
               "trimmed_mean": {"b"}, "bulyan": {"f"}}[name]
    given = {k for k in ("f", "b", "m") if getattr(defense, k) is not None}
    if given - allowed:
        raise ValueError(f"defense {name!r} takes no {sorted(given - allowed)}")
    if name in ("krum", "multi_krum", "bulyan") and f is None:
        raise ValueError(f"defense {name!r} requires f (assumed Byzantine clients per round)")
    if name == "trimmed_mean" and b is None:
        raise ValueError("trimmed_mean requires b (values trimmed per side)")
    if name in ("krum", "multi_krum") and n <= 2 * f + 2:
        raise ValueError(f"{name} requires n > 2f + 2, got n={n}, f={f}")
    if name == "multi_krum" and m is not None and not 1 <= m <= n:
        raise ValueError(f"multi_krum requires 1 <= m <= n, got m={m}, n={n}")
    if name == "trimmed_mean" and n - 2 * b < 1:
        raise ValueError(f"trimmed_mean requires n - 2b >= 1, got n={n}, b={b}")
    if name == "bulyan":
        if f < 1:
            raise ValueError(f"bulyan requires f >= 1, got {f}")
        if n < 4 * f + 3:
            raise ValueError(f"bulyan requires n >= 4f + 3, got n={n}, f={f}")


@dataclass(frozen=True)
class AttackConfig:
    type: str = "none"
    num_malicious: int = 0
    start_round: int = 1
    seed: int = 0
    gamma: float | None = None
    std: float | None = None
    z: float | str | None = None          # a number, or "z_max" (alie_z_max(n, m))
    source_class: int | None = None
    target_class: int | None = None
    poison_fraction: float | None = None
    trigger_mode: str | None = None       # "distributed" (part i mod 4) or "global"
    speech_amplitude: float | None = None
    speech_freq_hz: float | None = None


@dataclass(frozen=True)
class SimConfig:
    seed: int
    torch_threads: int
    dataset: str
    model: str
    num_clients: int
    rounds: int
    clients_per_round: int
    local_epochs: int
    batch_size: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    protocol: str
    sampling: str                        # "uniform" or "stratified"
    malicious_per_round: int | None      # stratified only
    eval_batch_size: int
    max_eval_samples: int | None
    defense: DefenseConfig
    attack: AttackConfig
    crypto: dict
    dataset_params: dict
    num_classes: int


def _parse_attack(a: dict, dataset: str, num_classes: int, num_clients: int,
                  rounds: int, clients_per_round: int,
                  malicious_per_round: int | None = None) -> AttackConfig:
    _check_keys(a, _ATTACK_KEYS, "attack")
    kind = a.get("type", "none")
    if kind not in ATTACKS:
        raise ValueError(f"unknown attack type {kind!r}; expected one of {ATTACKS}")
    num_mal = _int(a.get("num_malicious", 0), "attack.num_malicious", 0)
    start = _int(a.get("start_round", 1), "attack.start_round", 1)
    seed = _int(a.get("seed", 0), "attack.seed", 0)
    used = {"type", "num_malicious", "start_round", "seed"}
    if kind == "none":
        if num_mal != 0:
            raise ValueError("attack.type 'none' requires num_malicious = 0")
        extra = {k for k in a if k not in used}
        if extra:
            raise ValueError(f"attack.type 'none' takes no {sorted(extra)}")
        return AttackConfig()
    if not 1 <= num_mal <= num_clients:
        raise ValueError(f"attack.num_malicious must lie in [1, {num_clients}], got {num_mal}")
    if num_mal == num_clients:
        raise ValueError("at least one client must be honest")
    if start > rounds:
        raise ValueError(f"attack.start_round {start} is after the last round {rounds}")
    kw: dict = {}
    if kind == "scaling":
        kw["gamma"] = _float(a.get("gamma"), "attack.gamma", positive=True)
        used.add("gamma")
    elif kind == "gaussian":
        kw["std"] = _float(a.get("std"), "attack.std", positive=True)
        used.add("std")
    elif kind == "alie":
        if num_mal < 2:
            raise ValueError("alie estimates mean/std from >= 2 colluding malicious clients")
        if malicious_per_round is not None and malicious_per_round < 2:
            raise ValueError("alie needs malicious_per_round >= 2")
        z = a.get("z")
        if z != "z_max":
            z = _float(z, "attack.z")
        elif malicious_per_round is not None:
            alie_z_max(clients_per_round, malicious_per_round)   # raises if undefined
        elif clients_per_round == num_clients:
            alie_z_max(clients_per_round, num_mal)   # raises if undefined for (n, m)
        kw["z"] = z
        used.add("z")
    elif kind == "label_flip":
        src = _int(a.get("source_class"), "attack.source_class", 0)
        tgt = _int(a.get("target_class"), "attack.target_class", 0)
        if src >= num_classes or tgt >= num_classes or src == tgt:
            raise ValueError(f"need distinct source/target classes in [0, {num_classes})")
        kw.update(source_class=src, target_class=tgt)
        used |= {"source_class", "target_class", "poison_fraction"}
    elif kind == "backdoor":
        if dataset not in BACKDOOR_DATASETS:
            raise ValueError(f"no trigger is defined for dataset {dataset!r}")
        tgt = _int(a.get("target_class"), "attack.target_class", 0)
        if tgt >= num_classes:
            raise ValueError(f"attack.target_class must be < {num_classes}")
        mode = a.get("trigger_mode")
        if mode not in ("distributed", "global"):
            raise ValueError("attack.trigger_mode must be 'distributed' or 'global'")
        kw.update(target_class=tgt, trigger_mode=mode)
        used |= {"target_class", "poison_fraction", "trigger_mode"}
        if dataset == "speech_commands":
            kw["speech_amplitude"] = _float(a.get("speech_amplitude"), "attack.speech_amplitude",
                                            positive=True)
            kw["speech_freq_hz"] = _float(a.get("speech_freq_hz"), "attack.speech_freq_hz",
                                          positive=True)
            used |= {"speech_amplitude", "speech_freq_hz"}
    if kind in DATA_ATTACKS:
        frac = _float(a.get("poison_fraction"), "attack.poison_fraction", positive=True)
        if frac > 1:
            raise ValueError(f"attack.poison_fraction must lie in (0, 1], got {frac}")
        kw["poison_fraction"] = frac
    extra = {k for k in a if k not in used}
    if extra:
        raise ValueError(f"attack.type {kind!r} takes no {sorted(extra)}")
    return AttackConfig(type=kind, num_malicious=num_mal, start_round=start, seed=seed, **kw)


def parse_sim_config(raw: dict) -> SimConfig:
    """Validate the YAML dict and return a SimConfig (ValueError on any problem)."""
    seed = _int(raw.get("project", {}).get("seed"), "project.seed", 0)
    runtime = raw.get("runtime") or {}
    _check_keys(runtime, {"device", "num_workers", "torch_threads"}, "runtime")
    if runtime.get("device", "cpu") != "cpu":
        raise ValueError("only runtime.device 'cpu' is supported")
    if runtime.get("num_workers", 1) != 1:
        raise ValueError("only runtime.num_workers 1 is supported (no data-loader workers)")
    # thread count changes floating-point reduction order: results are
    # reproducible only for the same torch_threads
    torch_threads = _int(runtime.get("torch_threads", 1), "runtime.torch_threads", 1)
    e = raw.get("experiment")
    _check_keys(e, _EXPERIMENT_KEYS, "experiment")
    dataset, model = e.get("dataset"), e.get("model")
    if dataset not in DATASET_MODELS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASET_MODELS)}")
    if model not in DATASET_MODELS[dataset]:
        raise ValueError(f"model {model!r} is not paired with {dataset!r}; "
                         f"expected one of {DATASET_MODELS[dataset]}")
    dparams = dict(raw.get("datasets", {}).get(dataset) or {})
    num_clients = _int(dparams.get("num_clients"), f"datasets.{dataset}.num_clients", 1)
    if dataset == "synthetic":
        num_classes = _int(dparams.get("num_classes"), "datasets.synthetic.num_classes", 2)
    else:
        num_classes = NUM_CLASSES[dataset]

    rounds = _int(e.get("rounds"), "experiment.rounds", 1)
    k = _int(e.get("clients_per_round"), "experiment.clients_per_round", 1)
    if k > num_clients:
        raise ValueError(f"clients_per_round={k} exceeds num_clients={num_clients}")
    optimizer = e.get("optimizer", "sgd")
    if optimizer != "sgd":
        raise ValueError(f"only optimizer 'sgd' is implemented, got {optimizer!r}")
    protocol = e.get("protocol", "plaintext")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    d = raw.get("defense")
    _check_keys(d, _DEFENSE_KEYS, "defense")
    if d.get("name") not in RULES:
        raise ValueError(f"unknown defense {d.get('name')!r}; expected one of {sorted(RULES)}")
    weighting = d.get("fedavg_weighting", "samples")
    if weighting not in ("samples", "uniform"):
        raise ValueError("defense.fedavg_weighting must be 'samples' or 'uniform'")
    if d["name"] != "fedavg" and "fedavg_weighting" in d:
        raise ValueError("defense.fedavg_weighting applies only to fedavg")
    defense = DefenseConfig(
        name=d["name"],
        f=_opt_int(d.get("f"), "defense.f", 0),
        b=_opt_int(d.get("b"), "defense.b", 0),
        m=_opt_int(d.get("m"), "defense.m", 1),
        fedavg_weighting=weighting,
    )
    check_rule_params(defense, k)
    if protocol == "etf" and defense.name not in ETF_RULES:
        raise ValueError(f"protocol 'etf' supports only {ETF_RULES}; {defense.name!r} needs "
                         "coordinate-wise comparisons that CKKS does not provide")

    sampling = e.get("sampling", "uniform")
    mpr = e.get("malicious_per_round")
    if sampling not in ("uniform", "stratified"):
        raise ValueError(f"experiment.sampling must be 'uniform' or 'stratified', got {sampling!r}")
    if sampling == "uniform" and mpr is not None:
        raise ValueError("experiment.malicious_per_round requires sampling 'stratified'")
    if sampling == "stratified":
        mpr = _int(mpr, "experiment.malicious_per_round", 1)
    attack = _parse_attack(raw.get("attack") or {"type": "none"}, dataset, num_classes,
                           num_clients, rounds, k, mpr)
    if sampling == "stratified":
        if attack.type == "none":
            raise ValueError("stratified sampling needs malicious clients (attack.type != 'none')")
        if mpr > attack.num_malicious or mpr >= k:
            raise ValueError(f"malicious_per_round={mpr} must be <= num_malicious="
                             f"{attack.num_malicious} and < clients_per_round={k}")
        if k - mpr > num_clients - attack.num_malicious:
            raise ValueError("not enough honest clients for the honest share of each round")
    crypto = dict(raw.get("crypto") or {})
    if protocol != "plaintext":
        for key in ("poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale", "chunk_size"):
            if key not in crypto:
                raise ValueError(f"protocol {protocol!r} requires crypto.{key}")

    return SimConfig(
        seed=seed, torch_threads=torch_threads, dataset=dataset, model=model, num_clients=num_clients, rounds=rounds,
        clients_per_round=k,
        local_epochs=_int(e.get("local_epochs"), "experiment.local_epochs", 1),
        batch_size=_int(e.get("batch_size"), "experiment.batch_size", 1),
        optimizer=optimizer,
        lr=_float(e.get("lr"), "experiment.lr", positive=True),
        momentum=_float(e.get("momentum", 0.0), "experiment.momentum", minimum=0.0),
        weight_decay=_float(e.get("weight_decay", 0.0), "experiment.weight_decay", minimum=0.0),
        protocol=protocol, sampling=sampling, malicious_per_round=mpr,
        eval_batch_size=_int(e.get("eval_batch_size", 256), "experiment.eval_batch_size", 1),
        max_eval_samples=_opt_int(e.get("max_eval_samples"), "experiment.max_eval_samples", 1),
        defense=defense, attack=attack, crypto=crypto, dataset_params=dparams,
        num_classes=num_classes,
    )
