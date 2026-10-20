"""Simulator integration of the Milestone 3 attacks.

Threat model (recorded in every run's config):
  * A fixed set of `num_malicious` clients, drawn once from the population
    with the attack seed, is malicious for the whole run; they collude.
  * Before `start_round` malicious clients behave honestly (clean data,
    honest updates).
  * Update attacks transform a malicious client's honestly trained update:
      sign_flip, scaling, zero, gaussian  use only that client's own update
                                          (gaussian: only its length);
      alie  uses the coordinate-wise mean and sample std of the honest
            updates of the malicious clients participating in this round
            (>= 2 required; fewer raises).  It never reads honest clients'
            updates.  z is a number or z_max = alie_z_max(n, m) with n
            participants and m malicious participants this round.  Every
            participating malicious client sends the same ALIE vector.
  * Data attacks poison the malicious clients' local training data once,
    with a fixed seeded choice of samples (static poisoning):
      label_flip  a `poison_fraction` of the client's source-class samples
                  get the target label;
      backdoor    a `poison_fraction` of all the client's samples get the
                  trigger and the target label.  trigger_mode "distributed":
                  the k-th malicious client (sorted client index order)
                  uses DBA local trigger part k mod 4; "global": all use
                  the full trigger.
    This is DBA-STYLE poisoning, not a DBA reproduction: DBA poisons a
    fixed number of samples inside every local batch and uses its own
    attacker schedule, local epochs and learning rate, none of which are
    replicated here.
  * Malicious clients report their true sample counts to FedAvg.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from fte_fl.attacks import labels, triggers, updates
from fte_fl.fl.config import UPDATE_ATTACKS, AttackConfig
from fte_fl.fl.data import GAUSSIAN, MALICIOUS, POISON, ClientData, derive_seed

TRIGGER_FAMILY = {"femnist": "femnist", "cifar100": "cifar100", "speech_commands": "speech"}


def choose_malicious(num_clients: int, attack: AttackConfig) -> list[int]:
    """Sorted indices of the malicious clients (seeded by the attack seed)."""
    if attack.num_malicious == 0:
        return []
    rng = np.random.default_rng(derive_seed(attack.seed, MALICIOUS))
    return sorted(rng.choice(num_clients, size=attack.num_malicious, replace=False).tolist())


def apply_trigger(x: torch.Tensor, dataset: str, attack: AttackConfig, parts) -> torch.Tensor:
    """Trigger the inputs `x` (parts=None: the full global trigger)."""
    family = TRIGGER_FAMILY[dataset]
    if family == "speech":
        return triggers.apply_speech_trigger(x, parts, attack.speech_amplitude, attack.speech_freq_hz)
    return triggers.apply_image_trigger(x, family, parts)


def poison_client(client: ClientData, ordinal: int, client_index: int, dataset: str,
                  num_classes: int, attack: AttackConfig) -> tuple[ClientData, dict]:
    """Poisoned copy of a malicious client's data, plus a record of what changed.

    `ordinal` is the client's position among the malicious clients (sorted).
    """
    rng = np.random.default_rng(derive_seed(attack.seed, POISON, client_index))
    x, y = client.x.clone(), client.y.clone()
    if attack.type == "label_flip":
        eligible = torch.nonzero(y == attack.source_class).flatten().numpy()
    else:
        eligible = np.arange(len(y))
    k = math.floor(attack.poison_fraction * len(eligible) + 0.5)
    chosen = np.sort(rng.choice(eligible, size=k, replace=False)) if k else np.zeros(0, dtype=int)
    idx = torch.from_numpy(chosen.astype(np.int64))
    record = {"client_id": client.client_id, "eligible": int(len(eligible)), "poisoned": int(k)}
    if k:
        if attack.type == "label_flip":
            y[idx] = labels.flip_labels(y[idx], attack.source_class, attack.target_class, num_classes)
        else:
            parts = [ordinal % triggers.NUM_TRIGGER_PARTS] if attack.trigger_mode == "distributed" else None
            x[idx] = apply_trigger(x[idx], dataset, attack, parts)
            y[idx] = labels.backdoor_labels(y[idx], num_classes, attack.target_class)
            record["trigger_parts"] = "all" if parts is None else parts
    return ClientData(client.client_id, x, y), record


def attack_updates(deltas: np.ndarray, is_malicious: np.ndarray, client_indices: list[int],
                   round_id: int, attack: AttackConfig) -> np.ndarray:
    """Return a copy of the (n, d) update matrix with malicious rows replaced.

    Only rows with is_malicious[i] are read for the attack (ALIE) or changed.
    """
    if attack.type not in UPDATE_ATTACKS:
        raise ValueError(f"{attack.type!r} is not an update attack")
    out = deltas.copy()
    rows = np.flatnonzero(is_malicious)
    if len(rows) == 0:
        return out
    if attack.type == "alie":
        if len(rows) < 2:
            raise ValueError(f"round {round_id}: ALIE needs >= 2 malicious participants, "
                             f"got {len(rows)}")
        mu, sigma = updates.estimate_mean_std(torch.from_numpy(deltas[rows]))
        z = updates.alie_z_max(len(deltas), len(rows)) if attack.z == "z_max" else attack.z
        out[rows] = updates.alie(mu, sigma, z).numpy()
        return out
    for r in rows:
        u = torch.from_numpy(deltas[r])
        if attack.type == "sign_flip":
            v = updates.sign_flip(u)
        elif attack.type == "scaling":
            v = updates.scaling(u, attack.gamma)
        elif attack.type == "zero":
            v = updates.zero_update(u)
        else:
            v = updates.gaussian_noise(u, attack.std,
                                       derive_seed(attack.seed, GAUSSIAN, round_id, client_indices[r]))
        out[r] = v.numpy()
    return out
