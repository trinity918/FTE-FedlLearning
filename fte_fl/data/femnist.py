"""FEMNIST loader for LEAF-preprocessed data.

Each writer is one federated client.  The LEAF JSON format stores
flattened 28x28 grayscale images (784 floats) with integer labels 0-61
(10 digits + 26 lowercase + 26 uppercase).

References:
  LEAF benchmark: Caldas et al., "LEAF: A Benchmark for Federated Settings",
  arXiv:1812.01097, 2018.
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_leaf_json(path: str | Path) -> dict:
    """Parse a single LEAF-format JSON file.

    Returns dict with keys 'users' (list[str]) and 'user_data'
    (dict[user_id -> {'x': list[list[float]], 'y': list[int]}]).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


def _load_all_users(data_dir: str | Path) -> dict[str, dict]:
    """Aggregate user_data across all JSON files in a directory.

    Returns dict mapping user_id -> {'x': [...], 'y': [...]}.
    """
    data_dir = Path(data_dir)
    json_files = sorted(glob.glob(str(data_dir / "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    all_users: dict[str, dict] = {}
    for jf in json_files:
        data = parse_leaf_json(jf)
        for uid in data["users"]:
            if uid not in all_users:
                all_users[uid] = {"x": [], "y": []}
            all_users[uid]["x"].extend(data["user_data"][uid]["x"])
            all_users[uid]["y"].extend(data["user_data"][uid]["y"])
    return all_users


class FemnistDataset(Dataset):
    """Torch dataset wrapping a single writer's FEMNIST data.

    Each sample is reshaped to (1, 28, 28) float32.
    """

    def __init__(self, user_data: dict):
        self.x = np.array(user_data["x"], dtype=np.float32)  # (N, 784)
        self.y = np.array(user_data["y"], dtype=np.int64)    # (N,)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.x[idx]).reshape(1, 28, 28)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label


def build_femnist_clients(
    data_dir: str,
    num_clients: int = 500,
    seed: int = 42,
    split: str = "train",
) -> dict[str, FemnistDataset]:
    """Build per-writer datasets from LEAF-preprocessed FEMNIST.

    Args:
        data_dir: Path to the directory containing LEAF JSON files
                  (e.g. .../femnist/data/train).
        num_clients: Desired number of clients.  Capped at available
                     writers if fewer exist.
        seed: Random seed for reproducible client selection.
        split: 'train' or 'test'.

    Returns:
        Dict mapping writer_id -> FemnistDataset.
    """
    all_users = _load_all_users(data_dir)
    available = len(all_users)
    n = min(num_clients, available)

    rng = random.Random(seed)
    user_ids = sorted(all_users.keys())
    selected = rng.sample(user_ids, n)

    return {uid: FemnistDataset(all_users[uid]) for uid in selected}
