"""Experiment configuration loader with determinism support.

Loads YAML config files, computes a config hash, sets random seeds,
and records the git commit for reproducibility (anti-slop rule 5).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except ImportError:  # torch is optional at config-load time
    torch = None


def _git_commit(repo_dir: str | None = None) -> str:
    """Return the current git commit hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def config_hash(config: dict) -> str:
    """Compute a stable SHA-256 hash of the config dict (first 16 hex chars)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    """Structured experiment configuration loaded from YAML.

    Attributes:
        raw:        The raw parsed YAML dict.
        hash:       16-char SHA-256 hash of the config (for logging).
        git_commit: Current git commit hash (for reproducibility).
        seed:       Effective random seed.
    """

    raw: dict[str, Any]
    hash: str
    git_commit: str
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path, repo_dir: str | None = None) -> "Config":
        """Load a YAML config, set seeds, and capture git commit.

        Args:
            path:     Path to the YAML config file.
            repo_dir: Git repo root for commit lookup (default: auto-detect).
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        seed = int(raw.get("project", {}).get("seed", 42))
        set_seed(seed)

        if repo_dir is None:
            # Walk up from config file to find .git
            repo_dir = str(path.parent)
            while repo_dir != "/" and not Path(repo_dir, ".git").exists():
                repo_dir = str(Path(repo_dir).parent)

        return cls(
            raw=raw,
            hash=config_hash(raw),
            git_commit=_git_commit(repo_dir),
            seed=seed,
        )

    def to_dict(self) -> dict:
        """Return a flat dict suitable for logging into metrics.json."""
        return {
            "config_hash": self.hash,
            "git_commit": self.git_commit,
            "seed": self.seed,
            **self.raw,
        }

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
