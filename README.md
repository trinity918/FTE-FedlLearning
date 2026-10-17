# FTE-FL

Research code for comparing Filter-Then-Encrypt (FTE) and Encrypt-Then-Filter
(EtF) in Byzantine-robust federated learning.

## Environment

Developed on Ubuntu 22.04 under WSL2 with Python 3.10 and CPU-only PyTorch.

```bash
cd ~/fte-fl
python3.10 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch first (avoids pulling GPU wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the project and all declared dependencies (numpy, scipy,
# matplotlib, tenseal, timm, pyyaml, pytest)
pip install -e ".[dev]"
```

## Tests

```bash
source .venv/bin/activate
python -m pytest -v
```

The CKKS tests exercise chunked encryption/decryption and homomorphic
summation of 200k-float vectors (atol=1e-3). The Krum tests use synthetic
gradients (5 honest + sign-flip + noise) to verify honest-client selection.
The config tests verify YAML loading, deterministic hashing, and seed
reproducibility.

## Project Layout

```
fte_fl/
    configs/      YAML experiment configs + config loader
    crypto/       CKKS chunked encrypt/decrypt/homomorphic-sum (TenSEAL)
    defenses/     Krum + Multi-Krum (Blanchard 2017, Mhamdi 2018)
    attacks/      (Milestone 3) sign_flip, ALIE, DBA, etc.
    attest/       (Milestone 4) Pedersen commitment + Schnorr proof
    data/         (Milestone 1) dataset loaders
    models/       (Milestone 2) CNN, BC-ResNet, ViT, LoRA adapters
    fl/           (Milestone 5) single-process FL simulator
    eval/         (Milestone 7) stats and paper artifacts
    experiments/  (Milestone 6) experiment matrix runner
    results/      generated metrics, plots, PAPER_NOTES.md
tests/           pytest suite
configs/         experiment YAML files (one per experiment)
```

## Configuration

The default config is `fte_fl/configs/default.yaml`. The config loader
(`fte_fl/config.py`) computes a SHA-256 config hash and captures the git
commit hash for reproducibility logging into `metrics.json`.

```python
from fte_fl.config import Config

cfg = Config.from_yaml("fte_fl/configs/default.yaml")
print(cfg.hash, cfg.git_commit, cfg.seed)
```

## Data

FEMNIST and Speech Commands datasets are external to this repository
(already downloaded in the legacy repo). Do not modify the legacy
repository or regenerate datasets.
