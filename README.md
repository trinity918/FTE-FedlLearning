# FTE-FL

Research code comparing Filter-Then-Encrypt (FTE) and Encrypt-Then-Filter (EtF)
in Byzantine-robust federated learning.

## Environment

WSL2 Ubuntu 22.04, Python 3.10, CPU-only PyTorch. No GPU required.

```bash
cd ~/fte-fl
source .venv/bin/activate
```

Fresh setup (if starting from scratch):

```bash
cd ~/fte-fl
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Tests

Run the full offline suite (no dataset required):

```bash
python -m pytest -v -m "not realdata"
```

Run the real-data verification tests (requires datasets on disk):

```bash
python -m pytest -v -m realdata
```

Run everything:

```bash
python -m pytest -v
```

## Project Layout

```
fte_fl/
    configs/      YAML experiment configs + config loader
    crypto/       CKKS chunked encrypt/decrypt/homomorphic-sum (TenSEAL)
    defenses/     Krum + Multi-Krum (Blanchard 2017, Mhamdi 2018)
    data/         Dataset loaders: FEMNIST, Speech Commands, CIFAR-100
    attacks/      (Milestone 3) sign_flip, ALIE, DBA, etc.
    attest/       (Milestone 4) Pedersen commitment + Schnorr proof
    models/       (Milestone 2) CNN, BC-ResNet, ViT, LoRA adapters
    fl/           (Milestone 5) single-process FL simulator
    eval/         (Milestone 7) stats and paper artifacts
    experiments/  (Milestone 6) experiment matrix runner
    results/      generated metrics, plots, PAPER_NOTES.md
tests/           pytest suite (unit + realdata-marked real-data tests)
```

## Datasets

### FEMNIST (LEAF-preprocessed)

Source: `/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/justincase/leaf/data/femnist/data/`
Configured in: `fte_fl/configs/default.yaml` under `datasets.femnist`

**Verified counts (recomputed, not assumed):**
- Train writers: **198** (requested default of 500 is capped at 198)
- Train samples: **36,700**
- Test writers: **198** (all 198 train writers also appear in the test split)
- Test samples: **4,174**
- Classes: **62** (10 digits + 26 lowercase + 26 uppercase)
- Sample shape: **(1, 28, 28)** float32, pixels in [0, 1]
- Labels: integers in [0, 61]

**LEAF split protocol:** LEAF splits samples per writer — the same 198 writers
appear in both train and test sets. This is NOT a writer-independent split.
Each writer's samples are divided between train and test. This design is
appropriate for the FL setting where each client trains and evaluates on their
own writer's data.

Verify:
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset femnist
```

### Speech Commands v0.02

Source: `/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02/`
Configured in: `fte_fl/configs/default.yaml` under `datasets.speech_commands`

**Verified counts:**
- Total speakers (excluding `_background_noise_`): **2,618**
- Speakers selected by default (`num_clients=800`): **800**
- Total .wav files: **105,835** across 35 command words
- Audio format: 16 kHz, mono, 16-bit PCM
- Feature: log-mel spectrogram via `torch.stft` (no torchaudio)
  - 64 mel bins, n_fft=400, hop_length=160, 1-second clips
  - Output tensor: **(1, 64, 101)**
- Classes: **35** command words (see `SPEECH_LABELS` list)

**Speaker count note:** An earlier inspection reported 2,624 speakers. The
correct count is **2,618**. The discrepancy of 6 arises from files in
`_background_noise_/` (e.g., `doing_the_dishes.wav`, `dude_miaowing.wav`)
whose filenames produce false speaker IDs ("doing", "dude", "exercise",
"pink", "running", "white") when naively split on `_`. The loader correctly
excludes `_background_noise_` before extracting speaker IDs.

**Official split files:** `validation_list.txt` (9,981 clips, 256 speakers)
and `testing_list.txt` (11,005 clips, 250 speakers) are present. The two
lists have no speaker overlap. Of 2,618 total speakers: 256 validation-only,
250 test-only, **2,112 training-only**. The loader's `build_speech_clients`
does not currently implement this official split — it samples from the full
pool. This is a **known limitation** (see PAPER_NOTES.md).

Verify:
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset speech
```

### CIFAR-100

Cache location: `./data_cache/` (relative to repo root)
Configured in: `fte_fl/configs/default.yaml` under `datasets.cifar100`

**Download:** Not pre-downloaded. First use triggers automatic download via
`torchvision.datasets.CIFAR100(download=True)` to `./data_cache/`. This
downloads ~169 MB. Subsequent runs use the cached copy.

**Verified (against real torchvision data):**
- Train samples: 50,000 | Test samples: 10,000
- Classes: 100 fine classes, 20 superclasses (5 fine per superclass)
- Sample shape: (3, 32, 32) float32
- `_COARSE_OF_FINE` mapping verified to match the real torchvision pickle

**Partitioning strategies:**
1. **Dirichlet** (`split: dirichlet`): configurable `alpha` (0.1 or 0.5).
   Lower alpha = more non-IID. No duplicate index assignments.
2. **Superclass** (`split: superclass`): 20 official CIFAR-100 superclasses
   assigned round-robin to clients. Each client sees 1–3 superclasses.

Verify (requires download):
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset cifar100 --download
```

Verify partitioning logic only (no download):
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset cifar100
```

## Configuration

```python
from fte_fl.config import Config
cfg = Config.from_yaml("fte_fl/configs/default.yaml")
print(cfg.hash, cfg.git_commit, cfg.seed)
```

Every `metrics.json` output will contain `config_hash` and `git_commit` for
full reproducibility tracking.
