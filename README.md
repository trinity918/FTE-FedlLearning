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
    models/       Five model architectures (Milestone 2)
    attacks/      (Milestone 3) sign_flip, ALIE, DBA-style triggers, etc.
    attest/       (Milestone 4) Pedersen commitment + Schnorr proof
    fl/           (Milestone 5) single-process FL simulator
    eval/         (Milestone 7) stats and paper artifacts
    experiments/  (Milestone 6) experiment matrix runner
    results/      generated metrics, plots, PAPER_NOTES.md
tests/           pytest suite (unit + realdata-marked real-data tests)
```

## Models (Milestone 2)

Verify all five models and print a summary table:

```bash
python -m fte_fl.models.verify_models
```

Attempt pretrained-weight loading for LoRA ViT (requires internet):

```bash
python -m fte_fl.models.verify_models --pretrained
```

### Verified parameter counts (from `verify_models.py`, seed=42)

| Model     | Dataset          | Input              | Classes | Total params | Trainable | Frozen     | CKKS chunks (est.) |
|-----------|------------------|--------------------|---------|-------------|-----------|------------|--------------------|
| CnnSmall  | FEMNIST          | (1, 28, 28)        | 62      | 161,438     | 161,438   | 0          | 20                 |
| BCResNet1 | Speech Commands  | (1, 64, 101)       | 35      | 335,811     | 335,811   | 0          | 41                 |
| ViTTiny   | CIFAR-100        | (3, 32, 32)        | 100     | 5,380,132   | 5,380,132 | 0          | 657                |
| LoRAViT   | CIFAR-100        | (3, 32, 32)        | 100     | 86,539,108  | 740,452   | 85,798,656 | 91                 |
| LoRAAST   | Speech Commands  | (1, 64, 101)       | 35      | 5,483,555   | 89,699    | 5,393,856  | 11                 |

CKKS est. = `ceil(trainable_params / 8192)`. This is an estimate of the number
of CKKS ciphertext chunks needed to encrypt one model update.

### Model descriptions

**CnnSmall** (`fte_fl/models/cnn_small.py`)
- Two convolutional blocks (16→32 channels, 3×3 kernels, MaxPool) + two FC layers.
- 161,438 parameters, all trainable. Hard limit < 250,000 asserted at build.
- Reference: McMahan et al., AISTATS 2017.

**BCResNet1** (`fte_fl/models/bc_resnet.py`)
- **Implements BC-ResNet-8 (τ=8, base_c=64)** from Kim et al. 2021. The class
  name `BCResNet1` is a project file-naming convention — the implemented
  configuration is BC-ResNet-8. The module constant `BC_RESNET_TAU = 8` is
  the canonical record.
- Architecture ported from the official Qualcomm repo
  (https://github.com/Qualcomm-AI-research/bcresnet), inspecting `bcresnet.py`
  and `main.py`. The official CLI maps `--tau 8` to `base_c = 8×8 = 64`.
- 4 stages of BCResBlocks with frequency-axis strides; SubSpectralNorm with
  **S=8** sub-bands (upstream: S=5 at 40 bins). With 64 mel bins the SSN layers
  see F = 32/16/8, which S=5 does not divide; S=8 keeps upstream's 4/2/1
  frequency bins per sub-band at each stage. Indivisible F raises an error.
- Stem Conv2d(1→128, 5×5, stride=(2,1)) halves frequency dimension immediately.
- Adaptations: 64 mel bins (upstream: 40); 35 classes (upstream: 12-class
  task); the classifier's DW 5×5 leaves F=4 (upstream: 1), averaged by
  AdaptiveAvgPool.
- 335,811 parameters, all used in the forward pass (target 321k ± 10%);
  assertion enforced at build.
- Reference: Kim et al., Interspeech 2021 (arXiv:2106.04140).

**ViTTiny** (`fte_fl/models/vit_tiny.py`)
- ViT from scratch: embed_dim=192, 12 blocks, 3 heads, MLP ratio=4.
- Patch size=4 on 32×32 input → 64 patches + 1 CLS = 65 tokens.
  (Patch size 16 gives only 4 patches on 32×32 — too few for attention.)
- Randomly initialized; no pretrained weights.
- 5,380,132 parameters.
- Reference: Dosovitskiy et al., ICLR 2021 (arXiv:2010.11929).

**LoRAViT** (`fte_fl/models/lora_vit.py`)
- Frozen timm ViT-Base/16 (86M params) + LoRA r=8, α=16 on `qkv` and `fc1`.
- CIFAR-100 32×32 images are bilinearly resized to 224×224 inside `forward()`.
- 740,452 trainable parameters (adapters + head); 85,798,656 frozen.
- **Current initialization: random (`pretrained=False`, the default).** No
  pretrained weights have been downloaded or loaded in this environment.
  Pass `pretrained=True` for real pretrained weights; requires internet access
  (~330 MB). `pretrained_loaded` records the `pretrained` argument; loaded
  weights are not independently checked.
- With random init, training updates LoRA adapters + head on a **frozen random
  backbone**; this is not pretrained fine-tuning.
- References: Dosovitskiy et al. (ViT), Hu et al., ICLR 2022 (LoRA).

**LoRAAST** (`fte_fl/models/lora_ast.py`)
- **AST-style spectrogram transformer, NOT a faithful AST implementation.**
  This is a ViT-style backbone applied to 2-D log-mel patches (the same general
  approach as Gong et al. 2021), built from scratch using `SpectrogramTransformerBase`.
  It must not be compared with published AST numbers.
- Key differences from Gong 2021: random initialization (no ImageNet pretraining),
  non-overlapping 16×16 patches (vs. overlapping, stride=10), smaller backbone
  (embed_dim=192 vs. 768), 64 mel bins (vs. 128).
- LoRA r=8, α=16 applied to the top 6 blocks' `qkv` and `fc1`.
- 89,699 trainable parameters; 5,393,856 frozen. `pretrained_loaded = False`.
- The frozen backbone is **random**: patch/positional embeddings, CLS token and
  blocks 0–5 never train. This is LoRA on random features, not fine-tuning.
- References: Gong et al., Interspeech 2021 (arXiv:2104.01778); Hu et al. (LoRA).

## Attacks (Milestone 3)

Pure functions, independent of the (future) simulator; inputs are never
modified. Definitions, citations and caveats: `results/PAPER_NOTES.md`.

| Module | Operates on | Functions |
|---|---|---|
| `fte_fl/attacks/updates.py` | one client's 1-D float update tensor | `sign_flip`, `scaling` (γ=10), `zero_update`, `gaussian_noise` (replaces the update; explicit `std`, `seed`), `alie` (μ − z·σ, z=1), helpers `estimate_mean_std` (sample std), `alie_z_max` |
| `fte_fl/attacks/labels.py` | a malicious client's local training labels | `flip_labels` (targeted source→target), `backdoor_labels` (target class 0) |
| `fte_fl/attacks/triggers.py` | a malicious client's local training inputs | 4-part triggers: `apply_image_trigger` (DBA-style; CIFAR-100, FEMNIST; unnormalized [0, 1] images), `apply_speech_trigger` (DBA-inspired feature-space adaptation: a 1 kHz tone constructed directly in log-mel space, not injected as a physical waveform) |

```bash
python -m pytest -v tests/test_attacks.py
```

These tests check function contracts on synthetic data only. No attack
effectiveness has been measured. Only DBA's trigger decomposition is
implemented; DBA's training schedule, update scaling and poisoning-per-batch
procedure are not.

## Datasets

### FEMNIST (LEAF-preprocessed)

Source: `/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/justincase/leaf/data/femnist/data/`

**Verified counts:**
- Train writers: **198** (requested default of 500 is capped at 198)
- Train samples: **36,700**
- Test writers: **198** (all 198 train writers also appear in the test split)
- Test samples: **4,174**
- Classes: **62** (10 digits + 26 lowercase + 26 uppercase)
- Sample shape: **(1, 28, 28)** float32, pixels in [0, 1]

**LEAF split protocol:** LEAF splits samples per writer — the same 198 writers
appear in both train and test sets. This is NOT a writer-independent split.

Verify:
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset femnist
```

### Speech Commands v0.02

Source: `/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02/`

**Verified counts:**
- Total speakers (excluding `_background_noise_`): **2,618**
- Speakers selected by default (`num_clients=800`): **800**
- Total .wav files: **105,835** across 35 command words
- Audio format: 16 kHz, mono, 16-bit PCM
- Feature: log-mel spectrogram via `torch.stft` (no torchaudio)
  → **(1, 64, 101)** per clip
- Classes: **35** command words

**Known limitation:** `build_speech_clients` samples from all 2,618 speakers,
not restricted to the 2,112 official training-split speakers. Must be fixed
before reporting accuracy against the official test set.

Verify:
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset speech
```

### CIFAR-100

Cache: `./data_cache/` (git-ignored, torchvision auto-downloads ~169 MB on first use)

**Verified:** 50k train / 10k test, 100 fine classes, 20 superclasses.
Sample shape: (3, 32, 32). Superclass mapping verified against real torchvision pickle.

Verify (requires download):
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset cifar100 --download
```

## Configuration

```python
from fte_fl.config import Config
cfg = Config.from_yaml("fte_fl/configs/default.yaml")
print(cfg.hash, cfg.git_commit, cfg.seed)
```

Every `metrics.json` output will contain `config_hash` and `git_commit` for
full reproducibility tracking.
