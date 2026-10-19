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
    attacks/      (Milestone 3) sign_flip, ALIE, DBA, etc.
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
| BCResNet1 | Speech Commands  | (1, 64, 101)       | 35      | 329,923     | 329,923   | 0          | 41                 |
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
- BC-ResNet with τ=8 (base_c=64), the ~321k parameter variant from Kim et al. 2021.
- Architecture ported faithfully from the official Qualcomm repo
  (https://github.com/Qualcomm-AI-research/bcresnet).
- The parameter suffix "1" in the filename is a project naming convention;
  the 321k model corresponds to `--tau 8` in the official repo (base_c=τ×8=64).
- 4 stages of BCResBlocks with frequency-axis strides; SubSpectralNorm (S=5).
- Stem Conv2d(1→128, 5×5, stride=(2,1)) halves frequency dimension immediately.
- Adaptation: 64 mel bins (vs. 40 in the paper); head uses AdaptiveAvgPool.
- 329,923 parameters (target 321k ± 10%); assertion enforced at build.
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
- `pretrained=False` by default (offline use). Pass `pretrained=True` for
  real pretrained weights; requires internet access (~330 MB download).
  Pretrained-weight status reported in verify_models output.
- References: Dosovitskiy et al. (ViT), Hu et al., ICLR 2022 (LoRA).

**LoRAAST** (`fte_fl/models/lora_ast.py`)
- AST-style architecture (ViT applied to log-mel spectrograms) built from scratch.
- timm 1.0.30 does not include an AST model; this is implemented directly
  using ViT building blocks (embed_dim=192, depth=12, 3 heads, patch=16).
- Patch embedding: 16×16 patches on (1, 64, 101) input → 28 tokens (4×7,
  with time axis zero-padded to 112).
- LoRA r=8, α=16 applied to the top 6 blocks' `qkv` and `fc1`.
- 89,699 trainable parameters; 5,393,856 frozen.
- No pretrained weights (no offline AST checkpoint available). Documented as
  a limitation in PAPER_NOTES.md.
- References: Gong et al., Interspeech 2021 (arXiv:2104.01778); Hu et al. (LoRA).

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
