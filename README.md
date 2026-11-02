# FTE-FL

Research code comparing Filter-Then-Encrypt (FTE) and Encrypt-Then-Filter (EtF)
in Byzantine-robust federated learning.

## Environment

CPU-only PyTorch; no GPU required. Two platforms have produced results, and
each run records its own in `config.json`:

| Platform | Python | Used for |
|---|---|---|
| WSL2 Ubuntu 22.04 | 3.10.12 | Milestones 0-5, the first 45 runs of the main FEMNIST grid |
| Native Windows 11 | 3.11.9 | everything since 2026-11-01 |

Both use numpy 2.2.6, scipy 1.15.3, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
tenseal 0.3.16, timm 1.0.30 and PyYAML 6.0.3.

### Windows setup (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.2.6 scipy==1.15.3 tenseal==0.3.16 timm==1.0.30 PyYAML==6.0.3
pip install -e ".[dev]"
```

Datasets are found at the paths in `fte_fl/configs/default.yaml`. Those paths
are written `/mnt/c/...` and stay that way, because the raw config is part of
the config hash that resuming validates; `native_path` in `fte_fl/fl/data.py`
maps them to `C:/...` when the data is loaded. To use another location, pass
`--set datasets.<name>.<dir key>=...`.

Runs are not bitwise identical across the two platforms: see
`results/PAPER_NOTES.md`, Milestone 6, "Platform comparison".

### Linux / WSL setup

```bash
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
    defenses/     FedAvg, Krum, Multi-Krum, trimmed mean, Bulyan; EtF/FTE protocols (Milestone 4)
    data/         Dataset loaders: FEMNIST, Speech Commands, CIFAR-100
    models/       Five model architectures (Milestone 2)
    attacks/      (Milestone 3) sign_flip, ALIE, DBA-style triggers, etc.
    attest/       (Milestone 4) Pedersen commitment + Schnorr-style proof of opening
    fl/           (Milestone 5) single-process FL simulator: config validation,
                  update vectors, data, attacks, aggregation, metrics, loop
    eval/         (Milestone 7) stats and paper artifacts
    experiments/  run.py: run one simulation from a YAML config (Milestone 5)
    results/      generated metrics, plots, PAPER_NOTES.md
results/runs/    simulator outputs (git-ignored)
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

## Defenses and Protocols (Milestone 4)

Plaintext rules (`fte_fl/defenses/aggregators.py`; Krum scoring in `krum.py`).
Each returns a `Selection` (per-client weights + optional coordinate mask);
inputs are finite real (n, d) arrays.

| Rule | Definition | Constraints (else error) |
|---|---|---|
| `fedavg` | weighted mean, weights normalised (default equal) | weights finite, ≥ 0, sum > 0 |
| `krum` | update minimising the sum of squared distances to its n − f − 2 nearest neighbours; ties → smallest index | int f ≥ 0, n > 2f + 2 |
| `multi_krum` | mean of the m best Krum scores (default m = n − f) | as Krum, 1 ≤ m ≤ n |
| `trimmed_mean` | per coordinate: remove the b smallest and b largest values, mean of the other n − 2b | int b ≥ 0, n − 2b ≥ 1 |
| `bulyan` | repeated Krum picks θ = n − 2f updates; per coordinate, mean of the β = n − 4f values closest to the (lower) median | int f ≥ 1, n ≥ 4f + 3 |

Protocols (`fte_fl/defenses/protocols.py`):
- **EtF** `run_etf` (fedavg, krum, multi_krum): clients encrypt → the server
  computes encrypted pairwise squared distances → the key holder decrypts only
  those and selects with Krum → the server sums the selected ciphertexts →
  the key holder decrypts the aggregate.
- **FTE** `run_fte` (all five rules): clients send plaintext to a filter →
  the filter selects → selected clients encrypt their weighted, masked
  contributions → the server sums → the key holder decrypts.
- **FTE + attestation** `run_fte_attested`: FTE preceded by a Pedersen
  commitment to each update's SHA-256 digest and a Schnorr-style (Okamoto,
  Fiat–Shamir) proof of knowledge of its opening (`fte_fl/attest/pedersen.py`).
  The filter checks each proof and that the plaintext it receives opens the
  commitment.

Limitations:
- All parties run in one process with a single trusted key holder.
- The server holds no secret key and cannot decrypt (tested). Message order
  is recorded by the simulation itself.
- In FTE the filter sees every update in plaintext. EtF decrypts pairwise
  distances to filter, so neither is a fully homomorphic robust aggregation.
- CKKS is approximate (squared-distance error up to ~4e-6 observed), so EtF
  can select differently from plaintext Krum near score ties.
- Attestation does not prove that a ciphertext matches the committed update,
  that filtering was correct, or that an update is valid, and it gives no
  privacy from the filter. Binding, hiding, soundness and zero knowledge are
  standard assumptions, not tested properties.
- Robustness results are in `paper/` and `results/PAPER_NOTES.md`, Milestone 6.

Details and caveats: `results/PAPER_NOTES.md`.

```bash
python -m pytest -v tests/test_krum.py tests/test_aggregators.py tests/test_attest.py tests/test_protocols.py
```

## Simulator (Milestone 5)

One simulation per command; everything is read from the YAML config, and
`--set` overrides dotted keys (values parsed as YAML):

```bash
python -m fte_fl.experiments.run --config fte_fl/configs/default.yaml \
    --run-dir results/runs/<name> \
    --set experiment.rounds=2 --set defense.name=krum --set defense.f=2
# continue an interrupted run (same config, same run dir):
python -m fte_fl.experiments.run --config ... --run-dir results/runs/<name> --resume
```

An existing non-empty run directory is never overwritten (error unless
`--resume`, which also checks the config hash). Output files:

| File | Contents |
|---|---|
| `config.json` | raw config + hash, parsed settings, Python/package versions, git commit and whether tracked files were modified, model parameter counts, CKKS parameters, threat model |
| `manifest.json` | client ids and sample counts, partition id, malicious clients, poisoning record, evaluation-set sizes |
| `metrics.jsonl` | one record per round (below) |
| `checkpoint.pt` | global vector after the last completed round |
| `summary.json` | `running` / `completed` / `failed` (+ error and traceback) |

**Config** (`fte_fl/configs/default.yaml`, validated by `fte_fl/fl/config.py`;
unknown keys and invalid combinations raise, nothing is switched silently):
- `runtime`: `device` (cpu only), `torch_threads` (default 4; results
  reproduce bitwise only at the same thread count; on this 8-CPU WSL2
  machine 8 threads was 5–30× slower than 4).
- `experiment`: `dataset`, `model` (pairs: femnist–cnn_small;
  speech_commands–bc_resnet1/lora_ast; cifar100–vit_tiny/lora_vit;
  synthetic–mlp, test only), `rounds`, `clients_per_round`, `local_epochs`,
  `batch_size`, `optimizer` (sgd only), `lr`, `momentum`, `weight_decay`,
  `protocol` (`plaintext`, `etf`, `fte`, `fte_attested`), `eval_batch_size`,
  `max_eval_samples`.
- `defense`: `name` (a key of `aggregators.RULES`), `f` (krum, multi_krum,
  bulyan), `b` (trimmed_mean), `m` (multi_krum), `fedavg_weighting`
  (`samples` or `uniform`). Budgets are checked against `clients_per_round`
  (Krum n > 2f+2, trimmed mean n − 2b ≥ 1, Bulyan f ≥ 1 and n ≥ 4f+3); EtF
  rejects trimmed_mean and bulyan.
- `attack`: `type` (`none`, `sign_flip`, `scaling`, `zero`, `gaussian`,
  `alie`, `label_flip`, `backdoor`), `num_malicious` (fixed set for the run),
  `start_round`, `seed`, and per-type parameters (`gamma`, `std`, `z` or
  `"z_max"`, `source_class`/`target_class`, `poison_fraction`,
  `trigger_mode` `distributed`/`global`, `speech_amplitude`/`speech_freq_hz`).
  Each type accepts only its own parameters.
- `datasets.<name>`: paths, `num_clients`, partition (CIFAR-100: `iid`,
  `dirichlet` + `alpha`, `superclass`), optional `max_samples_per_client`.

**Round** (`fte_fl/fl/simulator.py`): seeded uniform client sampling → local
SGD from the global model (fresh optimizer per round) → update = local −
global → update attack on malicious rows → aggregation under the protocol →
global += aggregate → evaluation. All randomness is derived from
(seed, purpose, round, client), so runs are bitwise reproducible on the same
machine and a resumed run equals an uninterrupted one (tested).

**Update vector** (`fte_fl/fl/state.py`): trainable parameters, then
floating-point buffers (BatchNorm running statistics, BCResNet1 only), in
float64. Frozen parameters and integer buffers are excluded and stay at the
global model's values. The global vector is read back from the float32 model
after every update, so the aggregated vector is the evaluated model.

**Weights** are applied once, inside the rule's selection: FedAvg uses
n_k / Σ n_k (or 1/n); Krum, Multi-Krum, trimmed mean and Bulyan use their own
unweighted coefficients and ignore sample counts.

**Metrics per round**: clean test loss/accuracy; validation loss/accuracy
(Speech Commands only); local training loss (mean over local minibatches;
all clients and honest clients); backdoor ASR on a separate triggered set
(held-out samples whose label is not the target, full trigger); for label
flipping, the source-class flip rate and source-class accuracy; filtering
diagnostics from ground-truth labels (malicious selected/rejected, TPR, FPR,
precision, recall, kept-coordinate fractions); wall-clock times; payload
sizes; ciphertext counts; CKKS aggregate error versus plaintext; process
peak RSS. Payload sizes are computed sizes of serialized objects, not
measured network traffic.

```bash
python -m pytest -v tests/test_simulator.py
```

Definitions, threat model, limitations and smoke-test results:
`results/PAPER_NOTES.md`. Unit tests and smoke runs validate the
implementation; they are not evidence of robustness.

## Experiments (Milestone 6)

Grids of runs × seeds are YAML specs in `fte_fl/experiments/grids/`:
`femnist_lr_sweep.yaml` (hyperparameter selection by training loss),
`femnist_main.yaml` (lr 0.2; stopped, kept as the record of its divergence),
`femnist_stability_lr0.1.yaml` (diagnostic seeds 200–202) and
`femnist_main_lr0.1.yaml` (the main grid):

```bash
python -m fte_fl.experiments.grid fte_fl/experiments/grids/femnist_main_lr0.1.yaml --jobs 3 --only honest__
python -m fte_fl.experiments.grid fte_fl/experiments/grids/femnist_main_lr0.1.yaml --jobs 3
python -m fte_fl.eval.analyze results/runs/m6/femnist_main_lr0.1     # -> _analysis/
```

- `grid.py` runs each job as its own process; completed jobs are skipped,
  partial ones resumed, early failures (no `config.json`) renamed to
  `seed<N>.failed-<time>` and retried; a run whose `summary.json` records a
  failure (e.g. divergence) is a result and is not rerun. Every attempt is
  logged in `_grid_log.jsonl` as it happens, with the memory readings at
  each launch.
- Memory guard: launches are serialised. A job starts only if
  min(available) − `--job-mb` (700) ≥ `--min-available-mb` (1024), where
  "available" is the VM's MemAvailable and, under WSL, the Windows host's
  Available MBytes (read through `powershell.exe`; the VM's figure did not
  show the host pressure that stopped two grids). When memory is short the
  runner waits for one of its own jobs to finish; if none is running, or
  memory cannot be read, it launches nothing more and reports the rest as
  `not_started_low_memory` / `not_started_memory_unknown`. Other programs'
  memory use after a launch is not controlled. A FEMNIST process peaks at
  ~620 MB (1.5 GB before the loader converted each LEAF JSON file to arrays
  as it is read; arrays are bitwise identical).
- `analyze.py` refuses incomplete, failed or inconsistent groups (missing
  rounds, seeds whose configs differ beyond the seed) and writes
  `summary.csv`, `summary.md`, `problems.txt` and per-group plots with mean ±
  sample std across seeds.
- `protocol_bench.py` aggregates IDENTICAL client updates under plaintext,
  EtF, FTE and FTE + attestation for every rule, records unsupported pairs
  with their validation error, and compares selections and aggregates with
  plaintext:

```bash
python -m fte_fl.experiments.protocol_bench --config fte_fl/configs/default.yaml \
    --out results/runs/m6/protocol_bench --checkpoint <run>/checkpoint.pt \
    --rounds 101 102 103 --budget 4 --set ...   # exact command: results/PAPER_NOTES.md
```

`experiment.sampling: stratified` with `experiment.malicious_per_round`
fixes the number of malicious participants per round (uniform sampling
makes it random, so rounds can exceed the defense budget and ALIE, which
needs ≥ 2 malicious participants, fails). Results and their provenance:
`results/PAPER_NOTES.md`, Milestone 6.

Grids added for the Windows runs (outputs under `results/runs/m6_windows/`, full
commands in `docs/REPRODUCTION.md`):

- `femnist_stratified_control.yaml`: honest behaviour under the attacked runs'
  stratified sampling (`scaling` with γ = 1.0 is an exact no-op), to measure the
  uniform-versus-stratified confound in the main grid.
- `femnist_platform_attack_check.yaml`: attacked Windows replays of Linux runs.
- `femnist_stratified_control_rerun.yaml`, `femnist_platform_attack_check_rerun.yaml`:
  the same runs in new directories, replacing 14 runs invalidated by two queues
  writing to one directory; `analyze` refuses the invalid ones.
- `speech_lr_pilot.yaml` (stopped, kept), `speech_pilot_v2.yaml` (selects lr 0.2)
  and `speech_main_bcres_seed{0,1,2}.yaml` (the Speech main grid).

`scripts/run_queue.sh` runs grids one after another at 3 jobs (`JOBS`) with a
3.5 GB launch floor, retrying only stages that stopped on low memory. `fte_fl/eval/full_eval.py`
scores final Speech checkpoints on the full test split and refuses a checkpoint
that does not replay its recorded subset accuracy. `fte_fl/eval/paper_artifacts.py`
writes the paper's tables and figures; a diverged run prints as `div. round R`,
never as a missing cell.

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
43 test images are byte-identical to a training image of the same writer
(checked 2026-10-20); they are kept, as in LEAF.

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

**Official split (Milestone 5):** `official_split_files` reads the dataset's
`validation_list.txt` / `testing_list.txt` (the README's SHA-1-of-speaker
rule; every other clip is training) and raises if a list is missing, lists
absent files, overlaps the other, or a speaker spans two splits. Recomputed
2026-10-20:

| Split | Clips | Speakers |
|---|---|---|
| training | 84,843 | 2,112 |
| validation | 9,981 | 256 |
| testing | 11,005 | 250 |

No speaker is shared between splits, and recomputing the README's hash rule
reproduces both lists exactly. `build_speech_clients` draws clients only
from training speakers; `build_speech_eval` returns validation/testing.

Verify:
```bash
python -m fte_fl.data.verify_datasets --seed 42 --dataset speech
```

**Feature cache.** `wav_to_logmel` costs ~28 ms per clip, so full-scale runs
read precomputed features instead. Build once, then opt in with
`datasets.speech_commands.feature_cache: data_cache/speech_logmel`:

```bash
python -m fte_fl.data.speech_cache --data-dir <speech_commands_v0.02> --out data_cache/speech_logmel --workers 3
```

The cache stores one `.npy` shard per word (2.55 GB), and the build resumes.
After building it compares 500 random clips with `wav_to_logmel` bit for bit,
at one torch thread: FFT rounding depends on the thread count (differences up
to 1e-6 were measured at 4 and 8 threads).

### CIFAR-100

Cache: `./data_cache/` (git-ignored, torchvision auto-downloads ~169 MB on first use)

**Verified:** 50k train / 10k test, 100 fine classes, 20 superclasses.
Sample shape: (3, 32, 32). Superclass mapping verified against real torchvision pickle.
Partitions of the training set: `iid`, `dirichlet` (alpha), `superclass`; the
simulator evaluates on the official test set and rejects partitions with
empty clients. 10 test images are byte-identical to training images (a
property of the official split, kept).

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
