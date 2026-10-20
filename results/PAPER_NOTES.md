# PAPER_NOTES — FTE-FL Research Project

## Milestone 2: Model Architectures — Decisions and Limitations

Verified: October 2026. All parameter counts and forward/backward results come
from running `python -m fte_fl.models.verify_models` with seed=42.

---

### Model A — CnnSmall (FEMNIST)

- Architecture: Two Conv blocks (1→16→32 channels, 3×3, MaxPool) + FC (1568→96→62).
- Parameters: **161,438** total, all trainable.
- Hard limit < 250,000 asserted at construction. Range 150k–200k satisfied.
- Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017.
- No limitations.

---

### Model B — BCResNet1 (Speech Commands)

- Architecture: ported from the official Qualcomm repo
  (https://github.com/Qualcomm-AI-research/bcresnet), inspecting `bcresnet.py`
  and `main.py` directly.
- Parameters: **335,811** total, all trainable and all receiving gradients.
  Within ±10% of 321,000 target.
- Variant: BC-ResNet-8 adapted to 64 mel bins / 35 classes, SubSpectralNorm S=8.
- Reference: Kim et al., "Broadcasted Residual Learning for Efficient Keyword
  Spotting", Interspeech 2021 (arXiv:2106.04140).

**Correction (2026-10-20):** an earlier version reported 329,923 parameters.
That count included 14,720 SSN parameters that never affected the output:
with S=5 and F ∈ {32, 16, 8}, every SSN layer silently fell back to plain
BatchNorm, so the grouped BN was dead weight inside `trainable_vector()`.

**Naming clarification (research-critical):**
  The Python class is `BCResNet1` for file-naming convenience, but it
  implements **BC-ResNet-8 (τ=8, base_c=64)** — the ~321k-parameter variant.
  In the paper, τ is a channel-width multiplier: BC-ResNet-1 (τ=1) has ~10k
  params; BC-ResNet-8 (τ=8) has ~321k params.  The official Qualcomm
  repository maps `--tau 8` to `BCResNets(int(self.tau * 8))`, i.e. base_c=64.
  The source files inspected were `bcresnet.py` (BCResNets class, BCResBlock)
  and `main.py` (_load_model) at the URL above.  The module constant
  `BC_RESNET_TAU = 8` is the canonical record in the code.

  Any research report must describe the model as BC-ResNet-8 (τ=8), not
  BC-ResNet-1.

**Adaptations vs. paper:**
- Input: 64 mel bins (paper: 40). The stem Conv2d(1→128) with stride=(2,1)
  halves frequency from 64→32 immediately. Subsequent stages stride F further
  to 32→16→8.
- SubSpectralNorm: **S=8** (upstream: S=5). Upstream (40 bins) feeds SSN with
  F = [20, 20, 10, 10, 5×8], i.e. 4/2/1 bins per sub-band. Ours feeds
  F = [32, 32, 16, 16, 8×8]; S=5 divides none of these. S=8 reproduces the
  4/2/1 bins-per-sub-band granularity exactly (trade-off: 8 sub-bands per
  layer instead of 5, so each sub-band covers 1/8 rather than 1/5 of the
  spectrum). As upstream, indivisible F raises an error; no plain-BN fallback.
- Classes: 35 (full GSC v2 label set). Upstream trains the 12-class task
  (10 keywords + silence + unknown) for both v1 and v2; the paper's accuracy
  figures are for that 12-class task and are not comparable.
- Classifier: DW 5×5 without frequency padding maps F=8→4 (upstream 5→1);
  AdaptiveAvgPool averages the remaining rows.
- Time frames: 101 (paper: 98). Head uses AdaptiveAvgPool so this is irrelevant.
- Upstream settings (n_mels, label set, SSN group count and affine mode,
  stride stages, classifier) were re-checked against the Qualcomm repo's
  `bcresnet.py`, `subspectralnorm.py`, `utils.py`, `main.py` on 2026-10-20.
  This was a settings check, not a line-by-line diff.

---

### Model C — ViTTiny (CIFAR-100)

- Architecture: ViT from scratch, embed_dim=192, 12 blocks, 3 heads, MLP ratio=4.
- Parameters: **5,380,132** total, all trainable. Larger than the 150–200k target
  for cnn_small because ViT-Tiny was specified to serve as an upper HE-cost bound
  (most expensive model to encrypt per the project spec).
- Patch size=4 (not 16). Reason: 32×32 CIFAR-100 images yield only 4 patches
  with patch_size=16 — far too few for meaningful self-attention. Patch_size=4
  yields 64 patches, the standard choice for ViT on CIFAR (see Dosovitskiy et
  al. 2021 Appendix B.1 ablation).
- Randomly initialized at every construction. No pretrained weights used or
  loaded. The `pretrained_loaded` attribute is absent/False.
- Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
  for Image Recognition at Scale", ICLR 2021 (arXiv:2010.11929).

---

### Model D — LoRAViT (CIFAR-100)

- Base: timm vit_base_patch16_224 (85,875,556 parameters, all frozen by default).
- LoRA: r=8, α=16, applied to `qkv` (fused Q/K/V projection) and `fc1` in
  every transformer block. This implicitly covers q and v per the fused design.
- Trainable: **740,452** (adapters + classification head). < 2M asserted.
- Frozen: **85,798,656** (base ViT weights).
- Image handling: CIFAR-100 images are 32×32; ViT-Base/16 requires 224×224.
  Images are bilinearly resized inside `forward()` before passing to the base.
  This is the standard fine-tuning approach for CIFAR on ViT-Base (Touvron et
  al., DeiT, 2021).
- Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  ICLR 2022 (arXiv:2106.09685).

**Pretrained-weight status (important):**
  The constructor defaults to `pretrained=False`, meaning the base ViT-Base/16
  is **randomly initialized** in all current builds.  The `pretrained_loaded`
  attribute equals the `pretrained` flag passed at construction (timm raises
  if a requested download fails; loaded weights are not independently
  checked).  In the current environment all builds use
  `pretrained=False` — no pretrained weights have been downloaded or loaded.
  Passing `pretrained=True` requires internet access to the timm HuggingFace
  hub (~330 MB, vit_base_patch16_224.augreg_in21k_ft_in1k or similar).  This
  has not been verified in the current environment.

**Frozen random backbone (research-critical):**
  With `pretrained=False`, training updates only the LoRA adapters and the
  head on top of a frozen, randomly initialized ViT-Base.  Results from such
  runs must NOT be described as pretrained fine-tuning, and accuracy may be
  limited by the random frozen features.

---

### Model E — LoRAAST (Speech Commands)

**Important: this is an AST-*style* spectrogram transformer, NOT a faithful
reproduction of the Audio Spectrogram Transformer of Gong et al. 2021.**

- Architecture: A ViT-style transformer applied to 2-D log-mel spectrogram
  patches (the same general approach as AST), built from scratch using the ViT
  components in `vit_tiny.py`. The backbone class is `SpectrogramTransformerBase`.
- LoRA: r=8, α=16, applied to the top 6 blocks' `qkv` and `fc1`.
- Trainable: **89,699** (adapters + classification head). < 2M asserted.
- Frozen: **5,393,856** (base spectrogram transformer weights, randomly initialized).
- Input: (1, 64, 101) log-mel. Patch=16×16, time axis zero-padded to 112 →
  4×7=28 patches.
- `pretrained_loaded = False` always (no pretrained weights available offline).
- **Frozen random backbone:** the base is random AND frozen, so the patch and
  positional embeddings, CLS token, final norm and blocks 0–5 never train.
  Only the top-6-block adapters and the head learn. This is LoRA on random
  features, not fine-tuning; accuracy may be limited by this design.
- Reference: Gong et al., "AST: Audio Spectrogram Transformer",
  Interspeech 2021 (arXiv:2104.01778) — this is the inspiration, not the
  implementation source.

**timm AST availability:**
  timm 1.0.30 does NOT contain a native AST model. Registry search for
  'ast', 'audio_spectrogram', 'wav2vec', and 'hubert' returned zero matches.
  The substring 'ast' only matches FasteRNet/FastViT models coincidentally.

**Deviations from Gong 2021 (these are NOT mere adaptations — they are
fundamental differences that invalidate direct comparison with AST results):**
  1. Random base initialization: Gong et al. initialize from DeiT-Base
     pretrained on ImageNet-21k, then fine-tune. Here the backbone is randomly
     initialized and frozen; only the LoRA adapters and the classification
     head are trained. This substantially affects convergence and final accuracy.
  2. Non-overlapping patches (stride=16): The paper uses stride=10 (overlapping),
     giving ~248 tokens. We use stride=16 (non-overlapping), giving 28 tokens.
  3. Much smaller backbone: embed_dim=192, heads=3. Gong uses embed_dim=768, heads=12.
  4. 64 mel bins: The paper uses 128 for AudioSet. Our preprocessing yields 64.

---

## Milestone 3: Byzantine Attacks — Definitions and Caveats

Implemented in `fte_fl/attacks/` (`updates.py`, `labels.py`, `triggers.py`) and
tested in `tests/test_attacks.py` with synthetic tensors only. **No attack has
been run in federated training. Unit tests establish that each function
follows its stated contract; they establish nothing about attack effectiveness,
attack success rate, or accuracy impact.**

### Model-update attacks (operate on one client's flat update vector)

Input contract for all: a non-empty, finite, 1-D float32/float64 `torch.Tensor`
(e.g. `trainable_vector()`); output is a new tensor with the same shape,
dtype and device; the input is never modified.

| Function | Definition | Parameters / defaults |
|---|---|---|
| `sign_flip(u)` | `-u` | — |
| `scaling(u, gamma)` | `gamma * u`; raises on overflow | `gamma=10.0`, must be finite and > 0 |
| `zero_update(u)` | all zeros, same shape/dtype/device | — |
| `gaussian_noise(u, std, seed)` | **replaces** `u` with i.i.d. N(0, std²) noise; `u`'s values are not used | `std` required (no default), > 0; `seed` required, drawn from a private `torch.Generator` (global RNG untouched) |
| `alie(mu, sigma, z)` | `mu - z * sigma` coordinate-wise | `z=1.0` (project default), any finite `z`; `sigma >= 0`; `sigma = 0` returns `mu` exactly |
| `estimate_mean_std(updates)` | coordinate-wise mean and **sample** std (ddof=1) of a supplied (k, d) tensor, k >= 2 | optional helper for ALIE |
| `alie_z_max(n, m)` | NeurIPS z_max = Φ⁻¹((n − m − s)/(n − m)), s = ⌊n/2 + 1⌋ − m | optional helper; requires 1 ≤ s ≤ n − m − 1 |

The Gaussian-noise distribution and scale are not claimed to be optimal or
realistic; no default scale has been justified.

### ALIE

- Citation (verified 2026-10-20 against the NeurIPS proceedings page and
  PDF, and arXiv:1902.06156): Gilad Baruch, Moran Baruch, Yoav Goldberg,
  "A Little Is Enough: Circumventing Defenses For Distributed Learning",
  NeurIPS 2019. The proceedings list the authors in that order; arXiv lists
  Moran Baruch first.
- **Source of the implemented equations: the NeurIPS 2019 proceedings PDF**
  (Algorithm 3 and Sec. 3.3, checked 2026-10-20). Algorithm 3:
  s = ⌊n/2 + 1⌋ − m (line 1); z_max = max z such that
  Φ(z) < (n − m − s)/(n − m) (line 2); per coordinate, the corrupted
  workers compute μ_j and σ_j (lines 3–4) and all send
  **(p_mal)_j = μ_j − z_max·σ_j** (line 5). Sec. 3.1 states that the
  corrupted workers' data alone suffices to estimate μ and σ. The paper does
  not state a population/sample std convention. Worked example (Sec. 3.3):
  n = 50, m = 24, s = 2, (50 − 24 − 2)/(50 − 24) = 0.923, z_max = 1.43,
  value μ_j − 1.43·σ_j (`alie_z_max(50, 24)` = 1.4261).
- **The arXiv rendering (ar5iv) differs and is not used here:** it states the
  threshold Φ(z) < (n − s)/n (worked example 0.96, z_max = 1.75) and writes
  μ_j + z_max·σ_j. The two thresholds give different z_max for the same
  n and m; they are not interchangeable.
- This project: `alie` computes **μ − z·σ**, i.e. NeurIPS line 5 with z in
  place of z_max; a negative z moves the output the other way (μ + |z|·σ).
  `z = 1.0` is a configurable project default, not the paper's setting; the
  paper's z_max is computed from n and m by `alie_z_max`, valid for
  1 ≤ s ≤ n − m − 1 (so the probability lies strictly in (0, 1)); outside
  that range it raises. z_max ≤ 0 when the probability is ≤ 0.5.
  `alie` receives only the attacker's μ and σ; it cannot read other clients'
  updates. If statistics are estimated with `estimate_mean_std`, the caller
  must supply only updates the attacker legitimately holds (e.g. the colluding
  malicious clients' own current-round updates); the simulator (Milestone 5)
  must enforce this — no honest-client or future-round information.

### Label poisoning (operates on a malicious client's local training labels)

- `flip_labels(labels, source_class, target_class, num_classes)`: targeted
  mapping source → target; every other label is unchanged. Not a permutation.
- `backdoor_labels(labels, num_classes, target_class=0)`: sets the given
  labels to the target class; used together with, but separately from,
  trigger insertion.
- Labels: 0-D/1-D integer `torch.Tensor` or `numpy.ndarray`, all values in
  [0, num_classes); a new object is returned. Applying these only to a
  malicious client's local training data (never to test/evaluation data) is
  the simulator's responsibility.

### DBA-style backdoor triggers (operate on local training inputs)

- Citation (verified via the official repository README,
  github.com/AI-secure/DBA): Chulin Xie, Keli Huang, Pin-Yu Chen, Bo Li,
  "DBA: Distributed Backdoor Attacks against Federated Learning", ICLR 2020.
  The OpenReview page could not be read (browser verification), so the venue
  is confirmed from the repository citation only.
- Scope: only DBA's **trigger decomposition** is implemented. DBA's training
  schedule, model-update scaling, and poisoning-per-batch procedure are not
  implemented in Milestone 3, so nothing here is an exact DBA reproduction.
  The image triggers are DBA-style; the speech trigger is DBA-inspired.
- Distributed structure: the global trigger is the union of 4 local triggers.
  Malicious client i trains with local part i; the global trigger
  (`parts=None`) is the one evaluated at test time (DBA's
  `centralized_test_trigger`). Sequential application of parts equals the
  union (tested). Assigning parts to clients is the simulator's job.
- Image triggers (`apply_image_trigger`), on unnormalized [0, 1] float
  tensors (C, H, W) or (N, C, H, W); inputs outside [0, 1] are rejected, so
  the trigger must be applied before any normalization:
  - CIFAR-100: pixel coordinates copied from DBA's `cifar_params.yaml`
    (four 1×6 bars, rows 0 and 4, columns 0–5 and 9–14), set to 1.0 in all
    three channels. DBA's `image_helper.py` loads CIFAR with `ToTensor()`
    only (no `Normalize`), so DBA's value 1 is already white in [0, 1]; this
    project uses the same value in the same representation. The CIFAR-100
    loader currently returns PIL images; `to_tensor` conversion to [0, 1] is
    assumed.
  - FEMNIST: coordinates copied from DBA's `mnist_params.yaml` (four 1×4
    bars, rows 0 and 3, columns 0–3 and 6–9), set to **0.0** (a dark patch).
    Adaptation: DBA writes 1 (white) on un-normalized MNIST (its `Normalize`
    line is commented out), i.e. white on a black background, whereas the
    loaded LEAF FEMNIST images have a predominantly white background with
    dark ink, so a white trigger would be invisible. Audit observation
    (read-only probe on 2026-10-20, not a test-suite guarantee; the offline
    tests use synthetic images): 1,611 samples from one LEAF train file
    loaded via `FemnistDataset` had 89.0% of pixels exactly 1.0, 99.8% of
    samples had an all-1.0 trigger region (minimum 0.77), and the trigger
    changed exactly the 16 mask pixels per sample, all to 0.0.
- Speech Commands (`apply_speech_trigger`): **DBA-inspired feature-space
  adaptation, not a physical waveform.** The trigger is constructed directly
  in log-mel space; no waveform is injected during training. The loader
  yields only log-mel features x = log(mel + 1e-6), shape (1, 64, 101). The
  trigger adds the steady-state mel magnitude T of a 1 kHz tone
  (peak amplitude 0.1 ≈ −20 dBFS by default, a project choice) in the linear
  domain: x' = log(exp(x) + T) where T > 0, bit-exact x elsewhere. 1 kHz is
  exactly FFT bin 25 (n_fft = 400, 16 kHz); with the loader's periodic Hamming
  window the tone's STFT magnitude is 0.27·A·N at bin 25 and 0.115·A·N at bins
  24 and 26, exactly zero elsewhere, so T occupies mel rows 21–23 only.
  A test feeds a synthetic 1 kHz tone through the loader's own
  `wav_to_logmel` and confirms the trigger applied to silence reproduces it
  (linear-domain atol 1e-3, frames 2–98). Approximations: magnitudes are
  added, ignoring phase interaction with speech; the tone is idealized as
  steady in every frame (no onset/offset or edge-padding effects). Local
  parts are 4 contiguous time segments (26/25/25/25 frames) — a project
  adaptation, since DBA has no audio setting. Input validation only rejects
  values below the log(1e-6) floor; standardized features lie above it and
  are not detected, so the trigger must be applied before any normalization.
  Trigger stealth or audibility has not been evaluated.

### Not implemented in Milestone 3

Client selection, attack scheduling (including DBA's training schedule and
model-update scaling), number of poisoned samples per batch (DBA used
`poisoning_per_batch`), aggregation/defenses, training,
configuration fields for attacks, and any experiment. These belong to
later milestones.

---

## Milestone 1 Dataset Findings (carried forward)

### FEMNIST

- 198 writers, 36,700 train samples, 4,174 test samples.
- LEAF split protocol: per-sample, same writers in train and test.
- Not a writer-independent held-out split.

### Speech Commands v0.02

- 2,618 valid speakers (2,624 minus 6 false IDs from _background_noise_).
- Official train/val/test speaker split exists via validation_list.txt and
  testing_list.txt. Current loader does NOT enforce this split. Must be
  fixed before reporting accuracy against the official test set.

### CIFAR-100

- 50k train, 10k test, 100 fine classes, 20 superclasses verified.
- Partition logic verified on real data.
- CIFAR-100 download (~169 MB) completed and cached in `data_cache/`.

---

## What This Study Does NOT Include

- The EEG legacy experiments from `majorproj/fl_byzantine` are NOT part of
  this study and are not referenced or reproduced here.
- No model accuracy results have been measured yet (experiments are Milestone 6).
- No federated training or attack/defense evaluation has occurred yet. Attack
  functions exist (Milestone 3) but have not been run in any experiment.
