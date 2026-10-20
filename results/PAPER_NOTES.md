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

## Milestone 4: Aggregation Defenses and Protocols — Definitions and Caveats

Implemented in `fte_fl/defenses/` (`krum.py`, `aggregators.py`,
`protocols.py`), `fte_fl/attest/pedersen.py`, and `fte_fl/crypto/ckks_manager.py`;
tested with synthetic data in `tests/test_krum.py`, `test_aggregators.py`,
`test_attest.py`, `test_protocols.py`, `test_ckks_manager.py` and
`test_config.py`. **No
defense has been run in federated training; nothing here measures robustness,
accuracy or attack success.** All protocol roles run in one Python process.

### Plaintext rules (`aggregators.py`)

Each rule returns a `Selection` (per-client weights a, optional boolean
per-coordinate mask M) with aggregate = Σ_i a_i (M_i ⊙ u_i). Updates: real,
finite, (n, d) arrays; inputs are not modified; output is float64.

| Rule | Definition implemented | Constraints enforced | Source (checked 2026-10-20) |
|---|---|---|---|
| `fedavg` | Σ (w_i / Σw) u_i; default equal weights | w finite, ≥ 0, Σw > 0 | McMahan et al., AISTATS 2017 (standard; not re-checked this milestone) |
| `krum` | argmin of s(i) = Σ of the n−f−2 smallest **squared** distances; ties → smallest index | int f ≥ 0, n > 2f+2 | Blanchard et al., NeurIPS 2017, Sec. 4 (proceedings PDF) |
| `multi_krum` | mean of the m best-scoring updates, one pass; default m = n−f | 1 ≤ m ≤ n | same paper, Sec. 6 (m ∈ {1..n}; experiments use m = n−f) |
| `trimmed_mean` | per coordinate independently: sort the n values, remove the b smallest **and** the b largest (2b removed), return the arithmetic mean of the remaining n−2b (sum ÷ (n−2b)) | int b ≥ 0, n−2b ≥ 1 | Yin, Chen, Ramchandran, Bartlett, ICML 2018 (PMLR 80), Def. 2 (divisor (1−2β)m) |
| `bulyan` | θ = n−2f via repeated Krum with removal; per coordinate, mean of the β = θ−2f values of S closest to the median | int f ≥ 1, n ≥ 4f+3 | El Mhamdi, Guerraoui, Rouault, ICML 2018 (PMLR 80), Sec. 4 |

- Trimmed mean takes a per-side **count** b (the paper's βn). b = 0 is the
  plain mean; the largest valid b is ⌊(n−1)/2⌋ (for odd n, n−2b = 1 gives
  the coordinate-wise median). Robustness needs b ≥ the number of Byzantine
  clients (paper: β ≥ α); the caller chooses b.
- Bulyan, checked against the paper and against the Krum routine actually
  called. Supported: int f ≥ 1 and n ≥ 4f+3 (minimum n = 7, 11, 15 for
  f = 1, 2, 3); θ = n−2f is the size of the selection set S; β = n−4f ≥ 3
  values are averaged per coordinate. Adaptations:
  (1) The selection step scores the remaining set R with Krum's formula,
  k = |R|−f−2 nearest neighbours (n replaced by |R|), via
  `scores_from_distances`; |R| runs from n down to 2f+1. The paper claims a
  quorum of 2f+3 for every use of Krum, but its last two iterations see
  |R| = 2f+2 and 2f+1, where Krum's own condition |R| > 2f+2 fails (so the
  validating `krum_from_distances` cannot be used there, and Krum's guarantee
  does not cover those two steps). We keep the same formula there; the only
  clamp, k = max(·, 1), fires solely for f = 1 at |R| = 3 (0 → 1). A test
  records every step's (|R|, k).
  (2) The paper's median is an argmin over values of S; for even θ both
  middle values minimise, and we take the lower. (3) Ties among distances to
  the median go to the smaller client index. f = 0 is rejected.
- Tests compare FedAvg, trimmed mean and Bulyan with hand computations and
  with independent reference implementations written from the definitions.
- Configuration: `fte_fl/configs/default.yaml` now names the defense
  `multi_krum` (previously `multikrum`), the canonical key of
  `aggregators.RULES`; a test enforces this. No code reads the `defense`
  settings yet (the simulator, Milestone 5, will); `byzantine_fraction` is
  likewise unused.

### Krum / Multi-Krum audit (existing `krum.py`)

- The old docstring said the paper uses plain Euclidean distance and that
  squaring was an optimisation. **Incorrect:** the NeurIPS 2017 score is
  defined with squared distances; the implementation was already faithful,
  the docstring is corrected.
- The old docstring attributed Multi-Krum to "Mhamdi et al., ICDCS 2018".
  **Incorrect:** Multi-Krum is defined in Blanchard et al., NeurIPS 2017;
  El Mhamdi et al. (Bulyan) appeared at ICML 2018.
- **Behaviour change:** `multi_krum` previously restricted m to [1, n−f−2]
  and defaulted to n−f−2, which is not from the paper. It now accepts
  m ∈ [1, n] and defaults to m = n−f (the paper's experimental setting).
  The existing test of the default was updated accordingly.
- `f` must now be an int (floats/bools raise `TypeError`). Distance and
  selection steps are exposed (`pairwise_squared_distances`,
  `krum_from_distances`, `multi_krum_from_distances`) so that plaintext
  Krum, Bulyan and EtF share one selection code path.
- Distances use |a|²+|b|²−2a·b in float64 (clipped at 0), which can lose
  precision for nearly identical, large-norm vectors; tests compare against a
  direct difference-based reference on well-conditioned data only.

### CKKS manager fix

`CKKSManager.homomorphic_sum` previously added in place into the first
input's ciphertext chunks (TenSEAL `+=` is in-place), silently changing the
caller's first encrypted vector. It now starts from a copy; a regression test
checks that inputs are unchanged. `public_context_bytes()` was added: a
serialized context without the secret key (optionally without Galois keys).

### Protocol data flows (`protocols.py`)

Roles: clients; a single trusted **key holder** with the CKKS secret key
(the only role that decrypts; no threshold decryption); the **aggregator**
(server), built from a serialized public context; in FTE, a **filter** that
receives plaintext. All roles run in one Python process with no network.

- **EtF (encrypt, then filter)**, `run_etf`, rules fedavg / krum / multi_krum:
  1. client i → aggregator: Enc(u_i) (serialized chunks).
  2. aggregator: Enc(‖u_i − u_j‖²) for all i < j by homomorphic subtraction
     and dot product, summed over chunks.
  3. aggregator → key holder: encrypted distances; the key holder decrypts
     **only these**, runs Krum / Multi-Krum on them, and returns the selection.
  4. aggregator: Σ w_i Enc(u_i) with plaintext weights.
  5. aggregator → key holder: Enc(aggregate); decrypted.
  FedAvg skips steps 2–3. Trimmed mean and Bulyan are rejected (they need
  coordinate-wise comparisons, which CKKS does not provide here).
- **FTE (filter, then encrypt)**, `run_fte`, any of the five rules:
  1. client i → filter: plaintext u_i.
  2. filter: Selection (a, M) computed in plaintext; → each contributing
     client: its a_i and M_i.
  3. contributing client i → aggregator: Enc(a_i (M_i ⊙ u_i)); clients that
     do not contribute send nothing.
  4. aggregator: homomorphic sum; → key holder: Enc(aggregate); decrypted.
- **FTE + attestation**, `run_fte_attested`: before step 1 each client
  publishes (C_i, proof_i) to the filter and aggregator; in step 1 it also
  sends its opening. The filter excludes client i unless proof_i verifies
  under (round, i) and the opening opens C_i with m = digest(u_i). The
  aggregator re-verifies each contributor's public proof.

### CKKS approximation and near ties

CKKS is approximate. Observed: absolute error up to ~4e-6 on decrypted
squared distances in the tests (ring degree 8192; ~5e-4 relative for nearly
identical updates), and up to ~1.5e-6 on aggregates in development runs
(ring degrees 8192 and 16384; the tests allow 1e-4). EtF therefore selects from
**approximate** distances: when plaintext Krum scores are tied or nearly
tied (gap below roughly (n−f−2) × the distance error), the EtF selection can
differ from plaintext Krum. Tests that expect identical selections first
assert that the plaintext score gaps exceed 1e-4; a separate test builds an
exact plaintext tie and asserts only that EtF picks one of the tied clients.
No exact equivalence between CKKS and plaintext aggregation is claimed.

### Guarantees: enforced, simulated, assumed, missing

1. **Enforced by code and verified by tests (real arithmetic, no mocks):**
   - The aggregator holds no secret key: it is built from serialized public
     context bytes, a context containing the secret key is rejected, and
     TenSEAL raises on decryption there. It accepts only serialized
     ciphertexts (plaintext arrays raise `TypeError`).
   - EtF and FTE aggregates equal the plaintext rules within CKKS tolerance;
     EtF uses the Krum selection computed from the decrypted distances.
   - `CKKSManager.homomorphic_sum` no longer modifies its inputs.
   - Pedersen commitments C = g^m h^r mod p in the order-q subgroup of the
     RFC 3526 2048-bit MODP group (tests check p is a 2048-bit safe prime and
     g = 2 and h have order q); m = SHA-256 of the update's exact float64
     bytes. Honest openings verify; altered updates and wrong openings fail.
   - Okamoto/Schnorr-style Fiat–Shamir proofs of knowledge of (m, r), bound
     to a context (round id, client id): honest proofs verify; altered,
     random, wrong-round, replayed-under-another-client and non-group proofs
     fail. In `run_fte_attested` the filter rejects a client whose plaintext
     differs from its commitment even when its proof is valid (the opening
     check is not skipped), and contributor indices map to original clients.
   - Sigma-protocol structure: special-soundness extraction and the
     honest-verifier simulation equation hold (interactive protocol).
2. **Demonstrated only by the in-process simulation / transcript:** the
   separation of parties (one process, no network), who sends what to whom
   and in which order (the transcript is recorded by the protocol code
   itself, so it documents but does not independently enforce the flow),
   the single trusted key holder, and honest clients that encrypt exactly
   their filtered contribution.
3. **Assumed, not established by tests:** computational binding of the
   commitment (discrete-log hardness in this group, log_g h unknown, SHA-256
   collision resistance), perfect hiding (uniform r from `secrets`), and
   soundness and zero knowledge of the Fiat–Shamir proof (random-oracle
   model); also CKKS/RLWE semantic security of TenSEAL with these parameters
   (not evaluated here).
4. **Missing guarantees (not implemented):** any proof that a CKKS ciphertext
   encrypts the committed or the filtered update; any proof that the filter
   applied its rule correctly; any update-validity check (e.g. norm bounds);
   privacy from the filter in FTE (it sees every plaintext update and
   opening); privacy of pairwise distances from the key holder in EtF; and
   protection against a key holder who decrypts individual ciphertexts.

### Not implemented in Milestone 4 / outstanding

Threshold or multi-party decryption, encrypted coordinate-wise filtering,
ciphertext–commitment consistency proofs, verifiable filtering, network or
timing simulation, and any experiment. (Client sampling, training loops and
the config wiring were added in Milestone 5.) **Robustness of any defense
has not been measured in training.**

---

## Milestone 5: Simulator and Experimental Pipeline — Definitions and Caveats

Implemented in `fte_fl/fl/` (`config.py`, `state.py`, `data.py`,
`models.py`, `attacks.py`, `aggregation.py`, `metrics.py`,
`simulator.py`) and `fte_fl/experiments/run.py`; tested in
`tests/test_simulator.py` (synthetic data) plus dataset tests. **Only smoke
runs (2 rounds, a handful of clients, capped data) have been executed. They
show that the pipeline runs; their accuracies, ASRs and filtering numbers are
not results and must not be reported as such.**

### Dataset partitions (changes in this milestone)

- **Speech Commands — official split now enforced.** The dataset archive's
  README ("Partitioning") assigns files by SHA-1 of the speaker part of the
  name and ships the result as `validation_list.txt` / `testing_list.txt`.
  `official_split_files` reads those lists (it does not invent or recompute
  a split) and fails if they are missing, inconsistent with the files on
  disk, overlapping, or not speaker-disjoint. Recomputed on 2026-10-20:
  training 84,843 clips / 2,112 speakers; validation 9,981 / 256; testing
  11,005 / 250; zero speakers shared; a re-implementation of the README's
  hash rule reproduces both lists with zero mismatches over 105,829 clips.
  Clients are drawn only from training speakers; validation/testing are used
  only for evaluation. Behaviour change: `build_speech_clients(split=...)`
  now accepts only "train" (it previously ignored the argument and sampled
  from all 2,618 speakers). The log-mel representation is unchanged.
- **FEMNIST** — unchanged (LEAF per-sample split; same 198 writers in train
  and test). Audit: 43 test images are byte-identical to a training image of
  the same writer. This comes from the LEAF data, not from our code; the
  images are kept so the split remains LEAF's. `num_clients` default changed
  500 → 198 (the loader already capped it at 198; the simulator now rejects
  a request for more writers than exist).
- **CIFAR-100** — official train/test split, unchanged; an `iid` partition
  was added beside `dirichlet` and `superclass`. Audit: 10 test images are
  byte-identical to training images (property of the official split; kept).
  The simulator never partitions the test set and rejects partitions that
  leave a client empty.
- Smoke runs use `max_samples_per_client` / `max_eval_samples` (seeded random
  subsets, recorded in `manifest.json`). Results on subsets are not results
  on the dataset.

### Update representation and weighting

- Vector = trainable parameters (`named_parameters` order, identical to
  `trainable_vector()`), then floating-point buffers (BatchNorm running
  mean/var; BCResNet1 only), float64. Frozen parameters (LoRA backbones)
  and integer buffers (`num_batches_tracked`) are excluded and keep the
  global values; BatchNorm with `momentum=None` is rejected.
- Buffers are included so that robust rules filter them; leaving them out
  would require an unfiltered average. Consequence: an attack can drive a
  running variance negative; evaluation then fails (non-finite loss) and the
  run is recorded as failed instead of being clamped.
- Note for the CKKS cost estimates of Milestone 2: the aggregated vector of
  BCResNet1 is longer than its trainable count by its BatchNorm buffers.
- Client update u_i = local − global; global ← global + aggregate (server
  learning rate 1); the global vector is re-read from the float32 model.
- Weights are applied once, inside the rule's `Selection`. FedAvg:
  n_i / Σ n_j over the round's participants (or uniform, configurable). The
  robust rules use their own unweighted coefficients and ignore sample
  counts, so "Multi-Krum" here is an unweighted mean of the selected
  updates, not a sample-weighted FedAvg of them. Malicious clients report
  true sample counts.
- FTE + attestation: the rule runs on the clients that pass attestation;
  FedAvg weights are renormalised over them.

### Client sampling and local training

Uniform sampling without replacement each round (seed, round). Local
training: SGD (lr, momentum, weight decay configurable), fresh optimizer per
round, cross-entropy, shuffled minibatches; seeds derived from (seed, round,
client). All hyperparameter defaults are untuned project choices.

### Threat model and attack integration (`fte_fl/fl/attacks.py`)

- A fixed set of `num_malicious` clients (seeded) for the whole run; they
  collude. Before `start_round` they behave honestly.
- Update attacks transform a malicious client's honestly trained update and
  read only that update (sign flip, scaling, zero; Gaussian uses only its
  length). ALIE uses the mean and sample standard deviation of the honest
  updates of the malicious participants of the round (≥ 2 required, else
  the run fails) and never reads honest clients' updates; z is a number or
  `alie_z_max(n, m)` for the round's n participants and m malicious ones.
  Whether this matches the knowledge assumption of the ALIE paper has not
  been re-checked for this milestone; it is the project's explicit choice.
  All malicious participants send the same ALIE vector.
- Label flipping: a `poison_fraction` of a malicious client's source-class
  samples get the target label (static, chosen once with the attack seed).
- Backdoor: a `poison_fraction` of a malicious client's samples get the
  trigger and the target label (static). `distributed`: the k-th malicious
  client (sorted index order) uses DBA local part k mod 4; `global`: the
  full trigger. **DBA-style, not a DBA reproduction**: DBA poisons a fixed
  number of samples inside each local batch, uses its own attacker
  schedule, local epochs, learning rates and scaling, none of which is
  replicated. Speech Commands uses the documented feature-space log-mel
  trigger (not waveform poisoning), applied to unnormalized features (the
  simulator applies no normalization).
- The defenses never see the malicious labels; filtering metrics use them
  only as ground truth for diagnostics.

### Metrics (`fte_fl/fl/metrics.py`, `simulator.py`)

- Clean: test loss/accuracy on the untouched held-out set; validation
  loss/accuracy where an official validation split exists (Speech Commands).
  FEMNIST and CIFAR-100 have no validation split here (none was carved out,
  to keep the established partitions).
- Training loss: LOCAL, the mean minibatch cross-entropy over a client's
  local training; averaged over all participants and over honest ones.
- Backdoor ASR: held-out samples with true label ≠ target, full trigger,
  fraction predicted as target. Never mixed into clean accuracy.
- Label flipping: on held-out source-class samples, flip rate (predicted
  target) and source-class accuracy.
- Filtering (diagnostic): rejected = contributes nothing (weight 0 or all
  coordinates masked); malicious selected/rejected, TPR (= recall), FPR,
  precision; for trimmed mean and Bulyan, mean kept-coordinate fractions and
  the malicious clients' effective weight Σ a_i·kept_i; whether the round's
  malicious count exceeded the assumed f (or b).
- Efficiency: wall-clock local training (sum and max), aggregation,
  protocol steps (client encryption, server distance/sum, key-holder
  decryption, filtering, attestation and verification), evaluation, round;
  payload sizes (serialized ciphertexts and public contexts; plaintext
  updates as 8 bytes per coordinate; commitments/proofs/openings as
  fixed-width integers; model broadcast as 4 bytes per coordinate);
  ciphertext counts; CKKS aggregate error versus plaintext evaluation of the
  same selection; process peak RSS. Payload sizes are not measured network
  traffic: there is no network, framing is excluded, and public-context
  sizes are one copy each (key distribution). Serialized TenSEAL contexts
  vary by a few hundred bytes between serializations.
- All parties run in one process, so per-party times are sequential
  wall-clock times on one machine, not a distributed-system latency.

### Reproducibility

Every run records the raw config and its hash, parsed settings, Python and
package versions, torch thread count, git commit and whether tracked files
were modified, model parameter counts, CKKS parameters, the client manifest
and partition id. The torch thread count is a config setting
(`runtime.torch_threads`, default 4) because it changes floating-point
reduction order. Runs with the same config are bitwise identical on the
same machine and thread count (tested on synthetic data; also observed for
a repeated FEMNIST smoke run), and an
interrupted run resumed from its checkpoint equals an uninterrupted one
(tested). Pedersen commitments use `secrets` randomness, which does not
affect any aggregate or metric. Output directories are never overwritten.

### Smoke matrix (2026-10-20, git-ignored outputs in `results/runs/smoke-20261020/`)

2 rounds each, torch_threads 4, seed 42, capped data: FEMNIST 10 writers ×
≤40 samples, 500 test samples; Speech Commands 5 training speakers × ≤16
clips, 100 test + 100 validation clips; CIFAR-100 5 clients (dirichlet 0.5)
× ≤32 images, 200 test images (LoRA-ViT: 3 clients × 4 images, 16 test
images). These runs check that every path executes; **after 2 rounds on
this little data no accuracy, ASR or filtering number below means
anything about the methods.** Elapsed = whole process (data loading
dominates for FEMNIST and Speech Commands).

| Run | Exit | Elapsed (s) | Note |
|---|---|---|---|
| FEMNIST CnnSmall FedAvg (and an identical rerun) | 0, 0 | 27.3, 24.6 | rerun bitwise identical (metrics without timings, final vector) |
| Speech BCResNet1 FedAvg | 0 | 44.7 | validation metrics recorded |
| Speech LoRA-AST FedAvg | 0 | 36.2 | |
| CIFAR-100 ViT-Tiny FedAvg | 0 | 16.5 | |
| CIFAR-100 LoRA-ViT FedAvg | 0 | 22.3 | random (not pretrained) backbone |
| FEMNIST Krum / Multi-Krum / trimmed mean / Bulyan | 0 each | 23.4 / 22.4 / 24.7 / 25.1 | |
| FEMNIST EtF Multi-Krum (7 clients) | 0 | 160.1 | per round: distances ≈ 30.6 s, encrypted weighted sum ≈ 30.6–31.3 s, encryption ≈ 3 s; CKKS aggregate error ≤ 4.9e-8 |
| FEMNIST FTE trimmed mean (7 clients) | 0 | 38.3 | encrypted sum ≈ 2 s, encryption ≈ 2.9 s; error ≤ 6.3e-8 |
| FEMNIST FTE+attestation Bulyan (7 clients) | 0 | 43.6 | attest ≈ 1 s, verification ≈ 2.4 s per round; 0 rejected |
| FEMNIST sign flip vs Krum (2 malicious) | 0 | 27.4 | both malicious rejected each round |
| FEMNIST ALIE (z_max) vs Multi-Krum f=3 (3 malicious) | 0 | 24.4 | all 3 malicious selected, 3 honest rejected, each round |
| FEMNIST label flip 1→7 (3 malicious) | 0 | 24.0 | malicious clients had 2, 5, 1 source samples |
| FEMNIST DBA-style backdoor, 4 parts (4 malicious) | 0 | 28.5 | |
| CIFAR-100 global-trigger backdoor ViT-Tiny | 0 | 22.2 | ASR 1.0 with clean accuracy 0.0: the model predicts the target class for everything; not backdoor evidence |
| Speech feature-space backdoor BCResNet1 | 0 | 31.5 | same caveat (ASR 1.0, clean accuracy 0.0 in round 2) |
| Bulyan f=2 with 10 clients; EtF + trimmed mean; Krum f=2 with 6 clients | 1 each | 2.8 / 2.4 / 2.5 | rejected at config validation, no run directory created |

Also verified on real data: a FEMNIST Krum run killed with SIGKILL after
round 1 and resumed reproduced the uninterrupted 4-round run exactly. The
EtF encrypted weighted sum (plaintext-scalar × ciphertext per contributor)
costs ~15× the FTE sum of pre-weighted ciphertexts at these sizes.

### Pilot runs for runtime estimation (2026-10-20, `results/runs/pilot-20261020/`)

Full per-client data, 10 clients per round, Multi-Krum f=2, torch_threads 4,
CKKS ring 16384 / chunk 8192. Measured per-round wall-clock (round 1
includes ~1.5 s of one-time warm-up in local training):

| Pilot | Data | Round time (s) | Breakdown |
|---|---|---|---|
| FEMNIST plaintext | 198 writers, full test (4,174) | 2.8, then 0.9 | load 10 s |
| FEMNIST EtF | same | 94.3 | distances 56.9 (45 pairs), weighted sum 28.8, encryption 3.5, decryption 0.3 |
| FEMNIST FTE + attestation | same | 15.2 | encryption 3.4, sum 2.2, attest 1.7, verify 3.3 |
| CIFAR-100 ViT-Tiny plaintext | 100 clients, dirichlet 0.1 (no empty client at seed 42), full test (10,000) | 132.8 | training 85.3, evaluation 42.5, Multi-Krum 2.0; peak RSS 3.1 GB |
| Speech BCResNet1 plaintext | 20 speakers (697 clips), 1,000 test + 1,000 validation | 48.8 | training 13.1, evaluation 35.4; data build 88 s for 2,697 clips |

Payloads per FEMNIST round: EtF uploads 131.6 MB of ciphertexts + 20.7 MB
of encrypted distances; FTE+attestation 105.2 MB (8 contributors) and
12.9 MB of plaintext updates to the filter; the aggregator's EtF context
(with Galois keys) is 75.9 MB vs 3.7 MB without.

Extrapolations (NOT measured; linear in the measured per-unit costs):
- FEMNIST, 100 rounds, per run: plaintext ≈ 2 min; FTE ≈ 17 min (the attested round minus attestation);
  FTE + attestation ≈ 25 min; EtF ≈ 2.6 h.
- CIFAR-100 ViT-Tiny plaintext ≈ 2.2 min/round (≈ 3.7 h per 100 rounds);
  evaluating every round on all 10,000 test images is a third of it. With
  657 ciphertexts per update, FTE adds roughly 2 min/round of encryption
  and EtF roughly 45–50 min/round (distances alone ~30 min), i.e. EtF on
  ViT-Tiny is impractical on this CPU.
- Speech Commands: ~28 ms per clip of feature extraction, so 800 speakers
  (~32k training clips) plus full validation and test (21k clips) means
  ~25 min of data loading per run and ~6 min per round to evaluate on the
  full held-out sets. A feature cache and an evaluation interval would be
  needed before full Speech experiments (not implemented).
- LoRA-ViT (frozen ViT-B/16 at 224×224): from the smoke run, ~0.3 s per
  training image and ~0.14 s per test image, i.e. roughly 50 min per round
  at 10 clients × 500 images with full evaluation; impractical on this CPU.

---

## Milestone 1 Dataset Findings (carried forward)

### FEMNIST

- 198 writers, 36,700 train samples, 4,174 test samples.
- LEAF split protocol: per-sample, same writers in train and test.
- Not a writer-independent held-out split.

### Speech Commands v0.02

- 2,618 valid speakers (2,624 minus 6 false IDs from _background_noise_).
- Official train/val/test speaker split exists via validation_list.txt and
  testing_list.txt. Enforced since Milestone 5 (see above).

### CIFAR-100

- 50k train, 10k test, 100 fine classes, 20 superclasses verified.
- Partition logic verified on real data.
- CIFAR-100 download (~169 MB) completed and cached in `data_cache/`.

---

## What This Study Does NOT Include

- The EEG legacy experiments from `majorproj/fl_byzantine` are NOT part of
  this study and are not referenced or reproduced here.
- No model accuracy results have been measured yet (experiments are Milestone 6).
- No attack/defense evaluation has occurred yet. Milestone 5 ran only
  2-round smoke tests on capped data to check that the pipeline executes;
  they are not experiments.
