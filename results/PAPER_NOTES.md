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

- Architecture: ported faithfully from the official Qualcomm repo
  (https://github.com/Qualcomm-AI-research/bcresnet).
- Parameters: **329,923** total, all trainable. Within ±10% of 321,000 target.
- Reference: Kim et al., "Broadcasted Residual Learning for Efficient Keyword
  Spotting", Interspeech 2021 (arXiv:2106.04140).

**Naming clarification (important):**
  The project specification refers to "BC-ResNet-1 (~321k)". This is a naming
  ambiguity in the original requirement. In the paper, BC-ResNet-{τ} uses τ as
  a *channel-width multiplier* (not a model index). BC-ResNet-1 (τ=1) has
  approximately 10k parameters; BC-ResNet-8 (τ=8) has approximately 321k. The
  official Qualcomm codebase maps `--tau 8` to `BCResNets(base_c=8×8=64)`.
  We implement BC-ResNet with base_c=64 (the 321k configuration) and name the
  class `BCResNet1` to satisfy the filename convention.

**Adaptations vs. paper:**
- Input: 64 mel bins (paper: 40). The stem Conv2d(1→128) with stride=(2,1)
  halves frequency from 64→32 immediately. Subsequent stages stride F further
  to 32→16→8.
- SubSpectralNorm: S=5 (matching the paper). Falls back to plain BatchNorm2d
  when the frequency dimension is not divisible by 5 (this occurs in some
  intermediate layers due to our different input frequency).
- Time frames: 101 (paper: 98). Head uses AdaptiveAvgPool so this is irrelevant.

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

**Pretrained-weight status:**
  Built with `pretrained=False` by default (offline use). The `pretrained_loaded`
  flag is `False` in all reported results. Pretrained weights (~330 MB) can be
  loaded by passing `pretrained=True`, which requires internet access to the
  timm HuggingFace hub. This has NOT been verified in the current environment.
  The `verify_models.py --pretrained` flag exposes this path; it is a separate
  concern from the offline architecture verification.

---

### Model E — LoRAAST (Speech Commands)

- Architecture: Audio Spectrogram Transformer (AST) style, built from scratch.
- Base: ViT-style transformer (embed_dim=192, depth=12, 3 heads) with a 2-D
  patch embedding over the log-mel spectrogram, matching the architectural
  design of Gong et al. 2021.
- LoRA: r=8, α=16, applied to the top 6 blocks' `qkv` and `fc1`.
- Trainable: **89,699** (adapters + classification head). < 2M asserted.
- Frozen: **5,393,856** (base AST weights, randomly initialized).
- Input: (1, 64, 101) log-mel. Patch=16×16, time axis zero-padded to 112 →
  4×7=28 patches.
- Reference: Gong et al., "AST: Audio Spectrogram Transformer",
  Interspeech 2021 (arXiv:2104.01778).

**timm AST availability:**
  timm 1.0.30 does NOT contain a native AST model. A registry search for
  'ast', 'audio_spectrogram', 'wav2vec', and 'hubert' returned zero matches
  (the substring 'ast' only matches FasteRNet/FastViT models coincidentally).
  The AST architecture is mathematically equivalent to a ViT applied to 2-D
  spectrograms; we implement it directly from the ViT building blocks in
  `vit_tiny.py` (TransformerBlock, Attention, MLP), with a specialized
  SpectrogramPatchEmbed module.

**No pretrained AST weights:**
  Gong et al. fine-tune a DeiT-Base pretrained on ImageNet-21k. No offline
  checkpoint of this specific fine-tuned model is available in this environment.
  The base parameters are randomly initialized. This is a known limitation that
  will affect convergence in experiments — the LoRAAST model requires significant
  training from scratch rather than fast adaptation from a strong prior.
  Results from this model should be interpreted accordingly.

**Deviations from Gong 2021:**
  1. Non-overlapping patches (stride=16) instead of the paper's overlapping
     patches (stride=10). This reduces the sequence from ~248 tokens to 28,
     which lowers compute but reduces temporal resolution.
  2. Random base initialization instead of pretrained DeiT-Base weights.
  3. 64 mel bins instead of 128 (the paper uses 128 for AudioSet experiments).
     Our input (1, 64, 101) is defined by the Speech Commands preprocessing.

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
- No federated training or attack/defense evaluation has occurred yet.
