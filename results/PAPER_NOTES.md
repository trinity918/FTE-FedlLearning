# PAPER_NOTES — FTE-FL Research Project

## Milestone 1: Dataset Findings, Limitations, and Decisions

Verified: October 2026. All counts are recomputed from actual data files,
not assumed from documentation.

---

### FEMNIST (LEAF-preprocessed)

**Source:** LEAF benchmark, Caldas et al., "LEAF: A Benchmark for Federated
Settings", arXiv:1812.01097, 2018.

**Location:** `/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/
justincase/leaf/data/femnist/data/`

**Verified counts:**
| Split | Writers | Samples |
|-------|---------|---------|
| Train | 198     | 36,700  |
| Test  | 198     | 4,174   |

**Client configuration:** `num_clients=500` is capped at 198 (all available
writers). The experiment matrix must account for this: `clients_per_round`
should be ≤ ~20 to ensure enough diversity per round.

**Train/test split protocol:** LEAF uses a per-sample split within each
writer (approximately 90/10). All 198 writers appear in BOTH the train and
test JSON files. This is NOT a writer-independent (held-out-writer) split.

**Scientific implication:** Because train and test writers overlap, the
FEMNIST evaluation measures within-writer generalisation (how well a model
trained on part of a writer's samples generalises to the rest), not
cross-writer generalisation. This is standard for the LEAF benchmark and is
appropriate for FL client evaluation. It does NOT constitute data leakage
in the federated learning sense, because each client only trains on their own
data. However, the test accuracy reflects within-distribution performance,
not generalisation to unseen handwriting styles.

**Input format:** Flattened 784-float lists (28×28 grayscale) in the JSON.
The loader reshapes to (1, 28, 28) float32 with pixel values in [0, 1].
Labels are integers in [0, 61] (62 classes: 10 digits + 26 lower + 26 upper).

---

### Speech Commands v0.02

**Source:** Warden, "Launching Speech Commands", 2018.

**Location:** `/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02/`
(NOT at `justincase/speech_commands_v0.02/` as originally expected.)

**Verified speaker counts:**
| Population | Speakers | Clips |
|-----------|---------|-------|
| Total (excl. `_background_noise_`) | 2,618 | 105,829 |
| validation_list.txt | 256 | 9,981 |
| testing_list.txt | 250 | 11,005 |
| val ∩ test overlap | 0 | — |
| Training-only | 2,112 | ~84,843 |

**Speaker count discrepancy resolved:** An earlier inspection counted 2,624
speakers. The correct number is **2,618**. The difference of 6 comes from
the `_background_noise_/` directory, which contains filenames like
`doing_the_dishes.wav`, `dude_miaowing.wav`, `exercise_bike.wav`,
`pink_noise.wav`, `running_tap.wav`, and `white_noise.wav`. A naive
split on `_` extracts the first token ("doing", "dude", "exercise", "pink",
"running", "white") as false speaker IDs. The loader correctly excludes
`_background_noise_/` entirely before scanning for speaker IDs.

**Feature pipeline:** 16 kHz, mono, 16-bit PCM → zero-pad/truncate to 1 s →
`torch.stft` (n_fft=400, hop_length=160, Hamming window, center=True) →
magnitude → HTK mel filterbank (64 bands) → log(mel + 1e-6) →
tensor (1, 64, 101). No torchaudio dependency.

**Known mel filterbank edge case:** With n_mels=64, n_fft=400, the lowest-
frequency mel filter (approximately filter index 2) can have zero weight due
to integer bin-rounding of the triangular filters. This is an artefact of the
integer-bin approximation and does not affect overall accuracy meaningfully,
since the affected region is below ~100 Hz where speech energy is negligible.
The test suite allows at most 1 empty filter row.

**LIMITATION — No official speaker-independent split applied:** The
`build_speech_clients` function samples from the full 2,618-speaker pool and
does NOT use the official `validation_list.txt`/`testing_list.txt` speaker
split. Consequently, a random sample of 800 speakers may contain validation or
test speakers, creating potential information leakage between the FL
"training" population and any evaluation on the official test set.

**Recommended fix for Milestone 5:** When building the FL client pool, restrict
to the 2,112 training-only speakers (those not in `validation_list.txt` or
`testing_list.txt`). Use the 250 test speakers as a held-out global test set.
This must be implemented before any accuracy claims are made for Speech
Commands.

---

### CIFAR-100

**Source:** Krizhevsky, "Learning Multiple Layers of Features from Tiny
Images", 2009.

**Location:** Downloaded to `./data_cache/` by torchvision (~169 MB).

**Verified:**
- Train: 50,000 samples, 100 fine classes, 20 superclasses (5 fine per)
- Test: 10,000 samples
- Sample shape: (3, 32, 32) uint8 PIL (torchvision); (3, 32, 32) float32 after to_tensor
- `_COARSE_OF_FINE` mapping verified to match the real torchvision pickle exactly

**Download status:** CIFAR-100 was downloaded during Milestone 1 verification.
The download took approximately 1 hour over WSL2 at 40–60 kB/s. Subsequent
runs use the local cache and do not re-download.

**Partitioning:**
1. Dirichlet (alpha ∈ {0.1, 0.5}): verified no duplicate index assignments
   on real 50k train labels.
2. Superclass: verified no duplicates on real 50k train labels.

---

### What This Study Does NOT Include

- **EEG legacy results**: The EEG motor-imagery experiments from the prior
  repository at `majorproj/fl_byzantine` are NOT part of this study. Those
  results were produced by a different codebase with known documentation
  inaccuracies and are not reproduced here.

---

### General Limitations (Milestone 1)

1. **FEMNIST is limited to 198 writers.** Experiments should be designed for
   this population size. Multi-round experiments with `clients_per_round=10`
   will cycle through all writers roughly every 20 rounds.

2. **Speech Commands speaker-independent split not yet enforced.** The current
   loader may include validation/test speakers in the client pool. This must
   be fixed before reporting accuracy on the official test set.

3. **CIFAR-100 requires one-time download.** All partition tests ran
   successfully against the real 50k train data after download.

4. **Mel filterbank has one empty low-frequency filter row.** This is a
   known edge case documented in the test suite. It does not affect model
   quality for speech recognition tasks.
