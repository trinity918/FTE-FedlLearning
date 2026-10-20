"""DBA-style backdoor triggers (Xie et al., ICLR 2020).

Trigger insertion only: labels are poisoned separately by
`fte_fl.attacks.labels.backdoor_labels`.  Inputs are never modified.  This
module reproduces DBA's trigger decomposition, not the full DBA attack: its
training schedule, model-update scaling and poisoning-per-batch procedure
belong to the simulator and are not implemented here.

Distributed structure (as in DBA): each family's global trigger is the union
of NUM_TRIGGER_PARTS local triggers.  Malicious client i (i = 0..3) uses
part i during training; `parts=None` applies the full global trigger, as
used at test time.  Applying parts one after another equals applying their
union.

Image triggers (cifar100, femnist)
  Pixel coordinates are copied from the official DBA configs
  (github.com/AI-secure/DBA, utils/cifar_params.yaml and mnist_params.yaml):
  four horizontal bars in the top-left corner, 6 px long for CIFAR and 4 px
  long for 28x28 MNIST, which we reuse for FEMNIST.  Inputs must be
  unnormalized float images in [0, 1] of shape (C, H, W) or (N, C, H, W);
  apply the trigger before any normalization.
    cifar100: (3, 32, 32), trigger pixels set to 1.0 in all channels (white),
              as in DBA.  The CIFAR-100 loader currently returns PIL images;
              these coordinates assume `to_tensor` conversion to [0, 1].
    femnist:  (1, 28, 28), trigger pixels set to 0.0 (black).  Adaptation:
              DBA sets MNIST pixels to 1 (white on a black background), but
              LEAF FEMNIST is dark ink on a white (1.0) background, so a white
              trigger would be invisible there.

Speech trigger (speech_commands): DBA-inspired FEATURE-SPACE adaptation
  DBA has no audio setting.  The trigger is constructed directly in log-mel
  space; it is not injected as a physical waveform during training.
  The loader only exposes log-mel features x = log(mel + 1e-6) of shape
  (1, 64, 101) (64 HTK mel bins, n_fft=400 periodic Hamming, hop=160,
  16 kHz, 1 s).  No waveform is generated or modified.  Instead the trigger
  adds, in the linear mel-magnitude domain, the steady-state magnitude T of
  a 1 kHz tone of peak amplitude `amplitude` as the loader's STFT would see
  it:  x' = log(exp(x) + T) on cells where T > 0, x elsewhere.
  1 kHz is exactly FFT bin k0 = 25, and a periodic Hamming window
  (0.54 - 0.46 cos) gives STFT magnitudes 0.27*A*N at k0 and 0.115*A*N at
  k0 +- 1 (N = 400) and exactly 0 elsewhere; T = mel_filterbank @ that
  spectrum, repeated over the frames of the selected parts.
  Approximations: magnitudes are added, ignoring phase interaction with the
  clean signal; the tone is idealized as steady in every frame, so the
  onset/offset and reflect-padding edge effects a real tone would cause are
  absent.  Local parts are 4 contiguous time segments of the 101 frames
  (26, 25, 25, 25 frames); this split is a project adaptation.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from fte_fl.data.speech_commands import mel_filterbank

NUM_TRIGGER_PARTS: int = 4

# DBA local trigger i -> list of (row, col) pixels.
IMAGE_TRIGGER_PATTERNS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "cifar100": (
        tuple((0, c) for c in range(0, 6)),
        tuple((0, c) for c in range(9, 15)),
        tuple((4, c) for c in range(0, 6)),
        tuple((4, c) for c in range(9, 15)),
    ),
    "femnist": (
        tuple((0, c) for c in range(0, 4)),
        tuple((0, c) for c in range(6, 10)),
        tuple((3, c) for c in range(0, 4)),
        tuple((3, c) for c in range(6, 10)),
    ),
}
IMAGE_SHAPES: dict[str, tuple[int, int, int]] = {
    "cifar100": (3, 32, 32),
    "femnist": (1, 28, 28),
}
IMAGE_TRIGGER_VALUES: dict[str, float] = {"cifar100": 1.0, "femnist": 0.0}

# Must match fte_fl.data.speech_commands.wav_to_logmel defaults.
SPEECH_SHAPE: tuple[int, int, int] = (1, 64, 101)
SAMPLE_RATE: int = 16000
N_FFT: int = 400
LOG_MEL_EPS: float = 1e-6
_LOG_FLOOR_TOL: float = 1e-4


def _check_parts(parts: Iterable[int] | None) -> list[int]:
    if parts is None:
        return list(range(NUM_TRIGGER_PARTS))
    parts = list(parts)
    if not parts:
        raise ValueError("parts must be None (global trigger) or a non-empty list")
    for p in parts:
        if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < NUM_TRIGGER_PARTS:
            raise ValueError(f"trigger part {p!r} is not an int in [0, {NUM_TRIGGER_PARTS})")
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate trigger parts: {parts}")
    return sorted(parts)


def _check_input(x: torch.Tensor, shape: tuple[int, int, int]) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"input must be float32 or float64, got {x.dtype}")
    if x.ndim not in (3, 4) or tuple(x.shape[-3:]) != shape:
        raise ValueError(f"input must have shape {shape} or (N, *{shape}), got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("input contains NaN or infinity")


def image_trigger_mask(family: str, parts: Iterable[int] | None = None) -> torch.Tensor:
    """Boolean (H, W) mask of the trigger pixels for the given parts."""
    if family not in IMAGE_TRIGGER_PATTERNS:
        raise ValueError(f"unknown image family {family!r}; expected one of {sorted(IMAGE_SHAPES)}")
    _, h, w = IMAGE_SHAPES[family]
    mask = torch.zeros(h, w, dtype=torch.bool)
    for p in _check_parts(parts):
        for r, c in IMAGE_TRIGGER_PATTERNS[family][p]:
            mask[r, c] = True
    return mask


def apply_image_trigger(
    x: torch.Tensor, family: str, parts: Iterable[int] | None = None
) -> torch.Tensor:
    """Return a copy of image(s) `x` with the trigger pixels set in every channel."""
    mask = image_trigger_mask(family, parts)
    _check_input(x, IMAGE_SHAPES[family])
    if x.min() < 0 or x.max() > 1:
        raise ValueError("image values must lie in [0, 1] (apply before normalization)")
    out = x.clone()
    out[..., mask.to(x.device)] = IMAGE_TRIGGER_VALUES[family]
    return out


def speech_trigger_magnitude(
    parts: Iterable[int] | None = None,
    amplitude: float = 0.1,
    freq_hz: float = 1000.0,
) -> torch.Tensor:
    """Linear mel-magnitude pattern T of shape (64, 101) for the given parts.

    `amplitude` is the tone's peak amplitude on the loader's [-1, 1] waveform
    scale (default 0.1, i.e. -20 dBFS; a project choice, not from DBA).
    `freq_hz` must fall exactly on an FFT bin (multiples of 40 Hz).
    """
    if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)):
        raise TypeError("amplitude must be a real number")
    if not math.isfinite(amplitude) or not 0 < amplitude <= 1:
        raise ValueError(f"amplitude must lie in (0, 1], got {amplitude}")
    k0 = freq_hz * N_FFT / SAMPLE_RATE
    if not math.isfinite(k0) or k0 != int(k0) or not 2 <= k0 <= N_FFT // 2 - 2:
        raise ValueError(
            f"freq_hz={freq_hz} must be a multiple of {SAMPLE_RATE / N_FFT:g} Hz "
            f"between {2 * SAMPLE_RATE / N_FFT:g} and {(N_FFT // 2 - 2) * SAMPLE_RATE / N_FFT:g} Hz"
        )
    k0 = int(k0)
    _, n_mels, n_frames = SPEECH_SHAPE

    spectrum = torch.zeros(N_FFT // 2 + 1, dtype=torch.float64)
    spectrum[k0] = 0.27 * amplitude * N_FFT
    spectrum[k0 - 1] = spectrum[k0 + 1] = 0.115 * amplitude * N_FFT
    mel_column = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE).double() @ spectrum

    frames = torch.zeros(n_frames, dtype=torch.float64)
    segments = torch.tensor_split(torch.arange(n_frames), NUM_TRIGGER_PARTS)
    for p in _check_parts(parts):
        frames[segments[p]] = 1.0
    return torch.outer(mel_column, frames)


def apply_speech_trigger(
    x: torch.Tensor,
    parts: Iterable[int] | None = None,
    amplitude: float = 0.1,
    freq_hz: float = 1000.0,
) -> torch.Tensor:
    """Return a copy of log-mel feature(s) `x` with the feature-space tone trigger.

    `x` has shape (1, 64, 101) or (N, 1, 64, 101) and must be an
    unnormalized log-mel produced by the loader.  The only range check is the
    loader's floor: inputs with any value below log(1e-6) - 1e-4 are rejected.
    This does NOT detect normalized features in general: standardized
    (zero-mean, unit-variance) features lie above the floor and pass
    undetected, so the caller must apply the trigger before any
    normalization.  Cells outside the trigger are returned bit-for-bit
    unchanged.
    """
    _check_input(x, SPEECH_SHAPE)
    if x.min() < math.log(LOG_MEL_EPS) - _LOG_FLOOR_TOL:
        raise ValueError("input values fall below log(1e-6); expected unnormalized log-mel features")
    t = speech_trigger_magnitude(parts, amplitude, freq_hz).to(dtype=x.dtype, device=x.device)
    t = t.expand_as(x)
    return torch.where(t > 0, torch.log(torch.exp(x) + t), x)
