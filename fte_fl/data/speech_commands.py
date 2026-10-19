"""Speech Commands v0.02 loader with log-mel spectrogram features.

Uses torch.stft (no torchaudio) to compute log-mel spectrograms:
  64 mel bins, n_fft=400, hop=160, 1-second clips at 16 kHz
  -> tensor shape (1, 64, 101).

Each speaker (identified by the first 8 hex chars of the filename
before '_nohash_') is one federated client.

References:
  Warden, "Launching Speech Commands", 2018.
  Speech Commands v0.02: 35 command words, ~105k clips, 16 kHz mono.
"""
from __future__ import annotations

import glob
import os
import random
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 35 command words (excluding _background_noise_)
SPEECH_LABELS: list[str] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
]
_LABEL_TO_IDX = {label: i for i, label in enumerate(SPEECH_LABELS)}


def load_wav(path: str) -> np.ndarray:
    """Load a .wav file as a 1-D float32 array normalized to [-1, 1].

    Uses Python's built-in wave module (no torchaudio).
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"WAV file not found: {path}")
    with wave.open(path, "r") as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data /= 32768.0
    elif sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        data /= 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        data = data[::n_channels]  # take first channel

    return data


def mel_filterbank(
    n_mels: int = 64,
    n_fft: int = 400,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Create a mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).

    Uses the standard HTK mel scale: mel = 2595 * log10(1 + f / 700).
    """
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]
        for k in range(left, center):
            if center > left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m, k] = (right - k) / (right - center)
    return torch.from_numpy(fb)


def wav_to_logmel(
    path: str,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
    duration: float = 1.0,
) -> torch.Tensor:
    """Convert a .wav file to a log-mel spectrogram tensor.

    Returns shape (1, n_mels, n_frames) where n_frames = 101 for
    1-second clips at 16 kHz with hop=160 and center=True STFT.

    The clip is zero-padded or truncated to exactly `duration` seconds.
    """
    waveform = load_wav(path)

    target_len = int(sample_rate * duration)
    if len(waveform) > target_len:
        waveform = waveform[:target_len]
    elif len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))

    tensor = torch.from_numpy(waveform)

    window = torch.hamming_window(n_fft)
    stft = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    # stft shape: (n_fft // 2 + 1, n_frames) = (201, 101)

    magnitude = torch.abs(stft)
    fb = mel_filterbank(n_mels, n_fft, sample_rate)  # (n_mels, 201)
    mel_spec = fb @ magnitude  # (n_mels, n_frames)

    log_mel = torch.log(mel_spec + 1e-6)
    return log_mel.unsqueeze(0)  # (1, n_mels, n_frames)


class SpeechCommandsDataset(Dataset):
    """Torch dataset for a single speaker's Speech Commands data."""

    def __init__(self, filepaths: list[str]):
        self.filepaths = filepaths
        # Precompute labels from directory names
        self.labels = []
        for fp in filepaths:
            word = Path(fp).parent.name
            self.labels.append(_LABEL_TO_IDX.get(word, -1))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = wav_to_logmel(self.filepaths[idx])
        return spec, self.labels[idx]


def list_speakers(data_dir: str) -> list[str]:
    """Return sorted list of unique speaker IDs in the dataset.

    Speaker ID is the first 8-char hex prefix of the filename
    (before '_nohash_').
    """
    data_dir = Path(data_dir)
    speakers = set()
    for word_dir in sorted(data_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("."):
            continue
        if word_dir.name == "_background_noise_":
            continue
        for wav in word_dir.glob("*.wav"):
            basename = wav.name
            # Speaker ID is everything before the first '_'
            speaker_id = basename.split("_")[0]
            if speaker_id:
                speakers.add(speaker_id)
    return sorted(speakers)


def build_speech_clients(
    data_dir: str,
    num_clients: int = 800,
    seed: int = 42,
    split: str = "train",
) -> dict[str, SpeechCommandsDataset]:
    """Build per-speaker datasets from Speech Commands v0.02.

    Args:
        data_dir: Path to the speech_commands_v0.02 directory.
        num_clients: Desired number of clients. Capped at available speakers.
        seed: Random seed for reproducible client selection.
        split: Currently both splits use the same pool (LEAF-style
               per-writer partitioning is not available for this dataset).

    Returns:
        Dict mapping speaker_id -> SpeechCommandsDataset.
    """
    data_dir = Path(data_dir)

    # Group files by speaker
    speaker_files: dict[str, list[str]] = {}
    for word_dir in sorted(data_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("."):
            continue
        if word_dir.name == "_background_noise_":
            continue
        for wav in word_dir.glob("*.wav"):
            speaker_id = wav.name.split("_")[0]
            if speaker_id not in speaker_files:
                speaker_files[speaker_id] = []
            speaker_files[speaker_id].append(str(wav))

    available = len(speaker_files)
    n = min(num_clients, available)

    rng = random.Random(seed)
    speaker_ids = sorted(speaker_files.keys())
    selected = rng.sample(speaker_ids, n)

    return {sid: SpeechCommandsDataset(speaker_files[sid]) for sid in selected}
