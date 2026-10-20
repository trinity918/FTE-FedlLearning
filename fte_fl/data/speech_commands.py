"""Speech Commands v0.02 loader with log-mel spectrogram features.

Uses torch.stft (no torchaudio) to compute log-mel spectrograms:
  64 mel bins, n_fft=400, hop=160, 1-second clips at 16 kHz
  -> tensor shape (1, 64, 101).

Each speaker (identified by the first 8 hex chars of the filename
before '_nohash_') is one federated client.

Official partition: the dataset's README (v0.02, "Partitioning") assigns
each file to training / validation / testing by a SHA-1 hash of the part
of the file name before '_nohash_' (i.e. the speaker id), and ships the
result as validation_list.txt and testing_list.txt; every other file is
training.  `official_split_files` reads those two lists (it does not
recompute the hash) and raises if they are missing, list files that do not
exist, overlap, or put one speaker in two partitions.  Federated clients
are drawn only from training speakers; validation and testing files are
used only by `build_speech_eval`.

References:
  Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition", arXiv:1804.03209, 2018.
  Speech Commands v0.02: 35 command words, 105,829 clips, 16 kHz mono.
"""
from __future__ import annotations

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
            if word not in _LABEL_TO_IDX:
                raise ValueError(f"{fp}: directory {word!r} is not one of the 35 command words")
            self.labels.append(_LABEL_TO_IDX[word])

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


SPLITS = ("training", "validation", "testing")


def speaker_of(relpath: str) -> str:
    """Speaker id of a clip path such as 'yes/3cfc6b3a_nohash_2.wav'."""
    name = Path(relpath).name
    if "_nohash_" not in name:
        raise ValueError(f"clip name {name!r} has no '_nohash_' speaker marker")
    return name.split("_nohash_")[0]


def _list_clips(data_dir: Path) -> list[str]:
    """All command-word clips as sorted 'word/file.wav' paths (background noise excluded)."""
    clips = []
    for word_dir in sorted(data_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("."):
            continue
        if word_dir.name == "_background_noise_":
            continue
        clips.extend(f"{word_dir.name}/{wav.name}" for wav in word_dir.glob("*.wav"))
    return sorted(clips)


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"official split list {path} is missing; refusing to invent a split"
        )
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(set(entries)) != len(entries):
        raise ValueError(f"{path.name} contains duplicate entries")
    return entries


def official_split_files(data_dir: str) -> dict[str, list[str]]:
    """The official v0.02 partition as {split: sorted 'word/file.wav' paths}.

    validation/testing come from validation_list.txt / testing_list.txt;
    every other command-word clip is training (README, "Partitioning").
    Raises if a list is missing, has duplicates or entries absent on disk,
    if the two lists overlap, if a split is empty, or if any speaker
    appears in more than one split.
    """
    data_dir = Path(data_dir)
    clips = _list_clips(data_dir)
    on_disk = set(clips)
    val = _read_list(data_dir / "validation_list.txt")
    test = _read_list(data_dir / "testing_list.txt")
    for name, entries in (("validation_list.txt", val), ("testing_list.txt", test)):
        missing = [e for e in entries if e not in on_disk]
        if missing:
            raise ValueError(f"{name} lists {len(missing)} clips not on disk, e.g. {missing[0]!r}")
    if set(val) & set(test):
        raise ValueError("validation_list.txt and testing_list.txt overlap")
    held_out = set(val) | set(test)
    splits = {
        "training": [c for c in clips if c not in held_out],
        "validation": sorted(val),
        "testing": sorted(test),
    }
    speakers = {}
    for split, files in splits.items():
        if not files:
            raise ValueError(f"the official {split} split is empty")
        speakers[split] = {speaker_of(f) for f in files}
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            shared = speakers[SPLITS[a]] & speakers[SPLITS[b]]
            if shared:
                raise ValueError(
                    f"{len(shared)} speakers appear in both {SPLITS[a]} and {SPLITS[b]}"
                )
    return splits


def build_speech_clients(
    data_dir: str,
    num_clients: int = 800,
    seed: int = 42,
    split: str = "train",
) -> dict[str, SpeechCommandsDataset]:
    """Per-speaker client datasets drawn only from the official TRAINING split.

    Args:
        data_dir: Path to the speech_commands_v0.02 directory (with
                  validation_list.txt and testing_list.txt).
        num_clients: Desired number of clients; capped at the number of
                     training speakers.
        seed: Random seed for reproducible speaker selection
              (random.Random(seed).sample over the sorted speaker ids).
        split: must be "train"; validation and testing clips are never
               client data (use `build_speech_eval`).

    Returns:
        Dict mapping speaker_id -> SpeechCommandsDataset (that speaker's
        training clips, sorted by path), in selection order.
    """
    if split != "train":
        raise ValueError(
            f"clients are built from the training split only, got split={split!r}; "
            "use build_speech_eval for validation/testing"
        )
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients < 1:
        raise ValueError(f"num_clients must be a positive int, got {num_clients!r}")
    data_dir = Path(data_dir)
    speaker_files: dict[str, list[str]] = {}
    for rel in official_split_files(str(data_dir))["training"]:
        speaker_files.setdefault(speaker_of(rel), []).append(str(data_dir / rel))

    n = min(num_clients, len(speaker_files))
    rng = random.Random(seed)
    selected = rng.sample(sorted(speaker_files), n)
    return {sid: SpeechCommandsDataset(speaker_files[sid]) for sid in selected}


def build_speech_eval(data_dir: str, split: str) -> SpeechCommandsDataset:
    """All clips of the official 'validation' or 'testing' split, sorted by path."""
    if split not in ("validation", "testing"):
        raise ValueError(f"split must be 'validation' or 'testing', got {split!r}")
    data_dir = Path(data_dir)
    return SpeechCommandsDataset([str(data_dir / rel) for rel in official_split_files(str(data_dir))[split]])
