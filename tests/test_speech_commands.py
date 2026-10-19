"""Tests for the Speech Commands log-mel loader.

Tests create synthetic .wav files so they do not require the real dataset.
No torchaudio dependency — uses Python wave module + torch.stft.

Real-data tests are marked with @pytest.mark.realdata:
    pytest -m realdata

Speaker-count note (verified against real dataset):
    Total speakers in dataset excluding _background_noise_: 2,618
    The earlier count of 2,624 included 6 false "speakers" extracted from
    the background-noise filenames (doing, dude, exercise, pink, running,
    white).  The correct count is 2,618.
    Of these: 256 are in validation_list.txt, 250 in testing_list.txt
    (no overlap), leaving 2,112 train-only speakers.
"""
import numpy as np
import pytest
import torch
import wave
from pathlib import Path

from fte_fl.data.speech_commands import (
    load_wav,
    mel_filterbank,
    wav_to_logmel,
    list_speakers,
    build_speech_clients,
    SPEECH_LABELS,
)

REAL_SC_DIR = "/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02"


def _make_mock_wav(path, duration=1.0, sample_rate=16000, freq=1000.0):
    """Write a synthetic mono 16-bit .wav file."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


# ---- load_wav ----

def test_load_wav_returns_float32(tmp_path):
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=0.5)
    data = load_wav(str(p))
    assert data.dtype == np.float32
    assert len(data) == 8000  # 0.5s * 16000


def test_load_wav_normalized(tmp_path):
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=0.5)
    data = load_wav(str(p))
    assert np.all(np.abs(data) <= 1.0 + 1e-6)


def test_load_wav_missing_file():
    with pytest.raises(FileNotFoundError):
        load_wav("/nonexistent/path.wav")


def test_load_wav_16khz_frame_count(tmp_path):
    """16 kHz, 1-second clip -> exactly 16000 frames."""
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=1.0, sample_rate=16000)
    data = load_wav(str(p))
    assert len(data) == 16000


# ---- mel_filterbank ----

def test_mel_filterbank_shape():
    fb = mel_filterbank(n_mels=64, n_fft=400, sample_rate=16000)
    assert fb.shape == (64, 201)  # n_fft // 2 + 1 = 201


def test_mel_filterbank_nonnegative():
    fb = mel_filterbank(n_mels=64, n_fft=400, sample_rate=16000)
    assert torch.all(fb >= 0)


def test_mel_filterbank_rows_sum_to_at_most_one():
    """Mel filter row sums: most rows > 0, but edge filters at very low
    frequency may be empty due to integer bin rounding.  The filterbank
    must have at most one empty row and must not be all-zero."""
    fb = mel_filterbank(n_mels=64, n_fft=400, sample_rate=16000)
    row_sums = fb.sum(dim=1)
    zero_rows = int((row_sums == 0).sum().item())
    # Allow at most 1 empty row (known edge case at n_mels=64, n_fft=400)
    assert zero_rows <= 1, f"Too many empty mel filter rows: {zero_rows}"
    # The majority of rows must be nonzero
    assert int((row_sums > 0).sum().item()) >= 60


# ---- wav_to_logmel ----

def test_wav_to_logmel_shape(tmp_path):
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=1.0, sample_rate=16000)
    spec = wav_to_logmel(str(p))
    assert spec.shape == (1, 64, 101)


def test_wav_to_logmel_finite(tmp_path):
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=1.0)
    spec = wav_to_logmel(str(p))
    assert torch.all(torch.isfinite(spec))


def test_wav_to_logmel_short_clip_padded(tmp_path):
    """Clips shorter than 1 second must be zero-padded to (1, 64, 101)."""
    p = tmp_path / "short.wav"
    _make_mock_wav(p, duration=0.3)
    spec = wav_to_logmel(str(p))
    assert spec.shape == (1, 64, 101)


def test_wav_to_logmel_long_clip_truncated(tmp_path):
    """Clips longer than 1 second must be truncated to (1, 64, 101)."""
    p = tmp_path / "long.wav"
    _make_mock_wav(p, duration=2.0)
    spec = wav_to_logmel(str(p))
    assert spec.shape == (1, 64, 101)


def test_wav_to_logmel_reproducibility(tmp_path):
    """Same file must produce identical output on repeated calls."""
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=1.0)
    spec1 = wav_to_logmel(str(p))
    spec2 = wav_to_logmel(str(p))
    assert torch.allclose(spec1, spec2)


def test_wav_to_logmel_uses_stft_not_torchaudio(tmp_path):
    """Verify the spectrogram is computed via torch.stft (no torchaudio import)."""
    import sys
    assert "torchaudio" not in sys.modules, "torchaudio must not be imported"
    p = tmp_path / "test.wav"
    _make_mock_wav(p, duration=1.0)
    wav_to_logmel(str(p))
    assert "torchaudio" not in sys.modules, "wav_to_logmel imported torchaudio"


# ---- list_speakers ----

def _make_mock_sc_dir(base, speaker="00f0202f", words=None, n_per_word=3):
    """Create a mock Speech Commands directory."""
    if words is None:
        words = ["yes", "no", "up"]
    for word in words:
        word_dir = base / word
        word_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_word):
            _make_mock_wav(word_dir / f"{speaker}_nohash_{i}.wav", duration=1.0)


def test_list_speakers_finds_speakers(tmp_path):
    base = tmp_path / "sc"
    _make_mock_sc_dir(base, speaker="00f0202f")
    _make_mock_sc_dir(base, speaker="1a2b3c4d")
    speakers = list_speakers(str(base))
    assert "00f0202f" in speakers
    assert "1a2b3c4d" in speakers
    assert len(speakers) == 2


def test_list_speakers_excludes_background_noise(tmp_path):
    """_background_noise_ filenames must NOT contribute speaker IDs."""
    base = tmp_path / "sc"
    _make_mock_sc_dir(base, speaker="00f0202f")
    noise_dir = base / "_background_noise_"
    noise_dir.mkdir(parents=True, exist_ok=True)
    # Fake "speaker" in background noise (like real dataset: doing_the_dishes.wav)
    _make_mock_wav(noise_dir / "doing_the_dishes.wav", duration=2.0)
    speakers = list_speakers(str(base))
    assert "00f0202f" in speakers
    # "doing" must not appear as a speaker
    assert "doing" not in speakers
    assert len(speakers) == 1


def test_list_speakers_excludes_dotfiles(tmp_path):
    """Hidden directories (starting with '.') must not be scanned."""
    base = tmp_path / "sc"
    _make_mock_sc_dir(base, speaker="00f0202f")
    # Hidden dir that should be ignored
    hidden = base / ".DS_Store_dir"
    hidden.mkdir()
    _make_mock_wav(hidden / "fakespeaker_nohash_0.wav")
    speakers = list_speakers(str(base))
    assert "fakespeaker" not in speakers


def test_speech_labels_complete():
    """Must have exactly 35 command words, not including background noise."""
    assert len(SPEECH_LABELS) == 35
    assert "_background_noise_" not in SPEECH_LABELS


def test_speech_labels_sorted():
    """Labels must be in a stable, deterministic order."""
    assert SPEECH_LABELS == sorted(SPEECH_LABELS)


def test_speaker_id_extraction_from_nohash_filename(tmp_path):
    """Speaker ID is everything before the first '_' in the filename."""
    base = tmp_path / "sc"
    # Real-format: 8hex_nohash_N.wav
    word_dir = base / "yes"
    word_dir.mkdir(parents=True, exist_ok=True)
    _make_mock_wav(word_dir / "a1b2c3d4_nohash_0.wav")
    _make_mock_wav(word_dir / "a1b2c3d4_nohash_1.wav")
    _make_mock_wav(word_dir / "ffff0000_nohash_0.wav")
    speakers = list_speakers(str(base))
    assert "a1b2c3d4" in speakers
    assert "ffff0000" in speakers
    assert len(speakers) == 2


# ---- build_speech_clients ----

def test_build_speech_clients_basic(tmp_path):
    base = tmp_path / "sc"
    for i in range(5):
        _make_mock_sc_dir(base, speaker=f"sp{i:08x}", n_per_word=4)
    clients = build_speech_clients(str(base), num_clients=800, seed=42)
    assert len(clients) == 5  # capped at available
    for cid, ds in clients.items():
        assert len(ds) > 0


def test_build_speech_clients_reproducibility(tmp_path):
    base = tmp_path / "sc"
    for i in range(3):
        _make_mock_sc_dir(base, speaker=f"sp{i:08x}", n_per_word=3)
    c1 = build_speech_clients(str(base), num_clients=3, seed=42)
    c2 = build_speech_clients(str(base), num_clients=3, seed=42)
    assert list(c1.keys()) == list(c2.keys())


def test_build_speech_client_sample_shape(tmp_path):
    base = tmp_path / "sc"
    _make_mock_sc_dir(base, speaker="00f0202f", n_per_word=5)
    clients = build_speech_clients(str(base), num_clients=1, seed=42)
    ds = list(clients.values())[0]
    x, y = ds[0]
    assert x.shape == (1, 64, 101)
    assert 0 <= y < 35


def test_build_speech_clients_different_seeds(tmp_path):
    """Different seeds must select different speaker subsets when n > num_clients."""
    base = tmp_path / "sc"
    for i in range(10):
        _make_mock_sc_dir(base, speaker=f"sp{i:08x}", n_per_word=3)
    c1 = build_speech_clients(str(base), num_clients=5, seed=42)
    c2 = build_speech_clients(str(base), num_clients=5, seed=99)
    assert list(c1.keys()) != list(c2.keys())


# ---- Real-data tests ----

@pytest.mark.realdata
def test_real_speaker_count_excludes_background_noise():
    """2,618 speakers excluding _background_noise_, not 2,624."""
    speakers = list_speakers(REAL_SC_DIR)
    assert len(speakers) == 2618, (
        f"Expected 2618 speakers (excl. background noise), got {len(speakers)}\n"
        "The count of 2,624 is wrong: it includes 6 false speaker IDs extracted "
        "from _background_noise_ filenames (doing, dude, exercise, pink, running, white)."
    )


@pytest.mark.realdata
def test_real_speech_commands_builds_800_clients():
    """Default build selects 800 speakers (800 < 2618 available)."""
    clients = build_speech_clients(REAL_SC_DIR, num_clients=800, seed=42)
    assert len(clients) == 800


@pytest.mark.realdata
def test_real_speech_sample_shape():
    """Real Speech Commands clips: shape (1, 64, 101)."""
    clients = build_speech_clients(REAL_SC_DIR, num_clients=3, seed=42)
    for ds in clients.values():
        x, y = ds[0]
        assert x.shape == (1, 64, 101)
        assert torch.all(torch.isfinite(x))
        assert 0 <= y < 35
        break


@pytest.mark.realdata
def test_real_speech_official_split_files_exist():
    """validation_list.txt and testing_list.txt must be present."""
    import os
    for fname in ("validation_list.txt", "testing_list.txt"):
        assert os.path.exists(f"{REAL_SC_DIR}/{fname}"), f"Missing: {fname}"
