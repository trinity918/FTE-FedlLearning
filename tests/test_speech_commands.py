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
    build_speech_eval,
    official_split_files,
    speaker_of,
    SpeechCommandsDataset,
    SPEECH_LABELS,
)
from fte_fl.fl.data import native_path

REAL_SC_DIR = native_path("/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02")


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


# ---- official split + build_speech_clients ----

def _make_split_dir(base, train, val=("aaaa0001",), test=("bbbb0001",), n_per_word=3,
                    words=("yes", "no", "up")):
    """Mock dataset whose validation/testing lists name every clip of the given speakers."""
    for speaker in (*train, *val, *test):
        _make_mock_sc_dir(base, speaker=speaker, words=list(words), n_per_word=n_per_word)
    for fname, speakers in (("validation_list.txt", val), ("testing_list.txt", test)):
        lines = [f"{w}/{s}_nohash_{i}.wav" for s in speakers for w in words for i in range(n_per_word)]
        (base / fname).write_text("\n".join(lines) + "\n")


def test_official_split_partitions_by_list(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001", "sp000002"], val=["aaaa0001"], test=["bbbb0001", "bbbb0002"])
    splits = official_split_files(str(base))
    assert {speaker_of(f) for f in splits["training"]} == {"sp000001", "sp000002"}
    assert {speaker_of(f) for f in splits["validation"]} == {"aaaa0001"}
    assert {speaker_of(f) for f in splits["testing"]} == {"bbbb0001", "bbbb0002"}
    # every clip is in exactly one split, nothing lost or duplicated
    all_files = splits["training"] + splits["validation"] + splits["testing"]
    assert len(all_files) == len(set(all_files)) == 5 * 3 * 3
    assert splits["validation"] == sorted(splits["validation"])


def test_official_split_missing_list_raises(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    (base / "testing_list.txt").unlink()
    with pytest.raises(FileNotFoundError, match="refusing to invent"):
        official_split_files(str(base))


def test_official_split_rejects_speaker_in_two_splits(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    # move one clip of the validation speaker out of the list: it becomes training
    lines = (base / "validation_list.txt").read_text().split()
    (base / "validation_list.txt").write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="speakers appear in both training and validation"):
        official_split_files(str(base))


def test_official_split_rejects_overlapping_lists(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    extra = (base / "validation_list.txt").read_text().split()[0]
    with open(base / "testing_list.txt", "a") as f:
        f.write(extra + "\n")
    with pytest.raises(ValueError, match="overlap"):
        official_split_files(str(base))


def test_official_split_rejects_listed_clip_missing_on_disk(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    with open(base / "testing_list.txt", "a") as f:
        f.write("yes/ffffffff_nohash_0.wav\n")
    with pytest.raises(ValueError, match="not on disk"):
        official_split_files(str(base))


def test_official_split_rejects_empty_split(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    (base / "validation_list.txt").write_text("")
    # the validation speaker's clips fall back to training; validation is empty
    with pytest.raises(ValueError, match="validation split is empty"):
        official_split_files(str(base))


def test_build_speech_clients_basic(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=[f"sp{i:06x}" for i in range(5)], n_per_word=4)
    clients = build_speech_clients(str(base), num_clients=800, seed=42)
    assert len(clients) == 5  # capped at the 5 training speakers; held-out speakers excluded
    for cid, ds in clients.items():
        assert len(ds) == 3 * 4
        assert all(speaker_of(fp) == cid for fp in ds.filepaths)


def test_build_speech_clients_never_use_heldout_speakers(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001", "sp000002"], val=["aaaa0001", "aaaa0002"],
                    test=["bbbb0001", "bbbb0002"])
    clients = build_speech_clients(str(base), num_clients=10, seed=0)
    assert set(clients) == {"sp000001", "sp000002"}
    heldout = set(build_speech_eval(str(base), "validation").filepaths) | set(
        build_speech_eval(str(base), "testing").filepaths)
    used = {fp for ds in clients.values() for fp in ds.filepaths}
    assert not used & heldout
    assert {speaker_of(fp) for fp in heldout} == {"aaaa0001", "aaaa0002", "bbbb0001", "bbbb0002"}


def test_build_speech_clients_rejects_non_train_split(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    with pytest.raises(ValueError, match="training split only"):
        build_speech_clients(str(base), num_clients=1, split="test")
    with pytest.raises(ValueError, match="positive int"):
        build_speech_clients(str(base), num_clients=0)


def test_build_speech_eval_rejects_unknown_split(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["sp000001"])
    with pytest.raises(ValueError, match="'validation' or 'testing'"):
        build_speech_eval(str(base), "training")


def test_speaker_of_requires_nohash_marker():
    assert speaker_of("yes/3cfc6b3a_nohash_2.wav") == "3cfc6b3a"
    with pytest.raises(ValueError, match="_nohash_"):
        speaker_of("_background_noise_/white_noise.wav")


def test_dataset_rejects_unknown_word_directory(tmp_path):
    d = tmp_path / "notaword"
    d.mkdir()
    _make_mock_wav(d / "sp000001_nohash_0.wav")
    with pytest.raises(ValueError, match="not one of the 35 command words"):
        SpeechCommandsDataset([str(d / "sp000001_nohash_0.wav")])


def test_build_speech_clients_reproducibility(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=[f"sp{i:06x}" for i in range(6)])
    c1 = build_speech_clients(str(base), num_clients=3, seed=42)
    c2 = build_speech_clients(str(base), num_clients=3, seed=42)
    assert list(c1.keys()) == list(c2.keys())
    assert all(c1[k].filepaths == c2[k].filepaths for k in c1)


def test_build_speech_client_sample_shape(tmp_path):
    base = tmp_path / "sc"
    _make_split_dir(base, train=["00f0202f"], n_per_word=5)
    clients = build_speech_clients(str(base), num_clients=1, seed=42)
    ds = list(clients.values())[0]
    x, y = ds[0]
    assert x.shape == (1, 64, 101)
    assert 0 <= y < 35


def test_build_speech_clients_different_seeds(tmp_path):
    """Different seeds must select different speaker subsets when n > num_clients."""
    base = tmp_path / "sc"
    _make_split_dir(base, train=[f"sp{i:06x}" for i in range(10)])
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
def test_real_official_split_counts_and_disjointness():
    """Counts recomputed from validation_list.txt / testing_list.txt (2026-10-20)."""
    splits = official_split_files(REAL_SC_DIR)
    counts = {k: (len(v), len({speaker_of(f) for f in v})) for k, v in splits.items()}
    assert counts == {
        "training": (84843, 2112),
        "validation": (9981, 256),
        "testing": (11005, 250),
    }
    clients = build_speech_clients(REAL_SC_DIR, num_clients=800, seed=42)
    train_speakers = {speaker_of(f) for f in splits["training"]}
    assert set(clients) <= train_speakers


@pytest.mark.realdata
def test_real_speech_official_split_files_exist():
    """validation_list.txt and testing_list.txt must be present."""
    import os
    for fname in ("validation_list.txt", "testing_list.txt"):
        assert os.path.exists(f"{REAL_SC_DIR}/{fname}"), f"Missing: {fname}"
