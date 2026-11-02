"""Tests for the native-Windows path mapping and the Speech Commands feature cache."""
import sys
import wave

import numpy as np
import pytest
import torch

from fte_fl.data import speech_cache
from fte_fl.data.speech_commands import wav_to_logmel
from fte_fl.fl import data


@pytest.mark.parametrize("platform, given, expected", [
    ("win32", "/mnt/c/Users/x/data", "C:/Users/x/data"),
    ("win32", "/mnt/d/a", "D:/a"),
    ("win32", "./data_cache", "./data_cache"),
    ("win32", "C:/already/native", "C:/already/native"),
    ("linux", "/mnt/c/Users/x/data", "/mnt/c/Users/x/data"),
])
def test_native_path(monkeypatch, platform, given, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert data.native_path(given) == expected


def _write_wav(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.random.default_rng(seed).integers(-20000, 20000, 16000, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())


def _dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_cache, "SPEECH_LABELS", ["yes", "no"])
    root = tmp_path / "sc"
    for i, rel in enumerate(["yes/aaaa1111_nohash_0.wav", "yes/bbbb2222_nohash_0.wav", "no/aaaa1111_nohash_1.wav"]):
        _write_wav(root / rel, i)
    return root


def test_cache_is_bit_identical_to_wav_to_logmel(tmp_path, monkeypatch):
    root = _dataset(tmp_path, monkeypatch)
    out = tmp_path / "cache"
    speech_cache.build_cache(str(root), str(out))
    cache = speech_cache.SpeechFeatureCache(str(out))
    assert len(cache) == 3
    for wav in root.rglob("*.wav"):
        assert torch.equal(cache.get(str(wav)), wav_to_logmel(str(wav)))
    assert speech_cache.verify_cache(str(root), str(out), 3) == 3


def test_verify_detects_a_changed_shard(tmp_path, monkeypatch):
    root = _dataset(tmp_path, monkeypatch)
    out = tmp_path / "cache"
    speech_cache.build_cache(str(root), str(out))
    shard = np.load(out / "yes.npy")
    shard[0, 0, 0] += 1.0
    np.save(out / "yes.npy", shard)
    with pytest.raises(ValueError, match="differ"):
        speech_cache.verify_cache(str(root), str(out), 3)


def test_existing_shards_are_kept_when_a_build_continues(tmp_path, monkeypatch):
    root = _dataset(tmp_path, monkeypatch)
    out = tmp_path / "cache"
    speech_cache.build_cache(str(root), str(out))
    before = (out / "no.npy").stat().st_mtime_ns
    (out / "yes.npy").unlink()
    speech_cache.build_cache(str(root), str(out))
    assert (out / "no.npy").stat().st_mtime_ns == before
    assert (out / "yes.npy").exists()
