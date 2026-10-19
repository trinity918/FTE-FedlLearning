"""Tests for the FEMNIST LEAF loader.

Tests use mock JSON fixtures so they do not require the real dataset.
Real-data tests are marked with @pytest.mark.realdata and can be run via:
    pytest -m realdata
"""
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from fte_fl.data.femnist import parse_leaf_json, FemnistDataset, build_femnist_clients

REAL_TRAIN = (
    "/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/justincase"
    "/leaf/data/femnist/data/train"
)
REAL_TEST = (
    "/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/justincase"
    "/leaf/data/femnist/data/test"
)


def _make_mock_leaf(
    path: Path,
    num_users: int = 5,
    samples_per_user: int = 8,
    dim: int = 784,
    num_classes: int = 62,
    prefix: str = "u",
) -> Path:
    """Write a minimal LEAF-format JSON file."""
    users = [f"{prefix}{i:04d}" for i in range(num_users)]
    user_data = {}
    for u in users:
        user_data[u] = {
            "x": [[float(j % dim) / dim for j in range(dim)] for _ in range(samples_per_user)],
            "y": [i % num_classes for i in range(samples_per_user)],
        }
    data = {
        "users": users,
        "num_samples": [samples_per_user] * num_users,
        "user_data": user_data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---- parse_leaf_json ----

def test_parse_leaf_json_returns_users_and_data(tmp_path):
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=3, samples_per_user=4)
    result = parse_leaf_json(p)
    assert "users" in result
    assert "user_data" in result
    assert len(result["users"]) == 3
    assert set(result["user_data"].keys()) == set(result["users"])


def test_parse_leaf_json_sample_shapes(tmp_path):
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", dim=784)
    result = parse_leaf_json(p)
    first_user = result["users"][0]
    assert len(result["user_data"][first_user]["x"][0]) == 784
    assert isinstance(result["user_data"][first_user]["y"][0], int)


def test_parse_leaf_json_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_leaf_json(tmp_path / "nonexistent.json")


def test_parse_leaf_json_does_not_mutate_source(tmp_path):
    """Parsing must not write to or modify the source JSON file."""
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=2)
    import os
    mtime_before = os.path.getmtime(p)
    parse_leaf_json(p)
    mtime_after = os.path.getmtime(p)
    assert mtime_before == mtime_after, "parse_leaf_json modified the source file"


# ---- FemnistDataset ----

def test_femnist_dataset_shapes(tmp_path):
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=2, samples_per_user=5)
    result = parse_leaf_json(p)
    first_user = result["users"][0]
    ds = FemnistDataset(result["user_data"][first_user])
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert 0 <= int(y) < 62


def test_femnist_dataset_all_samples_accessible(tmp_path):
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=1, samples_per_user=10)
    result = parse_leaf_json(p)
    ds = FemnistDataset(result["user_data"][result["users"][0]])
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (1, 28, 28)


def test_femnist_dataset_label_range(tmp_path):
    """Labels must be integers in [0, 61]."""
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=1, samples_per_user=62)
    result = parse_leaf_json(p)
    ds = FemnistDataset(result["user_data"][result["users"][0]])
    labels = [int(ds[i][1]) for i in range(len(ds))]
    assert all(0 <= lbl <= 61 for lbl in labels)


def test_femnist_dataset_pixel_range(tmp_path):
    """Pixel values must be float32 in [0, 1]."""
    p = _make_mock_leaf(tmp_path / "train" / "mock.json", num_users=1, samples_per_user=4)
    result = parse_leaf_json(p)
    ds = FemnistDataset(result["user_data"][result["users"][0]])
    x, _ = ds[0]
    assert x.dtype == torch.float32
    assert float(x.min()) >= 0.0 - 1e-6
    assert float(x.max()) <= 1.0 + 1e-6


# ---- build_femnist_clients ----

def test_build_clients_basic(tmp_path):
    _make_mock_leaf(tmp_path / "train" / "mock_0.json", num_users=5, samples_per_user=4, prefix="u")
    _make_mock_leaf(tmp_path / "train" / "mock_1.json", num_users=3, samples_per_user=4, prefix="v")
    clients = build_femnist_clients(str(tmp_path / "train"), num_clients=500, seed=42)
    assert len(clients) == 8
    for cid, ds in clients.items():
        assert len(ds) > 0


def test_build_clients_reproducibility(tmp_path):
    _make_mock_leaf(tmp_path / "train" / "mock_0.json", num_users=5, samples_per_user=4)
    c1 = build_femnist_clients(str(tmp_path / "train"), num_clients=5, seed=42)
    c2 = build_femnist_clients(str(tmp_path / "train"), num_clients=5, seed=42)
    assert list(c1.keys()) == list(c2.keys())


def test_build_clients_different_seed_gives_different_selection(tmp_path):
    """Different seeds must produce a different client selection when n > num_clients."""
    # Create 10 users so there's real sampling
    _make_mock_leaf(tmp_path / "train" / "mock_0.json", num_users=10, samples_per_user=4)
    c1 = build_femnist_clients(str(tmp_path / "train"), num_clients=5, seed=42)
    c2 = build_femnist_clients(str(tmp_path / "train"), num_clients=5, seed=99)
    assert len(c1) == len(c2) == 5
    # With 10 writers sampled down to 5, different seeds should differ
    assert list(c1.keys()) != list(c2.keys())


def test_build_clients_caps_at_available(tmp_path):
    _make_mock_leaf(tmp_path / "train" / "mock_0.json", num_users=3, samples_per_user=4)
    clients = build_femnist_clients(str(tmp_path / "train"), num_clients=500, seed=42)
    assert len(clients) == 3


def test_build_clients_empty_dir(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        build_femnist_clients(str(tmp_path / "train"), num_clients=10, seed=42)


def test_build_clients_sample_count_matches_writer(tmp_path):
    """Dataset length per writer must match what was written."""
    _make_mock_leaf(tmp_path / "train" / "m.json", num_users=3, samples_per_user=7)
    clients = build_femnist_clients(str(tmp_path / "train"), num_clients=3, seed=42)
    for ds in clients.values():
        assert len(ds) == 7


# ---- Real-data tests (require FEMNIST dataset present) ----

@pytest.mark.realdata
def test_real_femnist_train_counts():
    """Verified counts: 198 writers, 36,700 train samples."""
    clients = build_femnist_clients(REAL_TRAIN, num_clients=500, seed=42)
    assert len(clients) == 198, f"Expected 198 writers, got {len(clients)}"
    total = sum(len(ds) for ds in clients.values())
    assert total == 36700, f"Expected 36700 train samples, got {total}"


@pytest.mark.realdata
def test_real_femnist_test_counts():
    """Verified counts: 198 writers, 4,174 test samples."""
    clients = build_femnist_clients(REAL_TEST, num_clients=500, seed=42)
    assert len(clients) == 198
    total = sum(len(ds) for ds in clients.values())
    assert total == 4174, f"Expected 4174 test samples, got {total}"


@pytest.mark.realdata
def test_real_femnist_writer_overlap():
    """All 198 writers appear in BOTH train and test (LEAF per-sample split)."""
    train_clients = build_femnist_clients(REAL_TRAIN, num_clients=500, seed=42)
    test_clients  = build_femnist_clients(REAL_TEST,  num_clients=500, seed=42)
    assert set(train_clients.keys()) == set(test_clients.keys()), (
        "LEAF splits samples per writer — all writers appear in both splits"
    )


@pytest.mark.realdata
def test_real_femnist_shapes_and_labels():
    """Real samples: shape (1,28,28), labels in [0,61], pixels in [0,1]."""
    clients = build_femnist_clients(REAL_TRAIN, num_clients=5, seed=42)
    for ds in clients.values():
        x, y = ds[0]
        assert x.shape == (1, 28, 28)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        assert 0 <= int(y) <= 61
        assert float(x.min()) >= 0.0 - 1e-6
        assert float(x.max()) <= 1.0 + 1e-6
