"""Tests for CIFAR-100 Dirichlet and superclass partitioning.

Offline unit tests use synthetic labels only (no download).
Real-data tests are marked with @pytest.mark.realdata:
    pytest -m realdata
"""
import numpy as np
import pytest
import torch

from fte_fl.data.cifar100 import (
    dirichlet_partition,
    iid_partition,
    superclass_partition,
    SUPERCLASS_MAP,
    _SUPERCLASS_NAMES,
    _COARSE_OF_FINE,
    build_cifar100_clients,
)

REAL_CIFAR_CACHE = "/home/shreesh/fte-fl/data_cache"


# ---- SUPERCLASS_MAP structure ----

def test_superclass_map_has_20_superclasses():
    assert len(SUPERCLASS_MAP) == 20


def test_superclass_map_covers_all_100_classes():
    all_classes = set()
    for classes in SUPERCLASS_MAP.values():
        all_classes.update(classes)
    assert all_classes == set(range(100))


def test_superclass_map_each_has_5_classes():
    for name, classes in SUPERCLASS_MAP.items():
        assert len(classes) == 5, f"{name} has {len(classes)} classes"


def test_superclass_map_no_duplicates():
    """No fine class appears in more than one superclass."""
    seen = set()
    for classes in SUPERCLASS_MAP.values():
        for c in classes:
            assert c not in seen, f"Class {c} appears in multiple superclasses"
            seen.add(c)


def test_coarse_of_fine_length():
    assert len(_COARSE_OF_FINE) == 100


def test_superclass_names_match_expected():
    """The 20 canonical superclass names must all be present."""
    expected_names = {
        "aquatic_mammals", "fish", "flowers", "food_containers",
        "fruit_and_vegetables", "household_electrical_devices",
        "household_furniture", "insects", "large_carnivores",
        "large_manmade_outdoor_things", "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores", "medium_sized_mammals",
        "non_insect_invertebrates", "people", "reptiles",
        "small_mammals", "trees", "vehicles_1", "vehicles_2",
    }
    assert set(SUPERCLASS_MAP.keys()) == expected_names


# ---- dirichlet_partition ----

def test_dirichlet_partition_basic():
    labels = np.array([i % 100 for i in range(5000)])
    parts = dirichlet_partition(labels, num_clients=10, alpha=0.1, num_classes=100, seed=42)
    assert len(parts) == 10
    assert all(len(p) > 0 for p in parts)
    assert sum(len(p) for p in parts) == 5000


def test_dirichlet_partition_no_overlap():
    labels = np.array([i % 100 for i in range(5000)])
    parts = dirichlet_partition(labels, num_clients=10, alpha=0.5, num_classes=100, seed=42)
    all_indices = set()
    for p in parts:
        for idx in p:
            assert idx not in all_indices, "Index assigned to multiple clients"
            all_indices.add(idx)
    assert len(all_indices) == 5000


def test_dirichlet_partition_reproducibility():
    labels = np.array([i % 100 for i in range(2000)])
    p1 = dirichlet_partition(labels, 10, 0.1, 100, seed=42)
    p2 = dirichlet_partition(labels, 10, 0.1, 100, seed=42)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a, b)


def test_dirichlet_partition_different_seed():
    labels = np.array([i % 100 for i in range(2000)])
    p1 = dirichlet_partition(labels, 10, 0.1, 100, seed=42)
    p2 = dirichlet_partition(labels, 10, 0.1, 100, seed=99)
    assert not all(np.array_equal(a, b) for a, b in zip(p1, p2))


def test_dirichlet_partition_alpha_affects_skew():
    """Lower alpha should produce more skewed (unequal) partitions."""
    labels = np.array([i % 10 for i in range(5000)])
    low_alpha = dirichlet_partition(labels, 10, 0.01, 10, seed=42)
    high_alpha = dirichlet_partition(labels, 10, 1.0, 10, seed=42)
    low_cv = np.std([len(p) for p in low_alpha]) / np.mean([len(p) for p in low_alpha])
    high_cv = np.std([len(p) for p in high_alpha]) / np.mean([len(p) for p in high_alpha])
    assert low_cv > high_cv


def test_dirichlet_partition_empty_clients_allowed():
    """Very low alpha with many clients can produce empty partitions; must not crash."""
    labels = np.array([i % 100 for i in range(500)])
    parts = dirichlet_partition(labels, 200, 0.001, 100, seed=42)
    assert len(parts) == 200
    assert sum(len(p) for p in parts) == 500
    # some partitions may be empty
    empty = [p for p in parts if len(p) == 0]
    assert len(empty) >= 0  # just must not crash


# ---- iid_partition ----

def test_iid_partition_disjoint_complete_balanced():
    parts = iid_partition(103, 10, seed=3)
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(103))
    assert {len(p) for p in parts} == {10, 11}
    assert all(np.all(np.diff(p) > 0) for p in parts)


def test_iid_partition_reproducible_and_seed_dependent():
    a = iid_partition(50, 5, seed=1)
    b = iid_partition(50, 5, seed=1)
    c = iid_partition(50, 5, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_iid_partition_rejects_more_clients_than_samples():
    with pytest.raises(ValueError, match="num_clients <= num_samples"):
        iid_partition(3, 4)


# ---- superclass_partition ----

def test_superclass_partition_basic():
    labels = np.array([i % 100 for i in range(5000)])
    parts = superclass_partition(labels, num_clients=10, seed=42)
    assert len(parts) == 10
    assert sum(len(p) for p in parts) == 5000


def test_superclass_partition_no_overlap():
    labels = np.array([i % 100 for i in range(5000)])
    parts = superclass_partition(labels, num_clients=10, seed=42)
    all_indices = set()
    for p in parts:
        for idx in p:
            assert idx not in all_indices
            all_indices.add(idx)
    assert len(all_indices) == 5000


def test_superclass_partition_reproducibility():
    labels = np.array([i % 100 for i in range(2000)])
    p1 = superclass_partition(labels, 10, seed=42)
    p2 = superclass_partition(labels, 10, seed=42)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a, b)


def test_superclass_partition_clients_see_subset_of_classes():
    """With 20 superclasses and 10 clients, each client sees at most 2 superclasses (10 fine classes)."""
    labels = np.array([i % 100 for i in range(5000)])
    parts = superclass_partition(labels, num_clients=10, seed=42)
    for p in parts:
        client_labels = set(labels[p])
        assert len(client_labels) <= 10


def test_superclass_partition_different_seed():
    labels = np.array([i % 100 for i in range(2000)])
    p1 = superclass_partition(labels, 10, seed=42)
    p2 = superclass_partition(labels, 10, seed=99)
    assert not all(np.array_equal(a, b) for a, b in zip(p1, p2))


# ---- build_cifar100_clients (no download) ----

def test_build_cifar100_clients_requires_download_or_cache(tmp_path):
    """If dataset is absent and download=False, must raise."""
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        build_cifar100_clients(
            root=str(tmp_path),
            num_clients=10,
            alpha=0.1,
            split="dirichlet",
            seed=42,
            download=False,
        )


def test_build_cifar100_clients_invalid_split(tmp_path):
    """Unknown split name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown split"):
        build_cifar100_clients(
            root=str(tmp_path),
            num_clients=10,
            split="random",
            seed=42,
            download=False,
        )


# ---- Real-data tests (require CIFAR-100 cached at data_cache/) ----

@pytest.mark.realdata
def test_real_cifar100_train_size():
    """CIFAR-100 train set must have 50,000 samples."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=True, download=False)
    assert len(ds) == 50000


@pytest.mark.realdata
def test_real_cifar100_test_size():
    """CIFAR-100 test set must have 10,000 samples."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=False, download=False)
    assert len(ds) == 10000


@pytest.mark.realdata
def test_real_cifar100_sample_shape():
    """Sample shape must be (3, 32, 32) after to_tensor, label in [0,99]."""
    import torchvision.transforms.functional as F
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=True, download=False)
    img, lbl = ds[0]
    img_t = F.to_tensor(img)
    assert img_t.shape == (3, 32, 32)
    assert img_t.dtype == torch.float32
    assert 0 <= lbl <= 99


@pytest.mark.realdata
def test_real_cifar100_label_count():
    """Must have exactly 100 unique fine labels."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=True, download=False)
    assert len(set(ds.targets)) == 100


@pytest.mark.realdata
def test_real_cifar100_coarse_mapping_matches_ours():
    """Our _COARSE_OF_FINE must match the real torchvision pickle."""
    import os, pickle
    pickle_path = os.path.join(REAL_CIFAR_CACHE, "cifar-100-python", "train")
    with open(pickle_path, "rb") as f:
        train_data = pickle.load(f, encoding="bytes")
    fine_labels   = train_data[b"fine_labels"]
    coarse_labels = train_data[b"coarse_labels"]
    # Build the real mapping
    real_map = {}
    for fine, coarse in zip(fine_labels, coarse_labels):
        if fine not in real_map:
            real_map[fine] = coarse
        else:
            assert real_map[fine] == coarse
    for i in range(100):
        assert _COARSE_OF_FINE[i] == real_map[i], (
            f"Mismatch at fine class {i}: "
            f"our={_COARSE_OF_FINE[i]}, real={real_map[i]}"
        )


@pytest.mark.realdata
def test_real_cifar100_dirichlet_partition_on_actual_data():
    """Dirichlet partition on real 50k train labels: no duplicates, all covered."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=True, download=False)
    labels = np.array(ds.targets, dtype=np.int64)
    for alpha in (0.1, 0.5):
        parts = dirichlet_partition(labels, 10, alpha, 100, seed=42)
        all_idx = np.concatenate(parts)
        assert len(all_idx) == 50000
        assert len(set(all_idx.tolist())) == 50000, f"Duplicates in alpha={alpha} partition"


@pytest.mark.realdata
def test_real_cifar100_superclass_partition_on_actual_data():
    """Superclass partition on real 50k train labels: no duplicates, all covered."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=REAL_CIFAR_CACHE, train=True, download=False)
    labels = np.array(ds.targets, dtype=np.int64)
    parts = superclass_partition(labels, 10, seed=42)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == 50000
    assert len(set(all_idx.tolist())) == 50000
