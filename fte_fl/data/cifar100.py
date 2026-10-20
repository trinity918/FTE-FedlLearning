"""CIFAR-100 loader with Dirichlet and superclass partitioning.

Supports two non-IID partitioning strategies:
  1. Dirichlet label-skew: draw Dirichlet(alpha) per class to split samples.
  2. Superclass-grouped: assign entire superclass groups to clients.

References:
  Krizhevsky, "Learning Multiple Layers of Features from Tiny Images",
  2009.  100 fine classes, 20 superclasses, 5 fine per superclass.
  Hsu et al., "Measuring the Effects of Non-Identical Data Distribution
  for Federated Visual Classification", arXiv:1909.06335, 2019 (Dirichlet).
"""
from __future__ import annotations

import random

import numpy as np
from torch.utils.data import Subset

# -- Official CIFAR-100 superclass -> fine-class-index mapping --
# Derived from the canonical fine_label -> coarse_label correspondence
# published at https://www.cs.toronto.edu/~kriz/cifar.html
#
# Fine label indices 0-99 correspond to the official alphabetical ordering.
# Coarse label indices 0-19 correspond to the 20 superclass groupings.

_SUPERCLASS_NAMES = [
    "aquatic_mammals",                    # 0
    "fish",                               # 1
    "flowers",                             # 2
    "food_containers",                     # 3
    "fruit_and_vegetables",                # 4
    "household_electrical_devices",        # 5
    "household_furniture",                 # 6
    "insects",                             # 7
    "large_carnivores",                    # 8
    "large_manmade_outdoor_things",        # 9
    "large_natural_outdoor_scenes",        # 10
    "large_omnivores_and_herbivores",      # 11
    "medium_sized_mammals",                # 12
    "non_insect_invertebrates",            # 13
    "people",                              # 14
    "reptiles",                            # 15
    "small_mammals",                       # 16
    "trees",                               # 17
    "vehicles_1",                          # 18
    "vehicles_2",                          # 19
]

# fine_label_index -> coarse_label_index (official CIFAR-100 ordering)
# Verified: each coarse index appears exactly 5 times, all 100 covered.
_COARSE_OF_FINE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,   # 0-9
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,   # 10-19
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,   # 20-29
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,   # 30-39
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,   # 40-49
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,   # 50-59
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,   # 60-69
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,   # 70-79
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,   # 80-89
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13,   # 90-99
]

SUPERCLASS_MAP: dict[str, list[int]] = {name: [] for name in _SUPERCLASS_NAMES}
for _fine_idx in range(100):
    _coarse_idx = _COARSE_OF_FINE[_fine_idx]
    SUPERCLASS_MAP[_SUPERCLASS_NAMES[_coarse_idx]].append(_fine_idx)

# Sanity checks (run at import time)
assert len(SUPERCLASS_MAP) == 20
assert all(len(v) == 5 for v in SUPERCLASS_MAP.values())
_all_fine = sorted(f for cls in SUPERCLASS_MAP.values() for f in cls)
assert _all_fine == list(range(100))


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition sample indices by Dirichlet label-skew (Hsu et al., 2019).

    For each class, draw a Dirichlet(alpha) distribution over clients,
    then assign samples of that class proportionally.

    Args:
        labels: 1-D array of integer class labels.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter. Lower = more non-IID.
        num_classes: Total number of classes.
        seed: Random seed.

    Returns:
        List of num_clients arrays, each containing sample indices.
    """
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        if len(class_indices) == 0:
            continue
        props = rng.dirichlet([alpha] * num_clients)
        client_assignments = rng.choice(
            num_clients, size=len(class_indices), p=props
        )
        for idx, client_id in zip(class_indices, client_assignments):
            partitions[client_id].append(int(idx))

    return [np.array(sorted(p), dtype=np.int64) for p in partitions]


def iid_partition(num_samples: int, num_clients: int, seed: int = 42) -> list[np.ndarray]:
    """IID partition: a seeded permutation split into num_clients near-equal parts.

    Client sizes differ by at most one; every index appears exactly once.
    """
    if num_clients < 1 or num_samples < num_clients:
        raise ValueError(f"need 1 <= num_clients <= num_samples, got {num_clients}, {num_samples}")
    perm = np.random.default_rng(seed).permutation(num_samples)
    return [np.sort(p).astype(np.int64) for p in np.array_split(perm, num_clients)]


def superclass_partition(
    labels: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition by assigning entire superclass groups to clients.

    Each client gets ceil(20 / num_clients) superclasses, and all
    samples belonging to those superclass fine classes are assigned
    to that client. This creates strong class-based non-IID-ness.

    Args:
        labels: 1-D array of integer class labels (0-99).
        num_clients: Number of clients.
        seed: Random seed for superclass ordering.

    Returns:
        List of num_clients arrays of sample indices.
    """
    rng = random.Random(seed)
    superclasses = list(SUPERCLASS_MAP.keys())
    rng.shuffle(superclasses)

    partitions: list[list[int]] = [[] for _ in range(num_clients)]

    for sc_idx, sc_name in enumerate(superclasses):
        client_id = sc_idx % num_clients
        fine_classes = set(SUPERCLASS_MAP[sc_name])
        for sample_idx, label in enumerate(labels):
            if int(label) in fine_classes:
                partitions[client_id].append(sample_idx)

    return [np.array(sorted(p), dtype=np.int64) for p in partitions]


def build_cifar100_clients(
    root: str,
    num_clients: int = 100,
    alpha: float = 0.1,
    split: str = "dirichlet",
    seed: int = 42,
    train: bool = True,
    download: bool = True,
) -> dict[int, Subset]:
    """Build per-client CIFAR-100 subsets.

    Args:
        root: Cache directory for torchvision.
        num_clients: Number of federated clients.
        alpha: Dirichlet alpha (used when split='dirichlet').
        split: 'dirichlet', 'superclass' or 'iid'.
        seed: Random seed.
        train: Whether to load train or test set.
        download: Whether to download if absent.

    Returns:
        Dict mapping client_id -> torch Subset.
    """
    from torchvision.datasets import CIFAR100

    # Validate split before attempting dataset load
    if split not in ("dirichlet", "superclass", "iid"):
        raise ValueError(f"Unknown split: {split!r}. Use 'dirichlet', 'superclass' or 'iid'.")

    dataset = CIFAR100(root=root, train=train, download=download)
    labels = np.array(dataset.targets, dtype=np.int64)

    if split == "iid":
        partitions = iid_partition(len(labels), num_clients, seed=seed)
    elif split == "dirichlet":
        partitions = dirichlet_partition(
            labels, num_clients, alpha, num_classes=100, seed=seed
        )
    elif split == "superclass":
        partitions = superclass_partition(labels, num_clients, seed=seed)
    else:
        raise ValueError(f"Unknown split: {split}. Use 'dirichlet' or 'superclass'.")

    return {i: Subset(dataset, partitions[i].tolist()) for i in range(num_clients)}
