"""Verification script for all three dataset loaders.

Prints per-client label histograms and sample tensor shapes for a sample.
Deterministic with --seed. Does not modify any dataset files.

Usage:
    python -m fte_fl.data.verify_datasets --seed 42
    python -m fte_fl.data.verify_datasets --seed 42 --dataset femnist
    python -m fte_fl.data.verify_datasets --seed 42 --dataset speech
    python -m fte_fl.data.verify_datasets --seed 42 --dataset cifar100 --download
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import torch

from fte_fl.config import set_seed, config_hash
from fte_fl.fl.data import native_path


# Default dataset paths (WSL2 mounts of Windows paths; drive paths on native Windows)
FEMNIST_DIR = native_path("/mnt/c/Users/Shreesh/Desktop/majorproj/fl_byzantine/justincase/leaf/data/femnist/data")
SPEECH_COMMANDS_DIR = native_path("/mnt/c/Users/Shreesh/Desktop/majorproj/speech_commands_v0.02")
CIFAR100_CACHE = "./data_cache"


def verify_femnist(seed: int):
    """Verify FEMNIST dataset: counts, shapes, label histograms."""
    from fte_fl.data.femnist import build_femnist_clients

    print("=" * 60)
    print("FEMNIST (LEAF-preprocessed)")
    print("=" * 60)

    train_dir = f"{FEMNIST_DIR}/train"
    test_dir = f"{FEMNIST_DIR}/test"

    try:
        clients = build_femnist_clients(train_dir, num_clients=500, seed=seed)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Train path: {train_dir}")
    print(f"  Requested clients: 500")
    print(f"  Available clients: {len(clients)}")
    total_samples = sum(len(ds) for ds in clients.values())
    print(f"  Total train samples: {total_samples}")

    # Sample shapes
    first_ds = list(clients.values())[0]
    x, y = first_ds[0]
    print(f"  Sample shape: {x.shape} (dtype={x.dtype})")
    print(f"  Label: {y} (dtype={y.dtype})")

    # Per-client label histogram (first 5 clients)
    print(f"\n  Per-client label histograms (first 5):")
    for i, (cid, ds) in enumerate(clients.items()):
        if i >= 5:
            break
        labels = [int(ds[j][1]) for j in range(len(ds))]
        hist = Counter(labels)
        top5 = sorted(hist.items())[:5]
        print(f"    Client {cid} ({len(ds)} samples): top-5 labels = {top5}")

    # Test set
    try:
        test_clients = build_femnist_clients(test_dir, num_clients=500, seed=seed)
        total_test = sum(len(ds) for ds in test_clients.values())
        print(f"\n  Test clients: {len(test_clients)}")
        print(f"  Total test samples: {total_test}")
    except Exception as e:
        print(f"  Test set error: {e}")


def verify_speech(seed: int):
    """Verify Speech Commands dataset: counts, shapes, speaker info."""
    from fte_fl.data.speech_commands import (
        build_speech_clients, list_speakers, official_split_files, speaker_of, SPEECH_LABELS,
    )

    print("\n" + "=" * 60)
    print("Speech Commands v0.02")
    print("=" * 60)

    speakers = list_speakers(SPEECH_COMMANDS_DIR)
    print(f"  Path: {SPEECH_COMMANDS_DIR}")
    print(f"  Total unique speakers: {len(speakers)}")
    print(f"  Labels: {len(SPEECH_LABELS)} words")

    # Official partition (validation_list.txt / testing_list.txt); raises on
    # missing lists or speaker overlap.
    for split, files in official_split_files(SPEECH_COMMANDS_DIR).items():
        print(f"  Official {split}: {len(files)} clips, "
              f"{len({speaker_of(f) for f in files})} speakers")

    clients = build_speech_clients(SPEECH_COMMANDS_DIR, num_clients=800, seed=seed)
    print(f"  Requested clients: 800 (training speakers only)")
    print(f"  Available clients: {len(clients)}")
    total_samples = sum(len(ds) for ds in clients.values())
    print(f"  Total samples: {total_samples}")

    # Sample shapes
    first_ds = list(clients.values())[0]
    x, y = first_ds[0]
    print(f"  Sample shape: {x.shape}")
    print(f"  Label: {y} ({SPEECH_LABELS[y]})")

    # Per-client label histogram (first 5)
    print(f"\n  Per-client label histograms (first 5):")
    for i, (sid, ds) in enumerate(clients.items()):
        if i >= 5:
            break
        labels = [ds[j][1] for j in range(min(len(ds), 50))]
        hist = Counter(labels)
        top5 = sorted(hist.items())[:5]
        print(f"    Speaker {sid} ({len(ds)} samples): top-5 labels = {top5}")


def verify_cifar100(seed: int, download: bool = False):
    """Verify CIFAR-100 dataset: counts, shapes, partitioning."""
    print("\n" + "=" * 60)
    print("CIFAR-100")
    print("=" * 60)

    try:
        from fte_fl.data.cifar100 import (
            build_cifar100_clients, SUPERCLASS_MAP,
            dirichlet_partition, superclass_partition,
        )
    except Exception as e:
        print(f"  Import error: {e}")
        return

    # Test partitioning logic with synthetic labels (no download needed)
    print("  [Synthetic partition tests — no download required]")
    labels = np.array([i % 100 for i in range(5000)])

    # Dirichlet alpha=0.1
    parts = dirichlet_partition(labels, 10, 0.1, 100, seed=seed)
    print(f"  Dirichlet(alpha=0.1): {len(parts)} clients, sizes = {[len(p) for p in parts]}")

    # Dirichlet alpha=0.5
    parts = dirichlet_partition(labels, 10, 0.5, 100, seed=seed)
    print(f"  Dirichlet(alpha=0.5): {len(parts)} clients, sizes = {[len(p) for p in parts]}")

    # Superclass
    parts = superclass_partition(labels, 10, seed=seed)
    print(f"  Superclass split: {len(parts)} clients, sizes = {[len(p) for p in parts]}")
    print(f"  Superclasses: {len(SUPERCLASS_MAP)}, fine classes: {sum(len(v) for v in SUPERCLASS_MAP.values())}")

    if download:
        print(f"\n  [Downloading CIFAR-100 to {CIFAR100_CACHE}]")
        try:
            clients = build_cifar100_clients(
                CIFAR100_CACHE, num_clients=10, alpha=0.1,
                split="dirichlet", seed=seed, download=True,
            )
            print(f"  Clients: {len(clients)}")
            first_ds = list(clients.values())[0]
            img, y = first_ds[0]
            # img is a PIL Image from raw CIFAR100; convert to check shape
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img)
            print(f"  Sample shape (tensor): {tuple(img_t.shape)}")
            print(f"  Label: {y}")

            for i, (cid, ds) in enumerate(clients.items()):
                if i >= 3:
                    break
                ys = [int(ds[j][1]) for j in range(min(len(ds), 100))]
                hist = Counter(ys)
                print(f"    Client {cid} ({len(ds)} samples): top-5 labels = {sorted(hist.items())[:5]}")
        except Exception as e:
            print(f"  Download/load error: {e}")
    else:
        print(f"  [Skipped actual data load — use --download to test]")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset loaders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", "femnist", "speech", "cifar100"],
                        default="all")
    parser.add_argument("--download", action="store_true",
                        help="Download CIFAR-100 if not cached")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    if args.dataset in ("all", "femnist"):
        verify_femnist(args.seed)
    if args.dataset in ("all", "speech"):
        verify_speech(args.seed)
    if args.dataset in ("all", "cifar100"):
        verify_cifar100(args.seed, download=args.download)

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
