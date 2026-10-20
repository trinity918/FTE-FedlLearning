"""Model verification CLI for FTE-FL Milestone 2.

Verifies all five model architectures and prints a summary table.
Exits with a nonzero status code if any required assertion fails.

Usage:
    python -m fte_fl.models.verify_models                # all models
    python -m fte_fl.models.verify_models --model cnn    # single model
    python -m fte_fl.models.verify_models --pretrained   # test real weight loading

Options:
    --model {cnn,bcresnet,vit,lora_vit,lora_ast,all}   Which model(s) to verify.
    --pretrained                                        Attempt pretrained-weight
                                                        loading for LoRA ViT.
                                                        Requires internet access.
    --seed SEED                                         Random seed (default: 42).
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

import torch

from fte_fl.models.base import count_parameters, estimated_ciphertext_chunks
from fte_fl.config import set_seed


@dataclass
class ModelResult:
    name: str
    dataset: str
    input_shape: tuple
    num_classes: int
    total_params: int = 0
    trainable_params: int = 0
    frozen_params: int = 0
    ckks_chunks: int = 0
    output_shape: Optional[tuple] = None
    forward_ok: bool = False
    backward_ok: bool = False
    pretrained_loaded: bool = False
    error: str = ""


def _verify_one(
    name: str,
    dataset: str,
    input_shape: tuple,
    num_classes: int,
    build_fn,
    check_pretrained: bool = False,
) -> ModelResult:
    """Build one model, run forward/backward, collect metrics."""
    result = ModelResult(name=name, dataset=dataset,
                         input_shape=input_shape, num_classes=num_classes)
    try:
        model = build_fn()
        total, trainable, frozen = count_parameters(model)
        result.total_params = total
        result.trainable_params = trainable
        result.frozen_params = frozen
        result.ckks_chunks = estimated_ciphertext_chunks(trainable)
        result.pretrained_loaded = getattr(model, "pretrained_loaded", False)

        # Forward pass
        x = torch.randn((2,) + input_shape)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, num_classes), (
            f"Expected (2, {num_classes}), got {out.shape}"
        )
        result.output_shape = tuple(out.shape)
        result.forward_ok = True

        # Backward pass (trainable params only)
        model.train()
        model.zero_grad()
        out = model(x)
        out.sum().backward()
        grads = [p for p in model.parameters()
                 if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, "No trainable parameters received gradients"
        result.backward_ok = True

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


MODEL_KEYS = ("cnn", "bcresnet", "vit", "lora_vit", "lora_ast")


def verify_all(
    pretrained_vit: bool = False, seed: int = 42, model: str = "all"
) -> list[ModelResult]:
    """Verify all models, or only the one named by `model` (a MODEL_KEYS entry)."""
    if model != "all" and model not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model!r}; expected 'all' or one of {MODEL_KEYS}")
    set_seed(seed)

    from fte_fl.models.cnn_small import CnnSmall
    from fte_fl.models.bc_resnet import BCResNet1
    from fte_fl.models.vit_tiny import ViTTiny
    from fte_fl.models.lora_vit import LoRAViT
    from fte_fl.models.lora_ast import LoRAAST

    specs = [
        dict(
            name="CnnSmall",
            dataset="FEMNIST",
            input_shape=(1, 28, 28),
            num_classes=62,
            build_fn=lambda: CnnSmall(),
        ),
        dict(
            name="BCResNet1",
            dataset="Speech Commands",
            input_shape=(1, 64, 101),
            num_classes=35,
            build_fn=lambda: BCResNet1(),
        ),
        dict(
            name="ViTTiny",
            dataset="CIFAR-100",
            input_shape=(3, 32, 32),
            num_classes=100,
            build_fn=lambda: ViTTiny(),
        ),
        dict(
            name="LoRAViT",
            dataset="CIFAR-100",
            input_shape=(3, 32, 32),
            num_classes=100,
            build_fn=lambda: LoRAViT(pretrained=pretrained_vit),
        ),
        dict(
            name="LoRAAST",
            dataset="Speech Commands",
            input_shape=(1, 64, 101),
            num_classes=35,
            build_fn=lambda: LoRAAST(),
        ),
    ]

    if model != "all":
        specs = [specs[MODEL_KEYS.index(model)]]

    results = []
    for spec in specs:
        print(f"  Verifying {spec['name']}...", end=" ", flush=True)
        r = _verify_one(**spec)
        status = "OK" if r.forward_ok and r.backward_ok and not r.error else "FAIL"
        print(status)
        results.append(r)
    return results


def _fmt(n: int) -> str:
    return f"{n:,}"


def print_table(results: list[ModelResult]) -> bool:
    col_widths = {
        "Model":            12,
        "Dataset":          16,
        "Input":            16,
        "Classes":           7,
        "Total":            10,
        "Trainable":        10,
        "Frozen":           10,
        "CKKS est.":         9,
        "Init":              8,
        "Fwd":               5,
        "Bwd":               5,
    }
    header = "  ".join(k.ljust(v) for k, v in col_widths.items())
    sep = "  ".join("-" * v for v in col_widths.values())

    print()
    print("=" * len(sep))
    print("FTE-FL — Model Verification Summary")
    print("=" * len(sep))
    print(header)
    print(sep)

    all_ok = True
    for r in results:
        ok = r.forward_ok and r.backward_ok and not r.error
        if not ok:
            all_ok = False
        init_label = "pretrained" if r.pretrained_loaded else "random"
        row = [
            r.name.ljust(col_widths["Model"]),
            r.dataset.ljust(col_widths["Dataset"]),
            str(r.input_shape).ljust(col_widths["Input"]),
            str(r.num_classes).ljust(col_widths["Classes"]),
            _fmt(r.total_params).ljust(col_widths["Total"]),
            _fmt(r.trainable_params).ljust(col_widths["Trainable"]),
            _fmt(r.frozen_params).ljust(col_widths["Frozen"]),
            str(r.ckks_chunks).ljust(col_widths["CKKS est."]),
            init_label.ljust(col_widths["Init"]),
            ("✓" if r.forward_ok else "✗").ljust(col_widths["Fwd"]),
            ("✓" if r.backward_ok else "✗").ljust(col_widths["Bwd"]),
        ]
        print("  ".join(row))
        if r.error:
            print(f"    ERROR: {r.error}")

    print(sep)
    print()
    print("Notes:")
    print("  CKKS est.  = ceil(trainable_params / 8192) — HE cost estimate only.")
    print("  Init       = 'random' (randomly initialized) or 'pretrained' (real")
    print("               pretrained weights loaded from hub). Default is random.")
    print("  Fwd/Bwd    = forward/backward execution check only. Does NOT verify")
    print("               architectural fidelity or model accuracy.")
    print("  BCResNet1  = implements BC-ResNet-8 (τ=8, base_c=64) from Kim et al.")
    print("               2021. Class name 'BCResNet1' is a project convention only.")
    print("  LoRAAST    = AST-style spectrogram transformer (NOT a faithful AST")
    print("               reproduction). Randomly initialized. See lora_ast.py.")
    print("  LoRAViT    = LoRA on timm ViT-Base/16. Use --pretrained for real")
    print("               pretrained weights (requires internet, ~330 MB).")
    print()

    if all_ok:
        print("All models: PASS")
    else:
        print("FAIL: one or more models did not pass verification.")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="FTE-FL model verification")
    parser.add_argument("--model", default="all",
                        choices=[*MODEL_KEYS, "all"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Attempt real pretrained-weight loading for LoRAViT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nRunning model verification (seed={args.seed})")
    results = verify_all(pretrained_vit=args.pretrained, seed=args.seed,
                         model=args.model)
    all_ok = print_table(results)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
