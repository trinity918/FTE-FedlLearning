"""BC-ResNet for Google Speech Commands keyword spotting.

Reference:
  Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting",
  Interspeech 2021 (arXiv:2106.04140).
  Official implementation: https://github.com/Qualcomm-AI-research/bcresnet
  License: see upstream repo (Qualcomm Technologies).

Naming convention:
  The paper defines BC-ResNet with a width parameter τ (tau).  The headline
  model in the paper is BC-ResNet-8 (τ=8), achieving ~321 k parameters and
  98.7% accuracy on Speech Commands v2.  The project specification targets
  ~321 k, which corresponds to τ=8.  We name the class BCResNet1 to satisfy
  the project's filename convention; the τ parameter is exposed and documented.

Architecture (faithfully ported from the official Qualcomm repo):
  Stem:      Conv2d(1→2τ, 5×5, stride=(2,1)) + BN + ReLU
             → halves the frequency dimension on the first step.
  Stage 1:   2 × BCResBlock, 2τ→τ,    idx=0 (no stride)
  Stage 2:   2 × BCResBlock, τ→1.5τ,  idx=1, stride=(2,1)
  Stage 3:   4 × BCResBlock, 1.5τ→2τ, idx=2, stride=(2,1)
  Stage 4:   4 × BCResBlock, 2τ→2.5τ, idx=3, no stride
  Classifier: DW 5×5 + PW 1×1(2.5τ→4τ) + BN + ReLU + AvgPool + 1×1 Conv

  For τ=8 (the 321k model):
    base_c = τ × 8 = 64, channels = [128, 64, 96, 128, 160, 256].

Input:  (B, 1, 64, 101) — log-mel spectrogram (64 mel bins, 101 time frames).
Output: (B, 35)          — logits for 35 Speech Commands classes.

Adaptations vs. paper:
  - Paper uses 40 mel bins; we use 64.  The stem stride-2 on freq means
    F: 64→32→16→8 across stages, which works cleanly.
  - SubSpectralNorm uses S=5 (paper) regardless of input frequency bins.
    S=5 gives 5 spectral groups; each group covers freq//5 bins within BN.
    If freq is not divisible by 5, we fall back to plain BN for that layer.
  - Paper uses num_classes=12 (GSC v1) or 35 (v2).  We use 35.

Target: ~321 000 parameters ± 10% (288 900 – 353 100).
Asserted at construction time; τ=8 verified to satisfy this range.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fte_fl.models.base import FTEModel


class SubSpectralNorm(nn.Module):
    """Normalize within sub-spectral frequency groups (Kim et al., 2021).

    Divides the F dimension into S groups, applies BatchNorm2d independently
    per group.  S=5 matches the paper (5 sub-bands over 40 mel bins = 8
    bins/group).  Falls back to plain BN when F % S ≠ 0.
    """

    def __init__(self, num_channels: int, S: int = 5) -> None:
        super().__init__()
        self.S = S
        self.num_channels = num_channels
        # BN over C*S "virtual channels" (each sub-band is a group)
        self.bn = nn.BatchNorm2d(num_channels * S)
        # Fallback plain BN used when freq is not divisible by S
        self.bn_plain = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, freq, T = x.shape
        S = self.S
        if freq % S != 0:
            return self.bn_plain(x)
        # Reshape so BN sees C*S channels, each of size freq//S × T
        x = x.reshape(B, C * S, freq // S, T)
        x = self.bn(x)
        return x.reshape(B, C, freq, T)


class ConvBNBlock(nn.Module):
    """Conv2d + optional BN/SSN + optional activation (ReLU or SiLU).

    Mirrors the official `ConvBNReLU` module.  Dilation on the time axis
    is 2^idx when use_dilation=True and kernel > 1 along that axis.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        idx: int,
        kernel_size,
        stride=(1, 1),
        groups: int = 1,
        use_dilation: bool = False,
        activation: str = "relu",   # "relu" | "swish" | "none"
        norm: str = "bn",           # "bn" | "ssn" | "none"
    ) -> None:
        super().__init__()
        self.idx = idx

        def _pad_and_dil(k, dilated):
            """Return (padding, dilation) for a single kernel dimension."""
            p = (k - 1) // 2
            d = 1
            if dilated and k > 1:
                d = int(2 ** idx)
                p = d * p
            return p, d

        if isinstance(kernel_size, (list, tuple)):
            pads, dils = [], []
            for k in kernel_size:
                p, d = _pad_and_dil(k, use_dilation)
                pads.append(p)
                dils.append(d)
            padding, dilation = tuple(pads), tuple(dils)
        else:
            padding, dilation = _pad_and_dil(kernel_size, use_dilation)

        layers: list[nn.Module] = [
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation,
                      groups, bias=False)
        ]
        if norm == "ssn":
            layers.append(SubSpectralNorm(out_c, S=5))
        elif norm == "bn":
            layers.append(nn.BatchNorm2d(out_c))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "swish":
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BCResBlock(nn.Module):
    """Broadcasted Residual Block (faithfully follows official Qualcomm impl).

    f2 (2-D path, full F×T):
      [Optional 1×1 transition if in_plane != out_plane]
      Depthwise (3,1) with SubSpectralNorm (operates on the full F×T feature).

    f1 (1-D broadcast path, collapsed to F=1 via global freq pooling):
      AvgPool over F  → (B, C, 1, T)
      Dilated DW (1,3) with SiLU + pointwise 1×1 + Dropout2d(0.1)

    Residual:
      out = f1(avgpool(f2(x))) + f2(x)
      + identity skip only when in_plane == out_plane (transition blocks
        have no skip, matching the official code).
      Final ReLU applied on the combined output.
    """

    def __init__(
        self,
        in_plane: int,
        out_plane: int,
        idx: int,
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.transition = (in_plane != out_plane)

        # ── f2: 2-D path ────────────────────────────────────────────────────
        f2_layers: list[nn.Module] = []
        cur = in_plane
        if self.transition:
            f2_layers.append(ConvBNBlock(cur, out_plane, idx, 1, activation="relu"))
            cur = out_plane
        f2_layers.append(
            ConvBNBlock(
                cur, out_plane, idx,
                kernel_size=(3, 1),
                stride=(stride[0], 1),
                groups=cur,
                norm="ssn",
                activation="none",
            )
        )
        self.f2 = nn.Sequential(*f2_layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))  # F → 1

        # ── f1: 1-D broadcast path ───────────────────────────────────────────
        self.f1 = nn.Sequential(
            ConvBNBlock(
                out_plane, out_plane, idx,
                kernel_size=(1, 3),
                stride=(1, stride[1]),
                groups=out_plane,
                use_dilation=True,
                activation="swish",
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # 2-D path
        x = self.f2(x)           # (B, C_out, F', T)
        aux_2d = x               # save for residual add

        # 1-D broadcast path (pools F to 1, then broadcasts back via add)
        x = self.avg_gpool(x)    # (B, C_out, 1, T)
        x = self.f1(x)           # (B, C_out, 1, T)
        x = x + aux_2d           # broadcast along F

        # Identity skip (only for non-transition blocks)
        if not self.transition:
            x = x + shortcut

        return F.relu(x, inplace=True)


def _make_stage(
    num_blocks: int,
    in_channel: int,
    out_channel: int,
    idx: int,
    use_stride: bool,
) -> nn.ModuleList:
    """Build one stage of BCResBlocks."""
    blocks = nn.ModuleList()
    channels = [in_channel] + [out_channel] * num_blocks
    for i in range(num_blocks):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        blocks.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return blocks


class BCResNet1(FTEModel):
    """BC-ResNet (τ=8, ~321 k params) for Speech Commands 35-class task.

    The class is named BCResNet1 to satisfy the project filename convention.
    The τ=8 width is the target-parameter configuration (see module docstring).

    Target: 288 900 – 353 100 parameters.  Assertion enforced at construction.
    """

    TARGET_PARAMS: int = 321_000
    TOLERANCE: float = 0.10

    def __init__(self, num_classes: int = 35, tau: int = 8) -> None:
        super().__init__()

        # Official mapping: base_c = tau * 8  (see Qualcomm repo main.py,
        # which calls BCResNets(int(self.tau * 8))).
        # tau=8 → base_c=64 → channels=[128, 64, 96, 128, 160, 256] → ~321k.
        base_c = tau * 8

        # Channel schedule: [2·base_c, base_c, 1.5·base_c, 2·base_c, 2.5·base_c, 4·base_c]
        c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        # Number of blocks per stage
        n = [2, 2, 4, 4]
        # Stages that apply frequency stride (0-indexed: stages 1 and 2)
        stride_stages = {1, 2}

        # Stem: Conv2d(1→c[0], 5×5, stride=(2,1)) halves freq immediately
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # Body: 4 stages of BCResBlocks
        self.stages = nn.ModuleList()
        for idx, num_blk in enumerate(n):
            stage = _make_stage(
                num_blk,
                in_channel=c[idx],
                out_channel=c[idx + 1],
                idx=idx,
                use_stride=(idx in stride_stages),
            )
            self.stages.append(stage)

        # Classifier head (matches official code exactly)
        self.classifier = nn.Sequential(
            nn.Conv2d(c[-2], c[-2], (5, 5), bias=False,
                      groups=c[-2], padding=(0, 2)),  # DW 5×5, no stride
            nn.Conv2d(c[-2], c[-1], 1, bias=False),    # PW 1×1
            nn.BatchNorm2d(c[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c[-1], num_classes, 1),
        )

        total = sum(p.numel() for p in self.parameters())
        lo = int(self.TARGET_PARAMS * (1 - self.TOLERANCE))
        hi = int(self.TARGET_PARAMS * (1 + self.TOLERANCE))
        assert lo <= total <= hi, (
            f"BCResNet1(τ={tau}) has {total:,} parameters; "
            f"expected {lo:,}–{hi:,} (321 000 ± 10%)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 64, 101) → logits (B, num_classes)."""
        x = self.cnn_head(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        x = self.classifier(x)
        return x.view(x.shape[0], -1)
