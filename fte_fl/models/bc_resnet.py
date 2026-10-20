"""BC-ResNet for Google Speech Commands keyword spotting.

Reference:
  Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting",
  Interspeech 2021 (arXiv:2106.04140).
  Official implementation: https://github.com/Qualcomm-AI-research/bcresnet
  License: see upstream repo (Qualcomm Technologies, Inc.).
  Source files inspected: bcresnet.py and main.py for the port;
  subspectralnorm.py and utils.py as well for the 2026-10-20 settings
  re-check (see below).

Naming clarification — important:
  In the BC-ResNet paper, the model suffix is a channel-width multiplier τ,
  not a ranked model index.  "BC-ResNet-8" means τ=8, base_c=64,
  giving ~321k parameters and the paper's best reported accuracy (98.7% on
  the 12-class GSCv2 task; a paper figure, not reproduced here).
  "BC-ResNet-1" means τ=1, base_c=8, giving ~10k parameters.

  This project's specification targets ~321k parameters, which corresponds
  to the BC-ResNet-8 (τ=8) configuration.  The Python class is named
  BCResNet1 for file-naming convenience; the `tau` constructor argument
  defaults to 8, and the module-level constant BC_RESNET_TAU = 8 is the
  canonical record.  Any researcher reading this code should treat the
  implemented model as **BC-ResNet-8 (τ=8, base_c=64)**.

Architecture (ported from the Qualcomm reference implementation):
  The implementation was derived by reading bcresnet.py (BCResNets class and
  BCResBlock class) and main.py (_load_model: BCResNets(int(self.tau * 8)))
  directly from https://github.com/Qualcomm-AI-research/bcresnet.

  Key components carried over (deviations listed below):
    - Stem: Conv2d(1→2·base_c, 5×5, stride=(2,1)) + BN + ReLU
    - BCResBlock with:
        f2 (2-D path): optional 1×1 transition + depthwise (3,1) + SSN
        f1 (1-D broadcast path): AvgPool(F→1) + dilated DW (1,3) + SiLU + PW 1×1 + Dropout2d
        residual: f1 + f2; identity skip only for non-transition blocks
    - Four stages, n=[2,2,4,4] blocks, frequency strides at stages 1 and 2
    - Classifier: DW (5,5) + PW 1×1 + BN + ReLU + AvgPool + Conv1×1(→classes)
    - Channel schedule with base_c=64: [128, 64, 96, 128, 160, 256]

  Upstream settings re-checked on 2026-10-20 against bcresnet.py,
  subspectralnorm.py and utils.py (main branch): 40 mel bins, 12-class
  label set (10 keywords + silence + unknown, for both GSC v1 and v2),
  SubSpectralNorm(out_plane, 5) with per-sub-band affine ("Sub" mode),
  and an assertion that the frequency size is divisible by the group count.

  Deviations from the upstream 40-bin / 12-class configuration:
    - Input: 64 mel bins (upstream: 40).  Frequency sizes seen by the SSN
      layers are [32, 32, 16, 16, 8 × 8] (upstream: [20, 20, 10, 10, 5 × 8]).
    - Classes: 35 (full GSC v2 label set; upstream trains the 12-class task).
    - SubSpectralNorm sub-bands: S=8 (upstream: S=5).  S=5 does not divide
      32/16/8.  S=8 reproduces upstream's per-stage sub-band width exactly
      (4, 2, 1 frequency bins per sub-band at the 1st/2nd/3rd-4th stage
      resolutions), at the cost of 8 rather than 5 sub-bands per layer.
    - Classifier: the depthwise (5,5) conv without frequency padding maps
      F=8 → 4 (upstream: F=5 → 1); the existing AdaptiveAvgPool averages
      the remaining 4 rows.

Target: 321,000 parameters ±10% (288,900 – 353,100).
Assertion enforced at construction time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fte_fl.models.base import FTEModel

# Canonical configuration for this project's BC-ResNet model.
# Corresponds to BC-ResNet-8 (τ=8) in Kim et al. 2021, base_c=64.
BC_RESNET_TAU: int = 8

# Number of SubSpectralNorm sub-bands for the 64-mel-bin input
# (upstream uses 5 for 40 bins; see module docstring for the justification).
SSN_SUBBANDS: int = 8


class SubSpectralNorm(nn.Module):
    """Normalize within sub-spectral frequency groups (Chang et al., 2021).

    Splits the F dimension into S contiguous sub-bands and applies BatchNorm2d
    with a separate mean/variance and affine (γ, β) per (channel, sub-band),
    matching the upstream "Sub" affine mode.  Like upstream, F must be
    divisible by S; there is no plain-BN fallback.
    """

    def __init__(self, num_channels: int, S: int = SSN_SUBBANDS) -> None:
        super().__init__()
        self.S = S
        self.num_channels = num_channels
        # BN over C*S "virtual channels" (each sub-band is a group)
        self.bn = nn.BatchNorm2d(num_channels * S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, freq, T = x.shape
        S = self.S
        if freq % S != 0:
            raise ValueError(
                f"SubSpectralNorm: frequency size {freq} is not divisible by "
                f"S={S} sub-bands."
            )
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
            layers.append(SubSpectralNorm(out_c, S=SSN_SUBBANDS))
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
    """BC-ResNet-8 (τ=8, base_c=64) keyword-spotting model for Speech Commands.

    Despite the class name BCResNet1 (kept for project file-naming convention),
    this implements the BC-ResNet-8 configuration — the ~321k-parameter variant
    from Kim et al. 2021.  See the module docstring for the full naming
    clarification and reference to the Qualcomm source code.

    Precise variant: BC-ResNet-8 adapted to 64 mel bins and 35 classes,
    with SubSpectralNorm using S=8 sub-bands (upstream: S=5 at 40 bins).

    Constructor argument `tau` defaults to 8 (BC-ResNet-8).  Setting tau=1
    would give the much smaller BC-ResNet-1 variant (~10k params), which does
    NOT meet the project's 321k ±10% parameter target.

    Target: 288,900 – 353,100 parameters.  Assertion enforced at construction.
    """

    TARGET_PARAMS: int = 321_000
    TOLERANCE: float = 0.10

    def __init__(self, num_classes: int = 35, tau: int = BC_RESNET_TAU) -> None:
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
