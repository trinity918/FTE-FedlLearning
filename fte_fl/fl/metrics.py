"""Evaluation and diagnostic metrics.

Clean metrics use the untouched held-out set only.  Attack metrics use
separately constructed sets:
  backdoor ASR      held-out samples whose true label != target, with the
                    FULL global trigger applied; ASR = fraction classified as
                    the target class.  (Samples already of the target class
                    are excluded so they cannot count as successes.)
  label flip        held-out samples of the source class, clean inputs:
                    flip_rate = fraction classified as the target class,
                    source_accuracy = fraction classified correctly.
Filtering metrics treat "rejected" as "contributes nothing to the
aggregate" (weight 0, or every coordinate masked).  They use the
simulator's ground-truth malicious labels, which no defense sees; they are
diagnostics, not something a deployed server could compute.  For
coordinate-wise rules (trimmed mean, Bulyan) a client is rarely rejected
outright, so the mean kept-coordinate fraction is also reported.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict:
    """Mean cross-entropy and accuracy of `model` (eval mode) on (x, y)."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for s in range(0, len(y), batch_size):
        logits = model(x[s:s + batch_size])
        loss_sum += F.cross_entropy(logits, y[s:s + batch_size], reduction="sum").item()
        correct += (logits.argmax(dim=1) == y[s:s + batch_size]).sum().item()
    loss = loss_sum / len(y)
    if not np.isfinite(loss):
        raise FloatingPointError("evaluation loss is not finite (the global model diverged)")
    return {"loss": loss, "accuracy": correct / len(y), "samples": int(len(y))}


@torch.no_grad()
def predictions(model, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[s:s + batch_size]).argmax(dim=1) for s in range(0, len(x), batch_size)])


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def filtering_metrics(is_malicious: np.ndarray, weights: np.ndarray, kept: np.ndarray,
                      assumed_f: int | None) -> dict:
    mal = np.asarray(is_malicious, dtype=bool)
    rejected = ~(kept > 0)
    tp = int(np.sum(rejected & mal))
    fp = int(np.sum(rejected & ~mal))
    n_mal, n_hon = int(mal.sum()), int((~mal).sum())
    return {
        "malicious_participants": n_mal,
        "honest_participants": n_hon,
        "malicious_selected": n_mal - tp,
        "malicious_rejected": tp,
        "honest_rejected": fp,
        "tpr": _rate(tp, n_mal),
        "fpr": _rate(fp, n_hon),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, n_mal),
        # sum over malicious clients of a_i * (fraction of coordinates kept)
        "malicious_effective_weight": float((weights * kept)[mal].sum()),
        "mean_kept_fraction_malicious": float(kept[mal].mean()) if n_mal else None,
        "mean_kept_fraction_honest": float(kept[~mal].mean()) if n_hon else None,
        "exceeds_assumed_f": None if assumed_f is None else bool(n_mal > assumed_f),
    }
