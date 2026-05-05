from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k > len(scores):
        raise ValueError(f"k must be in [1, {len(scores)}], got {k}")
    order = np.argsort(-scores, kind="stable")[:k]
    mask = np.zeros_like(scores, dtype=bool)
    mask[order] = True
    return mask


def precision_at_top_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    mask = _topk_mask(scores, k)
    return float(y_true[mask].mean())


def recall_at_top_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    pos = int(y_true.sum())
    if pos == 0:
        return 0.0
    mask = _topk_mask(scores, k)
    return float(y_true[mask].sum() / pos)


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    base = float(y_true.mean())
    if base == 0:
        return 0.0
    return precision_at_top_k(y_true, scores, k) / base


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    top_k_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    n = len(y_true)
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "prevalence": float(y_true.mean()),
        "n": float(n),
    }
    for frac in top_k_fractions:
        k = max(1, int(round(frac * n)))
        pct = int(frac * 100)
        out[f"precision_at_{pct}pct"] = precision_at_top_k(y_true, scores, k)
        out[f"recall_at_{pct}pct"] = recall_at_top_k(y_true, scores, k)
        out[f"lift_at_{pct}pct"] = lift_at_k(y_true, scores, k)
    return out
