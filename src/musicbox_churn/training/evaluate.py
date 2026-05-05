from __future__ import annotations

import numpy as np

from ..models.interface import ModelHandle
from .calibration import reliability_data
from .metrics import compute_metrics


def evaluate(
    handle: ModelHandle,
    X: np.ndarray,
    y: np.ndarray,
    *,
    top_k_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict:
    scores = handle.predict_proba(X)
    metrics = compute_metrics(y, scores, top_k_fractions=top_k_fractions)
    rel = reliability_data(y, scores, n_bins=10)
    return {"metrics": metrics, "reliability": rel, "scores_summary": _summary(scores)}


def _summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(scores.min()),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.median(scores)),
        "p75": float(np.quantile(scores, 0.75)),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
    }
