from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(scores).astype(float)))


def reliability_data(
    y_true: np.ndarray, scores: np.ndarray, *, n_bins: int = 10
) -> dict[str, list[float]]:
    """Return per-bin (mean predicted score, empirical positive rate, count).

    No Platt or isotonic fitting — we expose ranking scores honestly. See P3 in TODOS.md
    for the calibration unwind path if true probabilities are required downstream.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins[1:-1], right=False)
    mean_pred: list[float] = []
    emp_rate: list[float] = []
    counts: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            mean_pred.append(float("nan"))
            emp_rate.append(float("nan"))
            counts.append(0.0)
            continue
        mean_pred.append(float(scores[mask].mean()))
        emp_rate.append(float(y_true[mask].mean()))
        counts.append(float(mask.sum()))
    return {"mean_pred": mean_pred, "empirical_rate": emp_rate, "count": counts}
