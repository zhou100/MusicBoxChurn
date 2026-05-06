from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.metrics import precision_recall_curve


class ThresholdPolicy(str, Enum):
    MAX_F1 = "max_f1"
    TOP_K = "top_k"
    PRECISION_FLOOR = "precision_floor"
    RECALL_FLOOR = "recall_floor"


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    policy: ThresholdPolicy,
    top_k_fraction: float | None = None,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
) -> dict[str, float]:
    """Pick a decision threshold on validation scores. Returns chosen threshold + diagnostics."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if policy == ThresholdPolicy.MAX_F1:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = (2 * precision * recall) / np.where(precision + recall > 0, precision + recall, 1)
        idx = int(np.argmax(f1[:-1])) if len(thresholds) > 0 else 0
        thr = float(thresholds[idx]) if len(thresholds) > 0 else 0.5
        return {"threshold": thr, "f1": float(f1[idx])}

    if policy == ThresholdPolicy.TOP_K:
        if top_k_fraction is None:
            raise ValueError("top_k_fraction required for TOP_K policy")
        k = max(1, int(round(top_k_fraction * len(scores))))
        thr = float(np.sort(scores)[-k])
        return {"threshold": thr, "k": float(k)}

    if policy == ThresholdPolicy.PRECISION_FLOOR:
        if precision_floor is None:
            raise ValueError("precision_floor required")
        precision, _, thresholds = precision_recall_curve(y_true, scores)
        ok = np.where(precision[:-1] >= precision_floor)[0]
        if len(ok) == 0:
            return {"threshold": 1.0, "precision": float(precision.max())}
        idx = int(ok[0])
        return {"threshold": float(thresholds[idx]), "precision": float(precision[idx])}

    if policy == ThresholdPolicy.RECALL_FLOOR:
        if recall_floor is None:
            raise ValueError("recall_floor required")
        _, recall, thresholds = precision_recall_curve(y_true, scores)
        ok = np.where(recall[:-1] >= recall_floor)[0]
        if len(ok) == 0:
            return {"threshold": 0.0, "recall": float(recall.max())}
        idx = int(ok[-1])
        return {"threshold": float(thresholds[idx]), "recall": float(recall[idx])}

    raise ValueError(f"unknown policy: {policy}")
