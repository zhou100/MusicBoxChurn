from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class ScoreCalibrator:
    """Map raw ranking scores → calibrated probabilities.

    Two methods:
      - "isotonic": IsotonicRegression. Flexible, needs a few thousand
        validation examples. Default.
      - "sigmoid":  Platt scaling (LogisticRegression on raw score).
        Sturdier with small validation sets, less flexible.

    Fit on the *validation* split's (raw_score, label). Applied at scoring
    time before any expected-value math downstream.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")
        self.method = method
        self._estimator: IsotonicRegression | LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=int).ravel()
        if self.method == "isotonic":
            est = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            est.fit(scores, labels)
        else:
            est = LogisticRegression(C=1.0, solver="lbfgs")
            est.fit(scores.reshape(-1, 1), labels)
        self._estimator = est
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        assert self._estimator is not None, "fit() must be called before transform"
        scores = np.asarray(scores, dtype=float).ravel()
        if self.method == "isotonic":
            return np.clip(self._estimator.predict(scores), 0.0, 1.0)
        return self._estimator.predict_proba(scores.reshape(-1, 1))[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump({"method": self.method, "estimator": self._estimator}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreCalibrator":
        blob = joblib.load(path)
        c = cls(method=blob["method"])
        c._estimator = blob["estimator"]
        return c


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
