from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .interface import ModelHandle


class SklearnHandle(ModelHandle):
    def __init__(self, name: str, estimator: Any):
        self.name = name
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnHandle":
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump({"name": self.name, "estimator": self.estimator}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnHandle":
        blob = joblib.load(path)
        return cls(name=blob["name"], estimator=blob["estimator"])


def build_lr(cfg: dict[str, Any] | None = None) -> SklearnHandle:
    cfg = cfg or {}
    est = LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        solver=cfg.get("solver", "lbfgs"),
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
    )
    return SklearnHandle("lr", est)


def build_rf(cfg: dict[str, Any] | None = None) -> SklearnHandle:
    cfg = cfg or {}
    est = RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 5),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
    )
    return SklearnHandle("rf", est)


def build_gbm(cfg: dict[str, Any] | None = None) -> SklearnHandle:
    cfg = cfg or {}
    est = LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 500),
        learning_rate=cfg.get("learning_rate", 0.05),
        num_leaves=cfg.get("num_leaves", 31),
        min_child_samples=cfg.get("min_child_samples", 20),
        subsample=cfg.get("subsample", 0.9),
        colsample_bytree=cfg.get("colsample_bytree", 0.9),
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
        verbose=-1,
    )
    return SklearnHandle("gbm", est)
