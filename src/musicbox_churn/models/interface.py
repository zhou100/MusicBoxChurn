from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ModelHandle(ABC):
    """Unified model interface across sklearn-compatible and torch backends."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelHandle": ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(label=1) as a 1-D array of length len(X)."""

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ModelHandle": ...
