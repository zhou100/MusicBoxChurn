from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import TARGET_COL


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(
    df: pd.DataFrame,
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    target_col: str = TARGET_COL,
) -> SplitIndices:
    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    if min(ratios) <= 0:
        raise ValueError(f"all ratios must be > 0, got {ratios}")

    idx = np.arange(len(df))
    y = df[target_col].to_numpy()

    train_idx, temp_idx, _, y_temp = train_test_split(
        idx, y, test_size=(val_r + test_r), stratify=y, random_state=seed
    )
    val_share = val_r / (val_r + test_r)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - val_share), stratify=y_temp, random_state=seed
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)
