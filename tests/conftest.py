from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicbox_churn.data.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    ID_COL,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COL,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    data: dict[str, np.ndarray] = {
        ID_COL: np.arange(n),
        TARGET_COL: rng.integers(0, 2, size=n),
    }
    for col in NUMERIC_FEATURE_COLUMNS:
        data[col] = rng.random(n).astype(float)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        data[col] = rng.integers(1, 3, size=n)
    return pd.DataFrame(data)
