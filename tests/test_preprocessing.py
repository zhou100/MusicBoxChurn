from __future__ import annotations

import numpy as np

from musicbox_churn.data.schema import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from musicbox_churn.training.preprocessor import build_preprocessor


def test_preprocessor_output_is_finite_and_correct_shape(sample_df):
    pre = build_preprocessor()
    X_raw = sample_df[FEATURE_COLUMNS]
    X = pre.fit_transform(X_raw)
    assert X.shape[0] == len(sample_df)
    n_cat_levels = sum(sample_df[c].nunique() for c in CATEGORICAL_FEATURE_COLUMNS)
    assert X.shape[1] == len(NUMERIC_FEATURE_COLUMNS) + n_cat_levels
    assert np.isfinite(X).all()


def test_preprocessor_numeric_zero_mean_unit_var(sample_df):
    pre = build_preprocessor()
    X = pre.fit_transform(sample_df[FEATURE_COLUMNS])
    numeric_block = X[:, : len(NUMERIC_FEATURE_COLUMNS)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-2)
