from __future__ import annotations

import numpy as np
import pytest

from musicbox_churn.data.schema import TARGET_COL
from musicbox_churn.data.splits import stratified_split


def test_split_partitions_all_rows(sample_df):
    s = stratified_split(sample_df)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(all_idx) == len(sample_df)
    assert len(set(all_idx.tolist())) == len(sample_df)


def test_split_ratios_approximate(sample_df):
    n = len(sample_df)
    s = stratified_split(sample_df, ratios=(0.70, 0.15, 0.15), seed=42)
    assert abs(len(s.train) / n - 0.70) < 0.02
    assert abs(len(s.val) / n - 0.15) < 0.02
    assert abs(len(s.test) / n - 0.15) < 0.02


def test_split_preserves_class_balance(sample_df):
    s = stratified_split(sample_df, seed=42)
    base = sample_df[TARGET_COL].mean()
    for idx in (s.train, s.val, s.test):
        sub = sample_df.iloc[idx][TARGET_COL].mean()
        assert abs(sub - base) < 0.05


def test_split_is_deterministic(sample_df):
    a = stratified_split(sample_df, seed=42)
    b = stratified_split(sample_df, seed=42)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.test, b.test)


def test_split_rejects_bad_ratios(sample_df):
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(sample_df, ratios=(0.5, 0.3, 0.3))
    with pytest.raises(ValueError, match="> 0"):
        stratified_split(sample_df, ratios=(0.0, 0.5, 0.5))
