from __future__ import annotations

import numpy as np
import pytest

from musicbox_churn.training.thresholding import ThresholdPolicy, select_threshold


@pytest.fixture
def y_and_scores():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, size=n)
    s = (y * 0.6) + rng.random(n) * 0.4
    return y, s


def test_max_f1_returns_threshold_in_range(y_and_scores):
    y, s = y_and_scores
    out = select_threshold(y, s, policy=ThresholdPolicy.MAX_F1)
    assert 0.0 <= out["threshold"] <= 1.0


def test_top_k_selects_correct_count(y_and_scores):
    y, s = y_and_scores
    out = select_threshold(y, s, policy=ThresholdPolicy.TOP_K, top_k_fraction=0.10)
    above = (s >= out["threshold"]).sum()
    assert abs(above - 100) <= 5


def test_top_k_requires_fraction(y_and_scores):
    y, s = y_and_scores
    with pytest.raises(ValueError):
        select_threshold(y, s, policy=ThresholdPolicy.TOP_K)
