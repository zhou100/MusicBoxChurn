from __future__ import annotations

import numpy as np
import pytest

from musicbox_churn.training.metrics import (
    compute_metrics,
    lift_at_k,
    precision_at_top_k,
    recall_at_top_k,
    slice_metrics,
)


def test_perfect_ranker_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    assert precision_at_top_k(y, s, k=6) == 1.0
    assert recall_at_top_k(y, s, k=6) == 1.0
    assert lift_at_k(y, s, k=6) == pytest.approx(1.0 / 0.6)


def test_random_ranker_lift_near_one():
    rng = np.random.default_rng(0)
    n = 10_000
    y = rng.integers(0, 2, size=n)
    s = rng.random(n)
    k = n // 10
    assert abs(lift_at_k(y, s, k) - 1.0) < 0.1


def test_compute_metrics_shape():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, size=n)
    s = rng.random(n)
    out = compute_metrics(y, s)
    for key in ["pr_auc", "roc_auc", "brier", "prevalence", "n"]:
        assert key in out
    for frac in (5, 10, 20):
        assert f"precision_at_{frac}pct" in out
        assert f"recall_at_{frac}pct" in out
        assert f"lift_at_{frac}pct" in out


def test_slice_metrics_splits_by_group_and_skips_small_or_degenerate():
    rng = np.random.default_rng(2)
    n = 600
    y = rng.integers(0, 2, size=n)
    s = rng.random(n)
    groups = np.array(
        ["iphone"] * 300 + ["android"] * 290 + ["other"] * 10,
        dtype=object,
    )
    # Force one large group to be all-zero labels — should be skipped (degenerate).
    y_degenerate = y.copy()
    y_degenerate[300:590] = 0
    out = slice_metrics(y_degenerate, s, groups, min_group_size=50)
    assert "iphone" in out
    assert "android" not in out  # all-zero labels
    assert "other" not in out  # below min_group_size
    for key in ("pr_auc", "roc_auc", "brier", "n"):
        assert key in out["iphone"]


def test_topk_invalid_k_raises():
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        precision_at_top_k(y, s, k=0)
    with pytest.raises(ValueError):
        precision_at_top_k(y, s, k=3)
