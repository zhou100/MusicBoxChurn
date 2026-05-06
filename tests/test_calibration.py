from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from musicbox_churn.training.calibration import ScoreCalibrator, brier, reliability_data


@pytest.fixture
def miscalibrated_scores():
    """Generate scores that are systematically too low (need upward calibration)."""
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, size=n)
    raw = np.clip(0.3 * y + rng.beta(2, 5, size=n) * 0.5, 0, 1)
    return raw, y


def test_isotonic_improves_brier(miscalibrated_scores):
    raw, y = miscalibrated_scores
    raw_brier = brier_score_loss(y, raw)
    cal = ScoreCalibrator("isotonic").fit(raw, y)
    cal_brier = brier_score_loss(y, cal.transform(raw))
    assert cal_brier < raw_brier


def test_sigmoid_improves_brier(miscalibrated_scores):
    raw, y = miscalibrated_scores
    raw_brier = brier_score_loss(y, raw)
    cal = ScoreCalibrator("sigmoid").fit(raw, y)
    cal_brier = brier_score_loss(y, cal.transform(raw))
    assert cal_brier < raw_brier


def test_calibrated_in_unit_interval(miscalibrated_scores):
    raw, y = miscalibrated_scores
    cal = ScoreCalibrator("isotonic").fit(raw, y)
    out = cal.transform(raw)
    assert (out >= 0).all() and (out <= 1).all()


def test_isotonic_preserves_ranking_weakly(miscalibrated_scores):
    raw, y = miscalibrated_scores
    cal = ScoreCalibrator("isotonic").fit(raw, y).transform(raw)
    # Isotonic creates plateaus where calibrated values are tied; that breaks
    # exact argsort equality. The contract is weakly monotone: if raw[i] < raw[j]
    # then cal[i] <= cal[j].
    order = np.argsort(raw, kind="stable")
    cal_in_raw_order = cal[order]
    assert np.all(np.diff(cal_in_raw_order) >= 0)


def test_save_load_roundtrip(tmp_path, miscalibrated_scores):
    raw, y = miscalibrated_scores
    cal = ScoreCalibrator("isotonic").fit(raw, y)
    p = tmp_path / "cal.pkl"
    cal.save(p)
    loaded = ScoreCalibrator.load(p)
    np.testing.assert_array_equal(cal.transform(raw), loaded.transform(raw))


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="must be"):
        ScoreCalibrator("magic")


def test_brier_helper_matches_sklearn(miscalibrated_scores):
    raw, y = miscalibrated_scores
    assert brier(y, raw) == pytest.approx(brier_score_loss(y, raw))


def test_reliability_data_shape(miscalibrated_scores):
    raw, y = miscalibrated_scores
    rel = reliability_data(y, raw, n_bins=10)
    assert len(rel["mean_pred"]) == 10
    assert len(rel["empirical_rate"]) == 10
    assert sum(rel["count"]) == len(y)
