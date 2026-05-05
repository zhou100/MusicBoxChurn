from __future__ import annotations

import numpy as np
import pytest

from musicbox_churn.monitoring.drift_report import (
    population_stability_index,
    render_report,
    split_stability,
)


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    y_small = x + rng.normal(scale=0.05, size=10_000)
    y_large = rng.normal(loc=2.0, size=10_000)
    psi_small = population_stability_index(x, y_small)
    psi_large = population_stability_index(x, y_large)
    assert psi_small < psi_large
    assert psi_large > 0.25


def test_psi_handles_empty_input():
    assert population_stability_index(np.array([]), np.array([1.0, 2.0])) == 0.0


def test_split_stability_runs(sample_df):
    stats = split_stability(sample_df)
    assert set(stats["n"]) == {"train", "val", "test"}
    assert sum(stats["n"].values()) == len(sample_df)
    assert len(stats["feature_psi"]) > 0
    assert "label_rate_max_abs_delta" in stats


def test_render_report_contains_framing(sample_df):
    stats = split_stability(sample_df)
    text = render_report(stats)
    assert "NOT production drift" in text
    assert "PSI" in text


@pytest.fixture
def trained_run_for_monitor(tmp_path, sample_df):
    pytest.importorskip("mlflow")
    from musicbox_churn.training.train import train_from_config

    csv_path = tmp_path / "train.csv"
    sample_df.to_csv(csv_path, index=False)
    cfg = {
        "model_type": "lr",
        "seed": 42,
        "data": {"path": str(csv_path), "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "params": {"max_iter": 200},
        "threshold": {"policy": "max_f1"},
        "artifacts_dir": str(tmp_path / "artifacts"),
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "mlflow_experiment": "smoke",
    }
    return train_from_config(cfg)


def test_score_report_writes_markdown(trained_run_for_monitor):
    from musicbox_churn.monitoring.score_report import write_report

    out = write_report(trained_run_for_monitor)
    assert out.exists()
    text = out.read_text()
    assert "Evaluation report" in text
    assert "ranking scores" in text
