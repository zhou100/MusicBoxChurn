from __future__ import annotations

import json
from pathlib import Path

import pytest

from musicbox_churn.data.schema import FEATURE_COLUMNS, TARGET_COL


@pytest.fixture
def smoke_csv(tmp_path, sample_df):
    p = tmp_path / "smoke.csv"
    sample_df[
        [
            *([col for col in sample_df.columns if col not in FEATURE_COLUMNS + [TARGET_COL]]),
            TARGET_COL,
            *FEATURE_COLUMNS,
        ]
    ].to_csv(p, index=False)
    return p


def test_lr_smoke_end_to_end(tmp_path, smoke_csv):
    pytest.importorskip("mlflow")
    pytest.importorskip("yaml")
    from musicbox_churn.training.train import train_from_config

    cfg = {
        "model_type": "lr",
        "seed": 42,
        "data": {
            "path": str(smoke_csv),
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "params": {"max_iter": 200},
        "threshold": {"policy": "max_f1"},
        "artifacts_dir": str(tmp_path / "artifacts"),
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "mlflow_experiment": "smoke",
    }
    out_dir = train_from_config(cfg)
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "preprocessor.pkl").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "threshold.json").exists()
    assert (out_dir / "feature_schema.json").exists()
    assert (out_dir / "model_card.md").exists()

    metrics = json.loads(Path(out_dir / "metrics.json").read_text())
    assert "pr_auc" in metrics["metrics"]
    assert "pr_auc" in metrics["test_metrics"]


def test_gbm_smoke(tmp_path, smoke_csv):
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    from musicbox_churn.training.train import train_from_config

    cfg = {
        "model_type": "gbm",
        "seed": 42,
        "data": {"path": str(smoke_csv), "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "params": {"n_estimators": 30, "learning_rate": 0.1},
        "threshold": {"policy": "top_k", "top_k_fraction": 0.10},
        "artifacts_dir": str(tmp_path / "artifacts"),
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "mlflow_experiment": "smoke",
    }
    out_dir = train_from_config(cfg)
    assert (out_dir / "model.pkl").exists()


def test_mlp_smoke(tmp_path, smoke_csv):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from musicbox_churn.training.train import train_from_config

    cfg = {
        "model_type": "mlp",
        "seed": 42,
        "data": {"path": str(smoke_csv), "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "params": {"hidden": [16], "epochs": 2, "batch_size": 64},
        "threshold": {"policy": "max_f1"},
        "artifacts_dir": str(tmp_path / "artifacts"),
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "mlflow_experiment": "smoke",
    }
    out_dir = train_from_config(cfg)
    assert (out_dir / "model.pt").exists()
