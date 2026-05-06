from __future__ import annotations

import pandas as pd
import pytest

from musicbox_churn.data.schema import FEATURE_COLUMNS, ID_COL, TARGET_COL


@pytest.fixture
def trained_run(tmp_path, sample_df):
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


def test_batch_score_writes_outputs(tmp_path, sample_df, trained_run):
    from musicbox_churn.inference.batch_score import score

    inference_input = sample_df.drop(columns=[TARGET_COL])
    in_path = tmp_path / "infer.csv"
    inference_input.to_csv(in_path, index=False)

    out_dir = tmp_path / "output"
    paths = score(trained_run, in_path, out_dir, top_n=10)

    for k in ("csv", "parquet", "audience"):
        assert paths[k].exists()

    df_csv = pd.read_csv(paths["csv"])
    assert len(df_csv) == len(inference_input)
    assert {ID_COL, "score", "prob_churn", "rank", "above_threshold"} <= set(df_csv.columns)
    assert df_csv["rank"].is_monotonic_increasing
    assert df_csv["score"].is_monotonic_decreasing
    assert df_csv["score"].between(0.0, 1.0).all()
    assert df_csv["prob_churn"].between(0.0, 1.0).all()

    audience = pd.read_csv(paths["audience"])
    assert len(audience) == 10


def test_batch_score_works_when_label_present(tmp_path, sample_df, trained_run):
    from musicbox_churn.inference.batch_score import score

    in_path = tmp_path / "infer_with_label.csv"
    sample_df.to_csv(in_path, index=False)
    paths = score(trained_run, in_path, tmp_path / "output")
    assert paths["csv"].exists()


def test_batch_score_parquet_roundtrip(tmp_path, sample_df, trained_run):
    pytest.importorskip("pyarrow")
    from musicbox_churn.inference.batch_score import score

    in_path = tmp_path / "infer.csv"
    sample_df[[ID_COL, *FEATURE_COLUMNS]].to_csv(in_path, index=False)
    paths = score(trained_run, in_path, tmp_path / "output")
    df = pd.read_parquet(paths["parquet"])
    assert len(df) == len(sample_df)
