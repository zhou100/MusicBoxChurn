from __future__ import annotations

from pathlib import Path

import pytest

from musicbox_churn.data.load_data import load_csv
from musicbox_churn.data.schema import FEATURE_COLUMNS, ID_COL, TARGET_COL

DATA_PATH = Path("Processed_data/df_model_final.csv")


@pytest.mark.skipif(not DATA_PATH.exists(), reason=f"{DATA_PATH} not present")
def test_load_real_csv_passes_validation():
    df = load_csv(DATA_PATH)
    assert len(df) > 0
    assert {ID_COL, TARGET_COL, *FEATURE_COLUMNS}.issubset(df.columns)


def test_load_csv_strips_whitespace(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    padded = sample_df.rename(columns={c: f" {c} " for c in sample_df.columns})
    padded.to_csv(p, index=False)
    df = load_csv(p)
    assert all(c == c.strip() for c in df.columns)
