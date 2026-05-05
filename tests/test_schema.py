from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicbox_churn.data.schema import (
    FEATURE_COLUMNS,
    ID_COL,
    NUMERIC_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COL,
    SchemaError,
    validate,
)


def test_feature_column_count():
    assert len(NUMERIC_FEATURE_COLUMNS) == 44
    assert len(FEATURE_COLUMNS) == 45


def test_validate_passes_on_clean_frame(sample_df):
    validate(sample_df)


def test_validate_rejects_missing_column(sample_df):
    df = sample_df.drop(columns=[NUMERIC_FEATURE_COLUMNS[0]])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate(df)


def test_validate_rejects_duplicate_uid(sample_df):
    df = sample_df.copy()
    df.loc[0, ID_COL] = df.loc[1, ID_COL]
    with pytest.raises(SchemaError, match="duplicate"):
        validate(df)


def test_validate_rejects_bad_label(sample_df):
    df = sample_df.copy()
    df.loc[0, TARGET_COL] = 2
    with pytest.raises(SchemaError, match="must be in"):
        validate(df)


def test_validate_rejects_null_label(sample_df):
    df = sample_df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    df.loc[0, TARGET_COL] = np.nan
    with pytest.raises(SchemaError):
        validate(df)


def test_validate_rejects_non_numeric_feature(sample_df):
    df = sample_df.copy()
    df[NUMERIC_FEATURE_COLUMNS[0]] = "x"
    with pytest.raises(SchemaError, match="must be numeric"):
        validate(df)


def test_required_columns_have_no_duplicates():
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))


def test_validate_accepts_pandas_indexed_frame(sample_df):
    df = sample_df.set_index(np.arange(len(sample_df)) + 1000)
    validate(pd.DataFrame(df))
