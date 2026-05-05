from __future__ import annotations

import pandas as pd

ID_COL = "uid"
TARGET_COL = "label"

WINDOWS_SHORT = (1, 3, 7, 14, 30)
WINDOWS_RECENT = (7, 14, 30)

FREQ_COLUMNS = [
    f"freq_{e}_last_{w}" for e in ("P", "D", "S") for w in WINDOWS_SHORT
]
RECENT_COLUMNS = [
    f"recent_{e}_last_{w}" for e in ("P", "D", "S") for w in WINDOWS_RECENT
]
TOT_PLAYTIME_COLUMNS = [f"tot_playtime_last_{w}" for w in WINDOWS_SHORT]
SONGS_FULL_COLUMNS = [f"songs_full_last_{w}" for w in WINDOWS_SHORT]
MEAN_PERCENT_COLUMNS = [f"mean_percent_last_{w}" for w in WINDOWS_SHORT]
SD_PERCENT_COLUMNS = [f"sd_percent_last_{w}" for w in WINDOWS_SHORT]

NUMERIC_FEATURE_COLUMNS = (
    FREQ_COLUMNS
    + RECENT_COLUMNS
    + TOT_PLAYTIME_COLUMNS
    + SONGS_FULL_COLUMNS
    + MEAN_PERCENT_COLUMNS
    + SD_PERCENT_COLUMNS
)
CATEGORICAL_FEATURE_COLUMNS = ["device_type"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

REQUIRED_COLUMNS = [ID_COL, TARGET_COL, *FEATURE_COLUMNS]


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    if df[ID_COL].isna().any():
        raise SchemaError(f"{ID_COL} contains nulls")
    if df[ID_COL].duplicated().any():
        n = int(df[ID_COL].duplicated().sum())
        raise SchemaError(f"{ID_COL} has {n} duplicate values")

    label_values = set(df[TARGET_COL].dropna().unique().tolist())
    if not label_values.issubset({0, 1}):
        raise SchemaError(f"{TARGET_COL} must be in {{0, 1}}, got {sorted(label_values)}")
    if df[TARGET_COL].isna().any():
        raise SchemaError(f"{TARGET_COL} contains nulls")

    for col in NUMERIC_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"feature {col!r} must be numeric, got {df[col].dtype}")
