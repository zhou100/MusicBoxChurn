from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.schema import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
