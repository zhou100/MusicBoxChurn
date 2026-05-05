from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .schema import ID_COL, NUMERIC_FEATURE_COLUMNS, validate

logger = logging.getLogger(__name__)


def load_csv(
    path: str | Path,
    *,
    validate_schema: bool = True,
    require_target: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in NUMERIC_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if ID_COL in df.columns:
        dupes = int(df[ID_COL].duplicated().sum())
        if dupes:
            logger.warning("dropping %d duplicate %s rows (keeping first)", dupes, ID_COL)
            df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)

    if validate_schema:
        validate(df, require_target=require_target)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the modeling table.")
    parser.add_argument(
        "--path",
        default="Processed_data/df_model_final.csv",
        help="path to the modeling CSV",
    )
    args = parser.parse_args()

    df = load_csv(args.path)
    print(f"OK: {len(df):,} rows, {df.shape[1]} columns at {args.path}")


if __name__ == "__main__":
    main()
