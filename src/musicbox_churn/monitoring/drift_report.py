"""Split-stability diagnostics.

NOT production drift. Compares train vs holdout feature distributions on a
single snapshot to surface sampling variance and obvious split anomalies.
True production drift requires a second snapshot taken after deployment
(see TODOS.md, time-based-split entry).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.load_data import load_csv
from ..data.schema import NUMERIC_FEATURE_COLUMNS, TARGET_COL
from ..data.splits import stratified_split

logger = logging.getLogger(__name__)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10
) -> float:
    """PSI on quantile-bucketed numeric values. Returns a non-negative scalar.

    Convention: <0.1 = stable, 0.1–0.25 = moderate shift, >0.25 = large shift.
    Bins come from the *expected* (train) distribution.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 2:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    eps = 1e-6
    e_pct = e_counts / max(e_counts.sum(), 1) + eps
    a_pct = a_counts / max(a_counts.sum(), 1) + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def split_stability(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict:
    splits = stratified_split(df, ratios=ratios, seed=seed)
    train = df.iloc[splits.train]
    val = df.iloc[splits.val]
    test = df.iloc[splits.test]

    feature_psi: list[dict] = []
    for col in NUMERIC_FEATURE_COLUMNS:
        feature_psi.append(
            {
                "feature": col,
                "psi_train_vs_val": population_stability_index(train[col].to_numpy(), val[col].to_numpy()),
                "psi_train_vs_test": population_stability_index(
                    train[col].to_numpy(), test[col].to_numpy()
                ),
            }
        )

    label_rates = {
        "train": float(train[TARGET_COL].mean()),
        "val": float(val[TARGET_COL].mean()),
        "test": float(test[TARGET_COL].mean()),
    }
    return {
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "label_rate": label_rates,
        "label_rate_max_abs_delta": max(label_rates.values()) - min(label_rates.values()),
        "feature_psi": feature_psi,
    }


def _band(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "large"


def render_report(stats: dict) -> str:
    feats = sorted(stats["feature_psi"], key=lambda r: -max(r["psi_train_vs_val"], r["psi_train_vs_test"]))
    rows = [
        f"| {r['feature']} | {r['psi_train_vs_val']:.4f} ({_band(r['psi_train_vs_val'])}) "
        f"| {r['psi_train_vs_test']:.4f} ({_band(r['psi_train_vs_test'])}) |"
        for r in feats
    ]
    return f"""# Split-stability report

> **NOT production drift.** This report compares train vs val/test
> distributions on a single snapshot. It surfaces sampling variance and
> stratification anomalies, not shift over time. True drift monitoring
> requires multiple snapshots — see TODOS.md.

## Split sizes
- train: {stats["n"]["train"]:,}
- val:   {stats["n"]["val"]:,}
- test:  {stats["n"]["test"]:,}

## Label rate by split
- train: {stats["label_rate"]["train"]:.4f}
- val:   {stats["label_rate"]["val"]:.4f}
- test:  {stats["label_rate"]["test"]:.4f}
- max abs delta: {stats["label_rate_max_abs_delta"]:.4f}

## Feature PSI (train as reference)
PSI bands: <0.1 stable · 0.1–0.25 moderate · >0.25 large

| feature | PSI train↔val | PSI train↔test |
|---|---|---|
{chr(10).join(rows)}
"""


def write_report(
    data_path: str | Path,
    out_path: str | Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Path:
    df = load_csv(data_path)
    stats = split_stability(df, seed=seed, ratios=ratios)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(stats))
    (out.with_suffix(".json")).write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Write the split-stability monitoring report.")
    p.add_argument("--data", default="Processed_data/df_model_final.csv")
    p.add_argument("--out", default="output/monitoring_report.md")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    write_report(args.data, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
