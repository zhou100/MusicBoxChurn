from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.load_data import load_csv
from ..data.schema import ID_COL
from ..models.baselines import SklearnHandle
from ..models.interface import ModelHandle
from ..models.tabular_mlp import TabularMLPHandle
from ..training.calibration import ScoreCalibrator

logger = logging.getLogger(__name__)


def load_run(
    run_dir: str | Path,
) -> tuple[ModelHandle, object, ScoreCalibrator | None, dict]:
    run_dir = Path(run_dir)
    schema = json.loads((run_dir / "feature_schema.json").read_text())
    pre = joblib.load(run_dir / "preprocessor.pkl")
    model_type = schema.get("model_type")
    artifact = schema.get("model_artifact") or ("model.pt" if model_type == "mlp" else "model.pkl")
    model_path = run_dir / artifact

    if model_type == "mlp":
        handle: ModelHandle = TabularMLPHandle.load(model_path)
    else:
        handle = SklearnHandle.load(model_path)

    cal_path = run_dir / "calibrator.pkl"
    calibrator = ScoreCalibrator.load(cal_path) if cal_path.exists() else None
    return handle, pre, calibrator, schema


def score(
    run_dir: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 50,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handle, pre, calibrator, schema = load_run(run_dir)

    df = load_csv(input_path, require_target=False)
    feature_cols = schema["feature_columns"]
    X = np.asarray(pre.transform(df[feature_cols]), dtype=np.float32)
    scores = handle.predict_proba(X)
    prob_churn = calibrator.transform(scores) if calibrator is not None else None

    threshold_path = run_dir / "threshold.json"
    threshold = None
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text())["threshold"])

    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy(), "score": scores})
    if prob_churn is not None:
        out["prob_churn"] = prob_churn
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    if threshold is not None:
        out["above_threshold"] = (out["score"] >= threshold).astype(int)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"predictions_{stamp}.csv"
    parquet_path = output_dir / f"predictions_{stamp}.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)

    sample_dir = output_dir / "predictions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "audience_top50.csv"
    out.head(top_n).to_csv(sample_path, index=False)

    logger.info(
        "scored %d rows; wrote %s, %s, %s", len(out), csv_path, parquet_path, sample_path
    )
    return {"csv": csv_path, "parquet": parquet_path, "audience": sample_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Batch-score a CSV with a trained run.")
    p.add_argument("--run-dir", required=True, help="path to artifacts/<run_name>/")
    p.add_argument("--input", required=True, help="input CSV with uid + features")
    p.add_argument("--output-dir", default="output", help="directory for ranked outputs")
    p.add_argument("--top-n", type=int, default=50, help="audience_top50.csv size")
    args = p.parse_args()
    paths = score(args.run_dir, args.input, args.output_dir, top_n=args.top_n)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
