from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import yaml

from ..data.load_data import load_csv
from ..data.schema import FEATURE_COLUMNS, TARGET_COL
from ..data.splits import stratified_split
from ..models.baselines import build_gbm, build_lr, build_rf
from ..models.interface import ModelHandle
from ..models.tabular_mlp import TabularMLPHandle
from ..utils.seed import set_global_seed
from .calibration import ScoreCalibrator
from .evaluate import evaluate
from .metrics import compute_metrics
from .preprocessor import build_preprocessor
from .thresholding import ThresholdPolicy, select_threshold

logger = logging.getLogger(__name__)


def _build_model(model_type: str, params: dict[str, Any]) -> ModelHandle:
    if model_type == "lr":
        return build_lr(params)
    if model_type == "rf":
        return build_rf(params)
    if model_type == "gbm":
        return build_gbm(params)
    if model_type == "mlp":
        return TabularMLPHandle(**params)
    raise ValueError(f"unknown model_type: {model_type}")


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _model_artifact_name(model_type: str) -> str:
    return "model.pt" if model_type == "mlp" else "model.pkl"


def _write_model_card(
    out_dir: Path,
    cfg: dict,
    val_metrics: dict,
    test_metrics: dict,
    threshold_info: dict,
) -> None:
    text = f"""# Model card — {cfg["model_type"]}

Run: `{out_dir.name}`
Created: {datetime.now(timezone.utc).isoformat()}
Seed: {cfg.get("seed", 42)}

## Framing
This model emits a **ranking score**, not a calibrated probability.
Use it to prioritize an intervention audience (top-k by risk), not for
expected-value math without further calibration. See TODOS.md (P3).

## Validation metrics
```
{json.dumps(val_metrics, indent=2)}
```

## Test metrics
```
{json.dumps(test_metrics, indent=2)}
```

## Threshold
Policy: `{threshold_info.get("policy")}`
Selected: `{threshold_info.get("threshold")}`

## Config
```
{json.dumps(cfg, indent=2)}
```
"""
    (out_dir / "model_card.md").write_text(text)


def train_from_config(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path", "Processed_data/df_model_final.csv")
    df = load_csv(data_path)

    if "limit_rows" in data_cfg:
        df = df.sample(n=int(data_cfg["limit_rows"]), random_state=seed).reset_index(drop=True)

    split_cfg = data_cfg.get("split", {})
    ratios = (
        split_cfg.get("train", 0.70),
        split_cfg.get("val", 0.15),
        split_cfg.get("test", 0.15),
    )
    splits = stratified_split(df, ratios=ratios, seed=seed)

    pre = build_preprocessor()
    X_train_raw = df.iloc[splits.train][FEATURE_COLUMNS]
    X_val_raw = df.iloc[splits.val][FEATURE_COLUMNS]
    X_test_raw = df.iloc[splits.test][FEATURE_COLUMNS]
    y_train = df.iloc[splits.train][TARGET_COL].to_numpy()
    y_val = df.iloc[splits.val][TARGET_COL].to_numpy()
    y_test = df.iloc[splits.test][TARGET_COL].to_numpy()

    pre.fit(X_train_raw)
    X_train = np.asarray(pre.transform(X_train_raw), dtype=np.float32)
    X_val = np.asarray(pre.transform(X_val_raw), dtype=np.float32)
    X_test = np.asarray(pre.transform(X_test_raw), dtype=np.float32)

    model_type = cfg["model_type"]
    handle = _build_model(model_type, cfg.get("params", {}))

    t0 = time.time()
    handle.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    val_eval = evaluate(handle, X_val, y_val)
    test_eval = evaluate(handle, X_test, y_test)

    # Calibration: fit on val (raw_score, label); evaluate on val + test.
    cal_cfg = cfg.get("calibration", {"method": "isotonic"})
    cal_method = cal_cfg.get("method", "isotonic")
    val_scores_raw = handle.predict_proba(X_val)
    test_scores_raw = handle.predict_proba(X_test)
    calibrator = ScoreCalibrator(method=cal_method).fit(val_scores_raw, y_val)
    val_scores_cal = calibrator.transform(val_scores_raw)
    test_scores_cal = calibrator.transform(test_scores_raw)
    calibrated_metrics = {
        "method": cal_method,
        "val": compute_metrics(y_val, val_scores_cal),
        "test": compute_metrics(y_test, test_scores_cal),
    }

    thr_cfg = cfg.get("threshold", {"policy": "max_f1"})
    policy = ThresholdPolicy(thr_cfg.get("policy", "max_f1"))
    val_scores = handle.predict_proba(X_val)
    thr_info = select_threshold(
        y_val,
        val_scores,
        policy=policy,
        top_k_fraction=thr_cfg.get("top_k_fraction"),
        precision_floor=thr_cfg.get("precision_floor"),
        recall_floor=thr_cfg.get("recall_floor"),
    )
    thr_info["policy"] = policy.value

    run_name = f"{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(cfg.get("artifacts_dir", "artifacts")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    handle.save(out_dir / _model_artifact_name(model_type))
    joblib.dump(pre, out_dir / "preprocessor.pkl")
    calibrator.save(out_dir / "calibrator.pkl")
    _save_json(
        {
            "metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
            "calibrated": calibrated_metrics,
        },
        out_dir / "metrics.json",
    )
    _save_json(thr_info, out_dir / "threshold.json")
    _save_json(
        {
            "feature_columns": FEATURE_COLUMNS,
            "target_col": TARGET_COL,
            "preprocessor_output_dim": int(X_train.shape[1]),
            "model_type": model_type,
            "model_artifact": _model_artifact_name(model_type),
        },
        out_dir / "feature_schema.json",
    )
    _save_json(val_eval, out_dir / "val_eval.json")
    _save_json(test_eval, out_dir / "test_eval.json")
    _write_model_card(out_dir, cfg, val_eval["metrics"], test_eval["metrics"], thr_info)

    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(cfg.get("mlflow_experiment", "musicbox_churn"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model_type": model_type, "seed": seed, **cfg.get("params", {})})
        mlflow.log_metric("fit_seconds", fit_seconds)
        for k, v in val_eval["metrics"].items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_eval["metrics"].items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in calibrated_metrics["val"].items():
            mlflow.log_metric(f"val_cal_{k}", v)
        for k, v in calibrated_metrics["test"].items():
            mlflow.log_metric(f"test_cal_{k}", v)
        mlflow.log_metric("threshold", thr_info["threshold"])
        mlflow.log_param("calibration_method", cal_method)
        mlflow.log_artifacts(str(out_dir))

    logger.info("done: %s (val PR-AUC=%.4f)", out_dir, val_eval["metrics"]["pr_auc"])
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Train a churn model from a YAML config.")
    p.add_argument("--config", required=True, help="path to a configs/train_*.yaml file")
    args = p.parse_args()
    cfg = _load_config(args.config)
    out = train_from_config(cfg)
    print(out)


if __name__ == "__main__":
    main()
