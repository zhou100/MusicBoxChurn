# TODOs

Items considered but deferred. Captured for future revisit.

## Completed

### Bias slice analysis — shipped
**What:** Per-`device_type` PR-AUC / ROC-AUC / Brier / lift@10pct surfaced in every run's `val_eval.json` and `test_eval.json` and rendered as a table in `evaluation_report.md`.

**How shipped:** `slice_metrics()` in [src/musicbox_churn/training/metrics.py](src/musicbox_churn/training/metrics.py) (skips groups below `min_group_size=50` or with degenerate single-class labels). `evaluate()` takes an optional `slices` dict; `train.py` passes `device_type` from the unprocessed feature frames. `score_report.py` renders one table per slice dimension.

**Surprise finding:** the LR model is materially weaker on the smaller iPhone-ish cohort (`device_type=1`, n=1,114, PR-AUC 0.852) than on the larger Android cohort (`device_type=2`, n=7,556, PR-AUC 0.935). Same direction across models likely; worth a follow-up calibration check.

### Pin scikit-learn — shipped
**What:** [pyproject.toml](pyproject.toml) pins `scikit-learn==1.7.2` (was `>=1.4`). Eliminates the unpickling warning that surfaced when the Docker image resolved a newer minor than the training host.

### CI: GitHub Actions — shipped
**What:** [.github/workflows/ci.yml](.github/workflows/ci.yml) runs `ruff check`, `ruff format --check`, and `pytest` on every push to `master` and on every PR. The pytest job doubles as the training-smoke gate because `tests/test_training_smoke.py` fits LR / GBM / MLP end-to-end on synthetic fixtures. Two parallel jobs, ~3–5 min wall clock with the CPU torch wheel.

### Probability calibration (Platt or isotonic) — shipped
**What:** Fit a calibrator on the validation split's `(raw_score, label)` and apply it at scoring time so outputs are usable for expected-value math.

**How shipped:** `ScoreCalibrator` (isotonic by default, Platt option) in [src/musicbox_churn/training/calibration.py](src/musicbox_churn/training/calibration.py). `train.py` fits on val, saves `calibrator.pkl` per run. `batch_score.py` applies it and emits both `score` (raw ranking) and `prob_churn` (calibrated). Reliability and decision-curve figures in [REPORT.md](REPORT.md).

**Surprise finding:** these models are nearly calibrated out of the box — isotonic moves test Brier ≤0.2% per model. Atypical for tree ensembles, but the slot exists for future retrains that may be less well-behaved.

## P3

### Time-based split when richer data is available
**What:** Replace stratified train/val/test split with a time-based split keyed on a snapshot date. Update [src/musicbox_churn/data/splits.py](src/musicbox_churn/data/splits.py) to support both modes via config.

**Why:** The current dataset has a single snapshot (2017-04-28) and no timestamp column. Stratified split is the only honest option, but it cannot detect temporal drift, label leakage from time-correlated features, or stability under deployment-time delay. A multi-snapshot dataset would unlock all of these.

**Pros:** Aligns with how production churn models are actually validated. Lets `monitoring/drift_report.py` measure real drift instead of split-stability.

**Cons:** Requires data we do not have. Cannot be done on the current `df_model_final.csv` alone.

**Context:** Documented in [DATA.md](DATA.md) §5 as a structural limitation of the public processed table. The original feature-generation notebook had timestamps, but the raw logs are gone so re-snapshotting is impossible.

**Depends on:** New data source with multi-snapshot user-level features, OR access to raw event logs to regenerate snapshots at multiple cutoffs.

