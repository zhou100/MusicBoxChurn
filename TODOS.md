# TODOs

Items considered but deferred. Captured for future revisit.

## Completed

### Probability calibration (Platt or isotonic) — shipped
**What:** Fit a calibrator on the validation split's `(raw_score, label)` and apply it at scoring time so outputs are usable for expected-value math.

**How shipped:** `ScoreCalibrator` (isotonic by default, Platt option) in [src/musicbox_churn/training/calibration.py](src/musicbox_churn/training/calibration.py). `train.py` fits on val, saves `calibrator.pkl` per run. `batch_score.py` applies it and emits both `score` (raw ranking) and `prob_churn` (calibrated). Reliability and decision-curve figures in [REPORT.md](REPORT.md).

**Surprise finding:** these models are nearly calibrated out of the box — isotonic moves test Brier ≤0.2% per model. Atypical for tree ensembles, but the slot exists for future retrains that may be less well-behaved.

## P2

### CI: GitHub Actions
**What:** Wire `.github/workflows/ci.yml` to run `ruff check`, `ruff format --check`, `pytest`, and a 1-epoch training smoke on push and PR.

**Why:** Currently `make test` is the manual gate. CI prevents regressions sneaking in via merges and produces a visible green badge — strong "this person ships" signal.

**Pros:** Catches breakage early. Documents how the project is supposed to run. Required for any team setting.

**Cons:** None significant. Trivial to add.

## P3

### Time-based split when richer data is available
**What:** Replace stratified train/val/test split with a time-based split keyed on a snapshot date. Update [src/musicbox_churn/data/splits.py](src/musicbox_churn/data/splits.py) to support both modes via config.

**Why:** The current dataset has a single snapshot (2017-04-28) and no timestamp column. Stratified split is the only honest option, but it cannot detect temporal drift, label leakage from time-correlated features, or stability under deployment-time delay. A multi-snapshot dataset would unlock all of these.

**Pros:** Aligns with how production churn models are actually validated. Lets `monitoring/drift_report.py` measure real drift instead of split-stability.

**Cons:** Requires data we do not have. Cannot be done on the current `df_model_final.csv` alone.

**Context:** Documented in [DATA.md](DATA.md) §5 as a structural limitation of the public processed table. The original feature-generation notebook had timestamps, but the raw logs are gone so re-snapshotting is impossible.

**Depends on:** New data source with multi-snapshot user-level features, OR access to raw event logs to regenerate snapshots at multiple cutoffs.

### Bias slice analysis
**What:** Break down lift@k and calibration by `device_type` (iPhone vs Android/other) to surface whether the model systematically under-serves one device class.

**Why:** It's the only categorical feature we have, and the most realistic axis along which the model could be silently unfair. Slice metrics into per-device cohorts and compare.

**Pros:** Catches systematic bias before it becomes a product problem. Cheap — same eval, sliced.

**Cons:** Doesn't unlock new use cases; just guards existing ones.

### Pin sklearn version in pyproject.toml
**What:** Replace `scikit-learn>=1.4` with a tight pin (e.g., `scikit-learn==1.7.2`) to eliminate version skew between training host and the Docker image.

**Why:** The training host currently uses sklearn 1.7.2; the Docker image installs whatever the resolver picks (1.8.0 today). Predictions still work but the unpickling warning leaks into batch-score logs.

**Pros:** Bit-for-bit reproducibility. Cleaner logs.

**Cons:** Pin will bit-rot if not maintained.
