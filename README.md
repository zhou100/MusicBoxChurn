# Music Box churn-risk scoring

Modernized churn-prediction pipeline on the Music Box dataset (~58k users,
30-day feature window, 14-day label window). The model emits a **ranking
score** used to prioritize a finite intervention budget, not a calibrated
probability for expected-value math. See [§ Honest framing](#honest-framing).

This repo is a rewrite of an older capstone notebook chain. The original
report and notebooks are preserved under [archive/](archive/); the modern
stack lives in [src/musicbox_churn/](src/musicbox_churn/). For why the
table looks the way it does, see [DATA.md](DATA.md). For results and
visuals, see [REPORT.md](REPORT.md).

---

## Results (test set)

| model | PR-AUC | ROC-AUC | Brier | precision@10pct | recall@10pct | lift@10pct |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.929 | 0.901 | 0.120 | 0.972 | 0.157 | 1.57 |
| Random forest       | 0.940 | 0.913 | 0.112 | 0.987 | 0.159 | 1.59 |
| LightGBM            | 0.939 | 0.911 | 0.113 | 0.989 | 0.159 | 1.59 |
| MLP (PyTorch)       | 0.935 | 0.909 | 0.115 | 0.974 | 0.157 | 1.57 |

Prevalence = 0.621 (churn is the majority class). All four cluster within
~1 PR-AUC point; **RF and GBM are statistically indistinguishable** and beat
LR / MLP by a hair. Pick the model whose latency and ops profile fits — there
is no clear accuracy winner.

The load-bearing metrics here are PR-AUC and recall@top-k%, not accuracy:
the product question is *"of the budget for retention outreach, what
fraction of actual churners do we catch?"* not *"what fraction of all
predictions are correct?"*.

---

## Quickstart

```bash
# Python 3.11+
make install-dev
make validate-data       # schema check on Processed_data/df_model_final.csv
make test                # 37 tests

make train-lr            # also: train-rf, train-gbm, train-mlp
make mlflow-ui           # browse runs at http://localhost:5000

# score a CSV with any trained run
make batch-score RUN_DIR=artifacts/rf_<ts> INPUT=Processed_data/df_model_final.csv
# → output/predictions_<ts>.{csv,parquet} + output/predictions/audience_top50.csv

make score-report RUN_DIR=artifacts/rf_<ts>     # → <run>/evaluation_report.md
make monitoring-report                           # → output/monitoring_report.md
```

---

## What this is, and what it isn't

This was originally a 2017 capstone: PySpark feature generation +
notebook-driven sklearn modeling, with the writeup framing accuracy
(~0.92) as the headline. The modernization keeps the same processed
table but rewrites everything around it as a small batch-scoring system.

| | Original capstone | This repo |
|---|---|---|
| Headline metric | Accuracy (~0.92) | PR-AUC, recall@top-k, lift |
| Framing | Binary classifier | Ranked-audience decision system |
| Output semantics | "Probability of churn" | Ranking score (uncalibrated) |
| Code shape | 6 sequential notebooks | `src/musicbox_churn/` package + CLIs |
| Reproducibility | Seed-free, runtime path baked in | Pinned seed, YAML configs, MLflow runs |
| Validation | Train/test holdout | 70/15/15 stratified, fixed seed=42 |
| Tracking | None | MLflow (local file store) |
| Drift / monitoring | None | Split-stability report (not real drift — see below) |
| Tests | None | 37 (schema, splits, metrics, preprocessing, training smoke, batch score, monitoring) |
| Inference | Notebook cell | `batch_score.py` CLI → CSV + parquet + audience_top50 |

What didn't change: the processed table itself, the label definition
(inactivity 2017-04-29 → 2017-05-12), the feature families, and the
honest constraint that we have a single snapshot and the raw logs are
gone. Everything that *can't* be done with one snapshot stays out of scope.

---

## Honest framing

A lot of churn-modeling repos overclaim. Specific things this one does not do:

1. **Outputs are ranking scores, not probabilities.** Tree ensembles and
   MLPs are typically uncalibrated; we report Brier and a 10-bin reliability
   table per run for diagnosis, but we don't apply Platt or isotonic
   calibration. If a downstream caller needs `P(churn)` for expected-value
   math, see the calibration entry in [TODOS.md](TODOS.md).

2. **The "monitoring" report is split-stability, not production drift.**
   With one snapshot, comparing train vs holdout measures sampling
   variance and stratification anomalies — that's it. Real drift requires
   a second snapshot taken after deployment. The report carries an
   explicit banner saying so.

3. **No real-time inference.** The 30-day rolling features need raw event
   logs to recompute at request time, and the raw logs are gone. This
   repo ships **batch scoring** only, which matches how lifecycle teams
   actually consume churn scores in practice.

4. **No claim about cost / expected revenue.** A ranked audience plus an
   intervention budget is enough to prioritize outreach; converting to
   dollars requires real LTV inputs and a calibrated probability. Both
   are out of scope.

5. **Time-based split is impossible here.** The processed table has no
   timestamp column — it's one snapshot. Stratified split is the only
   honest option; documented as a structural limitation in DATA.md.

---

## Repo layout

```
src/musicbox_churn/
  config.py                   # pydantic configs
  data/
    schema.py                 # SchemaError, FEATURE_COLUMNS, validate()
    load_data.py              # load_csv() + CLI for `make validate-data`
    splits.py                 # stratified_split() (70/15/15, seed=42)
  models/
    interface.py              # ModelHandle ABC (fit / predict_proba / save / load)
    baselines.py              # LR, RF, LightGBM via SklearnHandle
    tabular_mlp.py            # PyTorch MLP, auto cuda/cpu
  training/
    preprocessor.py           # ColumnTransformer (StandardScaler + OHE device_type)
    metrics.py                # PR-AUC, ROC-AUC, brier, precision/recall/lift @ top-k%
    thresholding.py           # max_f1, top_k, precision_floor, recall_floor policies
    calibration.py            # brier + reliability bins (no Platt/isotonic; see TODOS)
    evaluate.py               # full eval bundle
    train.py                  # YAML-driven CLI; writes artifacts + MLflow run
  inference/
    batch_score.py            # load run → score CSV → ranked CSV/parquet + audience_top50
  monitoring/
    score_report.py           # evaluation_report.md per run
    drift_report.py           # split-stability report (NOT production drift)
  utils/seed.py               # set_global_seed (np, random, torch)

configs/                      # train_lr|rf|gbm|mlp.yaml + inference.yaml
tests/                        # 37 tests
artifacts/<run>/              # model.{pkl,pt}, preprocessor.pkl, metrics.json,
                              # threshold.json, feature_schema.json, model_card.md,
                              # val_eval.json, test_eval.json, evaluation_report.md
mlruns/                       # MLflow file store
output/                       # batch-scoring outputs + monitoring_report.md
```

---

## Run artifacts

Each `make train-*` writes a self-contained directory under `artifacts/`:

- `model.{pkl,pt}` — fitted model
- `preprocessor.pkl` — fitted ColumnTransformer (must travel with the model)
- `metrics.json` — val + test metrics
- `threshold.json` — selected decision threshold + policy
- `feature_schema.json` — feature columns, target, model type, output dim
- `model_card.md` — human-readable summary with the framing caveat
- `val_eval.json` / `test_eval.json` — full per-split eval (metrics + reliability + score summary)
- `evaluation_report.md` — generated by `make score-report`

Same directory is what `batch_score.py` consumes — runs are portable.

---

## Locked decisions (so future-you doesn't relitigate them)

These were settled in the CEO + eng plan reviews:

- **Splits:** 70/15/15 stratified by label, seed=42. No device_type
  stratification (sample-size risk). Same seed everywhere.
- **GBM:** LightGBM, not sklearn HistGradientBoosting (faster, better defaults).
- **MLP:** [128, 64], dropout 0.2, BatchNorm, BCEWithLogitsLoss, **no
  pos_weight** — churn is the majority class, so a naive `pos_weight = neg/pos`
  would overweight the majority and hurt calibration further.
- **Device:** auto-detect CUDA, fall back to CPU. The dataset is small enough
  that CPU is competitive; GPU is "free" if present, not load-bearing.
- **MLflow:** local file store at `./mlruns`. No remote registry.
- **Linter:** ruff (lint + format). Black is dropped — redundant.
- **Calibration:** Brier + reliability only. No Platt or isotonic until a
  downstream caller actually needs probabilities.
- **Cost-curve language:** banned. The honest framing is "ranked-audience
  economics under assumed unit costs", not "expected revenue".
- **Drift report:** named "split-stability diagnostics" with an explicit
  banner — train-vs-test on one snapshot is sampling variance, not drift.

---

## Deferred work

See [TODOS.md](TODOS.md). Highlights: GitHub Actions CI (P2), probability
calibration (P3, only if a downstream caller actually needs it), time-based
split (P3, requires data we don't have).

The deferred infra layers (Dockerfile + k8s CronJob; Feast feature
definitions; FastAPI serving stub) live under `extras/` once added — a
hard "doc-only" partition to keep the core honest.

---

## Acknowledgements

Original capstone (data ingestion in PySpark, feature definitions, label
window, EDA): see [archive/CAPSTONE.md](archive/CAPSTONE.md). The processed table
[Processed_data/df_model_final.csv](Processed_data/df_model_final.csv) is
the output of that pipeline and the offline source of truth for everything
in this repo.
