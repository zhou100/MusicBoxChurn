# CLAUDE.md — MusicBoxChurn

This file provides guidance for AI assistants (Claude and others) working on this codebase. The goal is consistent, project-aware behavior — not maximum verbosity.

---

## Project Overview

Churn-risk scoring pipeline for the Music Box dataset (~58k users). Goes from raw feature vectors → calibrated probabilities → a budget-aware decision rule that tells lifecycle marketing **who to contact and when it stops paying off**. Four interchangeable models (LR / RF / LightGBM / PyTorch MLP) sit behind a unified `ModelHandle` ABC; an isotonic calibrator turns ranking scores into meaningful `prob_churn`; batch scoring is the production interface (no real-time inference — raw event logs are gone).

The product intent that should never erode: **answer ranking + calibration + decision together**, not just rank users. The decision-curve analysis is the deliverable, not a side feature.

For full writeup see [REPORT.md](REPORT.md). For data provenance see [DATA.md](DATA.md). For deferred work see [TODOS.md](TODOS.md).

---

## Repository Structure

```
src/musicbox_churn/
  data/         schema, splits, load_csv
  models/       LR / RF / LightGBM / PyTorch MLP behind ModelHandle ABC
  training/     preprocessor, metrics, thresholding, calibration, train CLI
  inference/    batch_score CLI (raw score + calibrated prob_churn)
  monitoring/   score_report, drift_report (split-stability, not real drift)
  utils/        seeded RNG

configs/        train_{lr,rf,gbm,mlp}.yaml + inference.yaml
tests/          45 tests (end-to-end training smokes on synthetic fixtures)
k8s/            batch-score-cronjob.yaml
scripts/        report figure generation
extras/         doc-only stubs (Feast feature contract); core never imports from here
archive/        LEGACY — do not edit
Processed_data/ df_model_final.csv (the one input CSV)
artifacts/      per-run output dirs (model_<ts>/...) — generated, don't hand-edit
mlruns/         MLflow tracking store — generated
output/         batch-score outputs and monitoring reports — generated
report/         REPORT figures
```

---

## Tech Stack

- Python 3.11+
- scikit-learn 1.4+, LightGBM 4.3+, PyTorch 2.2+ (CPU)
- pandas 2.1+ / numpy 1.26+ / pyarrow 15+
- pydantic 2.6+ for schema validation
- MLflow 2.12+ (file-backed at `./mlruns`)
- ruff for lint + format (replaces black/isort/flake8)
- pytest
- Docker for the batch-score container; k8s CronJob for daily scoring

---

## Environment Setup

```bash
make install-dev          # editable install with dev extras
make validate-data        # schema check on Processed_data/df_model_final.csv
make test                 # 45 tests
```

No environment variables are required for local training or scoring.

---

## Workflow Philosophy

This repo treats AI-assisted development as a structured engineering workflow, not ad-hoc prompting.

### Before editing — classify the task

For anything non-trivial, produce a short plan naming: relevant files, system boundaries involved, what should NOT be touched, what to test afterward, and any risk to model behavior, calibration, or the run-directory contract.

Prefer small, reversible changes. If scope grows, stop and narrow rather than expand.

### Mode switching

- **Modeling mode** — new model, feature engineering, calibration. Respect the `ModelHandle` ABC. Keep raw score vs. calibrated probability distinct. Update tests in `tests/` when behavior changes. Re-run the relevant `make train-*` and confirm metrics didn't regress.
- **Pipeline / engineering mode** — preprocessor, splits, schema, CLIs. The 70/15/15 stratified split with `seed=42` is load-bearing for run comparability — don't change it casually. Keep the run directory contract stable (see Guardrails).
- **Reporting / docs mode** — REPORT.md, README.md, model cards. Numbers in docs are tied to specific runs; if you change a number, name the run it came from.
- **Deployment mode** — Dockerfile, k8s CronJob, Makefile container targets. Mirror the k8s mount layout in local docker runs. Don't introduce host-path assumptions.

### Context loading by task type

- **Model changes** — the relevant `models/<x>.py`, `training/train.py`, the YAML in `configs/`, and the matching test
- **Preprocessor / split / schema** — `data/`, `training/preprocessor.py`, plus every model test (preprocessor changes are cross-cutting)
- **Calibration / metrics / thresholding** — `training/{calibration,metrics,thresholding}.py` + `monitoring/score_report.py`
- **Batch scoring** — `inference/batch_score.py` + a sample run dir under `artifacts/` to confirm the contract
- **CI** — `.github/workflows/ci.yml`, `pyproject.toml` (ruff config), `Makefile`

### Review as a separate phase

Don't blend implementation and review. Implement the smallest viable patch, then review the diff against the original task, then run `make lint && make test` (and the relevant `make train-*` if behavior could shift), then summarize what changed and what's risky.

---

## Project-Specific Guardrails

- **Run directory contract.** Each `artifacts/<model>_<ts>/` must contain: `model.{pkl,pt}`, `preprocessor.pkl`, `calibrator.pkl`, `metrics.json`, `threshold.json`, `feature_schema.json`, `model_card.md`, `val_eval.json`, `test_eval.json`. `batch_score.py` consumes this layout. Don't add required files without updating `batch_score.py` and the relevant tests; don't rename existing ones.
- **Raw score vs. calibrated probability are distinct.** `score` = model ranking output, `prob_churn` = isotonic-calibrated probability. Both ship in batch-score output. Don't collapse them or swap their names.
- **Split is fixed.** 70/15/15 stratified, `seed=42`. Cross-run metric comparisons in REPORT.md depend on this. If you genuinely need to change it, update REPORT.md and re-run all four models in the same PR.
- **No temporal split.** The dataset is a single 2017-04-28 snapshot with no timestamp column; raw logs are gone (see DATA.md §5). Don't add a "time-based split" — it would be fake.
- **No real-time inference.** 30-day rolling features can't be recomputed at request time. Batch scoring is the only supported inference path. Don't add a serving endpoint without first solving the feature-recompute problem.
- **`extras/` is doc-only.** The Feast feature contract there is illustrative. Core code in `src/musicbox_churn/` must never import from `extras/`.
- **`archive/` is legacy.** Don't edit, don't import, don't reference in new code.
- **Don't remove the calibrator** even though models are nearly calibrated out of the box. It's part of the "no trust-us framing" contract for `prob_churn`.
- **Unit economics ($1/contact, $10/save, intervention-success scan) are sensitivity inputs, not measurements.** Don't promote them to "the answer" in code or docs. The decision curve is the deliverable.
- **MLflow logging mirrors the run directory.** If you add a metric or artifact to one, add it to the other. Don't let them drift.

---

## Common Commands

```bash
# Setup + checks
make install-dev
make validate-data
make test
make lint                 # ruff check + ruff format --check
make format               # ruff auto-fix

# Train (writes artifacts/<model>_<ts>/, logs to MLflow)
make train-lr
make train-rf
make train-gbm
make train-mlp
make mlflow-ui            # http://localhost:5000

# Score
make batch-score RUN_DIR=$(ls -td artifacts/gbm_* | head -1) \
                 INPUT=Processed_data/df_model_final.csv

# Reports
make score-report RUN_DIR=...
make monitoring-report
make report-figures

# Container
make docker-build
make docker-batch-score RUN_DIR=$(pwd)/artifacts/rf_<ts> \
                        INPUT=$(pwd)/Processed_data/df_model_final.csv
```

---

## Conventions

- **Configs are YAML in `configs/`,** one per model, consumed by `training/train.py --config`. New hyperparameters go in the YAML, not as CLI flags.
- **All randomness flows through `utils/` seeded RNG.** Don't call `np.random.seed` or `torch.manual_seed` directly in new code.
- **Schema validation via pydantic** at load time (`data/load_data.py`). Schema changes require updating `feature_schema.json` expectations and the corresponding tests.
- **Model implementations live behind `ModelHandle`** (`models/`). New models implement the ABC; `train.py` should not need a special case.
- **Tests mirror source layout under `tests/`.** End-to-end training smokes use synthetic fixtures, not the real CSV — keep it that way so CI stays ~2 min.
- **ruff config in `pyproject.toml`** is the single source of truth for lint + format. Don't add black/isort/flake8 back.

---

## Common Pitfalls

- **Forgetting to log to both MLflow and the run directory.** A new metric in `metrics.json` that's missing from MLflow (or vice versa) silently breaks cross-run comparison.
- **Calibrating on test instead of val.** The isotonic calibrator is fit on the val split; test is held out for honest reporting. Easy to swap by accident when refactoring.
- **Treating `score` as a probability.** It isn't until it goes through the calibrator. Anything user-facing should use `prob_churn`.
- **Editing `archive/`.** It's legacy; changes there have no effect on the active pipeline and confuse future readers.
