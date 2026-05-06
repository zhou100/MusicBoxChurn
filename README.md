# Music Box churn-risk scoring

[![CI](https://github.com/zhou100/MusicBoxChurn/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/zhou100/MusicBoxChurn/actions/workflows/ci.yml)

Churn-risk scoring on the Music Box dataset (~58k users) that goes from
raw feature vectors → calibrated probabilities → a budget-aware decision
rule that tells lifecycle marketing **who to contact and when it stops
paying off**.

For the full writeup with figures and the dollar-decision table, see
[REPORT.md](REPORT.md). For data provenance, see [DATA.md](DATA.md).
For deferred work, see [TODOS.md](TODOS.md).

---

## Why

Music Box loses ~62% of its users to inactivity in any 14-day window.
Retention outreach is finite, so marketing has to answer three questions
together:

1. **Who** is most likely to churn? (ranking)
2. **What's the probability** each user churns? (calibration)
3. **Who do we actually contact, and how big should the audience be**,
   given a unit cost per contact and a unit value per save? (decision)

A model that only answers #1 produces a leaderboard. This pipeline
answers all three.

---

## How

```
Processed_data/df_model_final.csv  (57,943 users × 45 features + label)
   │
   ├─→ Schema validation, dedupe, stratified 70/15/15 split (seed=42)
   │
   ├─→ Preprocessor: StandardScaler(numeric) + OneHotEncoder(device_type)
   │
   ├─→ Train: LR / RF / LightGBM / PyTorch MLP behind a unified ModelHandle
   │           ↓ raw ranking score
   │
   ├─→ Calibrate: IsotonicRegression on val (raw_score, label)
   │           ↓ calibrated P(churn)
   │
   └─→ batch_score.py → ranked CSV/parquet with both `score` and `prob_churn`
                       + audience_top50.csv export
```

Test-set performance — all four cluster within ~1 PR-AUC point:

| Model | PR-AUC | ROC-AUC | Brier | lift@10pct |
|---|---:|---:|---:|---:|
| Random Forest | **0.940** | 0.913 | **0.112** | 1.59 |
| LightGBM      | 0.939 | 0.911 | 0.113 | 1.59 |
| MLP (PyTorch) | 0.935 | 0.909 | 0.115 | 1.57 |
| Logistic Reg. | 0.929 | 0.901 | 0.120 | 1.57 |

Pick by ops profile (training time, memory, deployment surface), not
accuracy. Reliability check showed these models are nearly calibrated
out of the box; the calibrator still ships so `prob_churn` is meaningful
without "trust us" framing.

---

## So what

At the assumed unit economics ($1/contact, $10/saved user, 20% intervention
success), the model picks an optimal cutoff at the **top ~64% of risk-scored
users** and generates **$4,052 expected net retention value** on the
8,692-user test set — **+92% over untargeted outreach** to the same
audience size.

| Intervention success | Optimal audience % | Net (model) | Net (random) | Lift |
|---:|---:|---:|---:|---:|
| 10% | 0% (don't run) | $0 | $0 | — |
| 20% | 64% | **$4,052** | $2,106 | +92% |
| 35% | 72% | $11,615 | $10,204 | +14% |

The model isn't just predicting churn — it tells marketing **what
intervention success rate they need to clear** and **what audience size
maximizes net value at any given rate**. Below ~10% success, no outreach
pays off and the system says so. Full decision-curve analysis in
[REPORT.md § 3.2](REPORT.md#32-the-decision-question--net-value-vs-audience-size).

---

## Quickstart

```bash
# Python 3.11+
make install-dev
make validate-data       # schema check on Processed_data/df_model_final.csv
make test                # 45 tests

make train-lr            # also: train-rf, train-gbm, train-mlp
                         # → artifacts/<model>_<ts>/, logged to MLflow
make mlflow-ui           # browse runs at http://localhost:5000

# score a CSV with any trained run
make batch-score RUN_DIR=$(ls -td artifacts/gbm_* | head -1) \
                 INPUT=Processed_data/df_model_final.csv
# → output/predictions_<ts>.csv with `score` + `prob_churn` columns
#   plus output/predictions/audience_top50.csv

make score-report RUN_DIR=...    # → <run>/evaluation_report.md
make monitoring-report           # → output/monitoring_report.md
```

### What a training run produces

Each `make train-*` invocation:

1. Loads + validates the CSV, splits 70/15/15 (seed=42), fits the
   preprocessor and the model on train, evaluates on val and test, fits
   an isotonic calibrator on val.
2. Writes a self-contained run directory:
   `artifacts/<model>_<ts>/{model.pkl,preprocessor.pkl,calibrator.pkl,metrics.json,threshold.json,feature_schema.json,model_card.md,val_eval.json,test_eval.json}`.
3. Logs the same data — params, metrics (raw + calibrated), and every
   artifact — to MLflow at `./mlruns`.

Comparing across runs is one query (or one `make mlflow-ui`):

| run | model | val PR-AUC | test PR-AUC | test Brier | fit (s) |
|---|---|---:|---:|---:|---:|
| `rf_20260506T190027Z`  | rf  | **0.934** | **0.940** | **0.112** | 0.91 |
| `gbm_20260506T203818Z` | gbm | 0.934 | 0.939 | 0.114 | 43.97 |
| `mlp_20260506T190042Z` | mlp | 0.928 | 0.935 | 0.115 | 6.56 |
| `lr_20260506T190022Z`  | lr  | 0.918 | 0.929 | 0.120 | 1.47 |

Worked example of an MLflow-backed sweep in [REPORT.md § 2.6](REPORT.md#26-how-we-tracked--compared-runs--mlflow).

Each `make train-*` writes a self-contained, portable run directory
under `artifacts/<model>_<ts>/`: `model.{pkl,pt}`, `preprocessor.pkl`,
`calibrator.pkl`, `metrics.json`, `threshold.json`, `feature_schema.json`,
`model_card.md`, plus per-split eval JSONs. The same directory is what
`batch_score.py` consumes — copy it, score anywhere.

---

## Honest framing

1. **Unit economics are assumed, not measured.** $1/contact and $10/save
   are sensitivity-analysis inputs. Real LTV and channel cost will move
   the dollar numbers; the *shape* of the decision curves doesn't change.
2. **Intervention success is a model parameter, not data.** No A/B
   holdout exists. The decision curve scans 5–40% so you can read the
   answer at whatever rate your team believes.
3. **Calibration is on a single 2017-04-28 snapshot.** Production use
   requires periodic re-fit on fresh data. There is no temporal split
   because the dataset has no timestamp column and raw event logs are
   gone (see [DATA.md](DATA.md) §5).
4. **The "monitoring" report is split-stability, not real drift.** With
   one snapshot, train-vs-holdout PSI measures sampling variance. Real
   drift requires multiple snapshots.
5. **No real-time inference.** 30-day rolling features need raw logs to
   recompute at request time; logs are gone. Batch scoring matches how
   lifecycle teams consume churn scores in practice.

---

## Repo layout

```
src/musicbox_churn/
  data/      schema, splits, load_csv
  models/    LR / RF / LightGBM / PyTorch MLP behind ModelHandle ABC
  training/  preprocessor, metrics, thresholding, calibration, train CLI
  inference/ batch_score CLI (raw score + calibrated prob_churn)
  monitoring/ score_report (evaluation_report.md), drift_report (split-stability)
  utils/     seeded RNG

configs/     train_lr|rf|gbm|mlp.yaml + inference.yaml
tests/       45 tests
k8s/         batch-score-cronjob.yaml (daily scoring)
extras/      doc-only stubs (Feast feature contract); core never imports from here
```

Container build: `make docker-build` produces a CPU image that scores
57k users in ~1s; `make docker-batch-score` mirrors the k8s mount layout.

---

## Engineering hygiene

[![CI](https://github.com/zhou100/MusicBoxChurn/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/zhou100/MusicBoxChurn/actions/workflows/ci.yml)

**CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)): every push
to master and every PR runs two parallel jobs — `ruff check` + `ruff format --check`
(~10 s) and `pytest -q` on Python 3.11 (~2 min). The pytest job runs all
45 tests including end-to-end LR/GBM/MLP training smokes on synthetic
fixtures, so it catches both surface regressions and behavioral ones.
Green badge = "this still works on a clean machine."

**ruff** replaces the typical 3-tool Python stack (black + isort + flake8)
with one binary: ~10× faster, one config block in [pyproject.toml](pyproject.toml),
auto-fixes formatting and most lint violations. `make format` for the
fix loop, `make lint` for the check loop.

What's deliberately not in CI: full-data integration test (uses
synthetic fixtures), Docker build verification, benchmark regressions,
security scanning. Reasonable next-rung items, not load-bearing today.
Full breakdown including how to interpret a red badge in
[REPORT.md § 6](REPORT.md#6--engineering-hygiene--ci-and-ruff).
