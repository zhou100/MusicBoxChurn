# Music Box churn-risk scoring — results report

A modernized churn-prediction system on the Music Box dataset (~58k users,
single 2017-04-28 snapshot). Built as a productionized batch-scoring
service rather than a notebook chain. This document is the headline
result; for code see [src/musicbox_churn/](src/musicbox_churn/), for
provenance see [DATA.md](DATA.md), for the original capstone see
[archive/](archive/).

---

## TL;DR

> **Reaching the top-decile audience captures 16% of all churners; the
> top-half captures 74% — a 1.5× lift over random outreach.** Random
> Forest, LightGBM, MLP and Logistic Regression are all within ~1
> PR-AUC point of each other (0.929–0.940). The product question is
> not "which model wins", it's **how much retention budget to spend**.

| Decile contacted | Churners caught | Lift over random |
|---|---:|---:|
| top 5%  | 8.0%   | 1.6× |
| top 10% | 15.9%  | 1.6× |
| top 20% | 31.5%  | 1.6× |
| top 30% | 46.6%  | 1.6× |
| top 50% | 73.9%  | 1.5× |

*(LightGBM on a 8,692-user test set; prevalence = 0.621)*

---

## 1 · Motivation

Music Box is a music-streaming app. Roughly **62% of users go inactive
in any 14-day window** — typical for free-tier consumer apps. Retention
teams have a finite budget for outreach (push notifications, discount
nudges, re-engagement emails) and have to pick **who to spend it on**.

A churn-prediction model is only useful if it can answer the operational
question:

> *"If we have budget to contact N% of users, what fraction of
> actual churners do we catch?"*

That's a **ranking problem**, not a classification problem. Accuracy is
the wrong metric — at 62% prevalence, predicting "everyone churns" gets
62% accuracy and zero usefulness. The right metrics are **PR-AUC**,
**lift@top-k**, and the **cumulative-gains curve** below.

---

## 2 · Data

| | |
|---|---|
| Source | Music Box event logs, 2017-03-30 → 2017-05-12 (3 streams: play / download / search) |
| Modeling table | 57,943 users × 45 numeric features + `device_type` + `label` |
| Feature window | 30 days (2017-03-30 → 2017-04-28) |
| Label window | 14 days forward (2017-04-29 → 2017-05-12) |
| Label rule | Inactive throughout label window → churn (1) |
| Class balance | ≈ 60% churn / 40% active |

Features fall in five families, each computed at multiple windows
(1, 3, 7, 14, 30 days):

- **Frequency** — counts of plays / downloads / searches
- **Recency** — days since last event of each type
- **Total playtime** — seconds of audio played
- **Songs fully played** — events where ≥ 80% of song was completed
- **Play-completion %** — mean and stddev of `play_time / song_length`

Plus a single profile feature: `device_type` (iPhone vs Android/other).

> **Honest note on the data.** We have a single snapshot — no timestamp
> column on rows, raw event logs no longer available. This rules out
> time-based splits and real production-drift monitoring. See
> [DATA.md](DATA.md) §5 and [TODOS.md](TODOS.md) for the full constraint
> list and the unwind path if a multi-snapshot data source becomes
> available.

---

## 3 · Method

### 3.1 Pipeline

```
Processed_data/df_model_final.csv
   │
   ▼
[ load_csv → schema validation → dedupe → stratified 70/15/15 split (seed=42) ]
   │
   ▼
[ ColumnTransformer: StandardScaler(numeric) + OneHotEncoder(device_type) ]
   │
   ├──▶ Logistic Regression (sklearn)
   ├──▶ Random Forest        (sklearn, 300 trees)
   ├──▶ LightGBM             (500 rounds, lr=0.05)
   └──▶ MLP                  (PyTorch, [128, 64], BatchNorm, dropout=0.2)
            │
            ▼
[ predict_proba on val + test ]
   │
   ▼
[ metrics.json · threshold.json · feature_schema.json · model_card.md ]
   │
   ├──▶ MLflow run (params, metrics, artifacts)
   ├──▶ batch_score CLI → ranked CSV/parquet + audience_top50.csv
   └──▶ score_report + drift_report (split-stability, NOT real drift)
```

### 3.2 Locked design decisions

These were settled in a CEO + engineering plan review and codified in the
README so future-me doesn't relitigate them:

| Decision | Choice | Rationale |
|---|---|---|
| Splits | 70/15/15 stratified by label, seed=42 | Single snapshot ⇒ time split impossible (DATA.md §5) |
| GBM library | LightGBM (not sklearn HGBT) | Faster; better defaults for tabular |
| MLP `pos_weight` | **omitted** | Churn is the *majority* class; naive `neg/pos` would overweight majority and hurt calibration |
| Compute | Auto-detect CUDA → CPU | Dataset is small enough that CPU is competitive; GPU is "free if present" |
| Calibration | Brier + reliability only | No Platt/isotonic until a downstream caller actually needs `P(churn)` |
| Tracking | MLflow file store | No remote registry; runs portable as `artifacts/<run>/` directories |
| Linter | ruff (lint + format) | Drop black — redundant |

### 3.3 What's *not* in this pipeline (deliberately)

- **No real-time inference.** The 30-day rolling features need raw event
  logs to recompute on demand; raw logs are gone. We ship batch scoring,
  which is what lifecycle teams actually consume.
- **No probability calibration.** All four model families are
  uncalibrated rankers. We *measure* calibration (Brier + reliability)
  but don't *fix* it.
- **No expected-revenue / cost-curve framing.** That requires real LTV
  inputs and calibrated probabilities. Out of scope; would be misleading.
- **No drift monitoring in the production sense.** With one snapshot, the
  monitoring report measures sampling variance across splits — not shift
  over time. The report banner makes this explicit.

---

## 4 · Results

### 4.1 The headline question — how much budget catches how much churn?

![Cumulative-gains curve](report/figures/lift_curve.png)

All four models trace nearly the same curve. The operational shape:

- **Top decile** of risk-scored audience contains **16% of all churners**
- **Top half** contains **74% of all churners**
- Curve flattens past the 70-80% mark — diminishing returns on outreach

### 4.2 Score distribution — clean bimodal separation

![Score histogram by true label](report/figures/score_histogram.png)

Churned users (orange) pile up near 1.0; active users (blue) pile up near
0. The overlap in the middle (~0.4–0.7 range) is the population where
score-based ranking matters most — these are the borderline users where
the model adds the most value over random selection.

### 4.3 Model comparison — four-way tie

![Model comparison](report/figures/model_comparison.png)

| Metric | Random Forest | LightGBM | MLP | Logistic Reg. |
|---|---:|---:|---:|---:|
| **PR-AUC** | **0.940** | 0.939 | 0.935 | 0.929 |
| ROC-AUC | 0.913 | 0.911 | 0.909 | 0.901 |
| lift@10pct | 1.59 | 1.59 | 1.57 | 1.57 |
| Brier (lower = better) | **0.112** | 0.113 | 0.115 | 0.120 |

RF and GBM are statistically indistinguishable. **The model choice should
be driven by ops profile (training time, memory, deployment surface),
not accuracy** — there is no clear accuracy winner.

### 4.4 Precision–Recall curves

![Precision–recall curves](report/figures/pr_curves.png)

The curves overlap so tightly that the four lines are barely
distinguishable. PR-AUC area difference between best (RF) and worst (LR)
is 0.011 — well within "no clear winner" territory.

### 4.5 Reliability — surprisingly well-calibrated out of the box

![Reliability diagrams](report/figures/reliability.png)

All four models track close to the diagonal across the full 0–1 range,
with no Platt or isotonic calibration applied. This is unusually clean
for tree ensembles — most production GBMs need calibration before their
probabilities mean anything.

**Caveat:** the README still calls these "ranking scores", not probabilities.
This reliability diagram is a diagnostic, not a license. If a downstream
consumer wants to multiply by LTV, calibrate first
([TODOS.md](TODOS.md) P3).

### 4.6 What drives the scores — feature importance

![Feature importance](report/figures/feature_importance.png)

Both tree models agree on the top families:

- **Recency wins.** `recent_P_last_30` (days since last play in 30-day
  window) tops the RF chart — the original 2017 capstone reached the same
  conclusion.
- **Recent frequency matters.** `freq_P_last_30/14` are top-3 across
  both models.
- **Engagement-quality features** (`mean_percent_last_*`,
  `tot_playtime_last_*`) cluster in the top 10 — *how* a user listens is
  almost as informative as *whether* they listen.

The two libraries score importance differently (RF: Gini-impurity decrease;
LightGBM: split gain), but the ranking agrees on what matters.

---

## 5 · What's shipped

| Component | Status |
|---|---|
| Python 3.11 package + pyproject.toml | ✓ |
| Schema validation, stratified splits, deterministic seed | ✓ |
| 4 model families with unified `ModelHandle` interface | ✓ |
| YAML-driven training CLI | ✓ |
| MLflow experiment tracking (local file store) | ✓ |
| Batch scoring CLI (CSV + parquet + audience_top50) | ✓ |
| Score report + split-stability report (markdown + JSON) | ✓ |
| Dockerfile (CPU runtime, libgomp, tini) | ✓ |
| k8s CronJob manifest for daily scoring | ✓ |
| 37 tests (schema, splits, metrics, preprocessing, smoke, batch, monitoring) | ✓ |

---

## 6 · Future steps

Roughly ordered by ROI given the current dataset and constraints:

1. **CI** ([TODOS.md](TODOS.md) P2) — wire GitHub Actions to run ruff +
   pytest + a 1-epoch training smoke. Trivial; was scoped out of the
   modernization PR for focus.
2. **Probability calibration** (P3) — add `CalibratedClassifierCV` (Platt
   or isotonic) on the validation split if a downstream consumer requires
   true `P(churn)` for cost / expected-value math.
3. **Pin sklearn version** — the Dockerfile currently uses whatever the
   pip resolver picks; minor-version skew between training and serving
   produces unpickling warnings. Add an explicit pin.
4. **Bias slice analysis** — break down lift@k by `device_type` to verify
   we're not systematically under-serving one device class.
5. **Time-based split** (P3) — requires a multi-snapshot dataset or
   access to raw logs. Unlocks real drift monitoring and honest
   evaluation of feature freshness.
6. **Feast feature definitions** (P7, planned) — declarative feature spec
   in `extras/feast/` for documentation + future training/inference
   consistency. Doc-only stub; no online materialization.
7. **Slim the inference image** — strip mlflow + training-only deps from
   the batch-score image to drop from 1.74 GB to <500 MB.

---

## 7 · How to reproduce

```bash
make install-dev
make validate-data            # schema check on Processed_data/df_model_final.csv

make train-lr train-rf train-gbm train-mlp
make mlflow-ui                 # browse runs at http://localhost:5000

PYTHONPATH=src python3 scripts/generate_report_figures.py
# regenerates the six PNGs in report/figures/

make batch-score RUN_DIR=$(ls -td artifacts/rf_* | head -1) \
                 INPUT=Processed_data/df_model_final.csv
# → output/predictions_<ts>.{csv,parquet} + output/predictions/audience_top50.csv
```

---

## Appendix · Run artifacts

Each `make train-*` writes a self-contained, portable directory:

```
artifacts/<model>_<utc-timestamp>/
├── model.{pkl,pt}           # fitted estimator (sklearn pickle or torch state_dict)
├── preprocessor.pkl         # fitted ColumnTransformer (must travel with the model)
├── feature_schema.json      # feature columns, target, model type, output dim
├── metrics.json             # val + test metrics
├── val_eval.json            # full per-split eval (metrics + reliability + score summary)
├── test_eval.json           # ditto for test set
├── threshold.json           # selected decision threshold + policy
├── model_card.md            # human-readable summary with framing caveat
└── evaluation_report.md     # generated by `make score-report`
```

The same directory is what `batch_score.py` consumes. Runs are
self-contained and portable — copy the dir, score anywhere.
