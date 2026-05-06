# TODOs

Items considered during planning but deferred. Captured for future revisit.

## P2

### CI: GitHub Actions
**What:** Wire `.github/workflows/ci.yml` running `ruff check`, `ruff format --check`, `pytest`, and a 1-epoch training smoke on push and PR.

**Why:** Currently `make test` is the manual gate. CI prevents regressions sneaking in via merges and produces a visible green badge for the README — strong "this person ships" signal.

**Pros:** Catches breakage early. Documents how the project is supposed to run. Required for any team setting.

**Cons:** Trivial to add; the only reason it was declined was scope discipline during the initial modernization PR.

**Context:** During the 2026-05-04 plan review, GitHub Actions CI was declined to keep the first modernization PR focused. Add after the initial PR lands so the README badge demos the green CI state.

**Depends on:** Initial modernization PR merged.

## P3

### Probability calibration (Platt or isotonic)
**What:** Add `sklearn.calibration.CalibratedClassifierCV` (or equivalent isotonic step) on the validation split for GBM and MLP. Save the calibrated model alongside the raw model. Update the README to drop the "ranking score" framing and resume probability language.

**Why:** Currently outputs are framed as ranking scores, not probabilities. If downstream consumers (cost calculations with real LTV, A/B sims, expected-revenue framing) require true `P(churn)`, calibration is required — Brier + reliability plots diagnose miscalibration but do not fix it.

**Pros:** Re-enables honest probability semantics. Makes the cost curve a real expected-value tool, not a sensitivity simulation.

**Cons:** Adds another fitted artifact per run. Eats some of the "slim core" honesty the current plan locked in. Should not be added speculatively — only if a real downstream caller needs probabilities.

**Context:** Codex outside-voice flagged during 2026-05-04 plan review that GBM and MLP outputs are typically uncalibrated. Decision was to consistently call outputs "ranking scores" and rename `cost_curve` to `audience-budget sensitivity`. This TODO is the unwind path if probability semantics are ever required.

**Depends on:** Concrete downstream requirement for `P(churn)`. Do not add speculatively.

### Time-based split when richer data is available
**What:** Replace stratified train/val/test split with a time-based split keyed on a snapshot date. Update `data/splits.py` to support both modes via config.

**Why:** The current dataset has a single snapshot (2017-04-28) and no timestamp column. Stratified split is the only honest option, but it cannot detect temporal drift, label leakage from time-correlated features, or stability under deployment-time delay. A multi-snapshot dataset would unlock all of these.

**Pros:** Aligns with how production churn models are actually validated. Lets `monitoring/drift_report.py` measure real drift instead of split-stability.

**Cons:** Requires data we do not have. Cannot be done on the current `df_model_final.csv` alone.

**Context:** Documented in DATA.md §5 as a structural limitation of the public processed table. The original PySpark notebook (`4_feature_label_generation_with_spark.ipynb`) had timestamps but the raw logs are gone, so re-snapshotting is impossible.

**Depends on:** New data source with multi-snapshot user-level features, OR access to raw event logs to regenerate snapshots at multiple cutoffs.
