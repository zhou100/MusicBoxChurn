# Feast feature repository — doc-only stub

> **This is documentation in YAML form, not a working feature store.**
> The core training and batch-scoring pipeline does not depend on Feast.
> It uses [`../../Processed_data/df_model_final.csv`](../../Processed_data/df_model_final.csv)
> directly via [`musicbox_churn.data.load_data`](../../src/musicbox_churn/data/load_data.py).

## What this is

A Feast 0.40+ feature_repo that declares — *as a contract* — the entity,
features, and offline source the modernized pipeline reads from. Anyone
inspecting the repo can answer:

- What is the entity grain? (`uid`, int64)
- What features exist, what are their dtypes, what windows do they cover?
- Where is the offline source, and what column means what?

If you have Feast installed (`pip install feast`), you can run:

```bash
cd extras/feast
feast apply             # registers entity + feature view
feast feature-views list
feast describe feature-view user_churn_features
```

…and Feast will validate the schema. Nothing about the core pipeline
changes either way.

## What this is NOT

- **Not an online store.** No Redis / DynamoDB push, no `materialize`
  loop, no online retrieval. The whole point of an online store is
  recomputing rolling features at request time — which requires raw event
  logs we no longer have ([../../DATA.md](../../DATA.md) §1).
- **Not a training-time integration.** [`musicbox_churn.training.train`](../../src/musicbox_churn/training/train.py)
  reads the CSV directly. Wiring it through Feast would add a registry
  dependency without changing what features get fit on, and would imply
  a larger feature platform than we actually run. See
  [../../README.md](../../README.md) § Honest framing.
- **Not a substitute for [`musicbox_churn.data.schema`](../../src/musicbox_churn/data/schema.py).**
  That module is the runtime source of truth — it raises `SchemaError`
  on bad inputs. The Feast definitions here are the human-facing contract.
  They are kept in sync by hand (and by [test_schema.py](../../tests/test_schema.py)
  asserting feature-count invariants).

## Why ship this at all

Two reasons:

1. **It documents the contract** that any future production deploy would
   need to honor — the entity grain, the feature names, the offline source
   path. A Feast file is a more precise contract than English prose.
2. **It marks the unwind path.** If a multi-snapshot data source becomes
   available (see [`../../TODOS.md`](../../TODOS.md), time-based-split
   entry), turning this stub into a real feature store with a `materialize`
   loop is the natural next step.

## Files

| File | What |
|---|---|
| [`feature_store.yaml`](feature_store.yaml) | Feast registry config — local provider, file-based registry, no online store |
| [`feature_definitions.py`](feature_definitions.py) | Entity + FeatureView + FileSource declaration mirroring `musicbox_churn.data.schema` |
