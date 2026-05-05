# Archive — original capstone

This folder preserves the original 2017 Music Box capstone for provenance:

- [CAPSTONE.md](CAPSTONE.md) — original capstone report
- `0_…6_…(.sh|.ipynb)` — the original ingest → ETL → feature → model →
  recommender notebook chain (PySpark + sklearn). The shell scripts and
  notebooks 1–3 cannot be re-run because the raw event logs are no longer
  available; see [../DATA.md](../DATA.md) §1.
- `MusicBoxChurn.Rproj` — leftover RStudio project file
- `img/` — figures referenced from CAPSTONE.md

The current production-style stack lives in [`../src/musicbox_churn/`](../src/musicbox_churn/)
and uses [`../Processed_data/df_model_final.csv`](../Processed_data/df_model_final.csv)
as the offline source of truth. Nothing in this directory is on the runtime path.
