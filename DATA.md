# Data Provenance and Feature Definitions

This document explains how the modeling table [Processed_data/df_model_final.csv](Processed_data/df_model_final.csv) was produced from raw Music Box event logs. The pipeline in this repo treats this table as the **offline source of truth** and does not re-run the upstream raw-log processing. This file documents what the features mean, what label leakage looks like in this domain, and what limitations follow from only having the processed table.

## 1. Source data (not in this repo)

The original project ingested three streams of user-level event logs from the Music Box app:

| Stream | Fields | Meaning |
|---|---|---|
| `play` | `uid, device, song_id, song_type, song_name, singer, play_time, song_length, paid_flag, date` | a user played a song (`play_time` = seconds played; `song_length` = full song length) |
| `down` | `uid, device, song_id, song_name, singer, paid_flag, date` | a user downloaded a song |
| `search` | `uid, device, time_stamp, search_query, date` | a user issued a search |

Logs were daily-rotated tarballs (`YYYYMMDD_*.log.tar.gz`). The shell scripts [archive/0_create_data_folders.sh](archive/0_create_data_folders.sh) and [archive/2_unpack_and_clean_files.sh](archive/2_unpack_and_clean_files.sh) untar them, append the date (parsed from the filename) to each row, and concatenate to `all_play_log` / `all_down_log` / `all_search_log`.

**The raw logs are no longer available** — they are not in this repo and the original copies are gone. The original ingestion notebooks ([archive/1_download_data.ipynb](archive/1_download_data.ipynb), [archive/2_unpack_and_clean_files.sh](archive/2_unpack_and_clean_files.sh), [archive/3_etl_down_sample_by_user.ipynb](archive/3_etl_down_sample_by_user.ipynb)) are kept for documentation only; they cannot be re-run. Everything downstream of step 3 (the contents of [Processed_data/](Processed_data/)) is what we have, and that is what the current pipeline works with.

## 2. Cleaning and downsampling ([archive/3_etl_down_sample_by_user.ipynb](archive/3_etl_down_sample_by_user.ipynb))

> The contents of [Processed_data/](Processed_data/) are the **outputs of this step** (and step 4 below) — they are the starting point of the current pipeline. Steps 2 and 3 here describe how those files came to exist, but cannot be re-executed without the raw logs.

Two cleaning steps were applied at the user level before any feature generation:

1. **Bot removal.** Per-user play counts were thresholded at the 99.9th percentile; users above the threshold were treated as bots and dropped.
2. **User-level downsampling.** A random 10% of remaining `uid`s were retained so the rest of the pipeline could fit on a single machine. Downsampling is applied at the **user** level (not the row level) so that all events for a sampled user are kept — this preserves per-user feature integrity. The full pipeline is written in PySpark and is intended to scale without downsampling on a cluster.

After cleaning, the three streams are unioned into a single long event table with schema `(uid, event, song_id, date)` where `event ∈ {P, D, S}` for play / download / search.

## 3. Label definition ([archive/4_feature_label_generation_with_spark.ipynb](archive/4_feature_label_generation_with_spark.ipynb))

Churn is defined by **inactivity in a fixed forward window**:

```
feature window:  2017-03-30 → 2017-04-28   (30 days, used to build features)
label   window:  2017-04-29 → 2017-05-12   (14 days, used to assign label)
```

- **Modeling population** = all `uid`s with at least one event during the feature window.
- A user is **active (label = 0)** if they have at least one event of any type during the label window.
- A user is **churned (label = 1)** otherwise (left-join from modeling population to label-window actives, then fill nulls with 1).

Class balance in the resulting table is roughly **60% churn / 40% active**, which matches industry experience for short windows on free-tier music apps.

### Why this matters for leakage

Because the label is defined by activity **strictly after** `feature_window_end_date = 2017-04-28`, any feature built only from events on or before that date is leakage-safe by construction. The schema validator in this repo encodes this implicitly: it checks that no feature column is derived from events in the label window, but since we no longer have access to the raw event stream, this is a structural assumption inherited from the upstream pipeline rather than something we can re-verify here. **This is a documented limitation** — we can validate the processed table, but cannot re-prove leakage safety against logs we do not possess.

## 4. Feature engineering

All features are computed over the **30-day feature window** ending 2017-04-28 and joined on `uid`. There are 45 numeric features plus one categorical (`device_type`), grouped into five families.

Notation: `last_W` means "in the W days ending at the snapshot date" (W ∈ {1, 3, 7, 14, 30}).

### 4.1 Frequency (event counts) — 15 features
Count of events of each type in each window.
```
freq_P_last_{1,3,7,14,30}   # plays
freq_D_last_{1,3,7,14,30}   # downloads
freq_S_last_{1,3,7,14,30}   # searches
```
Captures **how active** a user has been across recent timescales.

### 4.2 Recency (days since last event) — 9 features
Days from the snapshot date back to the user's most recent event of that type within the window. Computed only for the longer windows because short-window recency is largely degenerate.
```
recent_P_last_{7,14,30}
recent_D_last_{7,14,30}
recent_S_last_{7,14,30}
```
Captures **how stale** a user's engagement is. The original analysis found 30-day recency to be the single most predictive feature.

### 4.3 Total playtime — 5 features
Sum of `play_time` (seconds) across all play events in the window. Negative or zero `play_time` is filtered out as dirty data before aggregation.
```
tot_playtime_last_{1,3,7,14,30}
```

### 4.4 Songs fully played — 5 features
Count of play events where `play_time / song_length >= 0.8` (i.e. the user listened to at least 80% of the song). Filters require `play_time > 0` and `song_length > 0`.
```
songs_full_last_{1,3,7,14,30}
```
Captures **engagement quality**, not just volume — a user who skips through 100 tracks looks different from one who finishes 10.

### 4.5 Play-completion percentage — 10 features
Per-event ratio `percent = play_time / song_length`, then aggregated per user per window:
```
mean_percent_last_{1,3,7,14,30}    # mean of percent
sd_percent_last_{1,3,7,14,30}      # std-dev of percent
```
Captures listening style: high mean + low SD = consistent listener; low mean + high SD = sampler/skipper.

### 4.6 Profile — 1 feature
```
device_type   # 1 = iPhone, 2 = Android / other
```
Derived from the `device` field in the play log; the user profile data was otherwise sparse, so this is the only profile feature carried forward.

## 5. Final modeling table

[Processed_data/df_model_final.csv](Processed_data/df_model_final.csv) is the join of all the above on `uid`:

- **Rows:** 57,953 users (one row per user, after bot removal and 10% user-level downsample).
- **Columns:** 47 — `uid`, `label`, 45 numeric features above, `device_type`.
- **Target:** `label` ∈ {0, 1} where 1 = churned (no activity 2017-04-29 → 2017-05-12).
- **No timestamp column.** The snapshot date is implicit (2017-04-28); the table holds one row per user at that single snapshot.

### Implications for the pipeline

1. **Splitting.** With no timestamp column and a single snapshot, true temporal evaluation is impossible from this table alone. The pipeline uses **stratified train / val / test splits** and documents that this is a regression from the ideal time-based split a production system would use.
2. **Drift monitoring.** Without multiple snapshots, "drift" is simulated by comparing train vs. holdout feature distributions, not real production-vs-baseline drift over time.
3. **Feature store.** The Feast feature definitions in this repo describe the entity, schema, and offline source for documentation and training/inference consistency, but **raw-log materialization is intentionally out of scope** — the offline source is the processed table itself.
4. **Real-time inference is out of scope.** A real-time service would need access to raw event logs to recompute the 30-day rolling features at request time. This repo deliberately ships **batch scoring** (Kubernetes CronJob) instead, which matches how lifecycle / retention teams actually consume churn scores in practice.
