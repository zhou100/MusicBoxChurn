"""Feast feature declarations — doc-only stub.

Mirrors `musicbox_churn.data.schema` as a *human-facing contract*. The
runtime pipeline does not read from Feast; it reads the CSV directly via
`musicbox_churn.data.load_data`. This file documents the entity grain
and feature schema in Feast's vocabulary so that a future deploy with a
real feature store has a starting point.

The snapshot is a single point in time (2017-04-28). Feast wants an
event_timestamp on every row; we synthesize one from a constant (see
README.md for why this is honest given the data we have).

To validate this file:
    pip install feast
    cd extras/feast
    feast apply
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

try:
    from feast import Entity, FeatureView, Field, FileSource, ValueType
    from feast.types import Float64, Int64
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "feast is not installed. This stub is doc-only — install with "
        "`pip install feast` to validate the schema. The core pipeline "
        "does not require it."
    ) from e


# Path to the offline source. Resolved relative to the feast working
# directory; symlink or override as needed for `feast apply`.
SOURCE_PATH = str(Path(__file__).resolve().parents[2] / "Processed_data" / "df_model_final.csv")

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

user = Entity(
    name="user",
    join_keys=["uid"],
    value_type=ValueType.INT64,
    description="Music Box user id (post-bot-removal, post-10pct downsample).",
)

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

# Feast requires an event_timestamp column. The processed table is a
# single 2017-04-28 snapshot with no row-level timestamp; if you actually
# wire this up, add a constant `event_timestamp = 2017-04-28` column to
# the CSV first. See README.md for the honest framing.
churn_source = FileSource(
    name="user_churn_snapshot",
    path=SOURCE_PATH,
    timestamp_field="event_timestamp",
    description="2017-04-28 snapshot of per-user features. Single snapshot — see DATA.md §5.",
)

# ---------------------------------------------------------------------------
# Feature view
# ---------------------------------------------------------------------------

WINDOWS_SHORT = (1, 3, 7, 14, 30)
WINDOWS_RECENT = (7, 14, 30)


def _freq_fields() -> list[Field]:
    return [
        Field(name=f"freq_{e}_last_{w}", dtype=Int64)
        for e in ("P", "D", "S")
        for w in WINDOWS_SHORT
    ]


def _recent_fields() -> list[Field]:
    return [
        Field(name=f"recent_{e}_last_{w}", dtype=Int64)
        for e in ("P", "D", "S")
        for w in WINDOWS_RECENT
    ]


def _engagement_fields() -> list[Field]:
    out: list[Field] = []
    out += [Field(name=f"tot_playtime_last_{w}", dtype=Float64) for w in WINDOWS_SHORT]
    out += [Field(name=f"songs_full_last_{w}", dtype=Int64) for w in WINDOWS_SHORT]
    out += [Field(name=f"mean_percent_last_{w}", dtype=Float64) for w in WINDOWS_SHORT]
    out += [Field(name=f"sd_percent_last_{w}", dtype=Float64) for w in WINDOWS_SHORT]
    return out


user_churn_features = FeatureView(
    name="user_churn_features",
    entities=[user],
    ttl=timedelta(days=14),  # label window length; features grow stale once that elapses
    schema=[
        *_freq_fields(),
        *_recent_fields(),
        *_engagement_fields(),
        Field(name="device_type", dtype=Int64),
    ],
    online=False,  # See README.md — no online materialization possible without raw logs.
    source=churn_source,
    description=(
        "Per-user churn features over a 30-day window ending 2017-04-28. "
        "44 numeric + 1 categorical (device_type). See ../../DATA.md."
    ),
    tags={
        "owner": "musicbox-churn",
        "tier": "offline-only",
        "snapshot_date": "2017-04-28",
    },
)
