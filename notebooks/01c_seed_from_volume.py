# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01c - Seed: weather and events history from the landing volume
# MAGIC
# MAGIC The weather and events sources were first pulled locally for the research in `offline_ml/`.
# MAGIC Re-pulling that history through the APIs would spend quota for nothing (setlist.fm allows
# MAGIC 1,440 requests a day, and the club history took days), so the **raw API responses** are
# MAGIC uploaded once and loaded here into the same bronze tables the nightly notebooks (`02b`,
# MAGIC `02c`) keep current. It's the crime pattern again: a bulk seed, then incremental pulls.
# MAGIC - Same parsers as the live pulls (`src/open_meteo.py`, `src/events_sources.py`), so a seeded
# MAGIC   row and a pulled row mean the same thing. Seeded rows carry `_source = 'seed'`.
# MAGIC - Safe to re-run: keyed tables MERGE and never overwrite a live (`_source = 'api'`) row, and
# MAGIC   window-replaced tables are skipped once a live pull has written to them.
# MAGIC - Not part of the nightly job.
# MAGIC
# MAGIC **Upload first**, from the repo root:
# MAGIC ```bash
# MAGIC databricks fs cp -r offline_data/weather/raw dbfs:/Volumes/<catalog>/<schema>/landing/seed/weather
# MAGIC databricks fs cp -r offline_data/events/raw dbfs:/Volumes/<catalog>/<schema>/landing/seed/events
# MAGIC ```
# MAGIC
# MAGIC | File | What | Goes to |
# MAGIC |---|---|---|
# MAGIC | `weather/observed_era5.json` | ERA5 daily weather, 1996-01-01 to 2026-09-25 | `bronze_weather_observed` |
# MAGIC | `weather/forecast_gfs_seamless.json` | GFS forecasts 1-7 days ahead, hourly, 2021-03 on | `bronze_weather_forecast` |
# MAGIC | `weather/forecast_jma_seamless.json` | JMA forecasts 1-7 days ahead, hourly, 2021-01 on | `bronze_weather_forecast` |
# MAGIC | `events/cdot_permits.parquet` | CDOT festival/block party/parade/run/rally permits, 2014 on | `bronze_cdot_permits` |
# MAGIC | `events/park_permits.parquet` | Park District event permits, 2014 on | `bronze_park_permits` |
# MAGIC | `events/park_polygons.geojson` | park boundaries | `bronze_park_polygons` |
# MAGIC | `events/setlistfm/<venue id>/p*.json` | setlist.fm pages for the 20 clubs, 2014 on | `bronze_setlists` |
# MAGIC | `events/ticketmaster/upcoming_<date>.parquet` | a Ticketmaster snapshot of upcoming events | `bronze_ticketmaster` |
# MAGIC
# MAGIC Checked locally against these exact files: the parsers reproduce the research tables row for
# MAGIC row.

# COMMAND ----------

import glob
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src import events_sources as es
from src import open_meteo as om
from src.spark_bronze_utils import replace_rows, upsert_rows

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("seed_dir", "")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SEED_DIR = dbutils.widgets.get("seed_dir") or f"/Volumes/{CATALOG}/{SCHEMA}/landing/seed"

T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
OBSERVED_TABLE = f"{CATALOG}.{SCHEMA}.bronze_weather_observed"
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.bronze_weather_forecast"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather

# COMMAND ----------

def load(name: str) -> dict:
    path = os.path.join(SEED_DIR, "weather", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- upload offline_data/weather/raw first (see the top of this notebook)")
    with open(path) as f:
        return json.load(f)


observed = om.parse_observed(load("observed_era5.json"))
n = upsert_rows(spark, observed, OBSERVED_TABLE, om.OBSERVED_SCHEMA, om.OBSERVED_KEYS, source="seed", replace_sources=("seed",))
print(f"bronze_weather_observed: {n:,} seed rows, {observed[0]['day']} .. {observed[-1]['day']}")

for model in om.FORECAST_MODELS:
    rows = om.parse_previous_runs(load(f"forecast_{model}.json"), model)
    n = upsert_rows(spark, rows, FORECAST_TABLE, om.FORECAST_SCHEMA, om.FORECAST_KEYS, source="seed", replace_sources=("seed",))
    print(f"bronze_weather_forecast: {n:,} seed rows for {model}")

# COMMAND ----------

# MAGIC %md
# MAGIC The observed table should run contiguously from 1996-01-01, and the forecast archive should
# MAGIC hold ~28.5k rows (2 models x ~2,000 days x 7 leads, minus archive gaps).

# COMMAND ----------

display(spark.sql(f"""
    SELECT 'observed' AS tbl, _source, count(*) AS rows, min(day) AS first, max(day) AS last,
           datediff(max(day), min(day)) + 1 - count(*) AS missing_days
    FROM {OBSERVED_TABLE} GROUP BY _source
    UNION ALL
    SELECT concat('forecast ', model), _source, count(*), min(target_day), max(target_day), NULL
    FROM {FORECAST_TABLE} GROUP BY model, _source
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Events
# MAGIC
# MAGIC Permits and Ticketmaster are replaced as a window by `02c`, not merged on a key, so a re-seed
# MAGIC after a live pull would double up that window: the seed skips any table a live pull has
# MAGIC written. Setlists MERGE on their id, so they're always safe.

# COMMAND ----------

EVENTS_DIR = os.path.join(SEED_DIR, "events")


def seeded_only(table: str) -> bool:
    """True if a live pull (`_source = 'api'`) hasn't touched `table` yet."""
    if not spark.catalog.tableExists(T(table)):
        return True
    live = spark.sql(f"SELECT count(*) AS n FROM {T(table)} WHERE _source = 'api'").collect()[0]["n"]
    if live:
        print(f"{table}: skipped, {live:,} rows already came from live pulls")
    return live == 0


if seeded_only("bronze_cdot_permits"):
    rows = es.soda_rows(pd.read_parquet(os.path.join(EVENTS_DIR, "cdot_permits.parquet")).to_dict("records"), es.CDOT_COLUMNS)
    n = replace_rows(spark, rows, T("bronze_cdot_permits"), es.CDOT_SCHEMA, "_source = 'seed'", source="seed")
    print(f"bronze_cdot_permits: {n:,} seed rows")
if seeded_only("bronze_park_permits"):
    rows = es.soda_rows(pd.read_parquet(os.path.join(EVENTS_DIR, "park_permits.parquet")).to_dict("records"), es.PARK_COLUMNS)
    n = replace_rows(spark, rows, T("bronze_park_permits"), es.PARK_SCHEMA, "_source = 'seed'", source="seed")
    print(f"bronze_park_permits: {n:,} seed rows")
if seeded_only("bronze_park_polygons"):
    with open(os.path.join(EVENTS_DIR, "park_polygons.geojson")) as f:
        rows = es.parse_park_polygons(json.load(f))
    n = replace_rows(spark, rows, T("bronze_park_polygons"), es.POLYGON_SCHEMA, None, source="seed")
    print(f"bronze_park_polygons: {n} parks")

rows = []
for vid in es.CLUB_VENUE_IDS:                      # the clubs in scope only
    for page in sorted(glob.glob(os.path.join(EVENTS_DIR, "setlistfm", vid, "p*.json"))):
        with open(page) as f:
            rows += es.parse_setlist_page(json.load(f), vid)
rows = list({r["setlist_id"]: r for r in rows}.values())     # page boundaries can repeat a setlist
n = upsert_rows(spark, rows, T("bronze_setlists"), es.SETLIST_SCHEMA, es.SETLIST_KEYS, source="seed", replace_sources=("seed",))
print(f"bronze_setlists: {n:,} seed setlists from {len(es.CLUB_VENUE_IDS)} venue ids")

for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "ticketmaster", "upcoming_*.parquet"))):
    snap = date.fromisoformat(os.path.basename(path)[len("upcoming_"):-len(".parquet")])
    rows = [{"snapshot_date": snap, **r} for r in es.soda_rows(pd.read_parquet(path).to_dict("records"), es.TM_FIELDS)]
    n = replace_rows(spark, rows, T("bronze_ticketmaster"), es.TM_SCHEMA,
                     f"snapshot_date = '{snap}' AND _source = 'seed'", source="seed")
    print(f"bronze_ticketmaster: {n:,} events in the {snap} snapshot")
