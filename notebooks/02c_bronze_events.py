# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02c - Bronze: nightly event pulls (permits, setlist.fm, Ticketmaster)
# MAGIC
# MAGIC Feeds `08b_events_lookahead`, the app's list of upcoming street festivals and club nights.
# MAGIC Pro sports are left out on purpose: everyone already plans around those.
# MAGIC
# MAGIC | Source | Table | How it stays current |
# MAGIC |---|---|---|
# MAGIC | CDOT street-use permits (`pubx-yq2d`) | `bronze_cdot_permits` | replaces every permit starting within the last 180 days or ahead: statuses move after filing ("in review" to complete or cancelled) |
# MAGIC | Park District permits (`pk66-w54g`) | `bronze_park_permits` | the same 180-day window; the dataset has no stable key |
# MAGIC | Park boundaries (`ejsh-fztr`) | `bronze_park_polygons` | a full refresh, one request |
# MAGIC | setlist.fm, the 20 clubs in `src/events_sources.CLUBS` | `bronze_setlists` | newest pages first, back to 60 days before the newest stored show (fans log late); MERGE on setlist id |
# MAGIC | Ticketmaster Discovery | `bronze_ticketmaster` | a dated snapshot of the next year's Chicago events; it has no history, so the snapshots become one |
# MAGIC
# MAGIC **API budgets:**
# MAGIC - setlist.fm allows 1,440 requests a day. A routine run needs ~25 (one page per club), and
# MAGIC   `setlistfm_max_requests` caps a run.
# MAGIC - Ticketmaster allows 5,000 a day, and a run needs ~40.
# MAGIC - The permit sources are the city portal (the Socrata key, as `02`).
# MAGIC - The history came from the research pulls via `01c`, not these APIs.
# MAGIC
# MAGIC **Keys** live in the secret scope (`00_setup`). A missing key skips its source with a message
# MAGIC rather than failing the run. The parsers are tested in `tests/test_events.py`.

# COMMAND ----------

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src import events_sources as es
from src.soda_client import SocrataClient
from src.spark_bronze_utils import replace_rows, upsert_rows

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("setlistfm_max_requests", "300")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SETLISTFM_MAX = int(dbutils.widgets.get("setlistfm_max_requests"))
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
SECRET_SCOPE = config.SECRET_SCOPE
TODAY = datetime.now(ZoneInfo("America/Chicago")).date()


def secret(key: str) -> str | None:
    try:
        return dbutils.secrets.get(scope=SECRET_SCOPE, key=key)
    except Exception:
        return None


soda = SocrataClient(key_id=secret("socrata_key_id"), key_secret=secret("socrata_key_secret"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Permits and park boundaries

# COMMAND ----------

since = es.permit_window_start(TODAY)
cdot = es.soda_rows(soda.query(es.CDOT_PERMITS, where=es.cdot_where(since), select=",".join(es.CDOT_COLUMNS), order=":id"), es.CDOT_COLUMNS)
n = replace_rows(spark, cdot, T("bronze_cdot_permits"), es.CDOT_SCHEMA, f"applicationstartdate >= '{since}'", source="api")
print(f"cdot permits: {n:,} starting {since[:10]} or later")

park = es.soda_rows(soda.query(es.PARK_PERMITS, where=es.park_where(since), select=",".join(es.PARK_COLUMNS), order=":id"), es.PARK_COLUMNS)
n = replace_rows(spark, park, T("bronze_park_permits"), es.PARK_SCHEMA, f"reservation_start_date >= '{since}'", source="api")
print(f"park permits: {n:,} starting {since[:10]} or later")

polygons = es.parse_park_polygons(es.fetch_park_polygons())
n = replace_rows(spark, polygons, T("bronze_park_polygons"), es.POLYGON_SCHEMA, None, source="api")
print(f"park boundaries: {n} parks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## setlist.fm

# COMMAND ----------

key = secret("setlistfm_api_key")
if not key:
    print(f"setlist.fm skipped: no setlistfm_api_key in the {SECRET_SCOPE} scope")
else:
    newest = {}
    if spark.catalog.tableExists(T("bronze_setlists")):
        newest = {r["venue_id"]: r["d"] for r in spark.sql(
            f"SELECT venue_id, max(event_date) AS d FROM {T('bronze_setlists')} GROUP BY venue_id").collect()}
    client = es.SetlistFm(key, max_requests=SETLISTFM_MAX)
    rows = []
    try:
        for vid, club in es.CLUB_VENUE_IDS.items():
            # shows get logged ahead of time, so look back from today, not from the newest listing
            known = min(newest[vid], TODAY) if vid in newest else None
            got = es.pull_venue(client, vid, known)
            rows += got
            print(f"  {club} [{vid}]: {len(got)} setlists{'' if known else ' (no history on file: backfilling)'}")
    except es.QuotaSpent as exc:
        print(f"  stopped early: {exc}. The next run picks up where this one left off.")
    unique = list({r["setlist_id"]: r for r in rows}.values())     # page boundaries can repeat a setlist
    n = upsert_rows(spark, unique, T("bronze_setlists"), es.SETLIST_SCHEMA, es.SETLIST_KEYS, source="api",
                    replace_sources=("seed", "api"))
    print(f"setlist.fm: {n:,} setlists merged, {SETLISTFM_MAX - client.left} requests used")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ticketmaster

# COMMAND ----------

key = secret("ticketmaster_api_key")
if not key:
    print(f"Ticketmaster skipped: no ticketmaster_api_key in the {SECRET_SCOPE} scope")
else:
    rows = [{"snapshot_date": TODAY, **r} for r in es.pull_ticketmaster(key)]
    n = replace_rows(spark, rows, T("bronze_ticketmaster"), es.TM_SCHEMA, f"snapshot_date = '{TODAY}'", source="api")
    print(f"ticketmaster: {n:,} upcoming events in today's snapshot")

# COMMAND ----------

checks = [("cdot permits", "bronze_cdot_permits", "applicationstartdate"),
          ("park permits", "bronze_park_permits", "reservation_start_date"),
          ("setlists", "bronze_setlists", "cast(event_date AS STRING)"),
          ("ticketmaster", "bronze_ticketmaster", "cast(snapshot_date AS STRING)")]
parts = [f"SELECT '{label}' AS tbl, _source, count(*) AS rows, max({col}) AS newest FROM {T(t)} GROUP BY _source"
         for label, t, col in checks if spark.catalog.tableExists(T(t))]
display(spark.sql(" UNION ALL ".join(parts)))
