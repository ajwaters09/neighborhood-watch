# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01b - Bronze: 311 history, chunked and resumable
# MAGIC
# MAGIC One-time load of "311 Service Requests" (`v6vf-nfxy`, ~14.7M rows) into `bronze_311`
# MAGIC through the JSON API, one month at a time: `$offset` resets every month (deep offsets are
# MAGIC slow on Socrata), and rows go to Delta every `page_size` rows instead of piling up on the
# MAGIC driver (`src/spark_bronze_utils.write_rows_in_chunks`). A crashed run resumes where it
# MAGIC stopped.
# MAGIC
# MAGIC Run `00_setup` first, and this before `02_bronze_incremental`, which only appends.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.soda_client import SocrataClient
from src.spark_bronze_utils import month_windows, write_rows_in_chunks

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("backfill_start", "2019-01-01T00:00:00")
dbutils.widgets.text("page_size", "50000")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
BACKFILL_START = dbutils.widgets.get("backfill_start")
PAGE_SIZE = int(dbutils.widgets.get("page_size"))

SR_TABLE = f"{CATALOG}.{SCHEMA}.bronze_311"

# COMMAND ----------

try:
    key_id = dbutils.secrets.get(scope=config.SECRET_SCOPE, key="socrata_key_id")
    key_secret = dbutils.secrets.get(scope=config.SECRET_SCOPE, key="socrata_key_secret")
except Exception:
    print("No secret scope found; falling back to unauthenticated (throttled) requests.")
    key_id, key_secret = None, None

client = SocrataClient(key_id=key_id, key_secret=key_secret)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resume
# MAGIC
# MAGIC `resume` (the default) picks up after `max(created_date)` already in the table: covered
# MAGIC months are skipped, and the half-loaded one continues with a `created_date >` filter so
# MAGIC nothing duplicates. `drop_and_reload` starts clean.

# COMMAND ----------

dbutils.widgets.dropdown("if_exists", "resume", ["resume", "drop_and_reload"])
IF_EXISTS = dbutils.widgets.get("if_exists")

if IF_EXISTS == "drop_and_reload" and spark.catalog.tableExists(SR_TABLE):
    print(f"Dropping {SR_TABLE} for a clean reload.")
    spark.sql(f"DROP TABLE {SR_TABLE}")

if spark.catalog.tableExists(SR_TABLE):
    resume_after = spark.table(SR_TABLE).selectExpr("max(created_date) as m").collect()[0]["m"]
    print(f"{SR_TABLE} already has data up to created_date={resume_after!r} -- resuming after that point.")
else:
    resume_after = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backfill, month by month

# COMMAND ----------

from datetime import datetime
import time

now = datetime.utcnow()
windows = list(month_windows(BACKFILL_START, now))
print(f"Backfilling 311 service requests from {BACKFILL_START} to {now.isoformat()} "
      f"in {len(windows)} monthly chunks (page size {PAGE_SIZE:,})")

grand_total = 0
overall_start = time.time()

for i, (window_start, window_end) in enumerate(windows, start=1):
    label = f"{window_start[:7]} ({i}/{len(windows)})"

    if resume_after and window_end <= resume_after:
        print(f"[{label}] already covered (up to {resume_after}) -- skipping")
        continue

    if resume_after and resume_after >= window_start:
        where = f"created_date > '{resume_after}' AND created_date < '{window_end}'"
    else:
        where = f"created_date >= '{window_start}' AND created_date < '{window_end}'"

    rows_iter = client.get_service_requests(where=where, order="created_date ASC", page_size=PAGE_SIZE)

    month_total = write_rows_in_chunks(
        spark, rows_iter, SR_TABLE, page_size=PAGE_SIZE, label=label, ingest_source="bulk_backfill"
    )
    grand_total += month_total
    elapsed = time.time() - overall_start
    print(f"[{label}] month total: {month_total:,} rows -- grand total so far: {grand_total:,} "
          f"({elapsed:,.0f}s elapsed)\n")

print(f"Backfill complete: {grand_total:,} rows loaded into {SR_TABLE} in {time.time() - overall_start:,.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compact
# MAGIC
# MAGIC Hundreds of chunked appends leave many small files; one `OPTIMIZE` after the backfill
# MAGIC compacts them. The nightly appends are small enough not to need it.

# COMMAND ----------

try:
    spark.sql(f"OPTIMIZE {SR_TABLE}")
    print(f"Ran OPTIMIZE on {SR_TABLE}.")
except Exception as e:
    print(f"OPTIMIZE skipped/failed ({e}) -- fine to run manually later: OPTIMIZE {SR_TABLE}")

# COMMAND ----------

display(spark.table(SR_TABLE).limit(5))
