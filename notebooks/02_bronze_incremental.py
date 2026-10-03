# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 - Bronze: nightly crime and 311 pulls (SODA)
# MAGIC
# MAGIC Appends only what's newer than each bronze table's watermark (`max(date)` for crimes,
# MAGIC `max(created_date)` for 311), filtered server-side with SoQL `$where`, in page-sized writes.
# MAGIC The tables come from `01` (crimes) and `01b` (311).
# MAGIC
# MAGIC Known limitation: crimes are watermarked on the offense date, so a record corrected later
# MAGIC is never re-fetched. A full fix would re-sync on `updated_on` and MERGE.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.soda_client import SocrataClient
from src.spark_bronze_utils import write_rows_in_chunks

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("page_size", "50000")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PAGE_SIZE = int(dbutils.widgets.get("page_size"))

CRIMES_TABLE = f"{CATALOG}.{SCHEMA}.bronze_crimes"
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
# MAGIC ## Crimes

# COMMAND ----------

if not spark.catalog.tableExists(CRIMES_TABLE):
    raise RuntimeError(f"{CRIMES_TABLE} doesn't exist yet -- run 01_bronze_bulk_load_crimes first.")

max_date = spark.table(CRIMES_TABLE).selectExpr("max(date) as m").collect()[0]["m"]
where = f"date > '{max_date}'"

rows_iter = client.get_crimes(where=where, order="date ASC", page_size=PAGE_SIZE)
total = write_rows_in_chunks(spark, rows_iter, CRIMES_TABLE, page_size=PAGE_SIZE, label="crimes", ingest_source="soda_incremental")
print(f"[crimes] {total:,} new rows (where: {where!r})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 311

# COMMAND ----------

if not spark.catalog.tableExists(SR_TABLE):
    raise RuntimeError(f"{SR_TABLE} doesn't exist yet -- run 01b_bronze_bulk_load_311 first.")

max_created = spark.table(SR_TABLE).selectExpr("max(created_date) as m").collect()[0]["m"]
where = f"created_date > '{max_created}'"

rows_iter = client.get_service_requests(where=where, order="created_date ASC", page_size=PAGE_SIZE)
total = write_rows_in_chunks(spark, rows_iter, SR_TABLE, page_size=PAGE_SIZE, label="311", ingest_source="soda_incremental")
print(f"[311] {total:,} new rows (where: {where!r})")
