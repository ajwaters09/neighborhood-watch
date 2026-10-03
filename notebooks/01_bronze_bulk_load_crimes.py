# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01 - Bronze: crimes, full history
# MAGIC
# MAGIC One-time load of the whole "Crimes - 2001 to Present" dataset (`ijzp-q8t2`, ~8.6M rows) into
# MAGIC `bronze_crimes`, from Socrata's CSV bulk export: far cheaper than paging the JSON API for
# MAGIC the full history. `02_bronze_incremental` appends to the same table from then on.
# MAGIC
# MAGIC Run `00_setup` first.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.dropdown("force_redownload", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FORCE_REDOWNLOAD = dbutils.widgets.get("force_redownload") == "true"
LANDING_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
BULK_CSV_URL = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
BULK_CSV_LOCAL = f"{LANDING_PATH}/crimes_bulk_export.csv"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_crimes"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download
# MAGIC
# MAGIC Streams the export (about 2 GB) straight onto the volume. The endpoint is public; the API
# MAGIC key only lifts shared-IP throttling. The download is the slow part, so a file already on the
# MAGIC volume is reused unless `force_redownload` is set.

# COMMAND ----------

import requests

if os.path.exists(BULK_CSV_LOCAL) and not FORCE_REDOWNLOAD:
    size_mb = os.path.getsize(BULK_CSV_LOCAL) / (1024 * 1024)
    print(f"{BULK_CSV_LOCAL} already exists ({size_mb:,.0f} MB) -- skipping download. "
          f"Set force_redownload=true to pull a fresh copy.")
else:
    try:
        SOCRATA_KEY_ID = dbutils.secrets.get(scope=config.SECRET_SCOPE, key="socrata_key_id")
        SOCRATA_KEY_SECRET = dbutils.secrets.get(scope=config.SECRET_SCOPE, key="socrata_key_secret")
        auth = (SOCRATA_KEY_ID, SOCRATA_KEY_SECRET)
    except Exception:
        print("No secret scope found; downloading unauthenticated (fine for this public bulk endpoint).")
        auth = None

    with requests.get(BULK_CSV_URL, auth=auth, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(BULK_CSV_LOCAL, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

    print(f"Downloaded to {BULK_CSV_LOCAL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load into bronze
# MAGIC
# MAGIC The bulk CSV and the incremental SODA JSON pulls write the same table, so this matches what
# MAGIC `02_bronze_incremental` appends:
# MAGIC 1. CSV headers (`"Primary Type"`) become SODA field names (`primary_type`).
# MAGIC 2. Every column is a string, as the JSON API returns them. Typing happens in silver.
# MAGIC 3. `date` and `updated_on` are rewritten from the CSV's `MM/dd/yyyy hh:mm:ss a` to the API's
# MAGIC    ISO format. `02` watermarks on `max(date)` as a string, which only sorts correctly in ISO,
# MAGIC    and passes it to SoQL, which rejects the US format.

# COMMAND ----------

from pyspark.sql.functions import col, lit, to_timestamp, date_format

CSV_TS_FORMAT = "MM/dd/yyyy hh:mm:ss a"
ISO_TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSS"
TIMESTAMP_COLUMNS = {"date", "updated_on"}

raw_df = spark.read.csv(BULK_CSV_LOCAL, header=True, inferSchema=True, multiLine=True, escape='"')

rename_map = {c: c.strip().lower().replace(" ", "_") for c in raw_df.columns}


def normalize(c: str, new_c: str):
    if new_c in TIMESTAMP_COLUMNS:
        return date_format(to_timestamp(col(c).cast("string"), CSV_TS_FORMAT), ISO_TS_FORMAT).alias(new_c)
    return col(c).cast("string").alias(new_c)


bronze_df = raw_df.select([normalize(c, new_c) for c, new_c in rename_map.items()])
bronze_df = bronze_df.withColumn("_ingest_source", lit("bulk_csv_export"))

# Reformatting shouldn't add nulls beyond those already in the CSV.
for ts_col in TIMESTAMP_COLUMNS:
    source_col = [c for c, new_c in rename_map.items() if new_c == ts_col][0]
    source_nulls = raw_df.filter(col(source_col).isNull()).count()
    reformatted_nulls = bronze_df.filter(col(ts_col).isNull()).count()
    if reformatted_nulls > source_nulls:
        print(f"WARNING: reformatting '{ts_col}' introduced {reformatted_nulls - source_nulls:,} new nulls "
              f"-- CSV_TS_FORMAT may not match every row's actual format.")

(bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(BRONZE_TABLE))

print(f"Loaded {bronze_df.count():,} rows into {BRONZE_TABLE}")

# COMMAND ----------

display(spark.table(BRONZE_TABLE).limit(5))
