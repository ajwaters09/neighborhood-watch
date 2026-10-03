# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03 - Silver: typed, cleaned, deduplicated crimes and 311
# MAGIC
# MAGIC Bronze is raw and all-string. Silver types the columns that matter downstream, drops rows
# MAGIC with no usable community area (null or the portal's `0`), and deduplicates: an
# MAGIC append-only bronze can repeat rows, and 311 flags its own duplicate requests. Everything
# MAGIC downstream reads silver, including the agent's `get_recent_activity`.
# MAGIC
# MAGIC A full recompute every run: deduplication has to see the whole key space, and at ~23M rows
# MAGIC it's cheap. Runs nightly after `02`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

CRIMES_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_crimes"
SR_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_311"
CRIMES_SILVER = f"{CATALOG}.{SCHEMA}.silver_crimes"
SR_SILVER = f"{CATALOG}.{SCHEMA}.silver_311"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Crimes
# MAGIC
# MAGIC - Types the columns downstream code uses; the rest stay strings.
# MAGIC - Drops rows with no usable `community_area`.
# MAGIC - One row per `id` (the dataset's key): the latest `updated_on`.

# COMMAND ----------

crimes_bronze = spark.table(CRIMES_BRONZE)

crimes_typed = (
    crimes_bronze
    .withColumn("id", F.col("id").cast("long"))
    .withColumn("date", F.to_timestamp("date"))
    .withColumn("updated_on", F.to_timestamp("updated_on"))
    .withColumn("arrest", F.col("arrest").cast("boolean"))
    .withColumn("domestic", F.col("domestic").cast("boolean"))
    .withColumn("beat", F.col("beat").cast("int"))
    .withColumn("district", F.col("district").cast("int"))
    .withColumn("ward", F.col("ward").cast("int"))
    .withColumn("community_area", F.col("community_area").cast("int"))
    .withColumn("year", F.col("year").cast("int"))
    .withColumn("x_coordinate", F.col("x_coordinate").cast("double"))
    .withColumn("y_coordinate", F.col("y_coordinate").cast("double"))
    .withColumn("latitude", F.col("latitude").cast("double"))
    .withColumn("longitude", F.col("longitude").cast("double"))
)

crimes_clean = crimes_typed.filter(F.col("community_area").isNotNull() & (F.col("community_area") != 0))

dedup_window = Window.partitionBy("id").orderBy(F.col("updated_on").desc())
crimes_silver = (
    crimes_clean
    .withColumn("_rn", F.row_number().over(dedup_window))
    .filter("_rn = 1")
    .drop("_rn")
)

(crimes_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CRIMES_SILVER))

bronze_n, silver_n = crimes_bronze.count(), crimes_silver.count()
print(f"silver_crimes: {silver_n:,} rows (from {bronze_n:,} bronze rows, "
      f"{bronze_n - silver_n:,} dropped as no-community-area/duplicate)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 311
# MAGIC
# MAGIC The same, plus dropping requests the city flags as `duplicate` (ten people reporting one
# MAGIC pothole), which would otherwise inflate the leading-indicator counts. One row per
# MAGIC `sr_number`: the latest `last_modified_date`.

# COMMAND ----------

sr_bronze = spark.table(SR_BRONZE)

sr_typed = (
    sr_bronze
    .withColumn("created_date", F.to_timestamp("created_date"))
    .withColumn("last_modified_date", F.to_timestamp("last_modified_date"))
    .withColumn("closed_date", F.to_timestamp("closed_date"))
    .withColumn("duplicate", F.col("duplicate").cast("boolean"))
    .withColumn("legacy_record", F.col("legacy_record").cast("boolean"))
    .withColumn("community_area", F.col("community_area").cast("int"))
    .withColumn("ward", F.col("ward").cast("int"))
    .withColumn("created_hour", F.col("created_hour").cast("int"))
    .withColumn("created_day_of_week", F.col("created_day_of_week").cast("int"))
    .withColumn("created_month", F.col("created_month").cast("int"))
    .withColumn("x_coordinate", F.col("x_coordinate").cast("double"))
    .withColumn("y_coordinate", F.col("y_coordinate").cast("double"))
    .withColumn("latitude", F.col("latitude").cast("double"))
    .withColumn("longitude", F.col("longitude").cast("double"))
)

sr_clean = (
    sr_typed
    .filter(F.col("community_area").isNotNull() & (F.col("community_area") != 0))
    .filter(~F.coalesce(F.col("duplicate"), F.lit(False)))
)

dedup_window = Window.partitionBy("sr_number").orderBy(F.col("last_modified_date").desc())
sr_silver = (
    sr_clean
    .withColumn("_rn", F.row_number().over(dedup_window))
    .filter("_rn = 1")
    .drop("_rn")
)

(sr_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SR_SILVER))

bronze_n, silver_n = sr_bronze.count(), sr_silver.count()
print(f"silver_311: {silver_n:,} rows (from {bronze_n:,} bronze rows, "
      f"{bronze_n - silver_n:,} dropped as no-community-area/flagged-duplicate)")

# COMMAND ----------

display(spark.table(CRIMES_SILVER).limit(5))

# COMMAND ----------

display(spark.table(SR_SILVER).limit(5))
