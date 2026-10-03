# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 06 - `subscription_events`: app writes, from Lakebase Change Data Feed
# MAGIC
# MAGIC Lakebase CDF (enabled once in the Lakebase UI for the `public` schema, landing in this
# MAGIC Unity Catalog schema) keeps a Delta history table per Postgres table, `lb_<table>_history`,
# MAGIC within seconds of each write. This view turns the two that hold user actions into one event
# MAGIC stream for analytics:
# MAGIC
# MAGIC | event_type | source |
# MAGIC |---|---|
# MAGIC | `subscription` | an insert into `area_subscriptions` |
# MAGIC | `unsubscription` | a delete from it (`REPLICA IDENTITY FULL` keeps the deleted row's values) |
# MAGIC | `report` | an insert into `resident_reports` |
# MAGIC
# MAGIC A view, so it's always current with nothing to schedule. Nothing updates these rows, so
# MAGIC update images are left out. Run once, after CDF has taken its first snapshot.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

SUBSCRIPTIONS_HISTORY = f"{CATALOG}.{SCHEMA}.lb_area_subscriptions_history"
REPORTS_HISTORY = f"{CATALOG}.{SCHEMA}.lb_resident_reports_history"
VIEW_NAME = f"{CATALOG}.{SCHEMA}.subscription_events"

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {VIEW_NAME} AS
    SELECT
        'subscription' AS event_type,
        _timestamp AS event_timestamp,
        user_id,
        community_area,
        CAST(NULL AS STRING) AS detail
    FROM {SUBSCRIPTIONS_HISTORY}
    WHERE _pg_change_type = 'insert'

    UNION ALL

    SELECT
        'unsubscription' AS event_type,
        _timestamp AS event_timestamp,
        user_id,
        community_area,
        CAST(NULL AS STRING) AS detail
    FROM {SUBSCRIPTIONS_HISTORY}
    WHERE _pg_change_type = 'delete'

    UNION ALL

    SELECT
        'report' AS event_type,
        _timestamp AS event_timestamp,
        user_id,
        community_area,
        description AS detail
    FROM {REPORTS_HISTORY}
    WHERE _pg_change_type = 'insert'
""")

print(f"Created view {VIEW_NAME}")

# COMMAND ----------

display(spark.table(VIEW_NAME).orderBy("event_timestamp", ascending=False).limit(20))
