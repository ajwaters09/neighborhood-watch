# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02b - Bronze: nightly weather pulls (Open-Meteo)
# MAGIC
# MAGIC Feeds `08_next_week_outlook`, whose citywide model uses next week's weather forecast (about
# MAGIC +1% crime per °C above normal).
# MAGIC - **Observed** (`bronze_weather_observed`):
# MAGIC   - Re-pulls the last `observed_lookback_days` before the newest stored day, through
# MAGIC     yesterday, and MERGEs on `day`. The archive's newest days are preliminary and get revised.
# MAGIC   - Always starting behind the newest stored day means the series can't get a gap, even if
# MAGIC     some runs are missed.
# MAGIC   - On an empty table it pulls from 1996, one request.
# MAGIC - **Forecast** (`bronze_weather_forecast`):
# MAGIC   - Today's forecast for the next 10 days: GFS for temperature, JMA for precipitation.
# MAGIC   - Stored as (target_day, issue_date = today, lead_days, model), so the table is also a
# MAGIC     growing archive of what the forecast said when. That archive is what lets `08` score its
# MAGIC     backtest on forecasts that were really issued.
# MAGIC   - It picks up where the seed (`01c`) left off.
# MAGIC
# MAGIC Open-Meteo is free for non-commercial use and needs no key. `src/open_meteo.py` retries rate
# MAGIC limits and server errors, and rejects malformed responses (tested in
# MAGIC `tests/test_outlook_model.py`).

# COMMAND ----------

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src import open_meteo as om
from src.spark_bronze_utils import upsert_rows

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("observed_lookback_days", "10")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LOOKBACK = int(dbutils.widgets.get("observed_lookback_days"))

OBSERVED_TABLE = f"{CATALOG}.{SCHEMA}.bronze_weather_observed"
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.bronze_weather_forecast"
TODAY = datetime.now(ZoneInfo(om.TZ)).date()     # Chicago's date, not the cluster's UTC one

# COMMAND ----------

# MAGIC %md
# MAGIC ## Observed

# COMMAND ----------

newest = (spark.sql(f"SELECT max(day) AS d FROM {OBSERVED_TABLE}").collect()[0]["d"]
          if spark.catalog.tableExists(OBSERVED_TABLE) else None)
start = newest - timedelta(days=LOOKBACK) if newest else om.OBSERVED_START
end = TODAY - timedelta(days=1)
rows = om.parse_observed(om.fetch_observed(start, end))
n = upsert_rows(spark, rows, OBSERVED_TABLE, om.OBSERVED_SCHEMA, om.OBSERVED_KEYS, source="api", replace_sources=("seed", "api"))
print(f"observed: pulled {start} .. {end}, stored {n} days (newest with data: {rows[-1]['day'] if rows else 'none'})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast

# COMMAND ----------

for model in om.FORECAST_MODELS:
    rows = om.parse_forecast(om.fetch_forecast(model), model, issue_date=TODAY)
    n = upsert_rows(spark, rows, FORECAST_TABLE, om.FORECAST_SCHEMA, om.FORECAST_KEYS, source="api", replace_sources=("seed", "api"))
    print(f"forecast {model}: issued {TODAY}, {n} days ({rows[0]['target_day']} .. {rows[-1]['target_day']})")

# COMMAND ----------

display(spark.sql(f"""
    SELECT model, _source, count(*) AS rows, min(issue_date) AS first_issue, max(issue_date) AS last_issue,
           max(target_day) AS furthest_target
    FROM {FORECAST_TABLE} GROUP BY model, _source ORDER BY model, _source
"""))
