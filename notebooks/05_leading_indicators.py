# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 - Leading indicators: does 311 activity precede crime?
# MAGIC
# MAGIC The "broken windows" idea is that physical disorder (streetlights out, vacant buildings,
# MAGIC dumping, graffiti) comes before some kinds of crime. This measures it, area by area and month
# MAGIC by month, and writes `indicator_lead_lag` for the Insights heatmap, the area panel's Signals
# MAGIC card and the agent's `get_leading_indicators`.
# MAGIC
# MAGIC **Method** (`src/leading_indicators.py`, tested in `tests/test_leading_indicators.py`):
# MAGIC 1. Monthly counts per area for every 311 and crime metric (`area_trend_metrics`), complete
# MAGIC    months only, as `log(1 + n)`.
# MAGIC 2. Two-way demeaning per metric: remove each area's own level and each month's citywide
# MAGIC    swing (seasons lift 311 and crime everywhere, which would otherwise look like a link).
# MAGIC 3. For lags of 0-3 months, correlate a 311 metric's residual in month t-k with a crime
# MAGIC    metric's in month t, pooled across areas. Then the reverse: crime leading 311.
# MAGIC 4. A pair leads at its best lag k >= 1 when that correlation is Bonferroni-significant across
# MAGIC    every test, at least `MIN_R`, and stronger than both the same-month and the reverse one.
# MAGIC
# MAGIC **Caveats, shown in the app:** correlation, not causation; a citywide pattern, not a claim
# MAGIC about any one area; and months within an area aren't independent, so the p-values are
# MAGIC optimistic. The shuffled-area placebo at the bottom should find nothing.
# MAGIC
# MAGIC Steps 1-3 run in Spark here. Runs nightly after `04`.

# COMMAND ----------

import os
import sys
from functools import reduce

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.leading_indicators import ALPHA, LAGS, MIN_R, flag_leading

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_metrics"
GOLD_ROLLING_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_rolling"
LEAD_LAG_TABLE = f"{CATALOG}.{SCHEMA}.indicator_lead_lag"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly residuals

# COMMAND ----------

as_of_date = spark.table(GOLD_ROLLING_TABLE).agg(F.max("as_of_date")).first()[0]
# Drop the as-of month unless the data runs through its last day.
first_partial = F.trunc(F.lit(as_of_date), "month")
is_month_end = F.last_day(F.lit(as_of_date)) == F.lit(as_of_date)

monthly = (
    spark.table(GOLD_TABLE)
    .filter((F.col("period") < first_partial) | is_month_end)
    .select("community_area", "period", "metric", "metric_count")
    .withColumn("y", F.log1p("metric_count"))
)

by_area = Window.partitionBy("metric", "community_area")
by_month = Window.partitionBy("metric", "period")
by_metric = Window.partitionBy("metric")
resid = monthly.withColumn(
    "resid",
    F.col("y") - F.avg("y").over(by_area) - F.avg("y").over(by_month) + F.avg("y").over(by_metric),
).select("community_area", "period", "metric", "resid")

sr = resid.filter(F.col("metric").startswith("311_")).withColumnRenamed("metric", "sr_metric").withColumnRenamed("resid", "sr_resid")
crime = resid.filter(F.col("metric").startswith("crime_")).withColumnRenamed("metric", "crime_metric").withColumnRenamed("resid", "crime_resid")

n_sr = sr.select("sr_metric").distinct().count()
n_crime = crime.select("crime_metric").distinct().count()
print(f"as_of {as_of_date}: {n_sr} 311 metrics x {n_crime} crime metrics x lags {LAGS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lagged correlations, both directions

# COMMAND ----------


def lagged_corr(leader: DataFrame, leader_col: str, follower: DataFrame, follower_col: str, k: int) -> DataFrame:
    """corr(leader[area, t-k], follower[area, t]) per (sr_metric, crime_metric)."""
    shifted = leader.withColumn("period", F.add_months("period", k))
    return (
        shifted.join(follower, ["community_area", "period"])
        .groupBy("sr_metric", "crime_metric")
        .agg(F.corr(leader_col, follower_col).alias("r"), F.count(F.lit(1)).alias("n"))
        .withColumn("lag_months", F.lit(k))
    )


forward = reduce(DataFrame.unionByName, [lagged_corr(sr, "sr_resid", crime, "crime_resid", k) for k in LAGS])
reverse = reduce(DataFrame.unionByName, [lagged_corr(crime, "crime_resid", sr, "sr_resid", k) for k in LAGS])
fwd, rev = forward.toPandas(), reverse.toPandas()
print(f"{len(fwd)} forward rows, {len(rev)} reverse rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Significance and the `is_leading` flag
# MAGIC
# MAGIC A few hundred rows, so this step is pandas (`flag_leading`).

# COMMAND ----------

df = flag_leading(fwd, rev)
print(f"Bonferroni alpha {ALPHA} over {len(df) * 2} tests, MIN_R {MIN_R}: {int(df['is_leading'].sum())} leading pairs")
display(df[df["is_leading"]].sort_values("r", ascending=False))

# COMMAND ----------

result = (
    spark.createDataFrame(df)
    .withColumn("as_of_date", F.lit(as_of_date))
    .withColumn("computed_at", F.current_timestamp())
)
result.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(LEAD_LAG_TABLE)
print(f"Wrote {result.count()} rows to {LEAD_LAG_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check: the shuffled-area placebo
# MAGIC
# MAGIC Pair each area's 311 history with a different, random area's crime. Any real relationship
# MAGIC should vanish: the placebo's largest |r| should be tiny next to the flagged pairs. If it
# MAGIC isn't, the demeaning is leaving a shared citywide signal in.

# COMMAND ----------

import random

areas = [r["community_area"] for r in sr.select("community_area").distinct().collect()]
shuffled = areas[:]
random.Random(42).shuffle(shuffled)
remap = spark.createDataFrame(list(zip(areas, shuffled)), ["community_area", "shuffled_area"])
sr_shuffled = sr.join(remap, "community_area").drop("community_area").withColumnRenamed("shuffled_area", "community_area")
placebo = lagged_corr(sr_shuffled, "sr_resid", crime, "crime_resid", 1).toPandas()
print(f"Placebo (lag 1, shuffled areas): max |r| = {placebo['r'].abs().max():.3f}, "
      f"median |r| = {placebo['r'].abs().median():.3f}")
print(f"Real (lag 1): max r among flagged = {df[df['is_leading']]['r'].max() if df['is_leading'].any() else float('nan'):.3f}")
