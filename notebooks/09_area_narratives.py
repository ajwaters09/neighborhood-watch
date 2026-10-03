# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 09 - Area narratives: "what's been happening here lately", with `ai_query`
# MAGIC
# MAGIC Turns each area's recent numbers into a short, factual paragraph once a night and stores it
# MAGIC in `area_narratives`. The app leads the area panel with it, and the agent reads it through
# MAGIC `get_area_summary`. A nightly batch rather than a live call: 77 calls a night is cheap and
# MAGIC predictable, readers never wait on a model, and the prompt is reviewable in one place.
# MAGIC
# MAGIC **What the model sees** (per area, the 30 days ending `as_of_date`), assembled in Spark and
# MAGIC stored as `context_json`, so every sentence traces back to a number:
# MAGIC - counts per crime and 311 category against normal for this time of year and against the
# MAGIC   prior 30 days, which the prompt has to reconcile when they disagree
# MAGIC - the area's usual level: violent and all crime per resident, and its citywide rank
# MAGIC - next week's forecast, its citywide part kept apart from the area's own lean (from `08`)
# MAGIC - top crime types, location types and blocks, the arrest share, and top 311 requests
# MAGIC - leading-indicator 311 types rising here now (from `05`)
# MAGIC
# MAGIC Only areas whose `as_of_date` moved are regenerated; set `force` after changing the prompt.
# MAGIC Runs nightly after `04`, `05` and `08`.

# COMMAND ----------

import os
import sys
from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.community_areas import AREA_NAMES
# Shared with the app, so the narratives' "rising" and usual-level words match the area panel's.
from src.constants import LEVEL_BANDS, PROFILE_SPAN, RISING_MIN_PRIOR, RISING_PCT

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("endpoint", config.SERVING_ENDPOINT)
dbutils.widgets.dropdown("force", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint")
FORCE = dbutils.widgets.get("force") == "true"

CRIMES_TABLE = f"{CATALOG}.{SCHEMA}.silver_crimes"
SR_TABLE = f"{CATALOG}.{SCHEMA}.silver_311"
ROLLING_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_rolling"
NORMALS_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_normals"
PROFILE_TABLE = f"{CATALOG}.{SCHEMA}.area_profile"
OUTLOOK_AREA_TABLE = f"{CATALOG}.{SCHEMA}.outlook_area_week"
OUTLOOK_CITY_TABLE = f"{CATALOG}.{SCHEMA}.outlook_city_week"
LEAD_LAG_TABLE = f"{CATALOG}.{SCHEMA}.indicator_lead_lag"
NARRATIVES_TABLE = f"{CATALOG}.{SCHEMA}.area_narratives"

WINDOW_DAYS = 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which areas need a (re)write?

# COMMAND ----------

rolling = spark.table(ROLLING_TABLE).filter(F.col("window_days") == WINDOW_DAYS)
as_of_date = rolling.agg(F.max("as_of_date")).first()[0]
since = as_of_date - timedelta(days=WINDOW_DAYS - 1)
print(f"as_of_date {as_of_date}; narrative window {since} .. {as_of_date}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {NARRATIVES_TABLE} (
        community_area INT NOT NULL,
        as_of_date     DATE,
        narrative      STRING,
        context_json   STRING,
        model          STRING,
        generated_at   TIMESTAMP
    )
""")

all_areas = spark.createDataFrame([(k, v) for k, v in AREA_NAMES.items()], ["community_area", "area_name"])
if FORCE:
    todo = all_areas
else:
    current = spark.table(NARRATIVES_TABLE).filter(F.col("as_of_date") == F.lit(as_of_date)).select("community_area")
    todo = all_areas.join(current, "community_area", "left_anti")
print(f"{todo.count()} areas to (re)generate")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-area context (Spark, one JSON object per area)

# COMMAND ----------


def top_n(df, group_cols, n, alias):
    """Top-n rows per area by count, as an array of structs."""
    counted = df.groupBy("community_area", *group_cols).agg(F.count(F.lit(1)).alias("count"))
    ranked = counted.withColumn("rk", F.row_number().over(Window.partitionBy("community_area").orderBy(F.desc("count"))))
    return (
        ranked.filter(F.col("rk") <= n)
        .groupBy("community_area")
        .agg(F.sort_array(F.collect_list(F.struct(F.col("count"), *[F.col(c) for c in group_cols])), asc=False).alias(alias))
    )


in_window = lambda col: F.to_date(col).between(F.lit(since), F.lit(as_of_date))

crimes = spark.table(CRIMES_TABLE).filter(in_window("date"))
srs = spark.table(SR_TABLE).filter(in_window("created_date"))

normals = (
    spark.table(NORMALS_TABLE).filter(F.col("window_days") == WINDOW_DAYS)
    .select("community_area", "metric", "normal_count", F.round("pct_vs_normal", 1).alias("pct_vs_normal"))
)
trend = (
    rolling.join(normals, ["community_area", "metric"], "left")
    .select(
        "community_area",
        F.struct(
            "metric", "window_count", "normal_count", "pct_vs_normal", "prior_window_count",
            F.round("pct_change_vs_prior_window", 1).alias("pct_change_vs_prior_30d"),
        ).alias("t"),
    )
    .groupBy("community_area").agg(F.collect_list("t").alias("trend_30d"))
)

level_word = F.lit(None).cast("string")
for floor, label in reversed(LEVEL_BANDS):
    level_word = F.when(F.col("citywide_percentile") >= floor, F.lit(label)).otherwise(level_word)
usual_level = (
    # The last 12 months: what the area is like now.
    spark.table(PROFILE_TABLE).filter(F.col("metric").isin("crime_violent", "crime_total") & (F.col("span") == PROFILE_SPAN))
    .groupBy("community_area")
    .agg(F.map_from_entries(F.collect_list(F.struct(
        "metric",
        F.struct(F.round("rate_per_1k", 1).alias("per_1000_residents_a_year"),
                 F.col("citywide_rank").alias("rank_of_77_highest_first"),
                 level_word.alias("level")),
    ))).alias("usual_level"))
)

# Next week's forecast (notebook 08), if it has run: the call, and its citywide part (the same
# for every area) kept apart from this area's own lean.
if spark.catalog.tableExists(OUTLOOK_AREA_TABLE) and spark.catalog.tableExists(OUTLOOK_CITY_TABLE):
    city_live = spark.table(OUTLOOK_CITY_TABLE).filter("is_live").select(
        F.round(F.col("pct_momentum") * 100, 1).alias("city_trend_pct"),
        F.round(F.col("pct_weather") * 100, 1).alias("weather_pct"),
        F.round(F.col("pct_calendar") * 100, 1).alias("holidays_pct"),
        F.round(F.col("fc_t_anom") * 9 / 5, 1).alias("forecast_temp_vs_normal_f"),   # the model's °C, shown in °F
    )
    next_week = (
        spark.table(OUTLOOK_AREA_TABLE).filter("is_live").crossJoin(city_live)
        .select("community_area", F.struct(
            F.col("cutoff").cast("string").alias("week_of"), "call",
            F.round("p_above", 2).alias("p_above_normal"),
            F.round("forecast", 0).alias("forecast"), F.round("normal", 0).alias("normal"),
            F.round(F.col("pct_citywide") * 100, 1).alias("citywide_part_pct"),
            F.round(F.col("pct_local") * 100, 1).alias("this_area_part_pct"),
            F.struct("city_trend_pct", "weather_pct", "holidays_pct", "forecast_temp_vs_normal_f").alias("citywide_drivers"),
        ).alias("next_week"))
    )
else:
    print("No outlook tables yet (notebook 08) -- narratives go without next_week")
    next_week = spark.createDataFrame([], "community_area INT, next_week STRING")

arrests = crimes.groupBy("community_area").agg(
    F.round(F.avg(F.col("arrest").cast("int")) * 100, 1).alias("arrest_share_pct")
)

leading = (
    spark.table(LEAD_LAG_TABLE).filter("is_leading")
    .select("sr_metric", "crime_metric", "lag_months", F.round("r", 3).alias("r"))
)
signals = (
    rolling.filter(
        F.col("metric").startswith("311_")
        & (F.col("prior_window_count") >= RISING_MIN_PRIOR)
        & (F.col("pct_change_vs_prior_window") >= RISING_PCT)
    )
    .select("community_area", F.col("metric").alias("sr_metric"), F.round("pct_change_vs_prior_window", 1).alias("sr_pct_change"))
    .join(leading, "sr_metric")
    .groupBy("community_area")
    .agg(F.collect_list(F.struct("sr_metric", "sr_pct_change", "crime_metric", "lag_months", "r")).alias("rising_leading_signals"))
)

context = (
    todo
    .join(trend, "community_area", "left")
    .join(usual_level, "community_area", "left")
    .join(next_week, "community_area", "left")
    .join(top_n(crimes, ["primary_type", "description"], 5, "top_crime_types"), "community_area", "left")
    .join(top_n(crimes, ["location_description"], 3, "top_crime_locations"), "community_area", "left")
    .join(top_n(crimes, ["block"], 5, "top_crime_blocks"), "community_area", "left")
    .join(arrests, "community_area", "left")
    .join(top_n(srs, ["sr_type"], 5, "top_311_types"), "community_area", "left")
    .join(signals, "community_area", "left")
    .withColumn("window", F.lit(f"{since} to {as_of_date}"))
    .withColumn("context_json", F.to_json(F.struct(
        "area_name", "window", "trend_30d", "usual_level", "next_week", "top_crime_types",
        "top_crime_locations", "top_crime_blocks", "arrest_share_pct", "top_311_types",
        "rising_leading_signals",
    )))
    .select("community_area", "area_name", "context_json")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt and `ai_query`
# MAGIC
# MAGIC The guardrails matter more than the style: residents read this about their own
# MAGIC neighborhood. Only facts from the JSON, numbers cited, no speculation about people, no
# MAGIC alarmism either way, and small counts called small.

# COMMAND ----------

PROMPT = """You write the "what's been happening lately" summary for one Chicago community area in a
neighborhood early-warning app. Residents read it. Use ONLY the facts in the JSON below.

Write 4 to 6 plain sentences (no headings, no bullet points, under 140 words):
- Lead with how the last 30 days compare with normal for this time of year: trend_30d's
  pct_vs_normal for crime_total (normal = what the area's own past year predicts, adjusted for
  the season), with numbers.
- Then the short-term swing vs. the 30 days before (pct_change_vs_prior_30d). If the two point
  different ways or differ a lot, say why in plain words, e.g. "up 34% from an unusually quiet
  stretch, but only about 10% above normal".
- Place the area in the city with usual_level: its violent-crime level per resident over the past
  12 months (e.g. "among the lowest in the city for violent crime over the past year, #63 of 77").
- If next_week is present, give the forecast call, and keep its citywide part (weather, the
  city's trend, holidays -- the same for every area) apart from this area's own lean.
- Mention the most common incident types and where they cluster (blocks or location types).
- Mention rising 311 requests; if rising_leading_signals is non-empty, say that this kind of
  request has historically come before a rise in that crime category -- a pattern worth
  watching, not a prediction.
- If counts are small (under about 10), say the numbers are too small to read much into.
Use imperial units (degrees Fahrenheit, inches, miles).
Metric names: crime_<category> is a crime category, 311_<type> a 311 request type;
write them in plain words. Never speculate about individuals, motives, or demographics.
Be even-handed and direct: say plainly when something is above normal or rising, and when it's
below normal or falling. Don't soften bad news or play up good news. No alarmist language.

JSON:
"""

prompts = context.withColumn("prompt", F.concat(F.lit(PROMPT), F.col("context_json")))
prompts.createOrReplaceTempView("narrative_prompts")

narratives = spark.sql(f"""
    SELECT community_area, context_json,
           ai_query('{ENDPOINT}', prompt) AS narrative
    FROM narrative_prompts
""").withColumn("as_of_date", F.lit(as_of_date)) \
   .withColumn("model", F.lit(ENDPOINT)) \
   .withColumn("generated_at", F.current_timestamp())

narratives.createOrReplaceTempView("new_narratives")
spark.sql(f"""
    MERGE INTO {NARRATIVES_TABLE} t
    USING new_narratives s ON t.community_area = s.community_area
    WHEN MATCHED THEN UPDATE SET
        as_of_date = s.as_of_date, narrative = s.narrative, context_json = s.context_json,
        model = s.model, generated_at = s.generated_at
    WHEN NOT MATCHED THEN INSERT (community_area, as_of_date, narrative, context_json, model, generated_at)
        VALUES (s.community_area, s.as_of_date, s.narrative, s.context_json, s.model, s.generated_at)
""")
print("Merged narratives into", NARRATIVES_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot-check a few

# COMMAND ----------

display(
    spark.table(NARRATIVES_TABLE)
    .filter(F.col("community_area").isin(8, 25, 32, 43, 68))
    .select("community_area", "as_of_date", "narrative", "context_json")
)
