# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 04 - Gold: area trend metrics, normals and per-resident profile
# MAGIC
# MAGIC Buckets silver crimes and 311 requests by community area into one metric-labeled event base,
# MAGIC then writes four tables:
# MAGIC
# MAGIC | Table | Grain | What it answers |
# MAGIC |---|---|---|
# MAGIC | `area_trend_metrics` | area x month x metric | the monthly history, for trend lines (synced to Lakebase) |
# MAGIC | `area_trend_rolling` | area x metric x 30/60/90 days | the last N days vs. the N days before (synced to Lakebase) |
# MAGIC | `area_trend_normals` | area x metric x 30/60/90 days | the last N days vs. normal for this time of year |
# MAGIC | `area_profile` | area x metric | crime per 1,000 residents over the last 12 months, ranked citywide |
# MAGIC
# MAGIC **Metrics.** `crime_total` plus `crime_<category>` for the seven categories from `03b`, and
# MAGIC `311_total` plus ten 311 types chosen as physical-disorder signals (`src/constants.SR_INDICATOR_MAP`).
# MAGIC `311_total` leaves out information-only calls and aircraft-noise complaints, which carry
# MAGIC placeholder addresses.
# MAGIC
# MAGIC **Design notes.**
# MAGIC - *One shared as-of date.* Every window ends on the earlier of crime's and 311's last
# MAGIC   **full** day, and the monthly table is cut there too, so windows and the app's
# MAGIC   partial-month projection line up. The crime feed's newest day is a stub (14 records
# MAGIC   against ~650), so a day counts as full once it has half the median of the 28 days before it.
# MAGIC - *Windows, not months, for "better or worse".* A partial current month against a full prior
# MAGIC   one always looks like an improvement. The rolling table compares two complete windows; the
# MAGIC   monthly one keeps `pct_change_vs_prior` only for the long view.
# MAGIC - *Normals, not just the prior window.* The prior window swings with whatever it happened to
# MAGIC   hold, and ignores the season. The normals table measures against the same baseline the
# MAGIC   next-week outlook uses, so the app, the map and the forecast agree.
# MAGIC - *Zero-filled grids.* A missing (area, month, metric) is zero incidents, not missing data,
# MAGIC   so the grids are built from a real month sequence and left-joined.
# MAGIC - *Gold starts where 311 starts* (late 2018), so crime and 311 share one grid.
# MAGIC - *Full recompute every run.* About 800k output rows: cheap, and no incremental-upsert bugs.
# MAGIC
# MAGIC Runs nightly after `03b`. Re-trigger both Synced Tables afterwards.

# COMMAND ----------

import os
import sys
from datetime import timedelta
from functools import reduce

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.community_areas import POPULATION
# Shared with the app, so the windows and the profile span can't drift apart.
from src.constants import PROFILE_SPAN, PROFILE_YEARS, SR_EXCLUDE_FROM_TOTAL, SR_INDICATOR_MAP, WINDOW_DAYS_OPTIONS

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

CRIMES_TABLE = f"{CATALOG}.{SCHEMA}.silver_crimes"
SR_TABLE = f"{CATALOG}.{SCHEMA}.silver_311"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_metrics"
GOLD_ROLLING_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_rolling"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shared helpers

# COMMAND ----------


def month_bucket(df: DataFrame, ts_col: str) -> DataFrame:
    return df.withColumn("period", F.trunc(F.col(ts_col), "month"))


def zero_fill_monthly_grid(long_df: DataFrame) -> DataFrame:
    """Every (area, month, metric) as a row, zero where nothing happened, so lag() always
    compares with the true prior month. The months are a real sequence from min to max, not the
    distinct months in the data, which could skip an empty one."""
    areas = long_df.select("community_area").distinct()
    metrics = long_df.select("metric").distinct()
    bounds = long_df.agg(F.min("period").alias("min_p"), F.max("period").alias("max_p")).first()
    periods = spark.range(1).select(
        F.explode(F.sequence(F.lit(bounds["min_p"]), F.lit(bounds["max_p"]), F.expr("interval 1 month"))).alias("period")
    )
    grid = areas.crossJoin(periods).crossJoin(metrics)
    return (
        grid.join(long_df, ["community_area", "period", "metric"], "left")
        .withColumn("metric_count", F.coalesce(F.col("metric_count"), F.lit(0)))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 311 events
# MAGIC
# MAGIC One row per (request, metric): `(community_area, event_date, metric)`. Built first because
# MAGIC 311's start date trims the crime side.

# COMMAND ----------

# The 311 types and the 311_total exclusions are in src/constants.py.
sr = spark.table(SR_TABLE).withColumnRenamed("created_date", "event_date")

sr_indicator_map_df = spark.createDataFrame(
    list(SR_INDICATOR_MAP.items()), ["sr_type", "metric"]
)

sr_indicator_events = (
    sr.join(F.broadcast(sr_indicator_map_df), on="sr_type", how="inner")
    .select("community_area", "event_date", "metric")
)
sr_total_events = (
    sr.filter(~F.col("sr_type").isin(list(SR_EXCLUDE_FROM_TOTAL)))
    .select("community_area", "event_date")
    .withColumn("metric", F.lit("311_total"))
)
sr_events = sr_indicator_events.unionByName(sr_total_events)

sr_start = sr.agg(F.min("event_date")).first()[0]
print(f"311 metrics: {sr_events.select('metric').distinct().count()} distinct metrics, "
      f"data starts {sr_start} -- crime will be trimmed to the same start")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Crime events
# MAGIC
# MAGIC `crime_total` plus `crime_<category>` from `crime_category_map` (`03b`), from 311's start
# MAGIC date on. A pair missing from the map lands in `other`, so the categories always add up to
# MAGIC the total.

# COMMAND ----------

CATEGORY_MAP_TABLE = f"{CATALOG}.{SCHEMA}.crime_category_map"
category_map = (
    spark.table(CATEGORY_MAP_TABLE)
    .select(
        F.col("primary_type").alias("m_primary_type"),
        F.col("description").alias("m_description"),
        F.coalesce("category_override", "ai_category", F.lit("other")).alias("category"),
    )
)

crimes = spark.table(CRIMES_TABLE).withColumnRenamed("date", "event_date")
crimes = crimes.filter(F.col("event_date") >= F.lit(sr_start))
crimes = (
    crimes.join(
        F.broadcast(category_map),
        (F.col("primary_type") == F.col("m_primary_type")) & F.col("description").eqNullSafe(F.col("m_description")),
        "left",
    )
    .withColumn("crime_metric", F.concat(F.lit("crime_"), F.coalesce(F.col("category"), F.lit("other"))))
    .drop("m_primary_type", "m_description")
)
unmapped = crimes.filter(F.col("category").isNull()).count()
if unmapped:
    print(f"WARNING: {unmapped:,} crime rows have no crime_category_map entry (counted as 'other') -- re-run 03b")

crime_by_type_events = crimes.select("community_area", "event_date", F.col("crime_metric").alias("metric"))
crime_total_events = (
    crimes.select("community_area", "event_date")
    .withColumn("metric", F.lit("crime_total"))
)
crime_events = crime_by_type_events.unionByName(crime_total_events)
print(f"Crime metrics: {crime_events.select('metric').distinct().count()} distinct metrics "
      f"(trimmed to event_date >= {sr_start})")

# COMMAND ----------

events = crime_events.unionByName(sr_events)      # no .cache(): serverless doesn't support it

# COMMAND ----------

# MAGIC %md
# MAGIC ## The shared as-of date
# MAGIC
# MAGIC The **earlier** of the two sources' last full days: crime runs about 8 days behind 311, and
# MAGIC a window running past a source's data would silently undercount it.
# MAGIC
# MAGIC **Full** days only: the crime feed's newest day is usually a stub (14 records against ~650),
# MAGIC which made every 30-day window read a day short. A day counts once it holds `FULL_DAY_SHARE`
# MAGIC of the median of the 28 days before it.
# MAGIC
# MAGIC **Everything is cut at this date, the monthly table included**, because the app projects the
# MAGIC current month from it (N events through day D -> N x days_in_month / D).

# COMMAND ----------

FULL_DAY_SHARE = 0.5


def last_full_day(source_events: DataFrame):
    """Newest day with at least FULL_DAY_SHARE of the median daily count of the 28 days before it."""
    newest = source_events.agg(F.max("event_date")).first()[0].date()
    daily = {
        r["day"]: r["n"]
        for r in source_events.filter(F.col("event_date") >= F.lit(newest - timedelta(days=60)))
        .groupBy(F.to_date("event_date").alias("day")).count().withColumnRenamed("count", "n").collect()
    }
    day = newest
    while day > newest - timedelta(days=14):
        prior = sorted(daily.get(day - timedelta(days=i), 0) for i in range(1, 29))
        if daily.get(day, 0) >= FULL_DAY_SHARE * prior[len(prior) // 2]:
            return day
        day -= timedelta(days=1)
    raise ValueError(f"no full day in the 14 days up to {newest} -- check the incremental pulls")


# crime_total / 311_total only: one row per crime or request, not one per metric.
crime_full = last_full_day(crime_events.filter(F.col("metric") == "crime_total"))
sr_full = last_full_day(sr_events.filter(F.col("metric") == "311_total"))
as_of_date = min(crime_full, sr_full)
print(f"last full day: crime {crime_full}, 311 {sr_full} -> as_of_date {as_of_date}")

events = events.filter(F.to_date("event_date") <= F.lit(as_of_date))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `area_trend_metrics`: the monthly grid

# COMMAND ----------

monthly_long = events.transform(lambda df: month_bucket(df, "event_date")).groupBy(
    "community_area", "period", "metric"
).agg(F.count(F.lit(1)).alias("metric_count"))

combined = zero_fill_monthly_grid(monthly_long)
print(f"Combined gold grid: {combined.count():,} zero-filled (area, period, metric) rows "
      f"across {combined.select('period').distinct().count()} months")

trend_window = Window.partitionBy("community_area", "metric").orderBy("period")
rolling_3mo_window = trend_window.rowsBetween(-2, 0)  # this month + 2 prior = 3-month window

gold = (
    combined
    .withColumn("prior_period_count", F.lag("metric_count").over(trend_window))
    .withColumn(
        "pct_change_vs_prior",
        F.when(F.col("prior_period_count").isNull() | (F.col("prior_period_count") == 0), F.lit(None))
         .otherwise(
             (F.col("metric_count") - F.col("prior_period_count")) / F.col("prior_period_count") * 100
         ),
    )
    .withColumn("rolling_3mo_avg", F.avg("metric_count").over(rolling_3mo_window))
    .withColumn("updated_at", F.current_timestamp())
)

# COMMAND ----------

(gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(GOLD_TABLE))

print(f"Wrote {gold.count():,} rows to {GOLD_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `area_trend_rolling`: the last 30/60/90 days vs. the same span before
# MAGIC
# MAGIC A snapshot as of `as_of_date`, not a history: one row per (area, metric, window), rewritten
# MAGIC every run. Only the last `2 x max(WINDOW_DAYS_OPTIONS)` days are scanned.

# COMMAND ----------

max_window = max(WINDOW_DAYS_OPTIONS)
recent_events = events.filter(
    F.col("event_date") >= F.lit(as_of_date - timedelta(days=2 * max_window - 1))
)


def rolling_window_counts(window_days: int) -> DataFrame:
    cur_start = as_of_date - timedelta(days=window_days - 1)
    prior_start = as_of_date - timedelta(days=2 * window_days - 1)
    prior_end = as_of_date - timedelta(days=window_days)
    return (
        recent_events
        .groupBy("community_area", "metric")
        .agg(
            F.sum(
                F.when(F.to_date("event_date").between(cur_start, as_of_date), 1).otherwise(0)
            ).alias("window_count"),
            F.sum(
                F.when(F.to_date("event_date").between(prior_start, prior_end), 1).otherwise(0)
            ).alias("prior_window_count"),
        )
        .withColumn("window_days", F.lit(window_days))
    )


rolling_long = reduce(
    DataFrame.unionByName,
    [rolling_window_counts(w) for w in WINDOW_DAYS_OPTIONS],
)

# Zero-fill, as for the monthly grid: a combination with no events in the lookback is a 0.
areas = events.select("community_area").distinct()
metrics = events.select("metric").distinct()
windows_df = spark.createDataFrame([(w,) for w in WINDOW_DAYS_OPTIONS], ["window_days"])
rolling_grid = areas.crossJoin(metrics).crossJoin(windows_df)

rolling_gold = (
    rolling_grid.join(rolling_long, ["community_area", "metric", "window_days"], "left")
    .withColumn("window_count", F.coalesce(F.col("window_count"), F.lit(0)))
    .withColumn("prior_window_count", F.coalesce(F.col("prior_window_count"), F.lit(0)))
    .withColumn(
        "pct_change_vs_prior_window",
        F.when(F.col("prior_window_count") == 0, F.lit(None))
         .otherwise(
             (F.col("window_count") - F.col("prior_window_count")) / F.col("prior_window_count") * 100
         ),
    )
    .withColumn("as_of_date", F.lit(as_of_date))
    .withColumn("updated_at", F.current_timestamp())
)

# COMMAND ----------

(rolling_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(GOLD_ROLLING_TABLE))

print(f"Wrote {rolling_gold.count():,} rows to {GOLD_ROLLING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `area_trend_normals`: the last 30/60/90 days vs. normal for this time of year
# MAGIC
# MAGIC The prior window swings with whatever it held: an area coming off an unusually quiet stretch
# MAGIC reads "+34%" while running about normal. This compares the window with what the area's own
# MAGIC past year predicts for this time of year, the baseline the next-week outlook calls "normal":
# MAGIC - **normal** = the area's daily rate over the 364 days before the window x the window's
# MAGIC   length x a seasonal factor
# MAGIC - **seasonal factor** = the citywide count in the same window 1, 2 and 3 years back (whole
# MAGIC   weeks, so weekdays line up), each divided by the citywide pace of the year before it. The
# MAGIC   median of the three, per metric, so rodent complaints get a rodent season.
# MAGIC
# MAGIC The app reads this through its cached Unity Catalog path, so it needs no Synced Table.

# COMMAND ----------

GOLD_NORMALS_TABLE = f"{CATALOG}.{SCHEMA}.area_trend_normals"
NORMAL_YEARS = 3
YEAR = timedelta(days=364)

normal_events = events.filter(
    F.col("event_date") >= F.lit(as_of_date - timedelta(days=364 * (NORMAL_YEARS + 1) + max_window))
)


def normal_spans(window_days: int) -> dict:
    """(first, last) day of the current window, the year before it, and the same pair k years back."""
    cur_start = as_of_date - timedelta(days=window_days - 1)
    spans = {"cur": (cur_start, as_of_date), "trail": (cur_start - YEAR, cur_start - timedelta(days=1))}
    for k in range(1, NORMAL_YEARS + 1):
        shift = timedelta(days=364 * k)
        spans[f"ref{k}"] = (cur_start - shift, as_of_date - shift)
        spans[f"reftrail{k}"] = (cur_start - shift - YEAR, cur_start - shift - timedelta(days=1))
    return spans


def normal_counts(window_days: int) -> DataFrame:
    day = F.to_date("event_date")
    return (
        normal_events.groupBy("community_area", "metric")
        .agg(*[F.sum(F.when(day.between(a, b), 1).otherwise(0)).alias(name)
               for name, (a, b) in normal_spans(window_days).items()])
        .withColumn("window_days", F.lit(window_days))
    )


span_cols = list(normal_spans(30))
ref_cols = [c for c in span_cols if c.startswith("ref")]
area_spans = (
    rolling_grid.join(reduce(DataFrame.unionByName, [normal_counts(w) for w in WINDOW_DAYS_OPTIONS]),
                      ["community_area", "metric", "window_days"], "left")
    .fillna(0, subset=span_cols)
)

season = area_spans.groupBy("metric", "window_days").agg(*[F.sum(c).alias(c) for c in ref_cols])
for k in range(1, NORMAL_YEARS + 1):
    season = season.withColumn(
        f"s{k}",
        F.when(F.col(f"reftrail{k}") > 0,
               F.col(f"ref{k}") / (F.col(f"reftrail{k}") * F.col("window_days") / 364)),
    )
s_list = ", ".join(f"s{k}" for k in range(1, NORMAL_YEARS + 1))
season = season.withColumn("s", F.expr(f"array_sort(filter(array({s_list}), x -> x IS NOT NULL))")).withColumn(
    "season_factor",
    F.expr("CASE size(s) WHEN 0 THEN NULL WHEN 1 THEN s[0] WHEN 2 THEN (s[0] + s[1]) / 2 "
           "ELSE s[CAST((size(s) - 1) / 2 AS INT)] END"),
).select("metric", "window_days", "season_factor")

normals_gold = (
    area_spans.join(season, ["metric", "window_days"], "left")
    .withColumn("normal_count", F.col("trail") / 364 * F.col("window_days") * F.col("season_factor"))
    .select(
        "community_area", "metric", "window_days",
        F.col("cur").alias("window_count"),
        F.round("normal_count", 1).alias("normal_count"),
        F.when(F.col("normal_count") > 0, (F.col("cur") / F.col("normal_count") - 1) * 100).alias("pct_vs_normal"),
        F.round("season_factor", 3).alias("season_factor"),
    )
    .withColumn("as_of_date", F.lit(as_of_date))
    .withColumn("updated_at", F.current_timestamp())
)

(normals_gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(GOLD_NORMALS_TABLE))
print(f"Wrote {normals_gold.count():,} rows to {GOLD_NORMALS_TABLE}")

# COMMAND ----------

display(
    spark.table(GOLD_NORMALS_TABLE)
    .filter("metric = 'crime_total' AND window_days = 30")
    .groupBy().agg(F.sum("window_count").alias("city_window"), F.sum("normal_count").alias("city_normal"),
                   F.sum(F.when(F.col("pct_vs_normal") > 0, 1).otherwise(0)).alias("areas_above_normal"),
                   F.first("season_factor").alias("season_factor"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `area_profile`: how each area stacks up, per resident
# MAGIC
# MAGIC Trends say whether an area is getting better or worse; this says where it sits. An area
# MAGIC running 10% below its normal can still be far above one running 10% over its own. It backs
# MAGIC the map's "Crime per capita" layer and the rank notes under an area's name.
# MAGIC - Count per 1,000 residents over the last 12 months (`PROFILE_SPAN` in `src/constants.py`),
# MAGIC   for every metric, with its citywide percentile and rank (1 = highest).
# MAGIC - Population is the ACS 2023 5-year estimate (`src/community_areas.POPULATION`).
# MAGIC - The Loop and Near North Side draw far more visitors and workers than residents, so their
# MAGIC   per-resident rates run high, especially for theft. The app says so.

# COMMAND ----------

GOLD_PROFILE_TABLE = f"{CATALOG}.{SCHEMA}.area_profile"

population = spark.createDataFrame([(int(a), int(p)) for a, p in POPULATION.items()], ["community_area", "population"])
by_rate = Window.partitionBy("metric").orderBy("rate_per_1k")


def profile_for(span: str, years: int) -> DataFrame:
    start = as_of_date - timedelta(days=365 * years - 1)
    counts = (
        events.filter(F.to_date("event_date") >= F.lit(start))
        .groupBy("community_area", "metric").agg(F.count(F.lit(1)).alias("n"))
    )
    return (
        areas.crossJoin(metrics)
        .join(counts, ["community_area", "metric"], "left")
        .fillna(0, subset=["n"])
        .join(population, "community_area")
        .withColumn("annual_count", F.col("n") / years)
        .withColumn("rate_per_1k", F.col("annual_count") / F.col("population") * 1000)
        .withColumn("citywide_percentile", F.round(F.percent_rank().over(by_rate) * 100).cast("int"))
        .withColumn("citywide_rank", F.rank().over(Window.partitionBy("metric").orderBy(F.desc("rate_per_1k"))))
        .select(
            "community_area", "metric", "population",
            F.round("annual_count", 1).alias("annual_count"),
            F.round("rate_per_1k", 2).alias("rate_per_1k"),
            "citywide_percentile", "citywide_rank",
        )
        .withColumn("span", F.lit(span))
        .withColumn("years", F.lit(years))
        .withColumn("period_start", F.lit(start))
    )


profile_gold = (
    profile_for(PROFILE_SPAN, PROFILE_YEARS)
    .withColumn("as_of_date", F.lit(as_of_date))
    .withColumn("updated_at", F.current_timestamp())
)

(profile_gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(GOLD_PROFILE_TABLE))
print(f"Wrote {profile_gold.count():,} rows to {GOLD_PROFILE_TABLE}")

# COMMAND ----------

display(
    spark.table(GOLD_PROFILE_TABLE).filter(f"metric = 'crime_violent' AND span = '{PROFILE_SPAN}'")
    .orderBy(F.desc("rate_per_1k")).select("community_area", "rate_per_1k", "citywide_rank", "citywide_percentile")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Primary keys for the Lakebase Synced Tables
# MAGIC
# MAGIC A Synced Table (reverse ETL into Lakebase Postgres) needs its source to declare a primary key
# MAGIC and have Change Data Feed on (set in the writes above). The keys hold by construction: the
# MAGIC zero-filled grids have exactly one row per combination. Databricks doesn't enforce them;
# MAGIC they're informational. Key columns must be `NOT NULL` first.

# COMMAND ----------

spark.sql(f"ALTER TABLE {GOLD_TABLE} ALTER COLUMN community_area SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD_TABLE} ALTER COLUMN period SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD_TABLE} ALTER COLUMN metric SET NOT NULL")
spark.sql(f"""
    ALTER TABLE {GOLD_TABLE}
    ADD CONSTRAINT area_trend_metrics_pk PRIMARY KEY (community_area, period, metric)
""")
print(f"{GOLD_TABLE} is ready as a Synced Table source: PK declared, CDF enabled.")

spark.sql(f"ALTER TABLE {GOLD_ROLLING_TABLE} ALTER COLUMN community_area SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD_ROLLING_TABLE} ALTER COLUMN metric SET NOT NULL")
spark.sql(f"ALTER TABLE {GOLD_ROLLING_TABLE} ALTER COLUMN window_days SET NOT NULL")
spark.sql(f"""
    ALTER TABLE {GOLD_ROLLING_TABLE}
    ADD CONSTRAINT area_trend_rolling_pk PRIMARY KEY (community_area, metric, window_days)
""")
print(f"{GOLD_ROLLING_TABLE} is ready as a Synced Table source: PK declared, CDF enabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-time: create the Synced Tables (in the UI)
# MAGIC
# MAGIC For `area_trend_metrics` and `area_trend_rolling`: **Catalog** -> the table -> **Create** ->
# MAGIC **Synced table**, into the Lakebase project's `databricks_postgres` database, named
# MAGIC `<table>_lb` (the names in `src/config.py`). Use **Triggered** mode: both tables are rewritten
# MAGIC nightly, so continuous sync would only churn. Re-trigger both after each run of this notebook.

# COMMAND ----------

display(
    spark.table(GOLD_TABLE)
    .filter("metric = 'crime_total'")
    .orderBy(F.desc("period"))
    .limit(10)
)

# COMMAND ----------

display(
    spark.table(GOLD_ROLLING_TABLE)
    .filter("metric = 'crime_total'")
    .orderBy("window_days")
)
