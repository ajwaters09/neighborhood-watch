# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 08b - Events look-ahead: upcoming festivals and club nights, and the extra crime they tend to bring
# MAGIC
# MAGIC The app's "Upcoming events" list (`src/events_model.py`; the method is in `docs/modeling.md`):
# MAGIC 1. **Build:** permits and setlists become one event table on H3 resolution-9 cells
# MAGIC    (`silver_events`, `silver_event_cells`). The build runs in Spark (`src/events_spark.py`),
# MAGIC    checked against the pandas reference by `tests/test_events_spark.py`.
# MAGIC 2. **Past events:** observed vs. expected crime in each event's cells and the ring around them.
# MAGIC    Expected comes from clean same-weekday control days, scaled by how the whole city did.
# MAGIC 3. **Upcoming events:** extra crime = baseline x (ratio - 1). The baseline is the crime
# MAGIC    normally expected there on those dates; the ratio is the event's own record pooled with a
# MAGIC    prior (Gamma-Poisson):
# MAGIC    - festivals: past editions of the same festival; prior = the average for festivals its size
# MAGIC    - club nights: the club's past shows; prior = the all-club average
# MAGIC
# MAGIC **What the numbers mean.** Street festivals raise crime about 31% in their own cells (52%
# MAGIC for ones closing 5+ street segments), and club nights about 11% before midnight. In absolute
# MAGIC terms that's about half a crime per typical festival and 0.2 per night at the busiest clubs,
# MAGIC so this is a short filter for busy corridors, not an alarm. In backtests the ranking's top 20%
# MAGIC held about 74% of the extra crime.
# MAGIC
# MAGIC Writes `silver_events`, `silver_event_cells`, `events_upcoming` (what the app reads) and
# MAGIC `events_priors`. Runs nightly after `02c` and `03`. The H3 SQL functions need Photon, which
# MAGIC serverless has.

# COMMAND ----------

# MAGIC %pip install "h3>=4.1" "scipy>=1.10"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pyspark.sql import functions as F

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src import events_model as em
from src import events_spark as ess
from src.community_areas import AREA_NAMES

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

TODAY = datetime.now(ZoneInfo("America/Chicago")).date()
as_date = lambda s: pd.to_datetime(s).dt.date     # Arrow can hand DATE columns back as timestamps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the event table

# COMMAND ----------

# The build runs in Spark and writes the silver tables. The look-ahead is pandas, so the built
# tables (~85k events, ~260k cells) come back to the driver, read from the saved tables so the
# build isn't computed twice.
events_sdf, cells_sdf = ess.build_events(
    spark.table(T("bronze_cdot_permits")), spark.table(T("bronze_park_permits")),
    spark.table(T("bronze_park_polygons")), spark.table(T("bronze_setlists")), TODAY)
(events_sdf.withColumn("as_of", F.lit(TODAY)).write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(T("silver_events")))
(cells_sdf.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true").saveAsTable(T("silver_event_cells")))

events, cells = ess.to_pandas(spark.table(T("silver_events")).drop("as_of"), spark.table(T("silver_event_cells")))
tm = spark.sql(f"""
    SELECT * FROM {T('bronze_ticketmaster')}
    WHERE snapshot_date = (SELECT max(snapshot_date) FROM {T('bronze_ticketmaster')})
""").toPandas()
print(f"{len(events):,} events, {len(cells):,} event cells; "
      f"Ticketmaster snapshot of {tm['snapshot_date'].max() if len(tm) else 'none'} ({len(tm):,} events)")
display(events.groupby(["source", "category", "status"]).size().rename("events").reset_index())

# COMMAND ----------

def save(df: pd.DataFrame, name: str) -> None:
    (spark.createDataFrame(df).write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(T(name)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Crime counts where they're needed
# MAGIC
# MAGIC Counts per r9 cell, only for the cells the look-ahead reads: every past festival's and club
# MAGIC show's cells and ring, and every upcoming event's. Plus the citywide series for scaling.
# MAGIC - **Day counts** leave out "sometime this month" records stamped on the 1st at 00:00 or 00:01.
# MAGIC - **Night counts** (6pm-3am, keyed to the evening's date) leave out any 00:00 or 00:01
# MAGIC   timestamp. That's the "time unknown" convention, which would otherwise land inside the night.

# COMMAND ----------

fest, _ = em.upcoming_festivals(events)
nights = em.upcoming_club_nights(events, tm, TODAY)
need = em.cells_needed(events, cells, fest, nights)
spark.createDataFrame(pd.DataFrame({"h3_r9": sorted(need)})).createOrReplaceTempView("_needed_cells")
print(f"{len(fest)} upcoming festivals, {len(nights)} upcoming club nights; counts needed for {len(need):,} cells")

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW _crime_ts AS
    SELECT h3_longlatash3(longitude, latitude, 9) AS h3_r9, date AS ts
    FROM {T('silver_crimes')}
    WHERE date >= '2013-01-01' AND latitude IS NOT NULL AND longitude IS NOT NULL
""")
spark.sql("""
    CREATE OR REPLACE TEMP VIEW _crime_day AS
    SELECT h3_r9, to_date(ts) AS date FROM _crime_ts
    WHERE NOT (dayofmonth(ts) = 1 AND hour(ts) = 0 AND minute(ts) <= 1)
""")
spark.sql("""
    CREATE OR REPLACE TEMP VIEW _crime_night AS
    SELECT h3_r9, CASE WHEN hour(ts) >= 18 THEN to_date(ts) ELSE date_sub(to_date(ts), 1) END AS date
    FROM _crime_ts
    WHERE (hour(ts) >= 18 OR hour(ts) < 3) AND NOT (hour(ts) = 0 AND minute(ts) <= 1)
""")
per_cell = lambda v: spark.sql(f"""
    SELECT c.h3_r9, c.date, count(*) AS crime FROM {v} c JOIN _needed_cells n ON c.h3_r9 = n.h3_r9 GROUP BY 1, 2
""").toPandas()
city = lambda v: spark.sql(f"SELECT date, count(*) AS crime FROM {v} GROUP BY 1").toPandas()
day_counts, day_city, night_counts, night_city = per_cell("_crime_day"), city("_crime_day"), per_cell("_crime_night"), city("_crime_night")
print(f"counts: {len(day_counts):,} cell-days, {len(night_counts):,} cell-nights; city through {max(as_date(day_city['date']))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score

# COMMAND ----------

out = em.lookahead(events, cells, tm, TODAY, day_counts, day_city, night_counts, night_city)
print(f"crime treated as known through {out['known_through'].iloc[0].to_dict()}")
display(out["priors"])

upcoming = em.upcoming_table(out, em.area_of_cells())
upcoming["area_name"] = upcoming["community_area"].map(AREA_NAMES)
save(upcoming.assign(as_of=TODAY), "events_upcoming")
save(out["priors"].assign(as_of=TODAY), "events_priors")

scored = upcoming[upcoming["kind"] != "multi_week"]
print(f"events_upcoming: {len(scored)} scored ({(scored['kind'] == 'festival').sum()} festivals, "
      f"{(scored['kind'] == 'club_night').sum()} club nights), {scored['extra'].sum():.1f} extra crimes expected in total; "
      f"{(upcoming['kind'] == 'multi_week').sum()} multi-week runs listed unscored")
display(scored.sort_values("extra", ascending=False).head(15)[
    ["start_date", "kind", "name", "place", "area_name", "history", "baseline_crime", "ratio", "extra", "extra_lo", "extra_hi"]])
