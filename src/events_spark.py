"""The event build (permits and setlists -> one event table on H3 cells) in Spark, for notebook 08b.

The same rules as the pandas build in src/events_model.py (`cdot_events`, `park_events`,
`club_events`, `build_events`), which stays the tested reference; tests/test_events_spark.py
checks that the two produce the same tables. The Spark version keeps the build on the cluster
instead of pulling every bronze permit to the driver, and writes the silver tables directly.

Uses the Databricks H3 SQL functions (`h3_longlatash3`, `h3_try_polyfillash3`,
`h3_centerasgeojson`). Open-source Spark doesn't have them; the parity test registers h3-py
stand-ins under the same names.

Two known differences from the pandas build, both in edge cases:
- A club headliner tied on set length goes to the alphabetically first act (pandas took the
  first in row order, which isn't stable across reads anyway).
- A CDOT permit with a latitude but no longitude is dropped here; pandas would fail on it.
"""

from __future__ import annotations

from datetime import date
from itertools import chain

import pandas as pd
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from src.events_model import (CDOT_CATEGORY, CDOT_MERGE_BY_NAME, EVENT_COLUMNS, H3_RES, MAX_EVENT_DAYS,
                              PARK_HELD, PARK_PLANNED)
from src.events_sources import CLUB_VENUE_IDS, CLUBS, EVENTS_START

# Types for the columns a source doesn't have, so the union lines up.
EVENT_TYPES = {"event_id": "string", "source": "string", "category": "string", "name": "string", "venue": "string",
               "place": "string", "start_date": "date", "end_date": "date", "n_days": "bigint", "long_run": "boolean",
               "status": "string", "permit_stage": "string", "size_level": "double", "n_segments": "bigint",
               "full_closure": "boolean", "n_facilities": "bigint", "n_acts": "bigint", "lat": "double",
               "lon": "double", "h3_r9": "bigint", "n_cells": "bigint"}


def _cell(lat: str | Column, lon: str | Column) -> Column:
    col = lambda c: F.col(c) if isinstance(c, str) else c
    return F.call_function("h3_longlatash3", col(lon), col(lat), F.lit(H3_RES))


def _date(c: str) -> Column:
    """SODA timestamps ('2025-07-12T00:00:00.000') -> DATE, null if unparseable."""
    return F.expr(f"try_cast({c} AS DATE)")


def _num(c: str) -> Column:
    return F.expr(f"try_cast({c} AS DOUBLE)")


def _strip(c: Column) -> Column:
    """Python's str.strip: all leading/trailing whitespace, not just spaces like F.trim."""
    return F.regexp_replace(c, r"(?U)^\s+|\s+$", "")


def _lookup(mapping: dict, c: Column) -> Column:
    return F.create_map(*[F.lit(x) for x in chain(*mapping.items())])[c]


def _first(c: str, order: list[str]) -> Column:
    """pandas groupby .first() after a sort: the first non-null value in `order`."""
    return F.min_by(F.col(c), F.when(F.col(c).isNotNull(), F.struct(*order))).alias(c)


def _status(start: Column, end: Column, held: Column, planned: Column, today: date) -> Column:
    """'held' if it's over and the permit went through, 'planned' if it's still ahead, else null.
    Planned wins when both hold, as in the pandas version."""
    return F.when((start >= F.lit(today)) & planned, "planned").when((end < F.lit(today)) & held, "held")


# ---------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------

def cdot_events(raw: DataFrame, today: date) -> tuple[DataFrame, DataFrame]:
    milestone = F.col("currentmilestone")
    p = (raw.filter(F.col("worktypedescription").isin(list(CDOT_CATEGORY))
                    & (milestone.isNull() | ~milestone.isin("Cancelled", "Denied")))
         .withColumn("category", _lookup(CDOT_CATEGORY, F.col("worktypedescription")))
         .withColumn("start_date", _date("applicationstartdate"))
         .withColumn("end_date", _date("applicationenddate"))
         .withColumn("name", _strip(F.col("applicationname")))
         .withColumn("lat", _num("latitude")).withColumn("lon", _num("longitude")))
    frm, to = F.col("streetnumberfrom"), F.col("streetnumberto")
    number = F.when(frm == to, frm).otherwise(F.concat(F.coalesce(frm, F.lit("")), F.lit("-"), F.coalesce(to, F.lit(""))))
    nonempty = lambda c: F.when(c != "", c)
    p = (p.withColumn("segment", F.concat_ws(" ", *[nonempty(c) for c in
                                                   [number, F.col("direction"), F.col("streetname"), F.col("suffix")]]))
         .withColumn("status", _status(F.col("start_date"), F.col("end_date"), F.coalesce(milestone == "Complete", F.lit(False)),
                                       F.lit(True), today)))
    horizon = date(today.year + 2, 12, 31)          # a few applications carry typo'd years (2112)
    p = p.filter(F.col("status").isNotNull() & F.col("lat").isNotNull() & F.col("lon").isNotNull()
                 & F.col("start_date").isNotNull() & (F.col("start_date") <= F.lit(horizon)))

    # Festivals (and the other non-block-party types) sharing a name and dates are one event.
    by_name = F.col("category").isin(list(CDOT_MERGE_BY_NAME)) & F.col("name").isNotNull()
    group = F.when(by_name, F.concat_ws("|", "category", F.lower("name"), F.col("start_date").cast("string"),
                                        F.coalesce(F.col("end_date").cast("string"), F.lit("NaT")))) \
        .otherwise(F.col("applicationnumber"))
    p = (p.withColumn("group", group)
         .withColumn("event_id", F.concat(F.lit("cdot:"), F.min("applicationnumber").over(Window.partitionBy("group"))))
         .withColumn("h3_r9", _cell("lat", "lon")))

    order = ["applicationnumber", "uniquekey"]
    stage = (p.filter(milestone.isNotNull()).groupBy("event_id", "currentmilestone").count()
             .groupBy("event_id")      # the most common milestone; ties go to the alphabetically first
             .agg(F.min_by("currentmilestone", F.struct(-F.col("count"), F.col("currentmilestone"))).alias("permit_stage")))
    events = (p.groupBy("event_id")
              .agg(*[_first(c, order) for c in ["category", "name", "start_date", "end_date", "status"]],
                   F.avg("lat").alias("lat"), F.avg("lon").alias("lon"), F.count(F.lit(1)).alias("n_segments"),
                   F.max(F.coalesce(F.col("streetclosure") == "Full", F.lit(False))).alias("full_closure"),
                   _first("segment", order).alias("place"))
              .join(stage, "event_id", "left")
              .withColumn("source", F.lit("cdot")))
    return events, p.select("event_id", "h3_r9").distinct()


def park_cells(polygons: DataFrame) -> DataFrame:
    """One row per park: park_no, its name, the r9 cells covering its polygon (the centroid's
    cell for parks too small to contain a cell center), and the mean of those cells' centers."""
    kind = F.from_json("geometry", "struct<type: string>").type
    ring = F.when(kind == "MultiPolygon",
                  F.from_json("geometry", "struct<coordinates: array<array<array<array<double>>>>>").coordinates[0][0]) \
        .otherwise(F.from_json("geometry", "struct<coordinates: array<array<array<double>>>>").coordinates[0])
    mean = lambda i: F.aggregate(ring, F.lit(0.0), lambda acc, pt: acc + pt[i]) / F.size(ring)
    parks = (polygons.select(F.expr("try_cast(park_no AS DOUBLE)").cast("int").alias("park_no"), "park", "geometry")
             .withColumn("polyfill", F.expr(f"h3_try_polyfillash3(geometry, {H3_RES})"))
             .withColumn("cells", F.when(F.size("polyfill") > 0, F.col("polyfill"))
                         .otherwise(F.array(_cell(mean(1), mean(0))))))
    centers = (parks.select("park_no", F.explode("cells").alias("h3_r9"))
               .withColumn("ll", F.from_json(F.call_function("h3_centerasgeojson", F.col("h3_r9")),
                                             "struct<coordinates: array<double>>").coordinates)
               .groupBy("park_no").agg(F.avg(F.col("ll")[1]).alias("lat"), F.avg(F.col("ll")[0]).alias("lon")))
    return parks.select("park_no", "park", "cells").join(centers, "park_no")


def park_events(raw: DataFrame, polygons: DataFrame, today: date) -> tuple[DataFrame, DataFrame]:
    et = F.col("event_type")
    p = (raw.filter(et.rlike("Event|Festival"))
         .withColumn("category", F.when(et.contains("Athletic"), "park_athletic")
                     .when(et.contains("Corporate"), "park_corporate")
                     .when(et.contains("Commemorative"), "park_commemorative").otherwise("park_event"))
         # "Event 3 Cluster 1", "Athletic Event Level 4" -> 3, 4. The 12,001+ tier was renamed
         # "Event 6/10,000+" in 2021, so both are 6. Cluster is the region, not the size.
         .withColumn("size_level", F.when(et.contains("12,001"), F.lit(6.0))
                     .otherwise(F.expr(r"try_cast(regexp_extract(event_type, '(?:Event|Level) (\\d)', 1) AS DOUBLE)")))
         .withColumn("day", _date("reservation_start_date"))
         .withColumn("park_no", _num("park_number").cast("int"))
         .withColumn("name", _strip(F.coalesce("event_description", "organization", F.lit("?"))))
         .filter(F.col("day").isNotNull() & F.col("park_no").isNotNull()))

    # One row per facility per day, so an event is a run of consecutive days with the same
    # park, name and category. A gap of more than a day starts a new event. Runs are numbered
    # in (park, category, name, day) order across the whole table, as the pandas cumsum does.
    key = ["park_no", "category", "name"]
    days = p.groupBy(*key, "day").agg(F.collect_set("permit_status").alias("statuses"),
                                      F.max("size_level").alias("size_level"),
                                      F.countDistinct("park_facility_name").alias("n_fac"))
    prev = F.lag("day").over(Window.partitionBy(*key).orderBy("day"))
    days = (days.withColumn("new_run", F.when(prev.isNull() | (F.datediff("day", prev) != 1), 1).otherwise(0))
            .withColumn("run", F.sum("new_run").over(Window.orderBy(*key, "day")
                                                     .rowsBetween(Window.unboundedPreceding, Window.currentRow))))
    subset = lambda allowed: F.size(F.array_except("statuses", F.array(*[F.lit(s) for s in sorted(allowed)]))) == 0
    events = (days.groupBy("run", *key)
              .agg(F.min("day").alias("start_date"), F.max("day").alias("end_date"),
                   F.max("size_level").alias("size_level"), F.max("n_fac").alias("n_facilities"),
                   F.array_distinct(F.flatten(F.collect_list("statuses"))).alias("statuses"))
              .withColumn("status", _status(F.col("start_date"), F.col("end_date"),
                                            subset(PARK_HELD), subset(PARK_PLANNED), today))
              .filter(F.col("status").isNotNull())
              .withColumn("event_id", F.concat_ws(":", F.lit("park"), F.col("park_no").cast("string"),
                                                  F.col("start_date").cast("string"), F.col("run").cast("string"))))

    parks = park_cells(polygons)
    events = (events.join(parks, "park_no")
              .select("event_id", "category", "name", "start_date", "end_date", "size_level", "n_facilities", "status",
                      "lat", "lon", F.col("park_no").cast("string").alias("venue"), F.col("park").alias("place"),
                      F.array_join(F.sort_array("statuses"), "/").alias("permit_stage"), F.lit("park").alias("source"),
                      "cells"))
    return events.drop("cells"), events.select("event_id", F.explode("cells").alias("h3_r9"))


def club_events(setlists: DataFrame, today: date) -> tuple[DataFrame, DataFrame]:
    """One event per club per date. setlist.fm lists each act separately, openers included; the
    act with the longest logged set is the best guess at the headliner."""
    spark = setlists.sparkSession
    clubs = spark.createDataFrame([(vid, key, name, lat, lon) for vid, key in CLUB_VENUE_IDS.items()
                                   for k, name, _, lat, lon, _ in CLUBS if k == key],
                                  "venue_id string, venue string, place string, lat double, lon double")
    s = (setlists.join(F.broadcast(clubs), "venue_id").dropDuplicates(["setlist_id"])
         .filter(F.col("event_date") >= F.lit(EVENTS_START)))
    headliner = F.when(F.col("artist").isNotNull(), F.struct(-F.coalesce("n_songs", F.lit(-1)), F.col("artist")))
    ev = (s.groupBy("venue", "event_date")
          .agg(F.min_by("artist", headliner).alias("name"), F.countDistinct("artist").alias("n_acts"),
               F.first("place").alias("place"), F.first("lat").alias("lat"), F.first("lon").alias("lon"))
          .withColumn("start_date", F.col("event_date")).withColumn("end_date", F.col("event_date"))
          .withColumn("status", F.when(F.col("event_date") < F.lit(today), "held").otherwise("planned"))
          .withColumn("event_id", F.concat_ws(":", F.lit("club"), "venue", F.col("event_date").cast("string")))
          .withColumn("category", F.lit("club_show")).withColumn("source", F.lit("setlistfm"))
          .drop("event_date"))
    return ev, ev.select("event_id", _cell("lat", "lon").alias("h3_r9"))


# ---------------------------------------------------------------------------------------------
# All together
# ---------------------------------------------------------------------------------------------

def build_events(cdot: DataFrame, park: DataFrame, polygons: DataFrame, setlists: DataFrame,
                 today: date) -> tuple[DataFrame, DataFrame]:
    """One row per event (EVENT_COLUMNS) and every r9 cell each event touches."""
    parts = [cdot_events(cdot, today), park_events(park, polygons, today), club_events(setlists, today)]
    cells = parts[0][1].unionByName(parts[1][1]).unionByName(parts[2][1]).distinct()
    events = parts[0][0].unionByName(parts[1][0], allowMissingColumns=True) \
        .unionByName(parts[2][0], allowMissingColumns=True)
    n_days = F.datediff("end_date", "start_date") + 1
    events = (events.withColumn("h3_r9", _cell("lat", "lon"))
              .join(cells.groupBy("event_id").agg(F.count(F.lit(1)).alias("n_cells")), "event_id")
              .withColumn("n_days", n_days)
              .withColumn("long_run", F.coalesce(n_days > MAX_EVENT_DAYS, F.lit(False))))
    return events.select(*[(F.col(c) if c in events.columns else F.lit(None)).cast(EVENT_TYPES[c]).alias(c)
                           for c in EVENT_COLUMNS]), cells


def to_pandas(events: DataFrame, cells: DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The built tables as the pandas look-ahead expects them (what events_model.build_events
    returns): plain dates, one row order."""
    ev, ce = events.toPandas(), cells.toPandas()
    for c in ["start_date", "end_date"]:
        ev[c] = pd.to_datetime(ev[c]).dt.date     # Arrow can hand DATE columns back as timestamps
    return ev.sort_values(["start_date", "event_id"]).reset_index(drop=True), \
        ce.sort_values(["event_id", "h3_r9"]).reset_index(drop=True)
