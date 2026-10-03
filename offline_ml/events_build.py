"""Collapse the raw event pulls into one event table keyed to H3 cells.

    offline_ml/.venv/bin/python offline_ml/events_build.py

Inputs (offline_data/events/raw/, from events_fetch.py). Outputs (offline_data/events/):
    events.parquet        one row per event: source, category, dates, status, size, centroid cell
    event_cells.parquet   every r9 cell each event touches (street segments, park polygons, venue)

Every rule below came from profiling the raw pulls; see offline_ml/README.md for the numbers.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
from h3.api import basic_int as h3

from config import EVENTS, EVENTS_START
from events_fetch import CLUBS, RAW

H3_RES = 9
TODAY = date.today()

# Permit dates are whole days (no CDOT permit has a start hour). A run longer than this is a
# reservation window (setup and teardown, a season-long market), not the days the crowd is
# there: the Park District holds Grant Park for 27 days around Lollapalooza's 4, while CDOT's
# street permits for it cover exactly the 4. Long runs stay in the table flagged long_run so
# analyses can skip them.
MAX_EVENT_DAYS = 4

# "Street Closure" is left out: its permits are bridge work, sign dedications and graduations,
# with a median span of 8 days. Everything else CDOT pulled is a gathering.
CDOT_CATEGORY = {"Festival": "festival", "Block Party": "block_party", "Parade": "parade",
                 "Athletic": "athletic", "Assembly": "assembly"}
# Block parties are all named "Block Party"-ish, so only the other types merge applications
# that share a name and dates (Lollapalooza files several a year).
CDOT_MERGE_BY_NAME = {"festival", "parade", "athletic", "assembly"}

# A few applications carry typo'd years (the latest block party is dated 2112).
PLANNED_HORIZON = date(TODAY.year + 2, 12, 31)

PARK_HELD = ["Approved", "Completed", "Issued"]
PARK_PLANNED = PARK_HELD + ["Tentative"]   # a past permit still Tentative never happened


def status_expr(held: pl.Expr, planned: pl.Expr) -> pl.Expr:
    """'held' if it's over and the permit went through, 'planned' if it's still ahead, else null."""
    return (pl.when((pl.col("end_date") < TODAY) & held).then(pl.lit("held"))
            .when((pl.col("start_date") >= TODAY) & planned).then(pl.lit("planned")))


def cell_of(lat: pl.Expr, lon: pl.Expr) -> pl.Expr:
    return pl.struct(lat.alias("a"), lon.alias("o")).map_elements(
        lambda s: h3.latlng_to_cell(s["a"], s["o"], H3_RES), return_dtype=pl.Int64)


# --- CDOT street permits -------------------------------------------------------------------

def cdot_events() -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pl.read_parquet(RAW / "cdot_permits.parquet")
    p = (raw
         .filter(pl.col("worktypedescription").is_in(list(CDOT_CATEGORY)))
         .filter(~pl.col("currentmilestone").is_in(["Cancelled", "Denied"]))
         .with_columns(
             category=pl.col("worktypedescription").replace_strict(CDOT_CATEGORY),
             start_date=pl.col("applicationstartdate").str.to_datetime(strict=False).dt.date(),
             end_date=pl.col("applicationenddate").str.to_datetime(strict=False).dt.date(),
             name=pl.col("applicationname").str.strip_chars(),
             lat=pl.col("latitude").cast(pl.Float64), lon=pl.col("longitude").cast(pl.Float64),
             segment=pl.concat_str(
                 pl.when(pl.col("streetnumberfrom") == pl.col("streetnumberto")).then(pl.col("streetnumberfrom"))
                 .otherwise(pl.concat_str("streetnumberfrom", "streetnumberto", separator="-")),
                 "direction", "streetname", "suffix", separator=" ", ignore_nulls=True))
         .with_columns(status=status_expr(pl.col("currentmilestone") == "Complete", pl.lit(True)))
         .filter(pl.col("status").is_not_null() & pl.col("lat").is_not_null()
                 & (pl.col("start_date") <= PLANNED_HORIZON)))

    by_name = pl.col("category").is_in(CDOT_MERGE_BY_NAME) & pl.col("name").is_not_null()
    p = p.with_columns(group=pl.when(by_name)
                       .then(pl.concat_str("category", pl.col("name").str.to_lowercase(),
                                           pl.col("start_date").cast(pl.String),
                                           pl.col("end_date").cast(pl.String), separator="|"))
                       .otherwise(pl.col("applicationnumber")))
    p = p.with_columns(event_id=pl.lit("cdot:") + pl.col("applicationnumber").min().over("group"),
                       h3_r9=cell_of(pl.col("lat"), pl.col("lon")))

    events = p.group_by("event_id").agg(
        pl.first("category"), pl.first("name"), pl.first("start_date"), pl.first("end_date"),
        pl.first("status"), pl.mean("lat"), pl.mean("lon"),
        n_segments=pl.len(), full_closure=(pl.col("streetclosure") == "Full").any(),
        place=pl.col("segment").sort_by("applicationnumber", "uniquekey").first(),
        permit_stage=pl.col("currentmilestone").mode().first())
    cells = p.select("event_id", "h3_r9").unique()
    return events.with_columns(source=pl.lit("cdot")), cells


# --- Park District permits -----------------------------------------------------------------

def park_cells() -> tuple[dict[int, list[int]], dict[int, str]]:
    """Park number -> r9 cells covering its polygon (the centroid's cell for parks too small
    to contain a cell center), and park number -> park name."""
    out, names = {}, {}
    for f in json.loads((RAW / "park_polygons.geojson").read_text())["features"]:
        no, geom = f["properties"].get("park_no"), f["geometry"]
        if no is None or geom is None:
            continue
        cells = list(h3.geo_to_cells(geom, H3_RES))
        if not cells:
            ring = geom["coordinates"][0][0] if geom["type"] == "MultiPolygon" else geom["coordinates"][0]
            cells = [h3.latlng_to_cell(sum(p[1] for p in ring) / len(ring),
                                       sum(p[0] for p in ring) / len(ring), H3_RES)]
        out[int(float(no))] = cells
        names[int(float(no))] = f["properties"].get("park")
    return out, names


def park_events() -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pl.read_parquet(RAW / "park_permits.parquet")
    p = (raw
         .filter(pl.col("event_type").str.contains(r"Event|Festival"))
         .with_columns(
             category=pl.when(pl.col("event_type").str.contains("Athletic")).then(pl.lit("park_athletic"))
             .when(pl.col("event_type").str.contains("Corporate")).then(pl.lit("park_corporate"))
             .when(pl.col("event_type").str.contains("Commemorative")).then(pl.lit("park_commemorative"))
             .otherwise(pl.lit("park_event")),
             # "Event 3 Cluster 1", "Athletic Event Level 4" -> 3, 4. The 12,001+ tier was renamed
             # "Event 6/10,000+" in 2021, so both are 6. Cluster is the region, not the size.
             size_level=pl.when(pl.col("event_type").str.contains("12,001")).then(6)
             .otherwise(pl.col("event_type").str.extract(r"(?:Event|Level) (\d)").cast(pl.Int8)).cast(pl.Int8),
             day=pl.col("reservation_start_date").str.to_datetime(strict=False).dt.date(),
             park_no=pl.col("park_number").cast(pl.Float64).cast(pl.Int64),
             name=pl.coalesce(pl.col("event_description"), pl.col("organization"), pl.lit("?")).str.strip_chars()))

    # One row per facility per day, so an event is a run of consecutive days with the same
    # park, name and category. A gap of more than a day starts a new event.
    key = ["park_no", "category", "name"]
    days = (p.group_by(key + ["day"])
            .agg(pl.col("permit_status"), pl.max("size_level"), n_fac=pl.col("park_facility_name").n_unique())
            .sort(key + ["day"])
            .with_columns(new=(pl.col("day").diff().dt.total_days() != 1).fill_null(True).over(key))
            .with_columns(run=pl.col("new").cum_sum()))
    events = (days.group_by("run").agg(
        pl.first("park_no"), pl.first("category"), pl.first("name"),
        start_date=pl.min("day"), end_date=pl.max("day"), size_level=pl.max("size_level"),
        n_facilities=pl.max("n_fac"), statuses=pl.col("permit_status").list.explode(keep_nulls=False, empty_as_null=False).unique()))
    events = (events
              .with_columns(status=status_expr(
                  pl.col("statuses").list.eval(pl.element().is_in(PARK_HELD)).list.all(),
                  pl.col("statuses").list.eval(pl.element().is_in(PARK_PLANNED)).list.all()))
              .filter(pl.col("status").is_not_null())
              .with_columns(event_id=pl.concat_str(pl.lit("park"), "park_no", "start_date", "run", separator=":")))

    by_park, park_names = park_cells()
    cells = (events.select("event_id", "park_no")
             .with_columns(h3_r9=pl.col("park_no").map_elements(lambda n: by_park.get(n), return_dtype=pl.List(pl.Int64)))
             .drop_nulls("h3_r9").explode("h3_r9", empty_as_null=False).select("event_id", "h3_r9"))
    centroids = (cells.with_columns(ll=pl.col("h3_r9").map_elements(
                     lambda c: list(h3.cell_to_latlng(c)), return_dtype=pl.List(pl.Float64)))
                 .group_by("event_id").agg(lat=pl.col("ll").list.get(0).mean(), lon=pl.col("ll").list.get(1).mean()))
    unplaced = events.join(centroids, on="event_id", how="anti").height
    print(f"  park events with no polygon for their park number (dropped): {unplaced:,}")
    events = (events.join(centroids, on="event_id")
              .with_columns(venue=pl.col("park_no").cast(pl.String), source=pl.lit("park"),
                            place=pl.col("park_no").replace_strict(park_names, default=None, return_dtype=pl.String),
                            permit_stage=pl.col("statuses").list.sort().list.join("/"))
              .drop("run", "statuses", "park_no"))
    return events, cells


# --- club shows (setlist.fm) ---------------------------------------------------------------

def club_events() -> tuple[pl.DataFrame, pl.DataFrame]:
    """One event per club per date. setlist.fm lists each act separately, openers included."""
    rows, done, pending = [], [], []
    for key, name, _, lat, lon, ids in CLUBS:
        if not all((RAW / "setlistfm" / vid / "_done").exists() for vid in ids):
            pending.append(name)   # a half-fetched club only has its recent years; leave it out
            continue
        done.append(name)
        for vid in ids:
            for page in sorted((RAW / "setlistfm" / vid).glob("p*.json")):
                for s in json.loads(page.read_text()).get("setlist", []):
                    rows.append({
                        "setlist_id": s["id"], "club": key, "venue_name": name, "lat": lat, "lon": lon,
                        "day": datetime.strptime(s["eventDate"], "%d-%m-%Y").date(),
                        "artist": s["artist"]["name"],
                        "n_songs": sum(len(st.get("song", [])) for st in s["sets"].get("set", []))})
    print(f"  clubs complete: {len(done)}; still fetching (left out): {', '.join(pending) or 'none'}")
    if not rows:
        return pl.DataFrame(), pl.DataFrame()

    s = (pl.from_dicts(rows).unique("setlist_id")
         .filter(pl.col("day") >= EVENTS_START)
         # the act with the longest logged set is the best guess at the headliner; most logged
         # setlists are placeholders with no songs, so this is often just the first act listed
         .sort("n_songs", descending=True))
    events = (s.group_by("club", "day").agg(
                  pl.first("venue_name"), pl.first("lat"), pl.first("lon"),
                  name=pl.first("artist"), n_acts=pl.col("artist").n_unique())
              .with_columns(start_date=pl.col("day"), end_date=pl.col("day"))
              .with_columns(status=status_expr(pl.lit(True), pl.lit(True)),
                            event_id=pl.concat_str(pl.lit("club"), "club", pl.col("day").cast(pl.String), separator=":"),
                            h3_r9=cell_of(pl.col("lat"), pl.col("lon")),
                            category=pl.lit("club_show"), source=pl.lit("setlistfm"))
              .filter(pl.col("status").is_not_null())
              .rename({"club": "venue", "venue_name": "place"}).drop("day"))
    return events.drop("h3_r9"), events.select("event_id", "h3_r9")


def main() -> None:
    EVENTS.mkdir(parents=True, exist_ok=True)
    parts = []
    for label, fn in [("cdot", cdot_events), ("park", park_events), ("clubs", club_events)]:
        print(label)
        ev, cells = fn()
        if ev.height:
            parts.append((ev, cells))

    cells = pl.concat([c for _, c in parts]).unique()
    centroid = (pl.concat([e.select("event_id", "lat", "lon") for e, _ in parts])
                .with_columns(h3_r9=cell_of(pl.col("lat"), pl.col("lon"))))
    events = (pl.concat([e for e, _ in parts], how="diagonal_relaxed")
              .drop("lat", "lon").join(centroid, on="event_id")
              .join(cells.group_by("event_id").agg(n_cells=pl.len()), on="event_id")
              .with_columns(n_days=(pl.col("end_date") - pl.col("start_date")).dt.total_days() + 1)
              .with_columns(long_run=pl.col("n_days") > MAX_EVENT_DAYS)
              .select("event_id", "source", "category", "name", "venue", "place", "start_date", "end_date",
                      "n_days", "long_run", "status", "permit_stage", "size_level", "n_segments", "full_closure",
                      "n_facilities", "n_acts", "lat", "lon", "h3_r9", "n_cells")
              .sort("start_date", "event_id"))

    events.write_parquet(EVENTS / "events.parquet")
    cells.write_parquet(EVENTS / "event_cells.parquet")

    pl.Config.set_tbl_rows(30)
    print(f"\nevents.parquet: {events.height:,} events, event_cells.parquet: {cells.height:,} rows")
    print(events.group_by("source", "category").agg(
        held=(pl.col("status") == "held").sum(), planned=(pl.col("status") == "planned").sum(),
        long_run=pl.col("long_run").sum(), first=pl.min("start_date"), last=pl.max("start_date"),
        median_cells=pl.median("n_cells")).sort("source", "category"))


if __name__ == "__main__":
    main()
