"""Measure how permitted events and club shows move crime and 311 in and around their cells.

    offline_ml/.venv/bin/python offline_ml/events_impact.py            # both runs, ~25 s
    offline_ml/.venv/bin/python offline_ml/events_impact.py clubs      # or one: permits | clubs

Two runs share one design:
    permits    CDOT and Park District events, counted over whole days.
    clubs      setlist.fm club shows, counted over the night (6pm-3am, keyed to the show's
               date), since setlist.fm has show dates but no times.

Design, observed vs. expected per event:
    exposure   Every r9 cell an event touches (ring 0) on each event day, plus the cells 1 and 2
               steps out (rings 1 and 2). Ring 0 is also checked the day before, the day after,
               and on a placebo date 5 weeks later, when nothing should show.
    expected   The same cell's mean on the same weekday 1-4 weeks either side (up to 8 days),
               skipping days with an event within one cell and one day. That mean is scaled by
               how the whole city's count on the day compared with its mean over those same
               control days, which absorbs holidays, weather and trend.
    result     Per event, observed and expected are summed over its cell-days. Per group,
               ratio = sum observed / sum expected, with a 90% interval from a Poisson bootstrap
               over events, plus excess incidents per event.

Outputs (offline_data/events/impact/), with a _clubs suffix for the club run:
    oe_by_event.parquet   observed and expected per event x ring x timing x outcome
    summary.csv           pooled ratios with intervals, per group x ring x timing x outcome
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import timedelta

import numpy as np
import polars as pl
from h3.api import basic_int as h3

from config import CLEAN, EVENTS

OUT = EVENTS / "impact"

RUNS = {  # name -> (event sources scored, count over the night instead of the whole day)
    "permits": (["cdot", "park"], False),
    "clubs": (["setlistfm"], True),
}
# Club nights run 6pm to 3am. Shifting timestamps back 3 hours puts the whole night on the
# show's date.
NIGHT_START, NIGHT_END = 18, 3
# How far around an event baseline days must stay clear. Permits cover whole days and can spill
# into setup and cleanup, so one day either side. A club show is one evening, and the busiest
# clubs log a show on a third of nights, so a buffer would leave almost nothing to compare with.
BUSY_BUFFER = {"setlistfm": [0]}
DEFAULT_BUFFER = [-1, 0, 1]
CONTROL_WEEKS = [-4, -3, -2, -1, 1, 2, 3, 4]
MIN_CONTROLS = 4                      # a cell-day with fewer clean control days is skipped
PLACEBO_DAYS = 35                     # outside the ±4-week control window, same weekday
BOOTSTRAP = 400
SEED = 0

# Crimes police mostly find by being there. A rise in these near events says more about
# deployment than about the crowd.
ENFORCEMENT = ["NARCOTICS", "WEAPONS VIOLATION", "PUBLIC PEACE VIOLATION", "LIQUOR LAW VIOLATION",
               "INTERFERENCE WITH PUBLIC OFFICER", "PROSTITUTION"]
# Every outcome must be a per-row expression. A bare literal like pl.lit(True) is evaluated once
# per group inside group_by().agg(), so its sum is 1 for every cell-day: it counted cell-days
# with any crime, not crimes.
CRIME_OUTCOMES = {
    "crime": pl.col("occurred_at").is_not_null(),
    "violent": pl.col("ucr_class") == "violent",
    "property": pl.col("ucr_class") == "property",
    "battery": pl.col("primary_type") == "BATTERY",
    "assault": pl.col("primary_type") == "ASSAULT",
    "theft": pl.col("primary_type") == "THEFT",
    "robbery": pl.col("primary_type") == "ROBBERY",
    "criminal_damage": pl.col("primary_type") == "CRIMINAL DAMAGE",
    "motor_vehicle_theft": pl.col("primary_type") == "MOTOR VEHICLE THEFT",
    "enforcement": pl.col("primary_type").cast(pl.String).is_in(ENFORCEMENT),
}
# When the excess lands. Rows with an unknown time (logged at 00:00/00:01) are in neither.
DAY_SPLITS = {
    "crime_10am_10pm": ~pl.col("time_unknown") & pl.col("occurred_at").dt.hour().is_between(10, 21),
    "crime_10pm_10am": ~pl.col("time_unknown") & ~pl.col("occurred_at").dt.hour().is_between(10, 21),
}
NIGHT_SPLITS = {   # night runs already drop unknown times
    "crime_6pm_midnight": pl.col("occurred_at").dt.hour() >= NIGHT_START,
    "crime_midnight_3am": pl.col("occurred_at").dt.hour() < NIGHT_END,
}
# Resident-reported requests only: city crews enter graffiti and other sweeps in bulk, which
# says nothing about what people nearby noticed.
SR_OUTCOMES = {
    "sr_resident": pl.col("created_at").is_not_null(),
    "sr_sanitation": pl.col("sr_type").cast(pl.String).is_in(
        ["Sanitation Code Violation", "Street Cleaning Request", "Fly Dumping Complaint"]),
    "sr_parking": pl.col("sr_type").cast(pl.String).is_in(
        ["Abandoned Vehicle Complaint", "Vehicle Parked in Bike Lane Complaint",
         "E-Scooter Parking Complaint", "Divvy Bike Parking Complaint"]),
    "sr_business": pl.col("sr_type").cast(pl.String).is_in(
        ["Business Complaints", "Liquor Establishment Complaint", "Restaurant Complaint",
         "Pushcart Food Vendor Complaint"]),
}


# --- outcomes --------------------------------------------------------------------------------

def daily_counts(path, ts: str, outcomes: dict[str, pl.Expr], where: pl.Expr, night: bool = False) -> pl.DataFrame:
    shifted = pl.col(ts) - pl.duration(hours=NIGHT_END) if night else pl.col(ts)
    lf = pl.scan_parquet(path).filter(where & pl.col("h3_r9").is_not_null())
    if night:
        lf = lf.filter(shifted.dt.hour() >= NIGHT_START - NIGHT_END)
    return (lf.with_columns(date=shifted.dt.date())
            .group_by("h3_r9", "date")
            .agg([e.cast(pl.UInt16).sum().alias(k) for k, e in outcomes.items()])
            .collect())


def load_outcomes(night: bool):
    crime_outcomes = CRIME_OUTCOMES | (NIGHT_SPLITS if night else DAY_SPLITS)
    crime_where = ~pl.col("date_is_month_placeholder") & (pl.col("occurred_at") >= pl.datetime(2013, 11, 1))
    if night:   # an unknown time is logged as midnight, which would land inside the night
        crime_where &= ~pl.col("time_unknown")
    crime = daily_counts(CLEAN / "crimes.parquet", "occurred_at", crime_outcomes, crime_where, night)
    sr = daily_counts(CLEAN / "sr311.parquet", "created_at", SR_OUTCOMES, pl.col("origin_group") == "resident", night)
    # the newest week of each file is still filling in (crime especially is reported late)
    crime_window = (crime["date"].min(), crime["date"].max() - timedelta(days=7))
    sr_window = (sr["date"].min(), sr["date"].max() - timedelta(days=7))
    return (crime, crime_window, list(crime_outcomes)), (sr, sr_window, list(SR_OUTCOMES))


# --- exposure --------------------------------------------------------------------------------

def event_cell_days(events: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    return (events.select("event_id", "start_date", "end_date")
            .with_columns(date=pl.date_ranges("start_date", "end_date"))
            .explode("date", empty_as_null=False)
            .join(cells, on="event_id")
            .select("event_id", "h3_r9", "date"))


def busy_cell_days(events: pl.DataFrame, cells: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(cell, date) pairs any held event occupies, and the wider busy set (within one cell, and
    within BUSY_BUFFER days) that baseline days must avoid. Long runs and club shows count here."""
    held = events.filter(pl.col("status") == "held")
    spans = (event_cell_days(held, cells).join(held.select("event_id", "source"), on="event_id")
             .select("h3_r9", "date", "source").unique())
    occupied = spans.select("h3_r9", "date").unique()
    disk = {c: h3.grid_disk(c, 1) for c in occupied["h3_r9"].unique().to_list()}
    near = pl.DataFrame([(c, n) for c, ns in disk.items() for n in ns],
                        schema={"src": pl.Int64, "h3_r9": pl.Int64}, orient="row")
    buffers = [(pl.col("source") == src, days) for src, days in BUSY_BUFFER.items()]
    spread = pl.concat_list([pl.col("date") + pl.duration(days=k) for k in DEFAULT_BUFFER])
    for is_src, days in buffers:
        spread = pl.when(is_src).then(pl.concat_list([pl.col("date") + pl.duration(days=k) for k in days])).otherwise(spread)
    busy = (spans.rename({"h3_r9": "src"}).join(near, on="src").drop("src")
            .with_columns(date=spread).explode("date", empty_as_null=False)
            .select("h3_r9", "date").unique())
    return occupied, busy


def build_exposures(events: pl.DataFrame, cells: pl.DataFrame, sources: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rows of (event_id, ring, timing, h3_r9, date) to score, and the busy (cell, date) set
    that control days must avoid."""
    occupied, busy = busy_cell_days(events, cells)
    held = events.filter(pl.col("status") == "held")
    scored = held.filter(pl.col("source").is_in(sources) & ~pl.col("long_run"))
    ring0 = cells.join(scored.select("event_id"), on="event_id")
    rows = []
    for eid, own in ring0.group_by("event_id").agg(pl.col("h3_r9")).iter_rows():
        own = set(own)
        r1 = set().union(*(h3.grid_ring(c, 1) for c in own)) - own
        r2 = set().union(*(h3.grid_ring(c, 2) for c in own)) - own - r1
        rows += [(eid, 1, c) for c in r1] + [(eid, 2, c) for c in r2]
    rings = pl.concat([ring0.with_columns(ring=pl.lit(0, pl.Int8)).select("event_id", "ring", "h3_r9"),
                       pl.DataFrame(rows, schema={"event_id": pl.String, "ring": pl.Int8, "h3_r9": pl.Int64},
                                    orient="row")])

    days = (scored.select("event_id", "start_date", "end_date")
            .with_columns(date=pl.date_ranges("start_date", "end_date")).explode("date", empty_as_null=False))
    on_days = rings.join(days.select("event_id", "date"), on="event_id").with_columns(timing=pl.lit("event"))
    shifted = []
    for timing, anchor, offset in [("day_before", "start_date", -1), ("day_after", "end_date", 1),
                                   ("placebo_5wk", "start_date", PLACEBO_DAYS)]:
        d = scored.select("event_id", date=pl.col(anchor) + pl.duration(days=offset))
        shifted.append(rings.filter(pl.col("ring") == 0).join(d, on="event_id").with_columns(timing=pl.lit(timing)))

    expo = pl.concat([on_days] + shifted, how="diagonal").select("event_id", "ring", "timing", "h3_r9", "date")
    # A ring cell or off-day that hosts some other event isn't a clean read of this one.
    clean = pl.concat([
        expo.filter((pl.col("ring") == 0) & (pl.col("timing") == "event")),
        expo.filter((pl.col("ring") > 0) | (pl.col("timing") != "event")).join(occupied, on=["h3_r9", "date"], how="anti"),
    ])
    return clean, busy


# --- observed vs expected --------------------------------------------------------------------

def observed_expected(expo, busy, daily, window, outcomes) -> pl.DataFrame:
    first, last = window
    expo = expo.filter(pl.col("date").is_between(first, last))
    city = daily.group_by("date").agg([pl.sum(k).cast(pl.Float64) for k in outcomes])
    offsets = pl.DataFrame({"k": [7 * w for w in CONTROL_WEEKS]})
    busy_c = busy.rename({"date": "ctrl_date"})
    daily_c = daily.rename({"date": "ctrl_date"})
    city_c = city.rename({"date": "ctrl_date"} | {k: f"city_{k}" for k in outcomes})

    parts = []
    keys_all = expo.select("h3_r9", "date").unique()
    for year in sorted(keys_all["date"].dt.year().unique().to_list()):
        keys = keys_all.filter(pl.col("date").dt.year() == year)
        ctrl = (keys.join(offsets, how="cross")
                .with_columns(ctrl_date=pl.col("date") + pl.duration(days=pl.col("k")))
                .filter(pl.col("ctrl_date").is_between(first, last))
                .join(busy_c, on=["h3_r9", "ctrl_date"], how="anti")
                .join(daily_c, on=["h3_r9", "ctrl_date"], how="left")
                .join(city_c, on="ctrl_date", how="left")
                .fill_null(0))
        base = (ctrl.group_by("h3_r9", "date")
                .agg(pl.len().alias("n_ctrl"), *[pl.mean(k) for k in outcomes],
                     *[pl.mean(f"city_{k}") for k in outcomes])
                .filter(pl.col("n_ctrl") >= MIN_CONTROLS)
                .join(city.rename({k: f"today_{k}" for k in outcomes}), on="date", how="left"))
        base = base.select("h3_r9", "date", *[
            (pl.col(k) * pl.when(pl.col(f"city_{k}") > 0)
             .then(pl.col(f"today_{k}") / pl.col(f"city_{k}")).otherwise(1.0)).alias(f"E_{k}")
            for k in outcomes])
        obs = keys.join(daily, on=["h3_r9", "date"], how="left").fill_null(0).rename({k: f"O_{k}" for k in outcomes})
        parts.append(base.join(obs, on=["h3_r9", "date"]))
    cd = pl.concat(parts)

    return (expo.join(cd, on=["h3_r9", "date"])
            .group_by("event_id", "ring", "timing")
            .agg(pl.len().alias("cell_days"), *[pl.sum(f"O_{k}").cast(pl.Float64) for k in outcomes],
                 *[pl.sum(f"E_{k}") for k in outcomes]))


# --- pooled summaries ------------------------------------------------------------------------

def permit_groups(events: pl.DataFrame) -> dict[str, pl.Expr]:
    g = {c: pl.col("category") == c for c in events.filter(pl.col("source").is_in(RUNS["permits"][0]))["category"].unique().sort()}
    g |= {
        "festival: full closure": (pl.col("category") == "festival") & pl.col("full_closure"),
        "festival: no full closure": (pl.col("category") == "festival") & ~pl.col("full_closure"),
        "festival: 1 street segment": (pl.col("category") == "festival") & (pl.col("n_segments") == 1),
        "festival: 2-4 segments": (pl.col("category") == "festival") & pl.col("n_segments").is_between(2, 4),
        "festival: 5+ segments": (pl.col("category") == "festival") & (pl.col("n_segments") >= 5),
        "park_event: level 6 (10,000+)": (pl.col("category") == "park_event") & (pl.col("size_level") == 6),
        "park_event: level 5": (pl.col("category") == "park_event") & (pl.col("size_level") == 5),
        "park_event: levels 1-4": (pl.col("category") == "park_event") & (pl.col("size_level") <= 4),
    }
    return g


def club_groups(events: pl.DataFrame) -> dict[str, pl.Expr]:
    show = pl.col("category") == "club_show"
    weekend = pl.col("start_date").dt.weekday().is_in([5, 6])     # Friday and Saturday nights
    g = {
        "club_show": show,
        # setlist.fm lists each act, so 2+ acts usually means a headliner with openers
        "club: 2+ acts": show & (pl.col("n_acts") >= 2),
        "club: 1 act": show & (pl.col("n_acts") == 1),
        "club: Fri-Sat": show & weekend,
        "club: Sun-Thu": show & ~weekend,
    }
    places = events.filter(pl.col("source") == "setlistfm")["place"].unique().sort()
    return g | {f"club: {p}": show & (pl.col("place") == p) for p in places}


def summarize(oe: pl.DataFrame, events: pl.DataFrame, outcomes: list[str], groups: dict[str, pl.Expr]) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    meta = events.select("event_id", "category", "size_level", "full_closure", "n_segments", "n_acts", "place", "start_date")
    oe = oe.join(meta, on="event_id")
    rows = []
    for label, mask in groups.items():
        sub = oe.filter(mask)
        for (ring, timing), s in sub.group_by("ring", "timing"):
            O = s.select([f"O_{k}" for k in outcomes]).to_numpy()
            E = s.select([f"E_{k}" for k in outcomes]).to_numpy()
            w = rng.poisson(1.0, size=(BOOTSTRAP, len(s))).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # outcomes with no expected count
                boot = (w @ O) / (w @ E)
                lo, hi = np.nanquantile(boot, [0.05, 0.95], axis=0)
            for i, k in enumerate(outcomes):
                rows.append({"group": label, "ring": ring, "timing": timing, "outcome": k,
                             "n_events": len(s), "cell_days": int(s["cell_days"].sum()),
                             "observed": O[:, i].sum(), "expected": E[:, i].sum(),
                             "ratio": O[:, i].sum() / E[:, i].sum() if E[:, i].sum() else np.nan,
                             "lo90": lo[i], "hi90": hi[i],
                             "excess_per_event": (O[:, i].sum() - E[:, i].sum()) / len(s)})
    return pl.DataFrame(rows)


def show(summary: pl.DataFrame, ring: int, timing: str, outcomes: list[str], title: str) -> None:
    s = (summary.filter((pl.col("ring") == ring) & (pl.col("timing") == timing) & pl.col("outcome").is_in(outcomes))
         .with_columns(cell=pl.format("{} [{}–{}]", pl.col("ratio").round(2), pl.col("lo90").round(2), pl.col("hi90").round(2))))
    wide = s.pivot("outcome", index=["group", "n_events"], values="cell").select("group", "n_events", *outcomes).sort("group")
    print(f"\n{title}"), print(wide)


def run(name: str, events: pl.DataFrame, cells: pl.DataFrame) -> None:
    t0 = time.time()
    sources, night = RUNS[name]
    suffix = "" if name == "permits" else f"_{name}"
    expo, busy = build_exposures(events, cells, sources)
    print(f"\n=== {name}: exposure rows {expo.height:,}, busy cell-days {busy.height:,}")

    groups = permit_groups(events) if name == "permits" else club_groups(events)
    oe_parts, summaries = [], []
    for daily, window, outcomes in load_outcomes(night):
        oe = observed_expected(expo, busy, daily, window, outcomes)
        print(f"{outcomes[0]}: window {window[0]}..{window[1]}, {oe.height:,} event x ring x timing rows ({time.time() - t0:.0f}s)")
        oe_parts.append(oe)
        summaries.append(summarize(oe, events, outcomes, groups))

    oe = oe_parts[0].join(oe_parts[1], on=["event_id", "ring", "timing"], how="full", coalesce=True, suffix="_sr")
    oe.write_parquet(OUT / f"oe_by_event{suffix}.parquet")
    summary = pl.concat(summaries)
    summary.write_csv(OUT / f"summary{suffix}.csv")

    main_cols = ["crime", "violent", "property", "enforcement"]
    when = ["crime_6pm_midnight", "crime_midnight_3am"] if night else ["crime_10am_10pm", "crime_10pm_10am"]
    sr_cols = ["sr_resident", "sr_sanitation", "sr_parking", "sr_business"]
    span = "nights" if night else "days"
    show(summary, 0, "event", main_cols, f"Ring 0, event {span}: crime observed/expected [90% interval]")
    show(summary, 0, "event", when + ["battery", "theft"], f"Ring 0, event {span}: when and what")
    show(summary, 0, "event", sr_cols, f"Ring 0, event {span}: resident 311 observed/expected (2019-03 on)")
    show(summary, 0, "placebo_5wk", ["crime", "violent"], "Placebo, 5 weeks later: crime (should be ~1.0)")
    show(summary, 0, "placebo_5wk", ["sr_resident"], "Placebo, 5 weeks later: 311 (should be ~1.0)")
    for ring in (1, 2):
        show(summary, ring, "event", main_cols[:2], f"Ring {ring}, event {span}: crime")
        show(summary, ring, "event", sr_cols[:2], f"Ring {ring}, event {span}: 311")
    for timing in ("day_before", "day_after"):
        show(summary, 0, timing, main_cols[:2], f"Ring 0, {timing}: crime")
        show(summary, 0, timing, sr_cols[:2], f"Ring 0, {timing}: 311")
    print(f"{name} done in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", choices=[*RUNS, "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    events = pl.read_parquet(EVENTS / "events.parquet")
    cells = pl.read_parquet(EVENTS / "event_cells.parquet")
    pl.Config.set_tbl_rows(40), pl.Config.set_tbl_width_chars(250), pl.Config.set_fmt_str_lengths(40)
    for name in RUNS if args.run == "all" else [args.run]:
        run(name, events, cells)


if __name__ == "__main__":
    main()
