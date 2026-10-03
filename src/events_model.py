"""The events look-ahead: upcoming street festivals and club nights, and the extra crime they tend
to bring. The logic behind notebook 08b, from the research in offline_ml/ (the story is in
offline_ml/events_walkthrough.ipynb).

- **Build:** permits and setlists become one event table keyed to H3 resolution-9 cells, by
  rules profiled on the raw data:
  - CDOT applications for the same festival (same name and dates) merge into one event.
  - Park permits (one row per facility per day) become runs of consecutive days.
  - A club's logged shows become one event per club per night.
  - Runs longer than 4 days are reservation windows (setup, a season-long market), not crowd
    days, so they're flagged `long_run` and not scored.
- **Past events:** observed vs expected crime in each event's own cells and the ring around
  them. Expected comes from clean control days (same weekday 1-4 weeks either side), scaled by
  how the whole city did that day.
- **Upcoming events:** extra crime = baseline x (ratio - 1).
  - baseline: each cell's mean on the same weekday +-4 weeks in each of the last 3 years
  - ratio: the event's own record pooled with a prior (Gamma-Poisson)
    - festivals: past editions (same name, within 1.5 km); prior = the average for festivals
      its size
    - club nights: the club's past nights; prior = the all-club average
  - The prior's weight comes from how much events truly differ beyond noise. Festivals differ a
    lot (pub crawls), so a few editions mostly speak for themselves. Clubs barely differ.
  - Backtest (2017-2025, scored as of each Jan 1): the top 20% of this ranking captured 74% of
    the extra crime, against 69% for size priors alone.

**Baseline and control days skip "busy" cell-days**: a day with any held event within one cell,
plus the day either side for permits. Block parties alone put ~48,000 cell-days in that set.

Pure pandas, h3 and scipy, so it runs (and is tested) anywhere. Notebook 08b runs the build in
Spark (src/events_spark.py, checked against this one by tests/test_events_spark.py); this
version is the reference, and runs the look-ahead.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from h3.api import basic_int as h3
from scipy.stats import gamma

from src.events_sources import CLUB_VENUE_IDS, CLUBS, EVENTS_START

H3_RES = 9
# The community-area boundaries the app draws (from the city data portal, igwz-8jzy).
COMMUNITY_AREAS_GEOJSON = Path(__file__).resolve().parent.parent / "webapp" / "static" / "data" / "community_areas.geojson"
MAX_EVENT_DAYS = 4
CDOT_CATEGORY = {"Festival": "festival", "Block Party": "block_party", "Parade": "parade",
                 "Athletic": "athletic", "Assembly": "assembly"}
# Block parties are all named "Block Party"-ish, so only the other types merge applications
# that share a name and dates (Lollapalooza files several a year).
CDOT_MERGE_BY_NAME = {"festival", "parade", "athletic", "assembly"}
PARK_HELD = {"Approved", "Completed", "Issued"}
PARK_PLANNED = PARK_HELD | {"Tentative"}   # a past permit still Tentative never happened

CONTROL_WEEKS = (-4, -3, -2, -1, 1, 2, 3, 4)   # a past event is compared with its cells on these weeks
MIN_CONTROLS = 4               # a cell-day with fewer clean control days is skipped
SETTLE_DAYS = 7 * max(CONTROL_WEEKS)   # a past event counts as history once its control weeks are known
OE_START = date(2013, 11, 1)   # crime counts used for past events' observed vs expected start here
MIN_CV = 0.05                  # floor on the between-group spread, so a prior never becomes absolute
MATCH_KM = 1.5                 # "Oktoberfest" in Edison Park isn't the one in Lincoln Square
BASELINE_YEARS = 3
BASELINE_WEEKS = range(-4, 5)
MIN_BASELINE_DAYS = 8          # of up to 27; fewer falls back to the cell's plain daily mean
PLACEHOLDER_NAMES = {"test"}   # applications named "test" or "k" aren't events
ADDRESS_SLACK = 10             # house numbers that still count as one venue (Reggies spans 2105-2109)

EVENT_COLUMNS = ["event_id", "source", "category", "name", "venue", "place", "start_date", "end_date", "n_days",
                 "long_run", "status", "permit_stage", "size_level", "n_segments", "full_closure", "n_facilities",
                 "n_acts", "lat", "lon", "h3_r9", "n_cells"]


def _cell(lat: float, lon: float) -> int:
    return h3.latlng_to_cell(lat, lon, H3_RES)


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _status(start: pd.Series, end: pd.Series, held: pd.Series, planned: pd.Series, today: date) -> pd.Series:
    """'held' if it's over and the permit went through, 'planned' if it's still ahead, else None."""
    out = pd.Series(None, index=start.index, dtype=object)
    out[(end < today) & held] = "held"
    out[(start >= today) & planned] = "planned"
    return out


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------

def cdot_events(raw: pd.DataFrame, today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = raw[raw["worktypedescription"].isin(CDOT_CATEGORY) & ~raw["currentmilestone"].isin(["Cancelled", "Denied"])].copy()
    p["category"] = p["worktypedescription"].map(CDOT_CATEGORY)
    p["start_date"], p["end_date"] = _date(p["applicationstartdate"]), _date(p["applicationenddate"])
    p["name"] = p["applicationname"].str.strip()
    p["lat"], p["lon"] = pd.to_numeric(p["latitude"], errors="coerce"), pd.to_numeric(p["longitude"], errors="coerce")
    number = np.where(p["streetnumberfrom"] == p["streetnumberto"], p["streetnumberfrom"],
                      p["streetnumberfrom"].fillna("") + "-" + p["streetnumberto"].fillna(""))
    p["segment"] = [" ".join(x for x in parts if isinstance(x, str) and x)
                    for parts in zip(number, p["direction"], p["streetname"], p["suffix"])]
    p["status"] = _status(p["start_date"], p["end_date"], p["currentmilestone"] == "Complete",
                          pd.Series(True, index=p.index), today)
    horizon = date(today.year + 2, 12, 31)          # a few applications carry typo'd years (2112)
    p = p[p["status"].notna() & p["lat"].notna() & p["start_date"].notna() & (p["start_date"] <= horizon)].copy()

    by_name = p["category"].isin(CDOT_MERGE_BY_NAME) & p["name"].notna()
    p["group"] = np.where(by_name, p["category"] + "|" + p["name"].str.lower().fillna("") + "|"
                          + p["start_date"].astype(str) + "|" + p["end_date"].astype(str), p["applicationnumber"])
    p["event_id"] = "cdot:" + p.groupby("group")["applicationnumber"].transform("min")
    p["h3_r9"] = [_cell(a, o) for a, o in zip(p["lat"], p["lon"])]
    p = p.sort_values(["applicationnumber", "uniquekey"])
    g = p.groupby("event_id", sort=False)
    events = pd.DataFrame({
        "category": g["category"].first(), "name": g["name"].first(), "start_date": g["start_date"].first(),
        "end_date": g["end_date"].first(), "status": g["status"].first(), "lat": g["lat"].mean(), "lon": g["lon"].mean(),
        "n_segments": g.size(), "full_closure": g["streetclosure"].agg(lambda s: bool((s == "Full").any())),
        "place": g["segment"].first(), "permit_stage": g["currentmilestone"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None),
    }).reset_index()
    events["source"] = "cdot"
    return events, p[["event_id", "h3_r9"]].drop_duplicates()


def park_cells(polygons: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Park number -> r9 cells covering its polygon (the centroid's cell for parks too small to
    contain a cell center), and park number -> name."""
    out, names = {}, {}
    for no, name, geom in polygons[["park_no", "park", "geometry"]].itertuples(index=False):
        g = json.loads(geom) if isinstance(geom, str) else geom
        cells = list(h3.geo_to_cells(g, H3_RES))
        if not cells:
            ring = g["coordinates"][0][0] if g["type"] == "MultiPolygon" else g["coordinates"][0]
            cells = [_cell(sum(pt[1] for pt in ring) / len(ring), sum(pt[0] for pt in ring) / len(ring))]
        out[int(float(no))] = cells
        names[int(float(no))] = name
    return out, names


def park_events(raw: pd.DataFrame, polygons: pd.DataFrame, today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = raw[raw["event_type"].str.contains(r"Event|Festival", regex=True, na=False)].copy()
    et = p["event_type"]
    p["category"] = np.select([et.str.contains("Athletic"), et.str.contains("Corporate"), et.str.contains("Commemorative")],
                              ["park_athletic", "park_corporate", "park_commemorative"], "park_event")
    # "Event 3 Cluster 1", "Athletic Event Level 4" -> 3, 4. The 12,001+ tier was renamed
    # "Event 6/10,000+" in 2021, so both are 6. Cluster is the region, not the size.
    level = pd.to_numeric(et.str.extract(r"(?:Event|Level) (\d)")[0], errors="coerce")
    p["size_level"] = np.where(et.str.contains("12,001"), 6, level)
    p["day"] = _date(p["reservation_start_date"])
    p["park_no"] = pd.to_numeric(p["park_number"], errors="coerce")
    p["name"] = p["event_description"].fillna(p["organization"]).fillna("?").str.strip()
    p = p[p["day"].notna() & p["park_no"].notna()]
    if p.empty:
        return pd.DataFrame(columns=["event_id", "category", "name", "start_date", "end_date", "status", "lat", "lon"]), \
            pd.DataFrame(columns=["event_id", "h3_r9"])
    p["park_no"] = p["park_no"].astype(int)

    # One row per facility per day, so an event is a run of consecutive days with the same
    # park, name and category. A gap of more than a day starts a new event.
    key = ["park_no", "category", "name"]
    days = (p.groupby(key + ["day"])
            .agg(statuses=("permit_status", lambda s: set(s.dropna())), size_level=("size_level", "max"),
                 n_fac=("park_facility_name", "nunique"))
            .reset_index().sort_values(key + ["day"]))
    gap = days.groupby(key)["day"].diff().map(lambda d: None if pd.isna(d) else d.days)
    days["run"] = (gap != 1).cumsum()
    g = days.groupby("run")
    events = pd.DataFrame({
        "park_no": g["park_no"].first(), "category": g["category"].first(), "name": g["name"].first(),
        "start_date": g["day"].min(), "end_date": g["day"].max(), "size_level": g["size_level"].max(),
        "n_facilities": g["n_fac"].max(), "statuses": g["statuses"].agg(lambda s: set().union(*s)),
    }).reset_index()
    events["status"] = _status(events["start_date"], events["end_date"],
                               events["statuses"].map(lambda s: s <= PARK_HELD),
                               events["statuses"].map(lambda s: s <= PARK_PLANNED), today)
    events = events[events["status"].notna()].copy()
    events["event_id"] = ("park:" + events["park_no"].astype(str) + ":" + events["start_date"].astype(str)
                          + ":" + events["run"].astype(str))

    by_park, names = park_cells(polygons)
    cells = pd.DataFrame([(e, c) for e, n in zip(events["event_id"], events["park_no"]) for c in by_park.get(n, [])],
                         columns=["event_id", "h3_r9"])
    centers = cells.assign(ll=[h3.cell_to_latlng(c) for c in cells["h3_r9"]])
    centroid = centers.groupby("event_id")["ll"].agg(lambda s: tuple(np.mean(list(s), axis=0))).rename("ll")
    events = events.join(centroid, on="event_id", how="inner")
    events["lat"] = [ll[0] for ll in events["ll"]]
    events["lon"] = [ll[1] for ll in events["ll"]]
    events["venue"] = events["park_no"].astype(str)
    events["place"] = events["park_no"].map(names)
    events["permit_stage"] = events["statuses"].map(lambda s: "/".join(sorted(s)))
    events["source"] = "park"
    return events.drop(columns=["run", "statuses", "park_no", "ll"]), cells


def club_events(setlists: pd.DataFrame, today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One event per club per date. setlist.fm lists each act separately, openers included; the
    act with the longest logged set is the best guess at the headliner."""
    clubs = {key: (name, lat, lon) for key, name, _, lat, lon, _ in CLUBS}
    s = setlists[setlists["venue_id"].isin(CLUB_VENUE_IDS)].drop_duplicates("setlist_id").copy()
    s["club"] = s["venue_id"].map(CLUB_VENUE_IDS)
    s = s[s["event_date"] >= EVENTS_START].sort_values("n_songs", ascending=False, kind="stable")
    g = s.groupby(["club", "event_date"])
    ev = pd.DataFrame({"name": g["artist"].first(), "n_acts": g["artist"].nunique()}).reset_index()
    ev["place"] = ev["club"].map(lambda k: clubs[k][0])
    ev["lat"], ev["lon"] = ev["club"].map(lambda k: clubs[k][1]), ev["club"].map(lambda k: clubs[k][2])
    ev["start_date"] = ev["end_date"] = ev["event_date"]
    ev["status"] = np.where(ev["event_date"] < today, "held", "planned")
    ev["event_id"] = "club:" + ev["club"] + ":" + ev["event_date"].astype(str)
    ev["h3_r9"] = [_cell(a, o) for a, o in zip(ev["lat"], ev["lon"])]
    ev["category"], ev["source"] = "club_show", "setlistfm"
    ev = ev.rename(columns={"club": "venue"}).drop(columns=["event_date"])
    return ev.drop(columns=["h3_r9"]), ev[["event_id", "h3_r9"]]


def build_events(cdot: pd.DataFrame, park: pd.DataFrame, polygons: pd.DataFrame, setlists: pd.DataFrame,
                 today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per event (EVENT_COLUMNS) and every r9 cell each event touches."""
    parts = [cdot_events(cdot, today), park_events(park, polygons, today), club_events(setlists, today)]
    cells = pd.concat([c for _, c in parts]).drop_duplicates()
    events = pd.concat([e for e, _ in parts], ignore_index=True)
    events["h3_r9"] = [_cell(a, o) for a, o in zip(events["lat"], events["lon"])]
    events = events.merge(cells.groupby("event_id").size().rename("n_cells"), on="event_id")
    events["n_days"] = [(e - s).days + 1 for s, e in zip(events["start_date"], events["end_date"])]
    events["long_run"] = events["n_days"] > MAX_EVENT_DAYS
    for c in EVENT_COLUMNS:
        if c not in events:
            events[c] = None
    return events[EVENT_COLUMNS].sort_values(["start_date", "event_id"]).reset_index(drop=True), cells


# ---------------------------------------------------------------------------------------------
# Upcoming
# ---------------------------------------------------------------------------------------------

def _street_key(address: str | None) -> tuple[int | None, list[str]]:
    words = re.sub(r"[^a-z0-9 ]", " ", (address if isinstance(address, str) else "").lower()).split()
    return (int(words[0]) if words and words[0].isdigit() else None), words[1:]


def upcoming_festivals(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Planned street festivals to score, and multi-week runs to list without a score."""
    planned = events[(events["status"] == "planned") & (events["category"] == "festival")
                     & (events["name"].fillna("").str.len() >= 3)
                     & ~events["name"].fillna("").str.lower().isin(PLACEHOLDER_NAMES)]
    return planned[~planned["long_run"]].copy(), planned[planned["long_run"]].copy()


def upcoming_club_nights(events: pd.DataFrame, tm: pd.DataFrame, today: date) -> pd.DataFrame:
    """One row per club per upcoming night, from a Ticketmaster snapshot plus any setlist.fm
    listings. Ticketmaster venues are matched on street address: its venue names vary
    ("Reggies Rock Club", "Reggie's Music Joint") and some of its coordinates are wrong (it puts
    the Empty Bottle in New York)."""
    tm = tm[tm["segment"] != "Sports"]
    tm_keys = [(_street_key(a), r) for a, r in zip(tm["venue_address"], tm.to_dict("records"))]
    rows = []
    for key, name, address, lat, lon, _ in CLUBS:
        number, words = _street_key(address)
        street = words[-2]   # "2105 S State St" -> "state"
        for (n, w), r in tm_keys:
            if n is not None and abs(n - number) <= ADDRESS_SLACK and street in w and r.get("local_date"):
                rows.append({"venue": key, "place": name, "lat": lat, "lon": lon,
                             "start_date": date.fromisoformat(r["local_date"]), "local_time": r["local_time"],
                             "act": r["name"], "source": "ticketmaster"})
    sfm = events[(events["source"] == "setlistfm") & (events["status"] == "planned")]
    rows += [{"venue": r.venue, "place": r.place, "lat": r.lat, "lon": r.lon, "start_date": r.start_date,
              "local_time": None, "act": r.name, "source": "setlistfm"} for r in sfm.itertuples()]
    if not rows:
        return pd.DataFrame(columns=["event_id", "venue", "place", "lat", "lon", "start_date", "end_date",
                                     "local_time", "acts", "sources", "h3_r9"])
    df = pd.DataFrame(rows)
    df = df[df["start_date"] >= today]
    g = df.groupby(["venue", "start_date"])
    out = pd.DataFrame({
        "place": g["place"].first(), "lat": g["lat"].first(), "lon": g["lon"].first(),
        "local_time": g["local_time"].agg(lambda s: min(s.dropna(), default=None)),
        "acts": g["act"].agg(lambda s: " | ".join(sorted(set(s.dropna())))),
        "sources": g["source"].agg(lambda s: "+".join(sorted(set(s)))),
    }).reset_index()
    out["end_date"] = out["start_date"]
    out["event_id"] = "night:" + out["venue"] + ":" + out["start_date"].astype(str)
    out["h3_r9"] = [_cell(a, o) for a, o in zip(out["lat"], out["lon"])]
    return out


# ---------------------------------------------------------------------------------------------
# Observed vs expected, for past events
# ---------------------------------------------------------------------------------------------

def area_cells(event_cells: pd.DataFrame) -> pd.DataFrame:
    """(event_id, ring, h3_r9): each event's own cells (ring 0) and the ring around them (1),
    where the lift is measured and scored."""
    rows = []
    for eid, cs in event_cells.groupby("event_id")["h3_r9"]:
        own = set(cs)
        ring = set().union(*(h3.grid_ring(c, 1) for c in own)) - own
        rows += [(eid, 0, c) for c in own] + [(eid, 1, c) for c in ring]
    return pd.DataFrame(rows, columns=["event_id", "ring", "h3_r9"])


def _event_days(events: pd.DataFrame) -> pd.DataFrame:
    out = events[["event_id", "start_date", "end_date"]].copy()
    out["date"] = [pd.date_range(a, b).date.tolist() for a, b in zip(out["start_date"], out["end_date"])]
    return out[["event_id", "date"]].explode("date")


def occupancy(events: pd.DataFrame, cells: pd.DataFrame, near: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For the cells in `near`: the (h3_r9, date) pairs a held event occupies, and the wider busy
    set that control and baseline days must avoid. Busy means within one cell of any held event
    (club shows and multi-week runs included), and for permits the day either side too (a club
    show only blocks its own night, or the busiest clubs would leave almost nothing to compare
    with)."""
    held = events.loc[events["status"] == "held", ["event_id", "start_date", "end_date", "source"]]
    reach = set().union(*(h3.grid_disk(c, 1) for c in near)) if near else set()
    spans = (cells[cells["h3_r9"].isin(reach)].merge(_event_days(held), on="event_id")
             .merge(held[["event_id", "source"]], on="event_id")[["h3_r9", "date", "source"]].drop_duplicates())
    occupied = spans.loc[spans["h3_r9"].isin(near), ["h3_r9", "date"]].drop_duplicates()
    disk = pd.DataFrame([(c, n) for c in spans["h3_r9"].unique() for n in h3.grid_disk(c, 1) if n in near],
                        columns=["h3_r9", "near"])
    spans["k"] = [[0] if s == "setlistfm" else [-1, 0, 1] for s in spans["source"]]
    spread = spans.explode("k")
    spread["date"] = [d + timedelta(days=int(k)) for d, k in zip(spread["date"], spread["k"])]
    busy = (spread[["h3_r9", "date"]].drop_duplicates().merge(disk, on="h3_r9")[["near", "date"]]
            .rename(columns={"near": "h3_r9"}).drop_duplicates())
    return occupied, busy


def _anti(df: pd.DataFrame, block: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    m = df.merge(block[on].drop_duplicates().assign(_x=True), on=on, how="left")
    return m[m["_x"].isna()].drop(columns="_x")


def past_oe(past: pd.DataFrame, event_cells: pd.DataFrame, counts: pd.DataFrame, city: pd.DataFrame,
            occupied: pd.DataFrame, busy: pd.DataFrame, window: tuple[date, date]) -> pd.DataFrame:
    """Observed and expected crime per past event, own cells + ring, on its event days (or
    nights, with night counts).

    Expected for a cell-day: the mean of its clean control days (the same weekday 1-4 weeks
    either side, not busy, at least MIN_CONTROLS of them) x how the whole city did that day
    against those control days. The city scaling absorbs holidays, weather and trend. A ring
    cell that hosts some other event that day isn't a clean read, so it's skipped. Returns
    (event_id, O, E).
    """
    first, last = window
    area = area_cells(event_cells.merge(past[["event_id"]], on="event_id"))
    expo = area.merge(_event_days(past), on="event_id")
    expo = expo[(expo["date"] >= first) & (expo["date"] <= last)]
    expo = pd.concat([expo[expo["ring"] == 0], _anti(expo[expo["ring"] > 0], occupied, ["h3_r9", "date"])])
    keys = expo[["h3_r9", "date"]].drop_duplicates()
    ctrl = keys.merge(pd.DataFrame({"k": [7 * w for w in CONTROL_WEEKS]}), how="cross")
    ctrl["ctrl"] = [d + timedelta(days=int(k)) for d, k in zip(ctrl["date"], ctrl["k"])]
    ctrl = ctrl[(ctrl["ctrl"] >= first) & (ctrl["ctrl"] <= last)]
    ctrl = _anti(ctrl, busy.rename(columns={"date": "ctrl"}), ["h3_r9", "ctrl"])
    ctrl = (ctrl.merge(counts.rename(columns={"date": "ctrl"}), on=["h3_r9", "ctrl"], how="left")
            .merge(city.rename(columns={"date": "ctrl", "crime": "city"}), on="ctrl", how="left")
            .fillna({"crime": 0.0, "city": 0.0}))
    base = ctrl.groupby(["h3_r9", "date"]).agg(n=("crime", "size"), mean=("crime", "mean"), city_mean=("city", "mean")).reset_index()
    base = base[base["n"] >= MIN_CONTROLS].merge(city.rename(columns={"crime": "today"}), on="date", how="left")
    scale = np.where(base["city_mean"] > 0, base["today"].fillna(0.0) / base["city_mean"], 1.0)
    base["E"] = base["mean"] * scale
    base = base.merge(counts, on=["h3_r9", "date"], how="left").fillna({"crime": 0.0}).rename(columns={"crime": "O"})
    return (expo.merge(base[["h3_r9", "date", "O", "E"]], on=["h3_r9", "date"])
            .groupby("event_id")[["O", "E"]].sum().reset_index())


# ---------------------------------------------------------------------------------------------
# Scoring upcoming events
# ---------------------------------------------------------------------------------------------

def baselines(targets: pd.DataFrame, area: pd.DataFrame, counts: pd.DataFrame, known_until: date,
              busy: pd.DataFrame) -> pd.Series:
    """Crime normally expected over each target's area and dates: each cell-date's mean on the
    same weekday +-4 weeks in each of the last 3 years, skipping busy days (the cell's plain
    daily mean when fewer than MIN_BASELINE_DAYS clean ones are on record)."""
    first = known_until - timedelta(days=365 * BASELINE_YEARS)
    cell_days = _event_days(targets).merge(area[["event_id", "h3_r9"]], on="event_id")
    keys = cell_days[["h3_r9", "date"]].drop_duplicates()
    past = keys.merge(pd.DataFrame({"k": [-364 * y + 7 * w for y in range(1, BASELINE_YEARS + 1) for w in BASELINE_WEEKS]}),
                      how="cross")
    past["past"] = [d + timedelta(days=int(k)) for d, k in zip(past["date"], past["k"])]
    past = past[(past["past"] >= first) & (past["past"] <= known_until)]
    past = _anti(past, busy.rename(columns={"date": "past"}), ["h3_r9", "past"])
    past = past.merge(counts.rename(columns={"date": "past"}), on=["h3_r9", "past"], how="left").fillna({"crime": 0.0})
    same = past.groupby(["h3_r9", "date"]).agg(n=("crime", "size"), mean=("crime", "mean")).reset_index()
    n_days = (known_until - first).days + 1
    plain = (counts[(counts["date"] >= first) & (counts["date"] <= known_until)]
             .groupby("h3_r9")["crime"].sum() / n_days).rename("plain")
    base = keys.merge(same, on=["h3_r9", "date"], how="left").merge(plain, on="h3_r9", how="left")
    base["baseline"] = np.where(base["n"] >= MIN_BASELINE_DAYS, base["mean"], base["plain"])
    base["baseline"] = base["baseline"].fillna(0.0)
    return (cell_days.merge(base[["h3_r9", "date", "baseline"]], on=["h3_r9", "date"])
            .groupby("event_id")["baseline"].sum())


def between_cv(items: pd.DataFrame, group: list[str], bucket: str, min_events: int = 1) -> float:
    """How much groups' true lifts differ, as a share of their bucket's average, beyond Poisson
    noise (method of moments on O/E)."""
    g = items.groupby(group + [bucket]).agg(O=("O", "sum"), E=("E", "sum"), n=("O", "size")).reset_index()
    g = g[(g["n"] >= min_events) & (g["E"] > 0)]
    m = g.groupby(bucket).apply(lambda x: x["O"].sum() / x["E"].sum(), include_groups=False).rename("m")
    g = g.join(m, on=bucket)
    w = g["E"] / g["E"].sum()
    rel = (g["O"] / g["E"] - g["m"]) / g["m"]
    var = (w * rel ** 2).sum() - (w / (g["m"] * g["E"])).sum()
    return max(var, MIN_CV ** 2) ** 0.5


def priors(items: pd.DataFrame, bucket: str, cv: float) -> pd.DataFrame:
    """Per bucket: the pooled lift and the prior's weight b (in expected crimes)."""
    p = items.groupby(bucket).agg(n_past=("O", "size"), O=("O", "sum"), E=("E", "sum")).reset_index()
    p["prior"] = p["O"] / p["E"]
    p["b"] = 1 / (p["prior"] * cv ** 2)
    return p[[bucket, "n_past", "prior", "b"]]


def posterior(df: pd.DataFrame) -> pd.DataFrame:
    """Gamma-Poisson: the event's own history (O, E) pooled with its prior (prior, b). Needs
    baseline_crime; adds ratio and extra crime with 90% intervals."""
    df = df.fillna({"O": 0.0, "E": 0.0})
    shape, rate = df["O"] + df["b"] * df["prior"], df["E"] + df["b"]
    out = df.assign(ratio=shape / rate, ratio_lo=gamma.ppf(0.05, shape, scale=1 / rate),
                    ratio_hi=gamma.ppf(0.95, shape, scale=1 / rate))
    for c in ("", "_lo", "_hi"):
        out[f"extra{c}"] = out["baseline_crime"] * (out[f"ratio{c}"] - 1)
    return out


def norm_name(s: str | None) -> str:
    """'Hyde Park Jazz Fest 2026' and 'HYDE PARK JAZZ FEST' -> 'hyde park jazz fest'."""
    s = re.sub(r"\b\d+(st|nd|rd|th)?\b", " ", (s if isinstance(s, str) else "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", s)).strip()


def size_bucket(n_segments: pd.Series) -> np.ndarray:
    return np.where(n_segments >= 5, "5+", np.where(n_segments >= 2, "2-4", "1"))


def score_festivals(targets: pd.DataFrame, events: pd.DataFrame, cells: pd.DataFrame, counts: pd.DataFrame,
                    city: pd.DataFrame, known_until: date, occupied: pd.DataFrame, busy: pd.DataFrame,
                    first: date) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Score target festivals from festivals and crime up to known_until.

    The prior is the average lift for festivals the same size. A festival's own history is its
    past editions: the same name once years and punctuation are dropped, within MATCH_KM. A past
    event counts as history only once the 4 weeks after it are known too, because its expected
    count compares it with the same weekday up to 4 weeks later."""
    keyed = lambda df: df.assign(key=df["name"].map(norm_name), bucket=size_bucket(df["n_segments"]))
    settled = known_until - timedelta(days=SETTLE_DAYS)
    past = keyed(events[(events["status"] == "held") & (events["category"] == "festival") & ~events["long_run"]
                        & (events["end_date"] <= settled)])
    oe = past_oe(past, cells, counts, city, occupied, busy, (first, known_until))
    items = oe.merge(past[["event_id", "key", "bucket"]].assign(
        place=[h3.cell_to_parent(int(c), 7) for c in past["h3_r9"]]), on="event_id")
    cv = between_cv(items, ["key", "place"], "bucket", min_events=2)   # a lift is judged across a festival's own editions
    pri = priors(items, "bucket", cv)

    t = keyed(targets)
    pairs = (t.loc[t["key"] != "", ["event_id", "key", "lat", "lon"]]
             .merge(past[["event_id", "key", "lat", "lon", "start_date"]], on="key", suffixes=("", "_past")))
    near = [h3.great_circle_distance((a, o), (a2, o2), unit="km") <= MATCH_KM
            for a, o, a2, o2 in zip(pairs["lat"], pairs["lon"], pairs["lat_past"], pairs["lon_past"])]
    pairs = pairs[near].merge(oe.rename(columns={"event_id": "event_id_past"}), on="event_id_past", how="left").fillna({"O": 0.0, "E": 0.0})
    pairs["year"] = [d.year for d in pairs["start_date"]]
    hist = pairs.groupby("event_id").agg(past_editions=("year", "nunique"), O=("O", "sum"), E=("E", "sum")).reset_index()

    area = area_cells(cells.merge(t[["event_id"]], on="event_id"))
    t["baseline_crime"] = t["event_id"].map(baselines(t, area, counts, known_until, busy)).fillna(0.0)
    df = t.merge(hist, on="event_id", how="left").merge(pri, on="bucket", how="left")
    df["past_editions"] = df["past_editions"].fillna(0).astype(int)
    return posterior(df), cv, pri


def score_club_nights(targets: pd.DataFrame, events: pd.DataFrame, cells: pd.DataFrame, counts: pd.DataFrame,
                      city: pd.DataFrame, known_until: date, occupied: pd.DataFrame, busy: pd.DataFrame,
                      first: date) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Score upcoming club nights from show nights and crime up to known_until. The prior is the
    all-club average; a club's own history is its past logged show nights. Night counts
    (6pm-3am) throughout."""
    settled = known_until - timedelta(days=SETTLE_DAYS)
    past = events[(events["source"] == "setlistfm") & (events["status"] == "held") & (events["end_date"] <= settled)]
    oe = past_oe(past, cells, counts, city, occupied, busy, (first, known_until))
    items = oe.merge(past[["event_id", "venue"]], on="event_id").assign(bucket="all")
    cv = between_cv(items, ["venue"], "bucket")
    pri = priors(items, "bucket", cv)
    hist = items.groupby("venue").agg(past_nights=("O", "size"), O=("O", "sum"), E=("E", "sum")).reset_index()
    area = area_cells(targets[["event_id", "h3_r9"]])
    t = targets.copy()
    t["baseline_crime"] = t["event_id"].map(baselines(t, area, counts, known_until, busy)).fillna(0.0)
    df = t.merge(hist, on="venue", how="left").merge(pri.drop(columns="bucket"), how="cross")
    df["past_nights"] = df["past_nights"].fillna(0).astype(int)
    return posterior(df), cv, pri


def lookahead(events: pd.DataFrame, cells: pd.DataFrame, tm: pd.DataFrame, today: date,
              day_counts: pd.DataFrame, day_city: pd.DataFrame, night_counts: pd.DataFrame,
              night_city: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Everything the look-ahead needs, from built events and crime counts.

    `*_counts` are (h3_r9, date, crime) for at least cells_needed(); `*_city` are citywide
    (date, crime). Day counts leave out "sometime this month" records stamped on the 1st; night
    counts (6pm-3am, keyed to the evening's date) leave out unknown times, which are logged at
    midnight. Crime is treated as known through the city series' newest date - 7 days, since the
    newest week is still filling in.
    """
    fest, multi = upcoming_festivals(events)
    nights = upcoming_club_nights(events, tm, today)
    near = cells_needed(events, cells, fest, nights)
    occupied, busy = occupancy(events, cells, near)
    norm = lambda df: df.assign(date=pd.to_datetime(df["date"]).dt.date)   # Arrow may hand back timestamps
    day_counts, day_city, night_counts, night_city = map(norm, (day_counts, day_city, night_counts, night_city))
    known_day = max(day_city["date"]) - timedelta(days=7)
    known_night = max(night_city["date"]) - timedelta(days=7)
    f, f_cv, f_pri = score_festivals(fest, events, cells, day_counts, day_city, known_day, occupied, busy, OE_START)
    n, n_cv, n_pri = score_club_nights(nights, events, cells, night_counts, night_city, known_night, occupied, busy, OE_START)
    return {"festivals": f, "club_nights": n, "multi_week": multi,
            "priors": pd.concat([f_pri.assign(kind="festival", cv=f_cv),
                                 n_pri.assign(kind="club_night", cv=n_cv)], ignore_index=True),
            "known_through": pd.DataFrame([{"day": known_day, "night": known_night}])}


def cells_needed(events: pd.DataFrame, cells: pd.DataFrame, fest: pd.DataFrame, nights: pd.DataFrame) -> set[int]:
    """Every r9 cell whose crime counts the look-ahead reads: the own cells and ring of every
    past held festival and club show (for their history) and of every upcoming target."""
    past = events[(events["status"] == "held") & (((events["category"] == "festival") & ~events["long_run"])
                                                  | (events["source"] == "setlistfm"))]
    own = pd.concat([cells.merge(past[["event_id"]], on="event_id"), cells.merge(fest[["event_id"]], on="event_id"),
                     nights[["event_id", "h3_r9"]]])
    cs = set(own["h3_r9"])
    return cs | set().union(*(h3.grid_ring(c, 1) for c in cs))


# ---------------------------------------------------------------------------------------------
# The table the app reads
# ---------------------------------------------------------------------------------------------

def area_of_cells(geojson: dict | None = None) -> dict[int, int]:
    """r9 cell -> the community area whose boundary holds the cell's center (5,629 cells, no
    overlaps)."""
    g = geojson or json.loads(COMMUNITY_AREAS_GEOJSON.read_text())
    out = {}
    for f in g["features"]:
        props = f["properties"]
        area = int(float(props.get("area_numbe") or props.get("area_num_1")))
        for c in h3.geo_to_cells(f["geometry"], H3_RES):
            out[c] = area
    return out


UPCOMING_COLUMNS = ["event_id", "kind", "name", "place", "venue", "start_date", "end_date", "n_days", "local_time",
                    "community_area", "n_segments", "permit_stage", "history", "baseline_crime", "ratio", "ratio_lo",
                    "ratio_hi", "extra", "extra_lo", "extra_hi", "sources", "lat", "lon"]


def upcoming_table(out: dict[str, pd.DataFrame], cell_area: dict[int, int]) -> pd.DataFrame:
    """One row per upcoming festival, club night and (unscored) multi-week run, in one shape.
    `history` is past festival editions or the club's past logged nights."""
    f = out["festivals"].assign(kind="festival", history=out["festivals"]["past_editions"], local_time=None,
                                sources="cdot")
    n = out["club_nights"].assign(kind="club_night", name=out["club_nights"]["acts"], history=out["club_nights"]["past_nights"],
                                  n_days=1, n_segments=None, permit_stage=None)
    m = out["multi_week"].assign(kind="multi_week", history=None, local_time=None, sources="cdot")
    rows = pd.concat([f, n, m], ignore_index=True)
    for c in UPCOMING_COLUMNS:
        if c not in rows:
            rows[c] = None
    centroid = [_cell(a, o) for a, o in zip(rows["lat"], rows["lon"])]
    rows["community_area"] = [cell_area.get(c) for c in centroid]
    return rows[UPCOMING_COLUMNS].sort_values(["start_date", "kind", "event_id"]).reset_index(drop=True)
