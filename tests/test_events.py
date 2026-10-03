"""Offline checks for src/events_sources.py and src/events_model.py.

Small hand-built inputs, so every expected number can be worked out by hand:
- the build rules (festival applications merging, block parties never merging, statuses,
  multi-week runs, park runs of consecutive days, one club event per night)
- Ticketmaster club matching by street address
- observed vs expected on a planted 2x event effect, with busy days kept out of the controls
- the Gamma-Poisson posterior at its two limits (no history, lots of history)
- baselines never reading crime after the known-through date
- parsers rejecting malformed responses

Run: python -m pytest tests/test_events.py (requirements-dev.txt).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from h3.api import basic_int as h3

from src import events_model as em
from src import events_sources as es

TODAY = date(2026, 6, 1)
LAT, LON = 41.94, -87.65


def cdot_row(app, name, start, end, category="Festival", milestone="Complete", lat=LAT, lon=LON, **kw):
    row = {c: None for c in es.CDOT_COLUMNS}
    row.update(uniquekey=f"u{app}{lat}", applicationnumber=str(app), applicationname=name, worktypedescription=category,
               currentmilestone=milestone, applicationstartdate=f"{start}T00:00:00.000",
               applicationenddate=f"{end}T00:00:00.000", latitude=str(lat), longitude=str(lon),
               streetnumberfrom="3400", streetnumberto="3500", direction="N", streetname="CLARK", suffix="ST",
               streetclosure="Full")
    row.update(kw)
    return row


def test_cdot_rules():
    rows = pd.DataFrame([
        cdot_row(1, "Taste of Clark 2025", "2025-07-12", "2025-07-13"),
        cdot_row(2, "Taste of Clark 2025", "2025-07-12", "2025-07-13", lat=41.941),   # same festival, second segment
        cdot_row(3, "Block Party", "2025-07-12", "2025-07-12", category="Block Party"),
        cdot_row(4, "Block Party", "2025-07-12", "2025-07-12", category="Block Party"),  # never merged
        cdot_row(5, "Cancelled Fest", "2025-08-01", "2025-08-01", milestone="Cancelled"),
        cdot_row(6, "Unfinished Fest", "2025-08-02", "2025-08-02", milestone="Application in Review"),  # past, never completed
        cdot_row(7, "Future Fest", "2026-08-01", "2026-08-02", milestone="Application in Review"),
        cdot_row(8, "Winter Market", "2026-11-20", "2026-12-24", milestone="Permit Active"),
    ])
    ev, cells = em.cdot_events(rows, TODAY)
    by_name = ev.set_index("name")
    assert by_name.loc["Taste of Clark 2025", "n_segments"] == 2 and by_name.loc["Taste of Clark 2025", "status"] == "held"
    assert (ev["category"] == "block_party").sum() == 2
    assert "Cancelled Fest" not in by_name.index and "Unfinished Fest" not in by_name.index
    assert by_name.loc["Future Fest", "status"] == "planned"
    built, _ = em.build_events(rows, pd.DataFrame(columns=es.PARK_COLUMNS), pd.DataFrame(columns=["park_no", "park", "geometry"]),
                               pd.DataFrame(columns=["setlist_id", "venue_id", "event_date", "artist", "n_songs"]), TODAY)
    assert built.set_index("name").loc["Winter Market", "long_run"]


def test_park_runs():
    square = {"type": "Polygon", "coordinates": [[[-87.66, 41.93], [-87.64, 41.93], [-87.64, 41.95], [-87.66, 41.95], [-87.66, 41.93]]]}
    polygons = pd.DataFrame([{"park_no": "7.0", "park": "TEST PARK", "geometry": __import__("json").dumps(square)}])
    row = lambda day, status="Approved": {"organization": "Org", "park_number": "7", "park_facility_name": "Field 1",
                                          "reservation_start_date": f"{day}T00:00:00.000", "reservation_end_date": f"{day}T00:00:00.000",
                                          "event_type": "Permit - Event 3 Cluster 1", "event_description": "Summer Fest",
                                          "permit_status": status}
    raw = pd.DataFrame([row("2025-07-01"), row("2025-07-02"), row("2025-07-10"), row("2026-07-01", "Tentative")])
    ev, cells = em.park_events(raw, polygons, TODAY)
    ev = ev.sort_values("start_date")
    assert list(ev["n_facilities"]) == [1, 1, 1]
    assert [(s.isoformat(), e.isoformat()) for s, e in zip(ev["start_date"], ev["end_date"])] == [
        ("2025-07-01", "2025-07-02"), ("2025-07-10", "2025-07-10"), ("2026-07-01", "2026-07-01")]
    assert list(ev["status"]) == ["held", "held", "planned"] and list(ev["size_level"]) == [3, 3, 3]
    assert cells["h3_r9"].nunique() > 5          # the polygon, not just its center


def test_clubs_and_ticketmaster_matching():
    vic = es.CLUB_VENUE_IDS and "1bd63da8"
    setlists = pd.DataFrame([
        {"setlist_id": "a", "venue_id": vic, "event_date": date(2025, 5, 1), "artist": "Opener", "n_songs": 5},
        {"setlist_id": "b", "venue_id": vic, "event_date": date(2025, 5, 1), "artist": "Headliner", "n_songs": 15},
        {"setlist_id": "c", "venue_id": "not-a-club", "event_date": date(2025, 5, 1), "artist": "Elsewhere", "n_songs": 9},
    ])
    ev, _ = em.club_events(setlists, TODAY)
    assert len(ev) == 1 and ev.iloc[0]["name"] == "Headliner" and ev.iloc[0]["n_acts"] == 2
    tm = pd.DataFrame([
        {"name": "Show", "local_date": "2026-06-05", "local_time": "20:00:00", "segment": "Music", "venue_address": "3145 N. Sheffield Ave."},
        {"name": "Also Show", "local_date": "2026-06-06", "local_time": "19:00:00", "segment": "Music", "venue_address": "3149 N Sheffield Avenue"},
        {"name": "Game", "local_date": "2026-06-05", "local_time": "19:00:00", "segment": "Sports", "venue_address": "3145 N Sheffield Ave"},
        {"name": "Far", "local_date": "2026-06-05", "local_time": "19:00:00", "segment": "Music", "venue_address": "3145 N Clark St"},
        {"name": "Past", "local_date": "2026-05-01", "local_time": "19:00:00", "segment": "Music", "venue_address": "3145 N Sheffield Ave"},
    ])
    nights = em.upcoming_club_nights(ev, tm, TODAY)
    assert list(nights["start_date"]) == [date(2026, 6, 5), date(2026, 6, 6)] and set(nights["venue"]) == {"vic"}


def _flat_counts(cells, first, last, per_day=2.0):
    days = pd.date_range(first, last).date
    return (pd.DataFrame([(c, d, per_day) for c in cells for d in days], columns=["h3_r9", "date", "crime"]),
            pd.DataFrame({"date": days, "crime": 1000.0}))


def test_past_oe_recovers_a_planted_effect():
    own = h3.latlng_to_cell(LAT, LON, 9)
    ring = list(h3.grid_ring(own, 1))
    event_day = date(2025, 7, 12)
    counts, city = _flat_counts([own, *ring], date(2025, 5, 1), date(2025, 9, 30))
    counts.loc[(counts["h3_r9"] == own) & (counts["date"] == event_day), "crime"] = 4.0   # 2x in its own cell
    past = pd.DataFrame([{"event_id": "e1", "start_date": event_day, "end_date": event_day}])
    event_cells = pd.DataFrame({"event_id": ["e1"], "h3_r9": [own]})
    empty = pd.DataFrame(columns=["h3_r9", "date"])
    oe = em.past_oe(past, event_cells, counts, city, empty, empty, (date(2025, 5, 1), date(2025, 9, 30))).iloc[0]
    assert oe["O"] == 4 + 2 * 6 and oe["E"] == 2 * 7          # own cell doubled, ring unchanged
    # A control day that's busy is skipped: make one control 10x and mark it busy -> same answer.
    counts.loc[(counts["h3_r9"] == own) & (counts["date"] == event_day + timedelta(days=7)), "crime"] = 20.0
    busy = pd.DataFrame({"h3_r9": [own], "date": [event_day + timedelta(days=7)]})
    oe2 = em.past_oe(past, event_cells, counts, city, empty, busy, (date(2025, 5, 1), date(2025, 9, 30))).iloc[0]
    assert (oe2["O"], oe2["E"]) == (oe["O"], oe["E"])


def test_posterior_limits():
    base = pd.DataFrame({"baseline_crime": [10.0, 10.0], "prior": [1.1, 1.1], "b": [4.0, 4.0],
                         "O": [np.nan, 3000.0], "E": [np.nan, 1000.0]})
    out = em.posterior(base)
    assert np.isclose(out["ratio"].iloc[0], 1.1) and np.isclose(out["extra"].iloc[0], 1.0)   # no history -> the prior
    assert abs(out["ratio"].iloc[1] - 3.0) < 0.01                                             # lots of history -> its own O/E
    assert (out["ratio_lo"] <= out["ratio"]).all() and (out["ratio"] <= out["ratio_hi"]).all()


def test_baselines_ignore_crime_after_known_through():
    own = h3.latlng_to_cell(LAT, LON, 9)
    known = date(2026, 5, 20)
    rng = np.random.default_rng(0)
    days = pd.date_range("2023-01-01", "2026-06-30").date
    counts = pd.DataFrame({"h3_r9": own, "date": days, "crime": rng.poisson(2, len(days)).astype(float)})
    targets = pd.DataFrame([{"event_id": "t", "start_date": date(2026, 6, 13), "end_date": date(2026, 6, 14)}])
    area = pd.DataFrame({"event_id": ["t"], "h3_r9": [own]})
    empty = pd.DataFrame(columns=["h3_r9", "date"])
    full = em.baselines(targets, area, counts, known, empty)
    cut = em.baselines(targets, area, counts[counts["date"] <= known], known, empty)
    assert np.isclose(full["t"], cut["t"]) and 3 < full["t"] < 5     # ~2 a day over 2 days


def test_parsers():
    page = {"setlist": [{"id": "x1", "eventDate": "26-09-2026", "artist": {"name": "Band"}, "lastUpdated": "t",
                         "sets": {"set": [{"song": [{}, {}]}, {"song": [{}]}]}}], "itemsPerPage": 20, "total": 1}
    r = es.parse_setlist_page(page, "v")[0]
    assert r["event_date"] == date(2026, 9, 26) and r["n_songs"] == 3
    assert es.soda_rows([{"a": 1}], ["a", "b"]) == [{"a": "1", "b": None}]
    assert es.soda_rows([{"a": float("nan")}], ["a"]) == [{"a": None}]
    flat = es.tm_flatten({"id": "e", "dates": {"start": {"localDate": "2026-10-01"}},
                          "_embedded": {"venues": [{"address": {"line1": "1 Main St"}, "location": {"latitude": "41.9"}}]}})
    assert flat["venue_address"] == "1 Main St" and flat["venue_lat"] == "41.9" and flat["segment"] is None
    for bad, fn in ((({"setlist": "nope"}, "v"), es.parse_setlist_page), (({"type": "x"},), es.parse_park_polygons)):
        try:
            fn(*bad)
        except ValueError:
            continue
        raise AssertionError(f"malformed input accepted by {fn.__name__}")
