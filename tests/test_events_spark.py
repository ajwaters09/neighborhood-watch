"""Parity checks: the Spark event build (src/events_spark.py) against the pandas one (src/events_model.py).

- the hand-built inputs from test_events.py give the same events and cells
- on the offline pulls (offline_data/events/raw, if present), the full build matches too:
  every event id, cell and column, with lat/lon to 1e-9 and club headliners allowed to differ
  where setlist lengths tie

Runs on local Spark. The Databricks H3 SQL functions aren't in open-source Spark, so h3-py
stand-ins are registered under the same names; everything else is the code 08b runs. ANSI mode
is on, as on serverless.

Run: python -m pytest tests/test_events_spark.py. Needs pyspark and a JDK on top of
requirements-dev.txt; skipped without pyspark.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
from h3.api import basic_int as h3  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

from src import events_model as em  # noqa: E402
from src import events_sources as es  # noqa: E402
from src import events_spark as ess  # noqa: E402
from test_events import LAT, TODAY, cdot_row  # noqa: E402

RAW = ROOT / "offline_data" / "events" / "raw"
SETLIST_COLUMNS = ["setlist_id", "venue_id", "event_date", "artist", "n_songs"]


def _polyfill(geojson, res):
    try:
        return [int(c) for c in h3.geo_to_cells(json.loads(geojson), res)]
    except Exception:
        return None


def _center(cell):
    lat, lon = h3.cell_to_latlng(cell)
    return json.dumps({"type": "Point", "coordinates": [lon, lat]})


_spark = None


def spark() -> SparkSession:
    global _spark
    if _spark is None:
        # Python workers must be this interpreter, and able to import this module's UDFs.
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests"), str(ROOT), os.environ.get("PYTHONPATH", "")])
        _spark = (SparkSession.builder.master("local[2]").appName("events_spark_parity")
                  .config("spark.sql.ansi.enabled", "true").config("spark.sql.shuffle.partitions", "4")
                  .config("spark.sql.session.timeZone", "UTC").getOrCreate())
        _spark.sparkContext.setLogLevel("ERROR")
        _spark.udf.register("h3_longlatash3", lambda lon, lat, res: None if lon is None or lat is None
                            else h3.latlng_to_cell(lat, lon, res), T.LongType())
        _spark.udf.register("h3_try_polyfillash3", _polyfill, T.ArrayType(T.LongType()))
        _spark.udf.register("h3_centerasgeojson", _center, T.StringType())
    return _spark


def sdf(df: pd.DataFrame, schema: list[tuple[str, str]]):
    """A pandas frame as a bronze-typed Spark table (all-string permits, as SODA lands them)."""
    cols = [c for c, _ in schema]
    rows = df.reindex(columns=cols).astype(object).where(df.reindex(columns=cols).notna(), None)
    return spark().createDataFrame(rows.values.tolist(), ", ".join(f"{c} {t}" for c, t in schema))


def both_builds(cdot, park, polygons, setlists, today):
    want = em.build_events(cdot, park, polygons, setlists, today)
    got = ess.to_pandas(*ess.build_events(sdf(cdot, es.CDOT_SCHEMA), sdf(park, es.PARK_SCHEMA),
                                          sdf(polygons, es.POLYGON_SCHEMA),
                                          sdf(setlists, [(c, t) for c, t in es.SETLIST_SCHEMA if c in SETLIST_COLUMNS]),
                                          today))
    return want, got


def differences(want: pd.DataFrame, got: pd.DataFrame, skip_club_names: bool = False) -> dict[str, int]:
    """Column -> number of events whose value differs (missing on either side counts as equal)."""
    assert set(want["event_id"]) == set(got["event_id"]), (
        f"only pandas: {sorted(set(want['event_id']) - set(got['event_id']))[:5]}, "
        f"only Spark: {sorted(set(got['event_id']) - set(want['event_id']))[:5]}")
    m = want.merge(got, on="event_id", suffixes=("_w", "_g"))
    out = {}
    for c in em.EVENT_COLUMNS[1:]:
        w, g = m[f"{c}_w"], m[f"{c}_g"]
        missing = w.isna() & g.isna()
        if c in ("lat", "lon"):
            same = np.isclose(w.astype(float), g.astype(float), rtol=0, atol=1e-9)
        elif w.dtype.kind in "fiub" or g.dtype.kind in "fiub":
            same = pd.to_numeric(w, errors="coerce").astype(float).eq(pd.to_numeric(g, errors="coerce").astype(float))
        else:
            same = w.astype(object).eq(g.astype(object))
        bad = ~(same | missing)
        if skip_club_names and c == "name":
            bad &= m["source_w"] != "setlistfm"
        if bad.any():
            out[c] = int(bad.sum())
    return out


def same_cells(want: pd.DataFrame, got: pd.DataFrame) -> bool:
    key = lambda df: set(zip(df["event_id"], df["h3_r9"].astype("int64")))
    return key(want) == key(got)


def test_hand_built():
    cdot = pd.DataFrame([
        cdot_row(1, "Taste of Clark 2025", "2025-07-12", "2025-07-13"),
        cdot_row(2, " Taste of Clark 2025 ", "2025-07-12", "2025-07-13", lat=41.941, streetclosure=None),
        cdot_row(3, "Block Party", "2025-07-12", "2025-07-12", category="Block Party"),
        cdot_row(4, "Block Party", "2025-07-12", "2025-07-12", category="Block Party", streetnumberto="3400"),
        cdot_row(5, "Cancelled Fest", "2025-08-01", "2025-08-01", milestone="Cancelled"),
        cdot_row(6, "Unfinished Fest", "2025-08-02", "2025-08-02", milestone="Application in Review"),
        cdot_row(7, "Future Fest", "2026-08-01", "2026-08-02", milestone="Application in Review"),
        cdot_row(8, "Winter Market", "2026-11-20", "2026-12-24", milestone="Permit Active"),
        cdot_row(9, "Typo Fest", "2112-08-01", "2112-08-02", milestone="Permit Active"),
    ])
    square = {"type": "Polygon", "coordinates": [[[-87.66, 41.93], [-87.64, 41.93], [-87.64, 41.95], [-87.66, 41.95], [-87.66, 41.93]]]}
    tiny = {"type": "MultiPolygon", "coordinates": [[[[-87.70, 41.90], [-87.6999, 41.90], [-87.6999, 41.9001], [-87.70, 41.90]]]]}
    polygons = pd.DataFrame([{"park_no": "7.0", "park": "TEST PARK", "geometry": json.dumps(square)},
                             {"park_no": "8.0", "park": "TINY PARK", "geometry": json.dumps(tiny)}])
    row = lambda day, status="Approved", park="7", fac="Field 1", et="Permit - Event 3 Cluster 1": {
        "organization": "Org", "park_number": park, "park_facility_name": fac,
        "reservation_start_date": f"{day}T00:00:00.000", "reservation_end_date": f"{day}T00:00:00.000",
        "event_type": et, "event_description": "Summer Fest", "permit_status": status}
    park = pd.DataFrame([row("2025-07-01"), row("2025-07-02", fac="Field 2"), row("2025-07-10"),
                         row("2026-07-01", "Tentative"), row("2025-06-01", "Tentative"),
                         row("2025-07-04", park="8", et="Athletic Event Level 4"),
                         row("2025-07-05", park="8", et="Event 6 - 12,001+")])
    venue = es.CLUBS[0][5][0]
    setlists = pd.DataFrame([
        {"setlist_id": "a", "venue_id": venue, "event_date": pd.Timestamp("2025-05-01").date(), "artist": "Opener", "n_songs": 8},
        {"setlist_id": "b", "venue_id": venue, "event_date": pd.Timestamp("2025-05-01").date(), "artist": "Headliner", "n_songs": 15},
        {"setlist_id": "c", "venue_id": venue, "event_date": pd.Timestamp("2026-07-01").date(), "artist": "Future Act", "n_songs": 0},
        {"setlist_id": "d", "venue_id": "not-a-club", "event_date": pd.Timestamp("2025-05-01").date(), "artist": "X", "n_songs": 3},
    ])
    (want, want_cells), (got, got_cells) = both_builds(cdot, park, polygons, setlists, TODAY)
    assert len(want) >= 8 and (want["source"] == "park").sum() >= 3
    assert differences(want, got) == {}
    assert same_cells(want_cells, got_cells)


@pytest.mark.skipif(not (RAW / "cdot_permits.parquet").exists(), reason="offline pulls not present")
def test_offline_pulls():
    cdot = pd.read_parquet(RAW / "cdot_permits.parquet")
    park = pd.read_parquet(RAW / "park_permits.parquet")
    polygons = pd.DataFrame(es.parse_park_polygons(json.loads((RAW / "park_polygons.geojson").read_text())))
    rows = []
    for vdir in sorted(p for p in (RAW / "setlistfm").iterdir() if p.is_dir()):
        for page in sorted(vdir.glob("p*.json")):
            rows += es.parse_setlist_page(json.loads(page.read_text()), vdir.name)
    setlists = pd.DataFrame(rows)[SETLIST_COLUMNS]
    # Offline pulls are plain object strings on Databricks' toPandas; match that.
    cdot, park = (df.astype(object).where(df.notna(), None) for df in (cdot, park))
    (want, want_cells), (got, got_cells) = both_builds(cdot, park, polygons, setlists, TODAY)
    print(f"{len(want):,} events, {len(want_cells):,} cells")
    assert differences(want, got, skip_club_names=True) == {}
    assert same_cells(want_cells, got_cells)
