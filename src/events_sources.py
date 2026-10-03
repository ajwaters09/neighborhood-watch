"""Event sources for the events look-ahead: clients, parsers and bronze schemas.

Four sources, first explored locally (offline_ml/events_fetch.py):
- **CDOT street-use permits** (`pubx-yq2d`): festivals, block parties, parades, runs and
  rallies since 2014, one row per closed street segment.
- **Park District event permits** (`pk66-w54g`): one row per facility per day. Plus the
  **park polygons** (`ejsh-fztr`), to place them.
- **setlist.fm**: logged shows at CLUBS, 20 small music clubs. It allows 1,440 requests a day.
- **Ticketmaster Discovery**: upcoming events only (a past week returns nothing), for the
  club nights ahead.

The first two come through the city's SODA API (src/soda_client.py). Pro sports are left out
on purpose: everyone already plans around those.

The parsers are shared by 01c (seeding from raw responses uploaded to the landing volume) and
02c (live pulls), so seeded and pulled rows mean the same thing. Pure Python (requests only), so it's
testable off Databricks.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

EVENTS_START = date(2014, 1, 1)

CDOT_PERMITS = "pubx-yq2d"
PARK_PERMITS = "pk66-w54g"
PARK_POLYGONS = "ejsh-fztr"
# Pulled broadly; the build step decides what counts as a gathering. Filming (54k permits)
# and the construction types are left out.
CDOT_WORK_TYPES = ["Festival", "Block Party", "Parade", "Athletic", "Assembly", "Street Closure"]
# The applicant contact columns (names, home addresses) are skipped: nothing needs them.
CDOT_COLUMNS = [
    "uniquekey", "applicationnumber", "applicationname", "applicationdescription",
    "worktypedescription", "applicationstatus", "currentmilestone", "applicationstartdate",
    "applicationenddate", "applicationissueddate", "streetnumberfrom", "streetnumberto",
    "direction", "streetname", "suffix", "placement", "streetclosure", "detail", "ward",
    "latitude", "longitude",
]
PARK_COLUMNS = [
    "organization", "park_number", "park_facility_name", "reservation_start_date",
    "reservation_end_date", "event_type", "event_description", "permit_status",
]
# Permits change after filing ("in review" becomes issued, complete or cancelled), so each pull
# replaces every permit starting within this many days back, plus everything ahead.
PERMIT_REFRESH_DAYS = 180

SETLISTFM = "https://api.setlist.fm/rest/1.0"
SETLISTFM_SPACING_S = 1.0      # published limit is 2/s, but 0.6 s spacing still drew a 429
SETLISTFM_MAX_PER_RUN = 300    # of 1,440/day; a routine run needs ~25 (one page per club)
SETLISTFM_LOOKBACK_DAYS = 60   # fans log shows late, so re-read this far behind the newest stored show

TICKETMASTER = "https://app.ticketmaster.com/discovery/v2/events.json"
TM_PAGE = 200
TM_MAX_RESULTS = 1000          # deep paging stops at page * size = 1,000, so big windows get split
TM_DAYS_AHEAD = 365
TM_SPACING_S = 0.25            # limit is 5/s

# (key, name, street address, lat, lon, setlist.fm venue ids), in priority order. Coordinates
# are the Census geocoder's match on the address, because setlist.fm only knows the city. A club
# with several ids is one address listed under several room names. Addresses were checked against
# Ticketmaster's venue records. Left out to save setlist.fm quota: Metro and Smartbar (~10,000
# setlists) and House of Blues (~7,500).
CLUBS = [
    ("schubas", "Schubas Tavern", "3159 N Southport Ave", 41.93923, -87.66372, ["3d6396f"]),
    ("hideout", "The Hideout", "1354 W Wabansia Ave", 41.91375, -87.66239, ["6bd62e26"]),
    ("beat_kitchen", "Beat Kitchen", "2100 W Belmont Ave", 41.93960, -87.68086, ["4bd63bc6"]),
    ("empty_bottle", "Empty Bottle", "1035 N Western Ave", 41.90040, -87.68688, ["63d62ec3"]),
    ("sleeping_village", "Sleeping Village", "3734 W Belmont Ave", 41.93923, -87.72115, ["73d2d2a5"]),
    ("martyrs", "Martyrs'", "3855 N Lincoln Ave", 41.95176, -87.67700, ["7bd5bee8"]),
    ("cobra_lounge", "Cobra Lounge", "235 N Ashland Ave", 41.88629, -87.66693, ["13d6d141"]),
    ("chop_shop", "Chop Shop", "2033 W North Ave", 41.91040, -87.67882, ["13d54d09"]),
    ("subterranean", "Subterranean", "2011 W North Ave", 41.91041, -87.67791, ["3bd6d078"]),
    ("reggies", "Reggies", "2105 S State St", 41.85415, -87.62703, ["13dfb591", "3bd1f044", "23d1f03b"]),
    ("double_door", "Double Door", "1572 N Milwaukee Ave", 41.90998, -87.67687, ["43d63f77"]),  # closed 2017
    ("joes_on_weed", "Joe's on Weed St", "940 W Weed St", 41.90986, -87.65165, ["5bd62fe0"]),
    ("promontory", "The Promontory", "5311 S Lake Park Ave", 41.79937, -87.58721, ["bd5d592"]),
    ("lincoln_hall", "Lincoln Hall", "2424 N Lincoln Ave", 41.92617, -87.64989, ["63d6128f"]),
    ("bottom_lounge", "Bottom Lounge", "1375 W Lake St", 41.88537, -87.66159, ["2bd6c4da"]),
    ("avondale_music_hall", "Avondale Music Hall", "3336 N Milwaukee Ave", 41.94173, -87.72829, ["43d2779b"]),
    ("thalia_hall", "Thalia Hall", "1807 S Allport St", 41.85777, -87.65757, ["3bd434fc"]),
    ("park_west", "Park West", "322 W Armitage Ave", 41.91840, -87.63762, ["4bd6375e"]),
    ("concord", "Concord Music Hall", "2047 N Milwaukee Ave", 41.91828, -87.68966, ["53d4875d"]),
    ("vic", "Vic Theatre", "3145 N Sheffield Ave", 41.93943, -87.65401, ["1bd63da8"]),
]
CLUB_VENUE_IDS = {vid: key for key, *_, ids in CLUBS for vid in ids}

TM_FIELDS = ["tm_event_id", "name", "local_date", "local_time", "start_utc", "status", "segment", "genre",
             "venue_id", "venue_name", "venue_address", "venue_lat", "venue_lon"]

# Bronze schemas. Permits and Ticketmaster stay all-string, the SODA bronze convention (see
# src/spark_bronze_utils.py); the build step types them.
CDOT_SCHEMA = [(c, "STRING") for c in CDOT_COLUMNS]
PARK_SCHEMA = [(c, "STRING") for c in PARK_COLUMNS]
POLYGON_SCHEMA = [("park_no", "STRING"), ("park", "STRING"), ("geometry", "STRING")]
SETLIST_SCHEMA = [("setlist_id", "STRING"), ("venue_id", "STRING"), ("event_date", "DATE"), ("artist", "STRING"),
                  ("n_songs", "INT"), ("last_updated", "STRING")]
SETLIST_KEYS = ["setlist_id"]
TM_SCHEMA = [("snapshot_date", "DATE")] + [(c, "STRING") for c in TM_FIELDS]


# ---------------------------------------------------------------------------------------------
# City portal (SODA)
# ---------------------------------------------------------------------------------------------

def permit_window_start(today: date) -> str:
    return (today - timedelta(days=PERMIT_REFRESH_DAYS)).isoformat() + "T00:00:00.000"


def cdot_where(since: str) -> str:
    types = ",".join(f"'{t}'" for t in CDOT_WORK_TYPES)
    return f"worktypedescription in ({types}) AND applicationstartdate >= '{since}'"


def park_where(since: str) -> str:
    return f"reservation_start_date >= '{since}'"


def _missing(v: Any) -> bool:
    """None, NaN or pandas' NA (what a parquet read hands back for a null string, by version)."""
    if v is None or type(v).__name__ == "NAType":
        return True
    return isinstance(v, float) and v != v


def soda_rows(rows, columns: list[str]) -> list[dict]:
    """SODA leaves null fields out of a row entirely. Put every column back, as a string (or None)."""
    return [{c: (None if _missing(r.get(c)) else str(r[c])) for c in columns} for r in rows]


def fetch_park_polygons() -> dict:
    """Every park's boundary as GeoJSON (~600 parks, one request)."""
    url = f"https://data.cityofchicago.org/resource/{PARK_POLYGONS}.geojson"
    for attempt in range(1, 4):
        r = requests.get(url, params={"$select": "park_no,park,acres,park_class,the_geom", "$limit": 5000}, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("park polygons request kept failing")


def parse_park_polygons(geojson: dict) -> list[dict]:
    """GeoJSON feature collection -> one row per park, geometry kept as a JSON string."""
    import json

    if not isinstance(geojson, dict) or not isinstance(geojson.get("features"), list):
        raise ValueError("malformed park polygons: no `features` list")
    rows = []
    for f in geojson["features"]:
        props, geom = f.get("properties") or {}, f.get("geometry")
        if props.get("park_no") is None or geom is None:
            continue
        rows.append({"park_no": str(props["park_no"]), "park": props.get("park"), "geometry": json.dumps(geom)})
    return rows


# ---------------------------------------------------------------------------------------------
# setlist.fm
# ---------------------------------------------------------------------------------------------

class SetlistFm:
    """setlist.fm client with a per-run request cap and polite spacing. 429s back off and retry."""

    def __init__(self, api_key: str, max_requests: int = SETLISTFM_MAX_PER_RUN):
        self.key, self.left = api_key, max_requests

    def venue_page(self, venue_id: str, page: int) -> dict | None:
        """One page of a venue's setlists, newest first. None past the last page."""
        for backoff in (5, 20, 60):
            if self.left <= 0:
                raise QuotaSpent(f"setlist.fm per-run cap reached at venue {venue_id} p{page}")
            self.left -= 1
            r = requests.get(f"{SETLISTFM}/venue/{venue_id}/setlists", params={"p": page}, timeout=60,
                             headers={"x-api-key": self.key, "Accept": "application/json"})
            time.sleep(SETLISTFM_SPACING_S)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"setlist.fm still rate-limiting at venue {venue_id} p{page}")


class QuotaSpent(RuntimeError):
    pass


def parse_setlist_page(body: dict, venue_id: str) -> list[dict]:
    """One setlist.fm page -> one row per setlist (one per act, openers included)."""
    if not isinstance(body, dict) or not isinstance(body.get("setlist", []), list):
        raise ValueError("malformed setlist.fm page: `setlist` is not a list")
    rows = []
    for s in body.get("setlist", []):
        rows.append({
            "setlist_id": s["id"], "venue_id": venue_id,
            "event_date": datetime.strptime(s["eventDate"], "%d-%m-%Y").date(),
            "artist": (s.get("artist") or {}).get("name"),
            "n_songs": sum(len(st.get("song", [])) for st in (s.get("sets") or {}).get("set", [])),
            "last_updated": s.get("lastUpdated"),
        })
    return rows


def pull_venue(client: SetlistFm, venue_id: str, newest_known: date | None) -> list[dict]:
    """New and updated setlists for one venue, newest first.

    With history on file, pages back until a page ends before newest_known -
    SETLISTFM_LOOKBACK_DAYS (fans log shows late). Usually that's one page. Without, it
    backfills to EVENTS_START. The client's per-run cap stops a long backfill, and the next
    run resumes it."""
    stop_before = (newest_known - timedelta(days=SETLISTFM_LOOKBACK_DAYS)) if newest_known else EVENTS_START
    rows, page = [], 1
    while True:
        body = client.venue_page(venue_id, page)
        if body is None:
            break
        got = parse_setlist_page(body, venue_id)
        rows += got
        oldest = min((r["event_date"] for r in got), default=None)
        if oldest is None or oldest < stop_before or page * body.get("itemsPerPage", 20) >= body.get("total", 0):
            break
        page += 1
    return rows


# ---------------------------------------------------------------------------------------------
# Ticketmaster
# ---------------------------------------------------------------------------------------------

def tm_page(api_key: str, start: datetime, end: datetime, page: int) -> dict:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    for attempt in range(1, 4):
        r = requests.get(TICKETMASTER, timeout=60, params={
            "apikey": api_key, "city": "Chicago", "stateCode": "IL", "countryCode": "US",
            "startDateTime": start.strftime(fmt), "endDateTime": end.strftime(fmt),
            "sort": "date,asc", "size": TM_PAGE, "page": page})
        time.sleep(TM_SPACING_S)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * attempt)
            continue
        r.raise_for_status()
        body = r.json()
        if "page" not in body:
            raise ValueError(f"malformed Ticketmaster page: {str(body)[:300]}")
        return body
    raise RuntimeError("Ticketmaster kept failing")


def tm_flatten(e: dict) -> dict:
    start = e.get("dates", {}).get("start", {})
    cls = (e.get("classifications") or [{}])[0]
    venue = (e.get("_embedded", {}).get("venues") or [{}])[0]
    loc = venue.get("location") or {}
    out = {
        "tm_event_id": e["id"], "name": e.get("name"),
        "local_date": start.get("localDate"), "local_time": start.get("localTime"),
        "start_utc": start.get("dateTime"), "status": e.get("dates", {}).get("status", {}).get("code"),
        "segment": (cls.get("segment") or {}).get("name"), "genre": (cls.get("genre") or {}).get("name"),
        "venue_id": venue.get("id"), "venue_name": venue.get("name"),
        "venue_address": (venue.get("address") or {}).get("line1"),
        "venue_lat": loc.get("latitude"), "venue_lon": loc.get("longitude"),
    }
    return {k: (None if v is None else str(v)) for k, v in out.items()}


def pull_ticketmaster(api_key: str, now: datetime | None = None) -> list[dict]:
    """Every upcoming Chicago event for the next year, one row per event. 30-day windows, halved
    whenever one holds more than deep paging can reach."""
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    windows = [(now + timedelta(days=d), now + timedelta(days=d + 30)) for d in range(0, TM_DAYS_AHEAD, 30)]
    events: dict[str, dict] = {}
    while windows:
        a, b = windows.pop(0)
        first = tm_page(api_key, a, b, 0)
        if first["page"]["totalElements"] > TM_MAX_RESULTS:
            mid = a + (b - a) / 2
            windows[:0] = [(a, mid), (mid, b)]
            continue
        pages = [first] + [tm_page(api_key, a, b, p) for p in range(1, first["page"]["totalPages"])]
        for pg in pages:
            for e in pg.get("_embedded", {}).get("events", []):
                events[e["id"]] = tm_flatten(e)   # windows share an edge, so key on id
    return list(events.values())
