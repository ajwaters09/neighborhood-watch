"""Pull raw event data into offline_data/events/raw/.

    offline_ml/.venv/bin/python offline_ml/events_fetch.py permits        # ~1 min
    offline_ml/.venv/bin/python offline_ml/events_fetch.py setlistfm      # resumable; re-run daily
    offline_ml/.venv/bin/python offline_ml/events_fetch.py ticketmaster   # upcoming-events snapshot

Sources:
    permits       CDOT street-use permits (festivals, block parties, parades, runs, rallies),
                  Chicago Park District outdoor event permits, park polygons, and community
                  area names, all from the city data portal. No key needed.
    setlistfm     Shows at the clubs in CLUBS since EVENTS_START. setlist.fm allows 1,440
                  requests a day, so each run spends up to the cap and the next run picks up
                  where it stopped. Every page is cached, so nothing is fetched twice.
    ticketmaster  Upcoming events in Chicago. The Discovery API has no history (a past week
                  returns zero events), so this is the look-ahead source only. Each run saves a
                  dated snapshot.

Keys come from offline_ml/.env (gitignored): SETLISTFM_API_KEY, TICKETMASTER_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests

from config import EVENTS, EVENTS_START

RAW = EVENTS / "raw"
PORTAL = "https://data.cityofchicago.org/resource"
SODA_PAGE = 50_000

CDOT_PERMITS = "pubx-yq2d"
PARK_PERMITS = "pk66-w54g"
PARK_POLYGONS = "ejsh-fztr"
COMMUNITY_AREAS = "igwz-8jzy"

# CDOT work types that can mean people gathering in the street. Pulled broadly; events_build.py
# decides what counts. Filming (54k permits) and the construction types are left out.
CDOT_WORK_TYPES = ["Festival", "Block Party", "Parade", "Athletic", "Assembly", "Street Closure"]
# The applicant contact columns (names, home addresses) are skipped: nothing downstream needs them.
CDOT_COLUMNS = [
    "uniquekey", "applicationnumber", "applicationname", "applicationdescription",
    "worktypedescription", "applicationstatus", "currentmilestone", "applicationstartdate",
    "applicationenddate", "applicationissueddate", "streetnumberfrom", "streetnumberto",
    "direction", "streetname", "suffix", "placement", "streetclosure", "detail", "ward",
    "latitude", "longitude",
]
# requestor_ (a person's name) is skipped for the same reason.
PARK_COLUMNS = [
    "organization", "park_number", "park_facility_name", "reservation_start_date",
    "reservation_end_date", "event_type", "event_description", "permit_status",
]

SETLISTFM = "https://api.setlist.fm/rest/1.0"
SETLISTFM_CAP = 1400        # published limit is 1,440/day; the gap covers hand-run lookups
SETLISTFM_SPACING_S = 1.0   # published limit is 2/s, but 0.6 s spacing still drew a 429

# Clubs to pull, in priority order: when a run hits the cap, the rest waits for the next run.
# (key, name, street address, lat, lon, setlist.fm venue ids). Coordinates are the Census
# geocoder's match on the address, since setlist.fm venues only carry their city's coordinates.
# A club with several ids is one address listed under several room names. Addresses were checked
# against Ticketmaster's venue records where the club sells through it, which corrected Concord's.
# Left out on purpose to save quota: Metro and Smartbar (one venue, 3730 N Clark, ~10,000
# setlists) and House of Blues (~7,500). A partial House of Blues pull sits in its cache
# folder (23d6206b) unused.
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

TICKETMASTER = "https://app.ticketmaster.com/discovery/v2/events.json"
TM_PAGE = 200
TM_MAX_RESULTS = 1000       # deep paging stops at page * size = 1,000, so big windows get split
TM_DAYS_AHEAD = 365
TM_SPACING_S = 0.25         # limit is 5/s


def env(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip()
    raise SystemExit(f"{name} is missing from offline_ml/.env")


# --- city portal -----------------------------------------------------------------------------

def soda(dataset: str, columns: list[str], where: str) -> pl.DataFrame:
    rows: list[dict] = []
    while True:
        r = requests.get(f"{PORTAL}/{dataset}.json", timeout=300, params={
            "$select": ",".join(columns), "$where": where, "$order": ":id",
            "$limit": SODA_PAGE, "$offset": len(rows)})
        r.raise_for_status()
        page = r.json()
        rows += page
        if len(page) < SODA_PAGE:
            return pl.from_dicts(rows, schema={c: pl.String for c in columns})


def fetch_permits() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    since = EVENTS_START.isoformat()

    types = ",".join(f"'{t}'" for t in CDOT_WORK_TYPES)
    cdot = soda(CDOT_PERMITS, CDOT_COLUMNS,
                f"worktypedescription in ({types}) AND applicationstartdate >= '{since}'")
    cdot.write_parquet(RAW / "cdot_permits.parquet")
    print(f"cdot_permits: {cdot.height:,} rows")

    parks = soda(PARK_PERMITS, PARK_COLUMNS, f"reservation_start_date >= '{since}'")
    parks.write_parquet(RAW / "park_permits.parquet")
    print(f"park_permits: {parks.height:,} rows")

    r = requests.get(f"{PORTAL}/{PARK_POLYGONS}.geojson", timeout=300, params={
        "$select": "park_no,park,acres,park_class,the_geom", "$limit": 5000})
    r.raise_for_status()
    (RAW / "park_polygons.geojson").write_text(r.text)
    print(f"park_polygons: {len(r.json()['features'])} parks")

    # names for the look-ahead list; the crime file already carries the area number
    areas = soda(COMMUNITY_AREAS, ["area_numbe", "community"], "area_numbe IS NOT NULL")
    areas.write_parquet(RAW / "community_areas.parquet")
    print(f"community_areas: {areas.height}")


# --- setlist.fm ------------------------------------------------------------------------------

class Quota:
    """Counts requests over a rolling 24 h. setlist.fm doesn't say when its day starts, and
    staying under the cap in every 24 h window keeps every calendar day under it too."""

    def __init__(self, path: Path, cap: int):
        self.path, self.cap = path, cap
        self.stamps: list[float] = json.loads(path.read_text()) if path.exists() else []

    def left(self) -> int:
        self.stamps = [t for t in self.stamps if t > time.time() - 86_400]
        return self.cap - len(self.stamps)

    def spend(self) -> None:
        self.stamps.append(time.time())
        self.path.write_text(json.dumps(self.stamps))


def setlistfm_get(path: str, key: str, quota: Quota, **params) -> requests.Response | None:
    for backoff in (5, 20, 60):
        quota.spend()
        r = requests.get(f"{SETLISTFM}{path}", params=params, timeout=60,
                         headers={"x-api-key": key, "Accept": "application/json"})
        time.sleep(SETLISTFM_SPACING_S)
        if r.status_code != 429:
            return r
        time.sleep(backoff)
    return None


def fetch_setlistfm(cap: int) -> None:
    """Page through each venue's setlists, newest first, until one predates EVENTS_START.

    Pages shift by a row whenever someone adds an older show between runs, so a resumed venue
    can repeat a few setlists at a page boundary. events_build.py dedupes on setlist id.
    """
    key = env("SETLISTFM_API_KEY")
    root = RAW / "setlistfm"
    root.mkdir(parents=True, exist_ok=True)
    quota = Quota(root / "_requests.json", cap)

    for _, name, _, _, _, ids in CLUBS:
        for vid in ids:
            d = root / vid
            if (d / "_done").exists():
                continue
            d.mkdir(exist_ok=True)
            page = len(list(d.glob("p*.json"))) + 1
            while True:
                if quota.left() <= 0:
                    print(f"cap of {cap} requests/24 h reached; re-run later to resume at {name} p{page}")
                    return
                r = setlistfm_get(f"/venue/{vid}/setlists", key, quota, p=page)
                if r is None:
                    print(f"still rate-limited after retries at {name} p{page}; stopping")
                    return
                if r.status_code == 404:        # past the last page
                    (d / "_done").write_text(f"404 at p{page}")
                    break
                r.raise_for_status()
                j = r.json()
                (d / f"p{page:04d}.json").write_text(r.text)
                setlists = j.get("setlist", [])
                oldest = min((datetime.strptime(s["eventDate"], "%d-%m-%Y").date()
                              for s in setlists), default=None)
                if oldest is None or oldest < EVENTS_START or page * j["itemsPerPage"] >= j["total"]:
                    (d / "_done").write_text(f"stopped at p{page}, oldest {oldest}, total {j['total']}")
                    print(f"{name} [{vid}]: {page} pages, {j['total']:,} setlists all-time, reached {oldest}")
                    break
                page += 1
    print(f"all clubs done; {quota.left()} requests left in this 24 h window")


# --- Ticketmaster ----------------------------------------------------------------------------

def tm_page(key: str, start: datetime, end: datetime, page: int) -> dict:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    r = requests.get(TICKETMASTER, timeout=60, params={
        "apikey": key, "city": "Chicago", "stateCode": "IL", "countryCode": "US",
        "startDateTime": start.strftime(fmt), "endDateTime": end.strftime(fmt),
        "sort": "date,asc", "size": TM_PAGE, "page": page})
    time.sleep(TM_SPACING_S)
    r.raise_for_status()
    return r.json()


def tm_flatten(e: dict) -> dict:
    start = e.get("dates", {}).get("start", {})
    cls = (e.get("classifications") or [{}])[0]
    venue = (e.get("_embedded", {}).get("venues") or [{}])[0]
    loc = venue.get("location") or {}
    return {
        "tm_event_id": e["id"], "name": e.get("name"),
        "local_date": start.get("localDate"), "local_time": start.get("localTime"),
        "start_utc": start.get("dateTime"), "status": e.get("dates", {}).get("status", {}).get("code"),
        "segment": cls.get("segment", {}).get("name"), "genre": cls.get("genre", {}).get("name"),
        "venue_id": venue.get("id"), "venue_name": venue.get("name"),
        "venue_address": venue.get("address", {}).get("line1"),
        "venue_lat": loc.get("latitude"), "venue_lon": loc.get("longitude"),
    }


def fetch_ticketmaster() -> None:
    key = env("TICKETMASTER_API_KEY")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    windows = [(now + timedelta(days=d), now + timedelta(days=d + 30)) for d in range(0, TM_DAYS_AHEAD, 30)]
    events: dict[str, dict] = {}
    while windows:
        a, b = windows.pop(0)
        first = tm_page(key, a, b, 0)
        if first["page"]["totalElements"] > TM_MAX_RESULTS:
            mid = a + (b - a) / 2
            windows[:0] = [(a, mid), (mid, b)]
            continue
        pages = [first] + [tm_page(key, a, b, p) for p in range(1, first["page"]["totalPages"])]
        for pg in pages:
            for e in pg.get("_embedded", {}).get("events", []):
                events[e["id"]] = tm_flatten(e)   # windows share an edge, so key on id
    out = RAW / "ticketmaster"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"upcoming_{date.today().isoformat()}.parquet"
    pl.from_dicts(list(events.values())).write_parquet(path)
    print(f"ticketmaster: {len(events):,} upcoming events -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["permits", "setlistfm", "ticketmaster"])
    ap.add_argument("--cap", type=int, default=SETLISTFM_CAP, help="setlist.fm requests per rolling 24 h")
    args = ap.parse_args()
    if args.source == "permits":
        fetch_permits()
    elif args.source == "setlistfm":
        fetch_setlistfm(args.cap)
    else:
        fetch_ticketmaster()


if __name__ == "__main__":
    main()
