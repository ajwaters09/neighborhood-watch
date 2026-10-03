"""Rank upcoming street festivals and club shows by the extra crime they're likely to bring.

    offline_ml/.venv/bin/python offline_ml/events_lookahead.py

Needs events_build.py and events_impact.py output (both runs), plus a Ticketmaster snapshot
for club shows (events_fetch.py ticketmaster). Each upcoming event is scored the same way:

    area       Its own r9 cells plus the ring around them, where events_impact.py found the lift.
    baseline   Crimes normally expected there over its dates (festivals: whole days; club shows:
               6pm-3am). That's each cell's mean on the same weekday within ±4 weeks of the same
               date in each of the last 3 years, skipping days with an event nearby.
    ratio      Its own track record pooled with a prior (Gamma-Poisson):
                   festivals   past editions (same name once years and punctuation are dropped,
                               within 1.5 km); prior = the average lift for festivals its size
                   club shows  the club's past show nights; prior = the average over all clubs
               The prior's weight comes from how much festivals (or clubs) truly differ beyond
               Poisson noise, by method of moments: b = 1 / (prior * cv^2) expected crimes.
    extra      baseline * (ratio - 1), with a 90% interval from the ratio's posterior.

Everything is scored "as of" a date and only sees events and crime before it, so
events_backtest.py can replay past years through the same code. A past event counts as history
once the 4 weeks after it (its comparison window) are known too.

Festivals running more than 4 days (Christkindlmarket, ZooLights) are listed but not scored.
Clubs that don't sell through Ticketmaster (Sleeping Village, Martyrs', Cobra Lounge, the
Hideout, the Promontory) only show up when setlist.fm already lists an upcoming show.

Outputs (offline_data/events/lookahead/): festivals_<today>.csv, club_nights_<today>.csv
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import polars as pl
from h3.api import basic_int as h3
from scipy.stats import gamma

from config import CLEAN, EVENTS
from events_fetch import CLUBS, RAW
from events_impact import CONTROL_WEEKS, CRIME_OUTCOMES, busy_cell_days, daily_counts, event_cell_days

OUT = EVENTS / "lookahead"
OUTCOMES = ["crime", "violent"]
BASELINE_YEARS = 3
BASELINE_WEEKS = range(-4, 5)
MIN_BASELINE_DAYS = 8          # of up to 27; fewer falls back to the cell's plain daily mean
MIN_CV = 0.05                  # floor on the between-group spread, so the prior never becomes absolute
MATCH_KM = 1.5                 # "Oktoberfest" in Edison Park isn't the one in Lincoln Square
PLACEHOLDER_NAMES = {"test"}   # applications with names like "test" or "k" aren't events
ADDRESS_SLACK = 10             # house numbers that still count as one venue (Reggies spans 2105-2109)
# A past event's measured O/E compares it with the same weekday up to 4 weeks *after* it, so it
# only counts as history once those weeks are known too. Without this, scoring "as of" a date
# would quietly use crime from after it.
SETTLE_DAYS = 7 * max(CONTROL_WEEKS)


# --- shared scoring ----------------------------------------------------------------------------

def load_counts(night: bool) -> tuple[pl.DataFrame, date]:
    """Crime per cell per day (or per 6pm-3am night), and the last date complete enough to use."""
    where = ~pl.col("date_is_month_placeholder") & (pl.col("occurred_at") >= pl.datetime(2013, 1, 1))
    if night:   # unknown times are logged at midnight, inside the night
        where &= ~pl.col("time_unknown")
    counts = daily_counts(CLEAN / "crimes.parquet", "occurred_at", {k: CRIME_OUTCOMES[k] for k in OUTCOMES},
                          where, night)
    return counts, counts["date"].max() - timedelta(days=7)   # the newest week is still filling in


def area_rows(targets: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    """(event_id, ring, h3_r9) for each event's own cells (ring 0) and the ring around them."""
    own = cells.join(targets.select("event_id"), on="event_id")
    rows = []
    for eid, cs in own.group_by("event_id").agg(pl.col("h3_r9")).iter_rows():
        ring1 = set().union(*(h3.grid_ring(c, 1) for c in cs)) - set(cs)
        rows += [(eid, 0, c) for c in cs] + [(eid, 1, c) for c in ring1]
    return pl.DataFrame(rows, schema={"event_id": pl.String, "ring": pl.Int8, "h3_r9": pl.Int64}, orient="row")


def baselines(targets: pl.DataFrame, cells: pl.DataFrame, busy: pl.DataFrame, counts: pl.DataFrame,
              known_until: date) -> pl.DataFrame:
    """Crime normally expected over each target's area and dates, from the years before known_until."""
    first = known_until - timedelta(days=365 * BASELINE_YEARS)
    area = area_rows(targets, cells).select("event_id", "h3_r9")
    cell_days = event_cell_days(targets, area)
    keys = cell_days.select("h3_r9", "date").unique()
    offsets = pl.DataFrame({"k": [-364 * y + 7 * w for y in range(1, BASELINE_YEARS + 1) for w in BASELINE_WEEKS]})
    same_days = (keys.join(offsets, how="cross")
                 .with_columns(past=pl.col("date") + pl.duration(days=pl.col("k")))
                 .filter(pl.col("past").is_between(first, known_until))
                 .join(busy.rename({"date": "past"}), on=["h3_r9", "past"], how="anti")
                 .join(counts.rename({"date": "past"}), on=["h3_r9", "past"], how="left").fill_null(0)
                 .group_by("h3_r9", "date").agg(pl.len().alias("n"), *[pl.mean(k) for k in OUTCOMES]))
    n_days = (known_until - first).days + 1
    plain = (counts.filter(pl.col("date").is_between(first, known_until)).group_by("h3_r9")
             .agg([(pl.sum(k) / n_days).alias(f"plain_{k}") for k in OUTCOMES]))
    base = (keys.join(same_days, on=["h3_r9", "date"], how="left")
            .join(plain, on="h3_r9", how="left")
            .select("h3_r9", "date", *[
                pl.when(pl.col("n") >= MIN_BASELINE_DAYS).then(pl.col(k))
                .otherwise(pl.col(f"plain_{k}")).fill_null(0).alias(k) for k in OUTCOMES]))
    return (cell_days.join(base, on=["h3_r9", "date"])
            .group_by("event_id").agg([pl.sum(k).alias(f"baseline_{k}") for k in OUTCOMES]))


def event_oe(oe: pl.DataFrame, past: pl.DataFrame) -> pl.DataFrame:
    """Observed and expected per past event, own cells + ring 1, on its event days."""
    return (oe.filter((pl.col("timing") == "event") & (pl.col("ring") <= 1))
            .join(past.select("event_id"), on="event_id")
            .group_by("event_id").agg([pl.sum(f"{p}_{k}") for p in "OE" for k in OUTCOMES]))


def between_cv(items: pl.DataFrame, group: list[str], bucket: str, min_events: int = 1) -> float:
    """How much groups' true lifts differ, as a share of their bucket's average, beyond Poisson
    noise (method of moments on O/E). items holds O_crime and E_crime per past event."""
    g = (items.group_by(group + [bucket]).agg(pl.sum("O_crime"), pl.sum("E_crime"), n=pl.len())
         .filter((pl.col("n") >= min_events) & (pl.col("E_crime") > 0)))
    g = g.join(g.group_by(bucket).agg(m=pl.sum("O_crime") / pl.sum("E_crime")), on=bucket)
    w = g["E_crime"] / g["E_crime"].sum()
    rel = (g["O_crime"] / g["E_crime"] - g["m"]) / g["m"]
    var = (w * rel**2).sum() - (w / (g["m"] * g["E_crime"])).sum()
    return max(var, MIN_CV**2) ** 0.5


def with_priors(items: pl.DataFrame, bucket: str, cv: float) -> pl.DataFrame:
    """Per bucket: the pooled lift and the prior's weight b (in expected crimes). The violent
    prior borrows the crime cv; violent counts are too thin to estimate their own."""
    return (items.group_by(bucket).agg(pl.len().alias("n_past"),
                                       *[(pl.sum(f"O_{k}") / pl.sum(f"E_{k}")).alias(f"prior_{k}") for k in OUTCOMES])
            .with_columns([(1 / (pl.col(f"prior_{k}") * cv**2)).alias(f"b_{k}") for k in OUTCOMES]))


def apply_posterior(df: pl.DataFrame) -> pl.DataFrame:
    """Needs O_k, E_k (the event's own history), prior_k, b_k and baseline_k for each outcome."""
    df = df.with_columns(*[pl.col(f"{p}_{k}").fill_null(0.0) for p in "OE" for k in OUTCOMES])
    for k in OUTCOMES:
        shape = df[f"O_{k}"] + df[f"b_{k}"] * df[f"prior_{k}"]
        rate = df[f"E_{k}"] + df[f"b_{k}"]
        df = df.with_columns(
            pl.Series(f"ratio_{k}", shape / rate),
            pl.Series(f"ratio_{k}_lo", gamma.ppf(0.05, shape, scale=1 / rate)),
            pl.Series(f"ratio_{k}_hi", gamma.ppf(0.95, shape, scale=1 / rate)),
            pl.when(pl.col(f"E_{k}") > 0).then(pl.col(f"O_{k}") / pl.col(f"E_{k}")).alias(f"past_ratio_{k}"))
        df = df.with_columns([(pl.col(f"baseline_{k}") * (pl.col(c) - 1)).alias(c.replace("ratio", "extra"))
                              for c in (f"ratio_{k}", f"ratio_{k}_lo", f"ratio_{k}_hi")])
    return df


# --- festivals ---------------------------------------------------------------------------------

def norm_name(s: str | None) -> str:
    """'Hyde Park Jazz Fest 2026' and 'HYDE PARK JAZZ FEST' -> 'hyde park jazz fest'."""
    s = re.sub(r"\b\d+(st|nd|rd|th)?\b", " ", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", s)).strip()


def size_bucket() -> pl.Expr:
    return (pl.when(pl.col("n_segments") >= 5).then(pl.lit("5+"))
            .when(pl.col("n_segments") >= 2).then(pl.lit("2-4")).otherwise(pl.lit("1")))


def festival_history(past: pl.DataFrame, targets: pl.DataFrame, past_oe: pl.DataFrame) -> pl.DataFrame:
    """Observed and expected crime summed over each target's past editions."""
    past = past.select("event_id", "key", "lat", "lon", year=pl.col("start_date").dt.year())
    pairs = (targets.select("event_id", "key", "lat", "lon").filter(pl.col("key") != "")
             .join(past, on="key", suffix="_past")
             .filter(pl.struct("lat", "lon", "lat_past", "lon_past").map_elements(
                 lambda r: h3.great_circle_distance((r["lat"], r["lon"]), (r["lat_past"], r["lon_past"]), unit="km") <= MATCH_KM,
                 return_dtype=pl.Boolean)))
    return (pairs.join(past_oe, left_on="event_id_past", right_on="event_id", how="left").fill_null(0)
            .group_by("event_id").agg(past_editions=pl.col("year").n_unique(),
                                      past_years=pl.col("year").unique().sort().cast(pl.String).str.join(","),
                                      *[pl.sum(f"{p}_{k}") for p in "OE" for k in OUTCOMES]))


def festival_scores(targets: pl.DataFrame, events: pl.DataFrame, cells: pl.DataFrame, oe: pl.DataFrame,
                    busy: pl.DataFrame, counts: pl.DataFrame, known_until: date) -> tuple[pl.DataFrame, float, pl.DataFrame]:
    """Score target festivals using only festivals and crime up to known_until."""
    keyed = lambda df: df.with_columns(key=pl.col("name").map_elements(norm_name, return_dtype=pl.String),
                                       bucket=size_bucket())
    settled = known_until - timedelta(days=SETTLE_DAYS)
    past = keyed(events.filter((pl.col("status") == "held") & (pl.col("category") == "festival")
                               & ~pl.col("long_run") & (pl.col("end_date") <= settled)))
    past_oe = event_oe(oe, past)
    items = past_oe.join(past.select("event_id", "key", "bucket",
                                     place=pl.col("h3_r9").map_elements(lambda c: h3.cell_to_parent(c, 7), return_dtype=pl.Int64)),
                         on="event_id")
    # a festival's lift is judged across its own editions, so only names seen twice count
    cv = between_cv(items, ["key", "place"], "bucket", min_events=2)
    priors = with_priors(items, "bucket", cv)

    targets = keyed(targets)
    df = (targets.join(baselines(targets, cells, busy, counts, known_until), on="event_id", how="left")
          .join(festival_history(past, targets, past_oe), on="event_id", how="left")
          .join(priors, on="bucket", how="left")
          .with_columns(pl.col("past_editions").fill_null(0)))
    return apply_posterior(df), cv, priors


# --- club shows --------------------------------------------------------------------------------

def street_key(address: str) -> tuple[int | None, list[str]]:
    words = re.sub(r"[^a-z0-9 ]", " ", (address or "").lower()).split()
    return (int(words[0]) if words and words[0].isdigit() else None), words[1:]


def upcoming_club_nights(events: pl.DataFrame, today: date) -> pl.DataFrame:
    """One row per club per upcoming night, from the newest Ticketmaster snapshot plus any
    setlist.fm listings. Ticketmaster venues are matched on street address: its venue names vary
    ("Reggies Rock Club", "Reggie's Music Joint") and some of its coordinates are wrong."""
    snap = sorted((RAW / "ticketmaster").glob("upcoming_*.parquet"))[-1]
    tm = pl.read_parquet(snap).filter(pl.col("segment") != "Sports")
    rows = []
    for key, name, address, lat, lon, _ in CLUBS:
        number, words = street_key(address)
        street = words[-2]   # "2105 S State St" -> "state"
        for r in tm.iter_rows(named=True):
            n, w = street_key(r["venue_address"])
            if n is not None and abs(n - number) <= ADDRESS_SLACK and street in w:
                rows.append({"venue": key, "place": name, "lat": lat, "lon": lon, "start_date": date.fromisoformat(r["local_date"]),
                             "local_time": r["local_time"], "act": r["name"], "source": "ticketmaster"})
    sfm = events.filter((pl.col("source") == "setlistfm") & (pl.col("status") == "planned"))
    rows += [{"venue": r["venue"], "place": r["place"], "lat": r["lat"], "lon": r["lon"], "start_date": r["start_date"],
              "local_time": None, "act": r["name"], "source": "setlistfm"} for r in sfm.iter_rows(named=True)]
    return (pl.from_dicts(rows).filter(pl.col("start_date") >= today)
            .group_by("venue", "start_date").agg(
                pl.first("place"), pl.first("lat"), pl.first("lon"), pl.col("local_time").drop_nulls().min(),
                acts=pl.col("act").unique().sort().str.join(" | "), sources=pl.col("source").unique().sort().str.join("+"))
            .with_columns(end_date=pl.col("start_date"),
                          event_id=pl.concat_str(pl.lit("night"), "venue", pl.col("start_date").cast(pl.String), separator=":"),
                          h3_r9=pl.struct("lat", "lon").map_elements(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 9), return_dtype=pl.Int64)))


def club_scores(targets: pl.DataFrame, events: pl.DataFrame, oe_clubs: pl.DataFrame, busy: pl.DataFrame,
                counts: pl.DataFrame, known_until: date) -> tuple[pl.DataFrame, float, pl.DataFrame]:
    """Score upcoming club nights using only show nights and crime up to known_until."""
    settled = known_until - timedelta(days=SETTLE_DAYS)
    past = events.filter((pl.col("source") == "setlistfm") & (pl.col("status") == "held")
                         & (pl.col("end_date") <= settled))
    items = event_oe(oe_clubs, past).join(past.select("event_id", "venue"), on="event_id").with_columns(bucket=pl.lit("all"))
    cv = between_cv(items, ["venue"], "bucket")
    priors = with_priors(items, "bucket", cv)
    hist = items.group_by("venue").agg(past_nights=pl.len(), *[pl.sum(f"{p}_{k}") for p in "OE" for k in OUTCOMES])
    cells = targets.select("event_id", "h3_r9")
    df = (targets.join(baselines(targets, cells, busy, counts, known_until), on="event_id", how="left")
          .join(hist, on="venue", how="left")
          .join(priors.drop("bucket"), how="cross")
          .with_columns(pl.col("past_nights").fill_null(0)))
    return apply_posterior(df), cv, priors


# --- live run ----------------------------------------------------------------------------------

def community_areas() -> pl.DataFrame:
    """r9 cell -> the community area most of its crimes are recorded in, with its name."""
    names = (pl.read_parquet(RAW / "community_areas.parquet")
             .select(community_area=pl.col("area_numbe").cast(pl.Int16), community=pl.col("community").str.to_titlecase()))
    by_cell = (pl.scan_parquet(CLEAN / "crimes.parquet").drop_nulls(["h3_r9", "community_area"])
               .group_by("h3_r9", "community_area").len()
               .sort("len", descending=True).group_by("h3_r9").first().collect())
    return by_cell.join(names, on="community_area").select("h3_r9", "community")


def interval(k: str, what: str) -> pl.Expr:
    return pl.format("{} [{}–{}]", *[pl.col(f"{what}_{k}{s}").round(2) for s in ("", "_lo", "_hi")])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    events = pl.read_parquet(EVENTS / "events.parquet")
    cells = pl.read_parquet(EVENTS / "event_cells.parquet")
    oe = pl.read_parquet(EVENTS / "impact" / "oe_by_event.parquet")
    oe_clubs = pl.read_parquet(EVENTS / "impact" / "oe_by_event_clubs.parquet")
    _, busy = busy_cell_days(events, cells)
    areas = community_areas()
    pl.Config.set_tbl_rows(30), pl.Config.set_tbl_width_chars(250), pl.Config.set_fmt_str_lengths(32)

    # festivals
    day_counts, day_known = load_counts(night=False)
    planned = events.filter((pl.col("status") == "planned") & (pl.col("category") == "festival")
                            & (pl.col("name").str.len_chars() >= 3)
                            & ~pl.col("name").str.to_lowercase().is_in(PLACEHOLDER_NAMES))
    upcoming, seasonal = planned.filter(~pl.col("long_run")), planned.filter(pl.col("long_run"))
    fest, cv, priors = festival_scores(upcoming, events, cells, oe, busy, day_counts, day_known)
    fest = (fest.join(areas, on="h3_r9", how="left").sort("extra_crime", descending=True).with_row_index("rank", offset=1)
            .select("rank", "start_date", "end_date", "n_days", "name", "community", "place",
                    "n_segments", "full_closure", "permit_stage", "past_editions", "past_years",
                    "baseline_crime", "past_ratio_crime", "ratio_crime", "ratio_crime_lo", "ratio_crime_hi",
                    "extra_crime", "extra_crime_lo", "extra_crime_hi", "baseline_violent", "extra_violent",
                    "event_id", "lat", "lon"))
    path = OUT / f"festivals_{today.isoformat()}.csv"
    fest.write_csv(path)
    print(f"festivals: between-festival cv {cv:.3f}, crime known through {day_known}")
    print(priors.sort("bucket"))
    print(f"{fest.height} upcoming festivals scored, {fest['extra_crime'].sum():.1f} extra crimes in total -> {path}")
    print(fest.head(15).select("rank", "start_date", "n_days", "name", "community", "n_segments", "past_editions",
                               pl.col("baseline_crime").round(1), interval("crime", "ratio").alias("ratio"),
                               interval("crime", "extra").alias("extra crimes")))
    print(f"not scored, multi-week runs ({seasonal.height}):",
          ", ".join(seasonal.sort("start_date")["name"].to_list()))

    # club shows
    night_counts, night_known = load_counts(night=True)
    nights = upcoming_club_nights(events, today)
    clubs, club_cv, club_prior = club_scores(nights, events, oe_clubs, busy, night_counts, night_known)
    clubs = (clubs.join(areas, on="h3_r9", how="left").sort("extra_crime", descending=True).with_row_index("rank", offset=1)
             .select("rank", "start_date", "local_time", "place", "community", "acts", "sources", "past_nights",
                     "baseline_crime", "past_ratio_crime", "ratio_crime", "ratio_crime_lo", "ratio_crime_hi",
                     "extra_crime", "extra_crime_lo", "extra_crime_hi", "baseline_violent", "extra_violent",
                     "event_id", "lat", "lon"))
    path = OUT / f"club_nights_{today.isoformat()}.csv"
    clubs.write_csv(path)
    print(f"\nclub nights: between-club cv {club_cv:.3f}, all-club prior {club_prior['prior_crime'][0]:.3f}, "
          f"crime known through {night_known}")
    print(f"{clubs.height} upcoming club nights scored, {clubs['extra_crime'].sum():.1f} extra crimes in total -> {path}")
    print(clubs.group_by("place").agg(
              nights=pl.len(), first=pl.min("start_date"), last=pl.max("start_date"),
              ratio=pl.format("{} [{}–{}]", *[pl.first(c).round(2) for c in ("ratio_crime", "ratio_crime_lo", "ratio_crime_hi")]),
              baseline_per_night=pl.mean("baseline_crime").round(2), extra_per_night=pl.mean("extra_crime").round(3),
              extra_total=pl.sum("extra_crime").round(2))
          .sort("extra_total", descending=True))


if __name__ == "__main__":
    main()
