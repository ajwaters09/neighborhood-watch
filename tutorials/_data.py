"""Data for the tutorials, without Databricks.

Each loader returns the same shape the pipeline's tables have, from the first source available:
1. the research exports in `offline_data/` (offline_ml/README.md), if they're on this machine;
2. otherwise the public APIs, aggregated server-side so only small tables come back, and cached
   in `tutorials/.cache/` so a re-run is instant.

Set NW_TUTORIAL_SOURCE=api to skip the local exports. The first API run takes about half an hour
(the 311 history and the daily crime counts since 2001 come a year at a time); after that it's seconds.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import open_meteo as om                                       # noqa: E402
from src.community_areas import AREA_NAMES                            # noqa: E402
from src.constants import CRIME_CATEGORY_RULES, SR_EXCLUDE_FROM_TOTAL, SR_INDICATOR_MAP   # noqa: E402
from src.soda_client import CRIMES_DATASET_ID, SERVICE_REQUESTS_DATASET_ID, SocrataClient  # noqa: E402

OFFLINE = ROOT / "offline_data"
CACHE = ROOT / "tutorials" / ".cache"
GEOJSON = ROOT / "webapp" / "static" / "data" / "community_areas.geojson"
USE_OFFLINE = os.environ.get("NW_TUTORIAL_SOURCE", "auto") != "api"


def source() -> str:
    """Where the loaders read from: 'local exports' or 'public APIs'."""
    return "local exports" if USE_OFFLINE and (OFFLINE / "clean" / "crimes.parquet").exists() else "public APIs"


def _cached(name: str, build) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(path, index=False)
    return df


def _soda(dataset: str, select: str, where: str, group: str) -> pd.DataFrame:
    rows = SocrataClient(timeout=300).query(dataset, select=select, where=where, group=group, order=group)
    return pd.DataFrame(list(rows))


def _soda_by_year(dataset: str, select: str, where: str, group: str, date_col: str, first: str) -> pd.DataFrame:
    """The same grouped query a year at a time: one query over the whole history can time out."""
    parts = []
    for year in range(int(first[:4]), date.today().year + 1):
        lo = max(first, f"{year}-01-01")
        span = f"{date_col} >= '{lo}T00:00:00' AND {date_col} < '{year + 1}-01-01T00:00:00'"
        parts.append(_soda(dataset, select, f"{where} AND {span}", group))
        print(f"  {dataset} {year}: {len(parts[-1]):,} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def _category(primary_type: pd.Series) -> pd.Series:
    lookup = {t: c for c, types in CRIME_CATEGORY_RULES.items() for t in types}
    return "crime_" + primary_type.astype(str).map(lookup).fillna("other")


def _offline_crimes(columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(OFFLINE / "clean" / "crimes.parquet", columns=columns)
    return df[df["community_area"].between(1, 77)]


# ---------------------------------------------------------------------------------------------
# Crime
# ---------------------------------------------------------------------------------------------

def crime_daily_by_area() -> pd.DataFrame:
    """(community_area, day, n): every crime per area per day since 2001, as notebook 08 reads
    silver. The newest day is usually part-loaded."""
    if source() == "local exports":
        df = _offline_crimes(["occurred_at", "community_area"])
        df = df.assign(day=df["occurred_at"].dt.date).groupby(["community_area", "day"]).size().rename("n").reset_index()
    else:
        df = _cached("crime_daily_by_area", lambda: _soda_by_year(
            CRIMES_DATASET_ID, "date_trunc_ymd(date) AS day, community_area, count(*) AS n",
            "community_area IS NOT NULL", "day, community_area", "date", "2001-01-01"))
        df["day"] = pd.to_datetime(df["day"]).dt.date
    df["community_area"], df["n"] = df["community_area"].astype(int), df["n"].astype(int)
    return df[df["community_area"].between(1, 77)].reset_index(drop=True)


def crime_monthly_by_category(since: str = "2018-12-01") -> pd.DataFrame:
    """(community_area, period, metric, metric_count): crime_total and the seven crime_<category>
    metrics per area per month, categories by primary_type (src/constants.CRIME_CATEGORY_RULES)."""
    if source() == "local exports":
        df = _offline_crimes(["occurred_at", "community_area", "primary_type"])
        df = df[df["occurred_at"] >= since]
        df = df.assign(period=df["occurred_at"].dt.to_period("M").dt.to_timestamp().dt.date)
        g = df.groupby(["community_area", "period", "primary_type"], observed=True).size().rename("n").reset_index()
    else:
        g = _cached("crime_monthly_by_type", lambda: _soda_by_year(
            CRIMES_DATASET_ID, "date_trunc_ym(date) AS period, community_area, primary_type, count(*) AS n",
            "community_area IS NOT NULL", "period, community_area, primary_type", "date", since))
        g["period"] = pd.to_datetime(g["period"]).dt.date
    g["n"], g["community_area"] = g["n"].astype(int), g["community_area"].astype(int)
    g = g[g["community_area"].between(1, 77)]
    g["metric"] = _category(g["primary_type"])
    cats = g.groupby(["community_area", "period", "metric"])["n"].sum()
    total = g.groupby(["community_area", "period"])["n"].sum().rename("n").reset_index().assign(metric="crime_total")
    out = pd.concat([cats.reset_index(), total], ignore_index=True)
    return _zero_fill(out.rename(columns={"n": "metric_count"}))


def crime_daily_citywide(days: int = 60) -> pd.DataFrame:
    """(day, n): the last `days` of citywide crime, live from the portal (small and fast), so the
    feed's part-loaded newest days show as they are today."""
    since = (date.today() - timedelta(days=days)).isoformat()
    df = _soda(CRIMES_DATASET_ID, "date_trunc_ymd(date) AS day, count(*) AS n", f"date >= '{since}T00:00:00'", "day")
    return pd.DataFrame({"day": pd.to_datetime(df["day"]).dt.date, "n": df["n"].astype(int)})


def last_full_day(daily: pd.DataFrame, share: float = 0.5) -> date:
    """The newest day with at least `share` of the median count of the 28 days before it, the rule
    notebook 04 uses to skip the feed's part-loaded newest days. `daily` is (day, n)."""
    counts = daily.groupby("day")["n"].sum().to_dict()
    day = max(counts)
    while True:
        prior = sorted(counts.get(day - timedelta(days=i), 0) for i in range(1, 29))
        if counts.get(day, 0) >= share * prior[len(prior) // 2]:
            return day
        day -= timedelta(days=1)


# ---------------------------------------------------------------------------------------------
# 311
# ---------------------------------------------------------------------------------------------

def sr_monthly(since: str = "2019-03-01") -> pd.DataFrame:
    """(community_area, period, metric, metric_count): 311_total and the ten indicator types
    (src/constants.SR_INDICATOR_MAP) per area per month, duplicates left out, as notebook 04
    builds them."""
    if source() == "local exports":
        df = pd.read_parquet(OFFLINE / "clean" / "sr311.parquet", columns=["created_at", "sr_type", "community_area", "is_duplicate"])
        df = df[~df["is_duplicate"] & df["community_area"].between(1, 77) & (df["created_at"] >= since)]
        df = df.assign(period=df["created_at"].dt.to_period("M").dt.to_timestamp().dt.date, sr_type=df["sr_type"].astype(str))
        g = df.groupby(["community_area", "period", "sr_type"]).size().rename("n").reset_index()
    else:
        types = ", ".join(f"'{t}'" for t in SR_INDICATOR_MAP)
        skip = ", ".join(f"'{t}'" for t in SR_EXCLUDE_FROM_TOTAL)
        base = "community_area BETWEEN 1 AND 77 AND duplicate = false"
        by_type = _cached("sr_monthly_by_type", lambda: _soda_by_year(
            SERVICE_REQUESTS_DATASET_ID, "date_trunc_ym(created_date) AS period, community_area, sr_type, count(*) AS n",
            f"{base} AND sr_type IN ({types})", "period, community_area, sr_type", "created_date", since))
        totals = _cached("sr_monthly_total", lambda: _soda_by_year(
            SERVICE_REQUESTS_DATASET_ID, "date_trunc_ym(created_date) AS period, community_area, count(*) AS n",
            f"{base} AND sr_type NOT IN ({skip})", "period, community_area", "created_date", since))
        g = pd.concat([by_type, totals.assign(sr_type="(all)")], ignore_index=True)
        g["period"] = pd.to_datetime(g["period"]).dt.date
    g["n"], g["community_area"] = g["n"].astype(int), g["community_area"].astype(int)
    types = g[g["sr_type"].isin(SR_INDICATOR_MAP)].assign(metric=lambda d: d["sr_type"].map(SR_INDICATOR_MAP))
    in_total = g["sr_type"] == "(all)" if (g["sr_type"] == "(all)").any() else ~g["sr_type"].isin(SR_EXCLUDE_FROM_TOTAL)
    total = (g[in_total].groupby(["community_area", "period"])["n"].sum()
             .rename("n").reset_index().assign(metric="311_total"))
    out = pd.concat([types[["community_area", "period", "metric", "n"]], total], ignore_index=True)
    return _zero_fill(out.rename(columns={"n": "metric_count"}))


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Every (area, month, metric) as a row, zero where nothing happened, as notebook 04 does."""
    months = pd.date_range(min(df["period"]), max(df["period"]), freq="MS").date
    grid = pd.MultiIndex.from_product([range(1, 78), months, sorted(df["metric"].unique())],
                                      names=["community_area", "period", "metric"])
    s = df.groupby(["community_area", "period", "metric"])["metric_count"].sum().reindex(grid, fill_value=0)
    return s.reset_index()


# ---------------------------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------------------------

def _weather_raw(name: str, url: str, params: dict) -> dict:
    local = OFFLINE / "weather" / "raw" / f"{name}.json"
    if USE_OFFLINE and local.exists():
        return json.loads(local.read_text())
    path = CACHE / f"{name}.json"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(om.get_json(url, params)))
    return json.loads(path.read_text())


def weather_observed() -> pd.DataFrame:
    """(day, t_mean, t_max, precip, snow): observed daily weather since 1996 (°C, mm, cm)."""
    end = (date.today() - timedelta(days=1)).isoformat()
    body = _weather_raw("observed_era5", om.ARCHIVE_URL, dict(
        latitude=om.LAT, longitude=om.LON, timezone=om.TZ, start_date=om.OBSERVED_START.isoformat(), end_date=end,
        daily="temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum"))
    return pd.DataFrame(om.parse_observed(body))


def weather_forecasts() -> pd.DataFrame:
    """(target_day, issue_date, lead_days, model, t_mean, precip): what each forecast said 1-7
    days ahead, since 2021, from Open-Meteo's Previous Runs archive."""
    end = (date.today() - timedelta(days=1)).isoformat()
    hourly = [f"{v}_previous_day{n}" for v in ("temperature_2m", "precipitation") for n in om.PREVIOUS_RUN_LEADS]
    parts = []
    for model in om.FORECAST_MODELS:
        body = _weather_raw(f"forecast_{model}", om.PREVIOUS_RUNS_URL, dict(
            latitude=om.LAT, longitude=om.LON, timezone=om.TZ, models=model, hourly=",".join(hourly),
            start_date="2021-01-01", end_date=end))
        parts.append(pd.DataFrame(om.parse_previous_runs(body, model)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------------------------

def choropleth(values: dict[int, float], ax=None, cmap: str = "RdBu_r", vmin=None, vmax=None, title: str = "",
               label: str = ""):
    """A quick community-area map: one color per area from `values` (area number -> number)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6.5))
    v = np.array([x for x in values.values() if x is not None and not np.isnan(x)])
    norm = Normalize(vmin if vmin is not None else v.min(), vmax if vmax is not None else v.max())
    cm = plt.get_cmap(cmap)
    for f in json.loads(GEOJSON.read_text())["features"]:
        area = int(float(f["properties"].get("area_numbe") or f["properties"].get("area_num_1")))
        x = values.get(area)
        color = "#e1e0d9" if x is None or np.isnan(x) else cm(norm(x))
        geom = f["geometry"]
        for poly in geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=color, edgecolor="white", linewidth=0.5))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(41.84)))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11)
    plt.colorbar(ScalarMappable(norm, cm), ax=ax, shrink=0.6, label=label)
    return ax


def area_name(a: int) -> str:
    return AREA_NAMES.get(int(a), f"Area {a}")
