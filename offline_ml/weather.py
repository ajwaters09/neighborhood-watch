"""Weather history, archived weather forecasts, and a holiday calendar for the 1-week track.

    offline_ml/.venv/bin/python offline_ml/weather.py            # ~30 s the first time, no key
    offline_ml/.venv/bin/python offline_ml/weather.py --refresh  # refetch instead of using the cache

Sources (Open-Meteo, free for non-commercial use, no key), all at one point near the middle of
the city. Weather is used as a citywide signal, so one point is enough:
    observed       ERA5 reanalysis, daily, from 1996: what the weather actually was. It trains the
                   weather effect and gives the normals that anomalies are measured against.
    forecasts      The Previous Runs API: for every hour, what the forecast issued 1-7 days
                   earlier said. This is what makes an honest backtest possible: each test week
                   gets the forecast that existed at its cutoff, not the weather that happened.
                   Coverage differs by model (checked 2026-09-26):
                     jma_seamless  temperature and precipitation, leads 1-7, from 2021
                     gfs_seamless  temperature from 2021, precipitation only from 2024
                     ecmwf / gem   from 2024 only
Holidays are computed from their rules. There's no source to fetch.

Outputs (offline_data/weather/):
    observed_daily.parquet   day, t_mean, t_max (deg C), precip (mm), snow (cm)
    forecast_daily.parquet   day, model, lead, t_mean, precip: the forecast for `day` issued
                             `lead` days earlier (24 hourly values each, local time)
    holidays.parquet         day, holiday
    raw/*.json               API responses, cached
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import requests

from config import PRECIP_FORECAST_MODEL, TEMP_FORECAST_MODEL, TEST_START, WEATHER

RAW = WEATHER / "raw"
LAT, LON = 41.84, -87.68          # near the city's geographic middle
TZ = "America/Chicago"
OBSERVED_START = date(1996, 1, 1)  # normals look back 10 years from the first training cutoffs (2007)
FORECAST_START = date(2021, 1, 1)  # nothing older exists in the forecast archive
FORECAST_MODELS = ["jma_seamless", "gfs_seamless"]
LEADS = range(1, 8)


# ---------------------------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------------------------

def cached_get(name: str, url: str, params: dict, refresh: bool) -> dict:
    path = RAW / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    r = requests.get(url, params=params, timeout=600)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise SystemExit(f"{name}: {body}")
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return body


def observed(refresh: bool) -> pl.DataFrame:
    body = cached_get("observed_era5", "https://archive-api.open-meteo.com/v1/archive", dict(
        latitude=LAT, longitude=LON, timezone=TZ, start_date=OBSERVED_START.isoformat(),
        end_date=(date.today() - timedelta(days=1)).isoformat(),
        daily="temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum"), refresh)
    d = body["daily"]
    return pl.DataFrame({
        "day": d["time"], "t_mean": d["temperature_2m_mean"], "t_max": d["temperature_2m_max"],
        "precip": d["precipitation_sum"], "snow": d["snowfall_sum"],
    }, schema_overrides={"t_mean": pl.Float64, "t_max": pl.Float64, "precip": pl.Float64, "snow": pl.Float64}
    ).with_columns(pl.col("day").str.to_date()).drop_nulls()   # ERA5 runs ~5 days behind


def forecasts(model: str, refresh: bool) -> pl.DataFrame:
    """Hourly previous-run values, reduced to one row per (day, lead).

    `temperature_2m_previous_dayN` at hour h is what the run from N days earlier forecast for
    hour h. A day's value at lead N needs all 24 of its hours, so a gap in the archive drops that
    day rather than averaging over part of it.
    """
    hourly = [f"{v}_previous_day{n}" for v in ("temperature_2m", "precipitation") for n in LEADS]
    body = cached_get(f"forecast_{model}", "https://previous-runs-api.open-meteo.com/v1/forecast", dict(
        latitude=LAT, longitude=LON, timezone=TZ, models=model, hourly=",".join(hourly),
        start_date=FORECAST_START.isoformat(), end_date=(date.today() - timedelta(days=1)).isoformat()), refresh)
    h = pl.DataFrame(body["hourly"], infer_schema_length=None).with_columns(
        pl.col("time").str.to_datetime().dt.date().alias("day"))
    per_lead = []
    for n in LEADS:
        t, p = f"temperature_2m_previous_day{n}", f"precipitation_previous_day{n}"
        per_lead.append(
            h.group_by("day").agg(
                pl.col(t).cast(pl.Float64).mean().alias("t_mean"), pl.col(p).cast(pl.Float64).sum().alias("precip"),
                pl.col(t).is_not_null().sum().alias("_nt"), pl.col(p).is_not_null().sum().alias("_np"))
            .with_columns(pl.when(pl.col("_nt") == 24).then("t_mean").alias("t_mean"),
                          pl.when(pl.col("_np") == 24).then("precip").alias("precip"))
            .filter(pl.col("t_mean").is_not_null() | pl.col("precip").is_not_null())
            .select("day", pl.lit(model).alias("model"), pl.lit(n, dtype=pl.Int8).alias("lead"), "t_mean", "precip"))
    return pl.concat(per_lead).sort("day", "lead")


def issued_forecast(fc: pl.DataFrame) -> pl.DataFrame:
    """The forecast the crime model uses, per (day, lead): TEMP_FORECAST_MODEL's temperature,
    with PRECIP_FORECAST_MODEL's filling its gaps, and PRECIP_FORECAST_MODEL's precipitation."""
    t = fc.filter(pl.col("model") == TEMP_FORECAST_MODEL).select("day", "lead", pl.col("t_mean").alias("_t"))
    p = fc.filter(pl.col("model") == PRECIP_FORECAST_MODEL).select("day", "lead", pl.col("t_mean").alias("_t_fill"), "precip")
    return (
        p.join(t, on=["day", "lead"], how="full", coalesce=True)
        .select("day", "lead", pl.coalesce("_t", "_t_fill").alias("t_mean"), "precip")
        .sort("day", "lead")
    )


# ---------------------------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------------------------

def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of the month; n = -1 is the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def easter(year: int) -> date:
    """Western Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l + 90) // 2530
    month = (h + l - 7 * m + 90) // 25
    return date(year, month, (h + l - 7 * m + 33 * month + 19) % 32)


def holidays(first_year: int, last_year: int) -> pl.DataFrame:
    """Federal holidays on their actual dates (not the observed Friday/Monday), plus a few days
    that plausibly change how the city moves: Easter, Halloween, Christmas Eve, New Year's Eve."""
    rows = []
    for y in range(first_year, last_year + 1):
        days = {
            "new_years_day": date(y, 1, 1),
            "mlk_day": nth_weekday(y, 1, 0, 3),
            "presidents_day": nth_weekday(y, 2, 0, 3),
            "easter": easter(y),
            "memorial_day": nth_weekday(y, 5, 0, -1),
            "independence_day": date(y, 7, 4),
            "labor_day": nth_weekday(y, 9, 0, 1),
            "columbus_day": nth_weekday(y, 10, 0, 2),
            "halloween": date(y, 10, 31),
            "veterans_day": date(y, 11, 11),
            "thanksgiving": nth_weekday(y, 11, 3, 4),
            "christmas_eve": date(y, 12, 24),
            "christmas": date(y, 12, 25),
            "new_years_eve": date(y, 12, 31),
        }
        if y >= 2021:                       # a federal holiday since 2021
            days["juneteenth"] = date(y, 6, 19)
        rows += [{"day": d, "holiday": name} for name, d in days.items()]
    return pl.DataFrame(rows).sort("day")


# ---------------------------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------------------------

NORMAL_YEARS = 10
NORMAL_HALF_WINDOW = 7    # +-7 days around the same date: 15 days x 10 years = 150 values


def daily_normals(obs: pl.DataFrame) -> pl.DataFrame:
    """Each day's normal temperature and precipitation: the mean over the same date +-7 days in
    each of the previous 10 years. Only earlier years go in, so a normal never includes the day
    it's the normal for, and a warming trend shows up as a small warm anomaly, not a leak."""
    obs = obs.sort("day")
    assert (obs["day"].diff().drop_nulls() == timedelta(days=1)).all()
    out = {}
    for col in ("t_mean", "precip"):
        x = obs[col].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(x)])
        win = 2 * NORMAL_HALF_WINDOW + 1
        total = np.zeros(len(x))
        ok = np.ones(len(x), dtype=bool)
        idx = np.arange(len(x))
        for k in range(1, NORMAL_YEARS + 1):
            centre = idx - round(365.25 * k)
            lo, hi = centre - NORMAL_HALF_WINDOW, centre + NORMAL_HALF_WINDOW + 1
            ok &= lo >= 0
            lo_c, hi_c = np.clip(lo, 0, len(x)), np.clip(hi, 0, len(x))
            total += (csum[hi_c] - csum[lo_c]) / win
        out[f"{col}_normal"] = np.where(ok, total / NORMAL_YEARS, np.nan)
    return obs.select("day").with_columns(
        pl.Series(k, v).fill_nan(None) for k, v in out.items())


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------

def forecast_skill(obs: pl.DataFrame, fc: pl.DataFrame) -> pl.DataFrame:
    """How far off each model's forecasts were, on the quantity the crime model uses: a
    Monday-start week's departure from normal, with day i of the week forecast at lead i + 1
    (the whole week forecast from the day before the cutoff). Test period only.

    Correlations are on anomalies. Raw weekly temperature correlates ~0.99 with anything that
    knows the seasons, which says nothing about forecasting.
    """
    norm = daily_normals(obs)
    both = pl.concat([fc, issued_forecast(fc).with_columns(pl.lit("used").alias("model")).select(fc.columns)])
    week = (
        both.with_columns(pl.col("day").dt.truncate("1w").alias("week"))
        .filter((pl.col("lead") == (pl.col("day") - pl.col("week")).dt.total_days() + 1)
                & (pl.col("week") >= TEST_START))
        .join(obs.select("day", pl.col("t_mean").alias("t_obs"), pl.col("precip").alias("p_obs")), on="day")
        .join(norm, on="day")
        .group_by("model", "week")
        .agg((pl.col("t_mean") - pl.col("t_mean_normal")).mean().alias("t_fc"),
             (pl.col("t_obs") - pl.col("t_mean_normal")).mean().alias("t_ob"),
             (pl.col("precip") - pl.col("precip_normal")).sum().alias("p_fc"),
             (pl.col("p_obs") - pl.col("precip_normal")).sum().alias("p_ob"),
             pl.col("precip").is_not_null().sum().alias("np"), pl.len().alias("n"))
        .filter(pl.col("n") == 7)
    )
    return (
        week.group_by("model").agg(
            pl.len().alias("weeks"),
            (pl.col("t_fc") - pl.col("t_ob")).mean().round(2).alias("temp_bias_C"),
            (pl.col("t_fc") - pl.col("t_ob")).abs().mean().round(2).alias("temp_mae_C"),
            pl.col("t_ob").std().round(2).alias("obs_anomaly_sd_C"),
            pl.corr("t_fc", "t_ob").round(3).alias("temp_anomaly_corr"),
            pl.corr(pl.col("p_fc").filter(pl.col("np") == 7), pl.col("p_ob").filter(pl.col("np") == 7))
            .round(3).alias("precip_anomaly_corr"))
        .sort("model")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch instead of reading raw/")
    args = ap.parse_args()
    WEATHER.mkdir(parents=True, exist_ok=True)

    obs = observed(args.refresh)
    fc = pl.concat([forecasts(m, args.refresh) for m in FORECAST_MODELS])
    hol = holidays(OBSERVED_START.year, date.today().year + 1)

    # Self-checks: observed days are unique and contiguous, holidays land where they should.
    assert obs["day"].is_unique().all()
    assert (obs["day"].diff().drop_nulls() == timedelta(days=1)).all(), "gap in the observed series"
    assert fc.select("day", "model", "lead").is_unique().all()
    known = {("thanksgiving", 2025): date(2025, 11, 27), ("easter", 2024): date(2024, 3, 31),
             ("memorial_day", 2026): date(2026, 5, 25), ("mlk_day", 2022): date(2022, 1, 17),
             ("labor_day", 2023): date(2023, 9, 4)}
    for (name, y), d in known.items():
        assert hol.filter((pl.col("holiday") == name) & (pl.col("day").dt.year() == y))["day"].item() == d, name

    obs.write_parquet(WEATHER / "observed_daily.parquet")
    fc.write_parquet(WEATHER / "forecast_daily.parquet")
    hol.write_parquet(WEATHER / "holidays.parquet")

    print(f"observed: {obs['day'].min()} .. {obs['day'].max()} ({obs.height:,} days)")
    cover = (fc.group_by("model", "lead").agg(pl.col("t_mean").is_not_null().sum().alias("t_days"),
                                              pl.col("precip").is_not_null().sum().alias("p_days"),
                                              pl.col("day").filter(pl.col("precip").is_not_null()).min().alias("precip_from"),
                                              pl.col("day").min().alias("from"), pl.col("day").max().alias("to"))
             .filter(pl.col("lead").is_in([1, 7])).sort("model", "lead"))
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=160, tbl_hide_dataframe_shape=True):
        print("forecast archive coverage (leads 1 and 7):")
        print(cover)
        print(f"\nweekly anomaly, forecast vs observed, {TEST_START} on "
              f"(the model uses {TEMP_FORECAST_MODEL} temperature, {PRECIP_FORECAST_MODEL} precipitation):")
        print(forecast_skill(obs, fc))
    print(f"\nholidays: {hol.height} dates, {hol['holiday'].n_unique()} kinds, {hol['day'].min().year}-{hol['day'].max().year}")


if __name__ == "__main__":
    main()
