"""Open-Meteo client and response parsers for the weather behind the next-week outlook.

Two feeds, both free for non-commercial use and keyless, at one point near the middle of
the city (weather is used as a citywide signal):
- **observed** (the ERA5 archive): daily mean/max temperature, precipitation, snowfall.
  It trains the weather effect and gives the normals that anomalies are measured against.
  The newest few days are preliminary and get revised, so 02b re-pulls a trailing window
  and MERGEs.
- **forecast**: what the forecast said for each day, and when it was issued. Temperature
  comes from GFS and precipitation from JMA. On 2022+ weeks, GFS tracked the weekly
  temperature anomaly far better (r = 0.94 vs 0.76), and JMA's precipitation was the better
  of the two (0.44 vs 0.39). See offline_ml/weather.py.

The parsers are shared by 01c (seeding from raw responses uploaded to the landing volume) and
02b (live pulls), so a seeded row and a pulled row always mean the same thing. Forecast rows are
(target_day, issue_date, lead_days, model):
- The seed comes from the Previous Runs API: `*_previous_dayN` is the forecast issued N days
  before each hour, so issue_date = target_day - N.
- Live pulls come from the ordinary forecast API, issued today.

Pure Python (requests only), so it's testable off Databricks.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import requests

LAT, LON = 41.84, -87.68
TZ = "America/Chicago"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

TEMP_MODEL = "gfs_seamless"
PRECIP_MODEL = "jma_seamless"
FORECAST_MODELS = (TEMP_MODEL, PRECIP_MODEL)
FORECAST_DAYS = 10           # enough for next Monday-Sunday from any weekday
PREVIOUS_RUN_LEADS = range(1, 8)

# Bronze tables: 01c seeds them from uploaded raw responses, and 02b keeps them current.
OBSERVED_START = date(1996, 1, 1)      # normals look back 10 years from the first training weeks (2007)
OBSERVED_SCHEMA = [("day", "DATE"), ("t_mean", "DOUBLE"), ("t_max", "DOUBLE"), ("precip", "DOUBLE"), ("snow", "DOUBLE")]
OBSERVED_KEYS = ["day"]
FORECAST_SCHEMA = [("target_day", "DATE"), ("issue_date", "DATE"), ("lead_days", "INT"), ("model", "STRING"),
                   ("t_mean", "DOUBLE"), ("precip", "DOUBLE")]
FORECAST_KEYS = ["target_day", "issue_date", "model"]

MAX_RETRIES = 4
RETRY_BACKOFF_S = 3
TIMEOUT_S = 120


# ---------------------------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------------------------

def get_json(url: str, params: dict[str, Any], session: requests.Session | None = None) -> dict:
    """GET with retries on rate limits, server errors and network failures.

    A 4xx other than 429 is a bad request (a wrong variable name or a date out of range) and
    won't fix itself, so it fails at once with Open-Meteo's own message. Open-Meteo can also
    answer 200 with {"error": true, "reason": ...}. That's raised too.
    """
    http = session or requests
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = http.get(url, params=params, timeout=TIMEOUT_S)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"{resp.status_code} from {url}: {resp.text[:300]}")
                time.sleep(RETRY_BACKOFF_S * attempt)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Open-Meteo rejected the request ({resp.status_code}): {resp.text[:500]}\nparams: {params}")
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                raise RuntimeError(f"Open-Meteo error: {body.get('reason')}\nparams: {params}")
            return body
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            last = exc
            time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Open-Meteo request to {url} failed after {MAX_RETRIES} attempts") from last


def fetch_observed(start: date, end: date) -> dict:
    return get_json(ARCHIVE_URL, dict(
        latitude=LAT, longitude=LON, timezone=TZ, start_date=start.isoformat(), end_date=end.isoformat(),
        daily="temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum"))


def fetch_forecast(model: str, days: int = FORECAST_DAYS) -> dict:
    return get_json(FORECAST_URL, dict(
        latitude=LAT, longitude=LON, timezone=TZ, models=model, forecast_days=days,
        daily="temperature_2m_mean,precipitation_sum"))


# ---------------------------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------------------------

def _block(body: dict, key: str, fields: list[str]) -> dict[str, list]:
    """Pull a `daily`/`hourly` block and check it's well formed: every field present and the
    same length as `time`."""
    block = body.get(key) if isinstance(body, dict) else None
    if not isinstance(block, dict) or "time" not in block:
        raise ValueError(f"malformed Open-Meteo response: no `{key}.time` block")
    n = len(block["time"])
    for f in fields:
        if f not in block or len(block[f]) != n:
            raise ValueError(f"malformed Open-Meteo response: `{key}.{f}` missing or not {n} long")
    return block


def _num(v: Any) -> float | None:
    return None if v is None else float(v)


def parse_observed(body: dict) -> list[dict]:
    """Archive response -> one row per day. Days without a temperature yet (the archive runs a
    few days behind) are dropped rather than stored as holes."""
    fields = ["temperature_2m_mean", "temperature_2m_max", "precipitation_sum", "snowfall_sum"]
    d = _block(body, "daily", fields)
    rows = []
    for i, day in enumerate(d["time"]):
        if d["temperature_2m_mean"][i] is None:
            continue
        rows.append({"day": date.fromisoformat(day), "t_mean": _num(d["temperature_2m_mean"][i]),
                     "t_max": _num(d["temperature_2m_max"][i]), "precip": _num(d["precipitation_sum"][i]),
                     "snow": _num(d["snowfall_sum"][i])})
    return rows


def parse_forecast(body: dict, model: str, issue_date: date) -> list[dict]:
    """Forecast API response (issued today) -> one row per future day. Days past the model's
    horizon come back null and are dropped."""
    d = _block(body, "daily", ["temperature_2m_mean", "precipitation_sum"])
    rows = []
    for i, day in enumerate(d["time"]):
        target = date.fromisoformat(day)
        t, p = _num(d["temperature_2m_mean"][i]), _num(d["precipitation_sum"][i])
        if target < issue_date or (t is None and p is None):
            continue
        rows.append({"target_day": target, "issue_date": issue_date, "lead_days": (target - issue_date).days,
                     "model": model, "t_mean": t, "precip": p})
    return rows


def parse_previous_runs(body: dict, model: str) -> list[dict]:
    """Previous Runs API response (hourly `*_previous_dayN`) -> one row per (day, lead).

    A day's value at lead N needs all 24 of its hours: mean for temperature, sum for
    precipitation. A gap in the archive drops that day rather than averaging over part of it.
    Temperature and precipitation are judged separately, so a row can carry one without the other.
    """
    names = [f"{v}_previous_day{n}" for v in ("temperature_2m", "precipitation") for n in PREVIOUS_RUN_LEADS]
    h = _block(body, "hourly", names)
    by_day: dict[str, list[int]] = {}
    for i, ts in enumerate(h["time"]):
        by_day.setdefault(ts[:10], []).append(i)
    rows = []
    for day, idx in by_day.items():
        if len(idx) != 24:
            continue
        target = date.fromisoformat(day)
        for n in PREVIOUS_RUN_LEADS:
            ts_ = [h[f"temperature_2m_previous_day{n}"][i] for i in idx]
            ps_ = [h[f"precipitation_previous_day{n}"][i] for i in idx]
            t = sum(ts_) / 24 if all(v is not None for v in ts_) else None
            p = float(sum(ps_)) if all(v is not None for v in ps_) else None
            if t is None and p is None:
                continue
            rows.append({"target_day": target, "issue_date": target - timedelta(days=n), "lead_days": n,
                         "model": model, "t_mean": t, "precip": p})
    return rows
