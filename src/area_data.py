"""Area data behind the app and the agent tools: Unity Catalog and Lakebase reads, and the pure
functions that shape their rows.

agent_tools.py builds the agent's tools on this module, and the app's routes (webapp/main.py)
call it directly for the views that aren't tools: the map layers, the "All of Chicago" panel and
the rolling 12-month change.

The shape_* functions (and the other pure helpers) take plain rows and do no I/O, so the tests
cover them offline. Values from the Statement Execution API arrive as strings, so they cast
everything.
"""

from __future__ import annotations

import calendar
import json
import math
import time
from datetime import date, timedelta
from typing import Any

from src import config, db_connect
from src.area_wiki import passage_text
from src.community_areas import area_name
from src.constants import (NORMAL_HALF_WINDOW, NORMAL_YEARS, PROFILE_SPAN, RISING_MIN_PRIOR, RISING_PCT,
                           WINDOW_DAYS_OPTIONS, level_band)
from src.open_meteo import PRECIP_MODEL, TEMP_MODEL


VALID_COMMUNITY_AREAS = range(1, 78)  # Chicago has 77 community areas, numbered 1-77
CITY = 0                              # "All of Chicago": the app's citywide view (area 0 in the UI)
CITY_NAME = "All of Chicago"


def validate_community_area(community_area: int) -> str | None:
    if community_area not in VALID_COMMUNITY_AREAS:
        return f"community_area must be between 1 and 77, got {community_area}"
    return None


def iso(value: Any) -> Any:
    """A date or datetime as an ISO string; strings pass through.

    The Spark path returns real datetimes and the Statement Execution API returns strings, so
    callers that can hit either normalize through this.
    """
    return value.isoformat() if hasattr(value, "isoformat") else value


def to_float(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)


def to_date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def as_bool(v: Any) -> bool:
    return v is True or str(v).lower() == "true"


# ---------------------------------------------------------------------------
# Unity Catalog reads
#
# The small UC tables the app reads (notebooks 04, 05, 08, 08b, 09) change once a night, and the
# deployed app reaches UC through the SQL Statement Execution API (seconds on a cold warehouse).
# So they're read through an in-process cache with a one-hour TTL instead of being synced into
# Lakebase: no extra Synced Tables to create and trigger.
# ---------------------------------------------------------------------------

UC_LEAD_LAG = config.uc("indicator_lead_lag")
UC_NARRATIVES = config.uc("area_narratives")
UC_NORMALS = config.uc("area_trend_normals")
UC_PROFILE = config.uc("area_profile")
UC_OUTLOOK_SUMMARY = config.uc("outlook_summary")
UC_OUTLOOK_CITY = config.uc("outlook_city_week")
UC_OUTLOOK_AREA = config.uc("outlook_area_week")
UC_OUTLOOK_COEFS = config.uc("outlook_coefficients")
UC_EVENTS_PRIORS = config.uc("events_priors")
UC_EVENTS_UPCOMING = config.uc("events_upcoming")
UC_WEATHER_FORECAST = config.uc("bronze_weather_forecast")
UC_WEATHER_OBSERVED = config.uc("bronze_weather_observed")
UC_WIKI_INDEX = config.uc("area_wiki_chunks_index")   # a vector search index, not a table (notebook 03c)
_UC_CACHE_TTL_S = 3600
_uc_cache: dict[str, tuple[float, list[dict]]] = {}


def query_uc(sql: str) -> list[dict]:
    """A Unity Catalog query: through Spark where a session exists (notebooks), otherwise
    through the Statement Execution API (the deployed app).

    A deployed app has no pyspark at all, so the fallback catches ImportError as well as the
    RuntimeError raised when pyspark is installed but no session is active.
    """
    try:
        return db_connect.query_uc_spark(sql)
    except (RuntimeError, ImportError):
        return db_connect.query_uc_api(sql)


def query_uc_cached(sql: str) -> list[dict]:
    hit = _uc_cache.get(sql)
    if hit and time.monotonic() - hit[0] < _UC_CACHE_TTL_S:
        return hit[1]
    rows = query_uc(sql)
    _uc_cache[sql] = (time.monotonic(), rows)
    return rows


def lead_lag_rows() -> list[dict[str, Any]]:
    rows = query_uc_cached(f"""
        SELECT sr_metric, crime_metric, lag_months, r, n, p_value, reverse_r, lag0_r, is_leading
        FROM {UC_LEAD_LAG}
    """)
    return [
        {
            "sr_metric": r["sr_metric"],
            "crime_metric": r["crime_metric"],
            "lag_months": int(r["lag_months"]),
            "r": float(r["r"]) if r["r"] is not None else None,
            "n": int(r["n"]),
            "p_value": float(r["p_value"]) if r["p_value"] is not None else None,
            "reverse_r": float(r["reverse_r"]) if r["reverse_r"] is not None else None,
            "lag0_r": float(r["lag0_r"]) if r["lag0_r"] is not None else None,
            "is_leading": as_bool(r["is_leading"]),
        }
        for r in rows
    ]


def normals_rows() -> list[dict[str, Any]]:
    return query_uc_cached(f"""
        SELECT community_area, metric, window_days, window_count, normal_count, pct_vs_normal, as_of_date
        FROM {UC_NORMALS}
    """)


def profile_rows() -> list[dict[str, Any]]:
    return query_uc_cached(f"""
        SELECT community_area, metric, population, annual_count, rate_per_1k, citywide_percentile,
               citywide_rank, years, as_of_date
        FROM {UC_PROFILE} WHERE span = '{PROFILE_SPAN}'
    """)


# ---------------------------------------------------------------------------
# Monthly history
# ---------------------------------------------------------------------------


def annotate_partial_month(history: list[dict[str, Any]], as_of: date | None) -> None:
    """Mark the as-of month's history entry as partial and project it to a full month, in place.

    A straight-line projection: 10 events through the 15th of a 30-day month -> 20. `as_of` is the
    trend tables' shared as-of date (the earlier of crime's and 311's last full days), which is
    how far into the month the data goes. Crime runs about 8 days behind, so it isn't today.
    Notebook 04 cuts the monthly counts at the same day, so count and divisor cover the same
    days. A no-op when the last entry isn't the as-of month or the month is complete.
    """
    if not history or as_of is None:
        return
    last = history[-1]
    period = date.fromisoformat(last["period"])
    if (period.year, period.month) != (as_of.year, as_of.month):
        return
    days_in_month = calendar.monthrange(as_of.year, as_of.month)[1]
    if as_of.day >= days_in_month:
        return
    last.update({
        "partial": True,
        "days_observed": as_of.day,
        "days_in_month": days_in_month,
        "projected_count": round(last["count"] * days_in_month / as_of.day),
    })


# ---------------------------------------------------------------------------
# Leading indicators: which 311 types historically lead which crime categories (notebook 05)
# ---------------------------------------------------------------------------

# "Approaching threshold": a leading 311 category already up at least NEAR_PCT (on the same
# minimum baseline as "rising") but not yet at RISING_PCT. The top NEAR_LIMIT are surfaced so a
# user can set an alert before it crosses.
NEAR_PCT = 5.0
NEAR_LIMIT = 3


def summarize_leading_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (311 metric, crime metric) that notebook 05 flagged as leading, at its best lag."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if not r["is_leading"] or r["lag_months"] < 1 or r["r"] is None:
            continue
        key = (r["sr_metric"], r["crime_metric"])
        if key not in best or r["r"] > best[key]["r"]:
            best[key] = r
    return sorted(
        (
            {"sr_metric": k[0], "crime_metric": k[1], "lag_months": v["lag_months"], "r": round(v["r"], 3),
             "p_value": v["p_value"], "reverse_r": v["reverse_r"], "lag0_r": v["lag0_r"], "n": v["n"]}
            for k, v in best.items()
        ),
        key=lambda x: x["r"],
        reverse=True,
    )


def split_signals(current_rows, pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Sort one area's leading 311 categories into "watch" (rising now) and "near" (close to it).

    `current_rows` are (metric, window_count, prior_window_count, pct) for the area's current
    window. Both lists need a prior count of at least RISING_MIN_PRIOR (3 -> 5 isn't a signal).
    "watch" is pct >= RISING_PCT; "near" is NEAR_PCT <= pct < RISING_PCT, the NEAR_LIMIT closest
    to crossing.
    """
    leads_by_sr: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        leads_by_sr.setdefault(p["sr_metric"], []).append(
            {"crime_metric": p["crime_metric"], "lag_months": p["lag_months"], "r": p["r"]}
        )
    watch, near = [], []
    for metric, window_count, prior, pct in current_rows:
        if pct is None or prior is None or int(prior) < RISING_MIN_PRIOR:
            continue
        pct = float(pct)
        entry = {
            "sr_metric": metric,
            "window_count": int(window_count),
            "prior_window_count": int(prior),
            "pct_change": round(pct, 1),
            "leads": leads_by_sr.get(metric, []),
        }
        if pct >= RISING_PCT:
            watch.append(entry)
        elif pct >= NEAR_PCT:
            near.append(entry)
    return {
        "watch": sorted(watch, key=lambda w: w["pct_change"], reverse=True),
        "near": sorted(near, key=lambda w: w["pct_change"], reverse=True)[:NEAR_LIMIT],
    }


# ---------------------------------------------------------------------------
# Area context: "vs normal" and "usual level"
#
# Two small tables from notebook 04, read through the cached UC path:
# - area_trend_normals: each 30/60/90-day window vs. what the area's own past year predicts for
#   this time of year. The next-week outlook uses the same normal, so the card, the map and the
#   forecast share one baseline.
# - area_profile: each area's crime per 1,000 residents over the last 12 months, with its
#   citywide percentile and rank.
# ---------------------------------------------------------------------------

LEVEL_METRIC = "crime_violent"   # the headline for "how does this area usually stack up"


def shape_normals(rows: list[dict[str, Any]], window_days: int, metric: str | None = None,
                  community_area: int | None = None) -> dict[int, dict[str, dict[str, Any]]]:
    """{area: {metric: {"window_count", "normal_count", "pct_vs_normal"}}} for one window. Pure."""
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        a = int(r["community_area"])
        if int(r["window_days"]) != window_days or (metric and r["metric"] != metric) or (community_area and a != community_area):
            continue
        pct = to_float(r["pct_vs_normal"])
        out.setdefault(a, {})[r["metric"]] = {
            "window_count": int(float(r["window_count"])),
            "normal_count": round(to_float(r["normal_count"]) or 0.0, 1),
            "pct_vs_normal": None if pct is None else round(pct, 1),
        }
    return out


def shape_profile(rows: list[dict[str, Any]], metric: str | None = None,
                  community_area: int | None = None) -> dict[int, dict[str, dict[str, Any]]]:
    """{area: {metric: {"rate_per_1k", "annual_count", "citywide_percentile", "citywide_rank",
    "band", "label"}}}. Pure."""
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        a = int(r["community_area"])
        if (metric and r["metric"] != metric) or (community_area and a != community_area):
            continue
        pctile = to_float(r["citywide_percentile"])
        out.setdefault(a, {})[r["metric"]] = {
            "rate_per_1k": round(to_float(r["rate_per_1k"]) or 0.0, 1),
            "annual_count": round(to_float(r["annual_count"]) or 0.0),
            "citywide_percentile": None if pctile is None else int(pctile),
            "citywide_rank": int(float(r["citywide_rank"])),
            "population": int(float(r["population"])),
            "years": int(float(r["years"])),
            **(level_band(pctile) or {}),
        }
    return out


def with_city_ratio(profile: dict[int, dict[str, dict[str, Any]]], metric: str) -> dict[int, dict[str, Any]]:
    """{area: that metric's entry + ratio_to_city} from shape_profile's output. Pure."""
    items = {a: m[metric] for a, m in profile.items() if metric in m}
    pop = sum(p["population"] for p in items.values())
    city = sum(p["annual_count"] for p in items.values()) / pop * 1000 if pop else 0
    return {a: {**p, "city_rate_per_1k": round(city, 1),
                "ratio_to_city": round(p["rate_per_1k"] / city, 2) if city else None} for a, p in items.items()}


def get_area_context(community_area: int, window_days: int = 30) -> dict[str, Any]:
    """An area's windows vs. normal (every metric) and its usual level (every metric).

    Each half is best-effort: if one table is missing, the other still comes back, with the
    miss in `errors`. ok is False only when both fail.

    Returns:
        {"ok": True, "community_area", "area_name", "window_days",
         "vs_normal": {metric: {"window_count", "normal_count", "pct_vs_normal"}},
         "usual_level": {metric: {"rate_per_1k", "annual_count", "citywide_percentile",
                                  "citywide_rank", "population", "years", "band", "label"}},
         "headline_level_metric": "crime_violent", "errors": [str]}
        usual_level covers the last 12 months.
    """
    err = validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}
    out: dict[str, Any] = {"ok": True, "community_area": community_area, "area_name": area_name(community_area),
                           "window_days": window_days, "vs_normal": {}, "usual_level": {},
                           "headline_level_metric": LEVEL_METRIC, "errors": []}
    try:
        out["vs_normal"] = shape_normals(normals_rows(), window_days, community_area=community_area).get(community_area, {})
    except Exception as exc:
        out["errors"].append(f"vs-normal table unavailable (has notebook 04 run?): {exc}")
    try:
        out["usual_level"] = shape_profile(profile_rows(), community_area=community_area).get(community_area, {})
    except Exception as exc:
        out["errors"].append(f"area profile unavailable (has notebook 04 run?): {exc}")
    if len(out["errors"]) == 2:
        return {"ok": False, "error": "; ".join(out["errors"])}
    return out


# ---------------------------------------------------------------------------
# Map layers (/api/map-data)
# ---------------------------------------------------------------------------

FORECAST_TURN_PCT = 5.0   # "lately" must be at least this far from normal for a call against it to count as a turn
OUTLOOK_EXTREMES = 10     # the Outlook layer's "higher" / "lower concern": this many areas at each end


def normals_snapshot(metric: str, window_days: int) -> dict[int, dict[str, Any]]:
    """Every area's window vs. normal for one metric (the map's "Trend Map" layer)."""
    return {a: m[metric] for a, m in shape_normals(normals_rows(), window_days, metric).items()}


def profile_snapshot(metric: str) -> dict[int, dict[str, Any]]:
    """Every area's crime per resident for one metric (the map's "Crime per capita" layer),
    with `ratio_to_city`: its rate over the city's (all crime / all residents)."""
    return with_city_ratio(shape_profile(profile_rows(), metric), metric)


def forecast_snapshot() -> dict[int, dict[str, Any]]:
    """Every area's next-week outlook (the map's "Outlook" layer): its call, its last 30 days vs.
    normal, whether the call goes against that (`turning`), and `concern`."""
    return shape_forecast_map(query_uc_cached(OUTLOOK_AREAS_SQL), shape_normals(normals_rows(), 30, "crime_total"))


def shape_forecast_map(area_rows: list[dict[str, Any]], normals: dict[int, dict[str, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """The Outlook map layer. Pure.

    `turning` is "easing" for an area running FORECAST_TURN_PCT+ above normal lately whose call is
    "down", "worsening" for one running that far below normal whose call is "up", else None. Only
    calls count, so a turn is one the model is confident about.

    `concern` ranks the other areas against each other: "high" for the OUTLOOK_EXTREMES most likely
    to run above their normal, "low" for the least likely. It's relative on purpose: most of a
    week's swing is citywide, so in a cold week every area can be "below normal", and the map
    should still pick out where next week looks worst and best."""
    out = {}
    for r in area_rows:
        a = int(float(r["community_area"]))
        recent = normals.get(a, {}).get("crime_total", {}).get("pct_vs_normal")
        call = r["call"]
        turning = None
        if recent is not None and recent >= FORECAST_TURN_PCT and call == "down":
            turning = "easing"
        elif recent is not None and recent <= -FORECAST_TURN_PCT and call == "up":
            turning = "worsening"
        out[a] = {"call": call, "p_above": round(to_float(r["p_above"]), 2), "forecast": round(to_float(r["forecast"]), 1),
                  "normal": round(to_float(r["normal"]), 1), "pct_vs_normal": round(to_float(r["pct_vs_normal"]) * 100, 1),
                  "recent_pct_vs_normal": recent, "turning": turning, "concern": None}
    ranked = sorted((a for a in out if not out[a]["turning"]), key=lambda a: out[a]["p_above"])   # flips show as flips
    for a in ranked[:OUTLOOK_EXTREMES]:
        out[a]["concern"] = "low"
    for a in ranked[-OUTLOOK_EXTREMES:]:
        out[a]["concern"] = "high"
    return out


# ---------------------------------------------------------------------------
# All of Chicago: the app's citywide view (area 0)
# ---------------------------------------------------------------------------


def city_trend(months: int = 6, window_days: int = 30) -> dict[str, Any]:
    """get_area_trend's shape for the whole city, every metric summed over the 77 areas: the
    app's default "All of Chicago" panel. Not an agent tool; rank_areas reports city totals."""
    if window_days not in WINDOW_DAYS_OPTIONS:
        return {"ok": False, "error": f"window_days must be one of {WINDOW_DAYS_OPTIONS}, got {window_days}"}
    try:
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    WITH bounds AS (SELECT max(period) AS latest FROM {config.TREND_METRICS_LB})
                    SELECT t.period, t.metric, sum(t.metric_count)
                    FROM {config.TREND_METRICS_LB} t, bounds
                    WHERE t.period >= bounds.latest - (CAST(%(months)s AS text) || ' months')::interval
                    GROUP BY t.period, t.metric
                    ORDER BY t.metric, t.period
                """, {"months": months})
                monthly_rows = cur.fetchall()
                cur.execute(f"""
                    SELECT metric, sum(window_count), sum(prior_window_count), max(as_of_date)
                    FROM {config.TREND_ROLLING_LB}
                    WHERE window_days = %(window_days)s
                    GROUP BY metric
                """, {"window_days": window_days})
                rolling_rows = cur.fetchall()
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}
    return shape_city_trend(monthly_rows, rolling_rows, window_days)


def shape_city_trend(monthly_rows, rolling_rows, window_days: int) -> dict[str, Any]:
    """(period, metric, total) and (metric, window, prior, as_of) rows -> get_area_trend's shape. Pure."""
    metrics: dict[str, dict[str, Any]] = {}
    for period, metric, total in monthly_rows:
        m = metrics.setdefault(metric, {"history": []})
        m["history"].append({"period": iso(period), "count": int(total)})
        m["latest_period"], m["latest_count"] = iso(period), int(total)
    as_of = None
    for metric, window, prior, row_as_of in rolling_rows:
        window, prior = int(window or 0), int(prior or 0)
        metrics.setdefault(metric, {"history": []}).update(
            window_count=window, prior_window_count=prior,
            pct_change_vs_prior_window=None if prior == 0 else round((window - prior) / prior * 100, 1))
        as_of = as_of or row_as_of
    for m in metrics.values():
        annotate_partial_month(m["history"], as_of if not isinstance(as_of, str) else date.fromisoformat(as_of))
    return {"ok": True, "community_area": CITY, "area_name": CITY_NAME, "window_days": window_days,
            "as_of_date": iso(as_of), "metrics": metrics}


def shape_city_context(normals_rows, profile_rows, window_days: int) -> dict[str, Any]:
    """get_area_context's shape for the whole city. Pure.

    vs_normal sums every area's window and normal. usual_level is the city's own rate per 1,000
    residents, with the highest and lowest areas for scale (a citywide rank means nothing here).
    """
    vs_normal: dict[str, dict[str, Any]] = {}
    for a, by_metric in shape_normals(normals_rows, window_days).items():
        for metric, v in by_metric.items():
            s = vs_normal.setdefault(metric, {"window_count": 0, "normal_count": 0.0})
            s["window_count"] += v["window_count"]
            s["normal_count"] += v["normal_count"]
    for s in vs_normal.values():
        s["pct_vs_normal"] = None if s["normal_count"] <= 0 else round((s["window_count"] / s["normal_count"] - 1) * 100, 1)
        s["normal_count"] = round(s["normal_count"], 1)
    by_metric: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for a, ms in shape_profile(profile_rows).items():
        for metric, p in ms.items():
            by_metric.setdefault(metric, []).append((a, p))
    usual: dict[str, dict[str, Any]] = {}
    for metric, items in by_metric.items():
        pop = sum(p["population"] for _, p in items)
        annual = sum(p["annual_count"] for _, p in items)
        hi, lo = max(items, key=lambda x: x[1]["rate_per_1k"]), min(items, key=lambda x: x[1]["rate_per_1k"])
        pick = lambda x: {"community_area": x[0], "area_name": area_name(x[0]), "rate_per_1k": x[1]["rate_per_1k"]}
        usual[metric] = {"rate_per_1k": round(annual / pop * 1000, 1) if pop else None, "annual_count": annual,
                         "population": pop, "years": items[0][1]["years"], "highest": pick(hi), "lowest": pick(lo)}
    return {"vs_normal": vs_normal, "usual_level": usual}


def get_city_context(window_days: int = 30) -> dict[str, Any]:
    """get_area_context for "All of Chicago". Best-effort per table, like get_area_context."""
    out: dict[str, Any] = {"ok": True, "community_area": CITY, "area_name": CITY_NAME, "window_days": window_days,
                           "vs_normal": {}, "usual_level": {},
                           "headline_level_metric": LEVEL_METRIC, "errors": []}
    normals, profile = [], []
    try:
        normals = normals_rows()
    except Exception as exc:
        out["errors"].append(f"vs-normal table unavailable (has notebook 04 run?): {exc}")
    try:
        profile = profile_rows()
    except Exception as exc:
        out["errors"].append(f"area profile unavailable (has notebook 04 run?): {exc}")
    if len(out["errors"]) == 2:
        return {"ok": False, "error": "; ".join(out["errors"])}
    out.update(shape_city_context(normals, profile, window_days))
    return out


# ---------------------------------------------------------------------------
# Next-week outlook (notebook 08)
# ---------------------------------------------------------------------------

OUTLOOK_AREAS_SQL = f"""
    SELECT community_area, call, p_above, forecast, normal, pct_vs_normal, pct_citywide, pct_local
    FROM {UC_OUTLOOK_AREA} WHERE is_live
"""
CITY_WEEKS_SHOWN = 52


def shape_outlook(summary: dict[str, Any], city: dict[str, Any], areas: list[dict[str, Any]],
                  community_area: int | None) -> dict[str, Any]:
    """Notebook 08's live-week rows -> get_next_week_outlook's answer. Pure."""
    by_area = {int(r["community_area"]): r for r in areas}
    calls = {"up": [], "down": []}
    for a, r in sorted(by_area.items(), key=lambda kv: -to_float(kv[1]["p_above"])):
        if r["call"] in calls:
            calls[r["call"]].append({"community_area": a, "area_name": area_name(a), "p_above": round(to_float(r["p_above"]), 2)})
    holidays = [h for h in (city.get("holidays") or "").split(", ") if h]
    out: dict[str, Any] = {
        "ok": True,
        "week_start": iso(summary["live_cutoff"]), "week_end": iso(summary["live_week_end"]),
        "data_through": iso(summary["data_through"]),
        "citywide": {
            "pct_vs_normal": round(to_float(city["pct_vs_normal"]) * 100, 1),
            "drivers_pct": {k: round(to_float(city[f"pct_{k}"]) * 100, 1) for k in ("momentum", "weather", "calendar")},
            # The model works in Open-Meteo's °C and cm; everything shown is imperial.
            "forecast_temp_vs_normal_f": round(to_float(city["fc_t_anom"]) * 9 / 5, 1),
            "forecast_rain_vs_normal_in": round(to_float(city["fc_p_anom_cm"]) / 2.54, 2),
            "holidays": holidays,
        },
        "calls": calls,
        "track_record": {
            "backtest_from": iso(summary["test_first"]), "backtest_to": iso(summary["test_last"]),
            "share_of_area_weeks_called": round(to_float(summary["share_called"]) * 100),
            "calls_right_pct": round(to_float(summary["hit_rate"]) * 100),
            "calls_right_ci_pct": [round(to_float(summary["hit_lo"]) * 100), round(to_float(summary["hit_hi"]) * 100)],
            "base_rate_above_normal_pct": round(to_float(summary["base_rate"]) * 100),
        },
    }
    if community_area is not None:
        r = by_area.get(int(community_area))
        if r is None:
            return {"ok": False, "error": f"no outlook for community area {community_area}"}
        out["area"] = {
            "community_area": int(community_area), "area_name": area_name(community_area),
            "call": r["call"], "p_above_normal": round(to_float(r["p_above"]), 2),
            "forecast": round(to_float(r["forecast"]), 1), "normal": round(to_float(r["normal"]), 1),
            "pct_vs_normal": round(to_float(r["pct_vs_normal"]) * 100, 1),
            "citywide_part_pct": round(to_float(r["pct_citywide"]) * 100, 1),
            "local_part_pct": round(to_float(r["pct_local"]) * 100, 1),
        }
    return out


def weather_effects(coefs: list[dict[str, Any]]) -> dict[str, Any]:
    """The live citywide model's effects (outlook_coefficients) as % changes in plain units. Pure.

    - per_degree_f_pct: a week 1 °F warmer than normal (the model's per-°C effect x 5/9)
    - per_inch_rain_pct: 1 inch more rain than normal over the week (per-cm effect x 2.54)
    - momentum_10pct_pct: the city ran 10% above normal over both the last 7 and 28 known days
    - month_start_pct: a week that includes the 1st of a month
    - holidays: {name: pct} for a week containing that holiday, biggest first
    """
    b = {r["feature"]: float(r["per_unit"]) for r in coefs}
    pct = lambda x: round(math.expm1(x) * 100, 1)
    hol = {f.removeprefix("hol_"): pct(v) for f, v in b.items() if f.startswith("hol_") and f != "hol_in_recent_7d"}
    return {
        "per_degree_f_pct": pct(b.get("t_anom", 0.0) * 5 / 9),
        "per_inch_rain_pct": pct(b.get("p_anom_cm", 0.0) * 2.54),
        "momentum_10pct_pct": pct((b.get("dep_7d", 0.0) + b.get("dep_28d", 0.0)) * math.log(1.1)),
        "month_start_pct": pct(b.get("month_start", 0.0)),
        "holidays": dict(sorted(hol.items(), key=lambda kv: -kv[1])),
    }


def forecast_days_sql(week_start: date) -> str:
    """SQL for the daily weather forecast behind the outlook for the week starting `week_start`,
    picked the way src/outlook/features.issued_forecast picks it: each model's newest issue by
    the day before the week, temperature from TEMP_MODEL (falling back to PRECIP_MODEL's),
    precipitation from PRECIP_MODEL. Each day comes with its normal: the observed mean over the
    same date +-NORMAL_HALF_WINDOW days in each of the NORMAL_YEARS years before (calendar years
    here, 365.25-day years in the model; a day's difference at most). °C and mm."""
    ws = iso(week_start)
    return f"""
        WITH picked AS (
          SELECT model, max(issue_date) AS issue_date FROM {UC_WEATHER_FORECAST}
          WHERE issue_date <= date_sub(DATE'{ws}', 1) GROUP BY model
        ),
        fc AS (
          SELECT f.target_day,
                 max(CASE WHEN f.model = '{TEMP_MODEL}' THEN f.t_mean END) AS t_main,
                 max(CASE WHEN f.model = '{PRECIP_MODEL}' THEN f.t_mean END) AS t_backup,
                 max(CASE WHEN f.model = '{PRECIP_MODEL}' THEN f.precip END) AS precip,
                 max(f.issue_date) AS issue_date
          FROM {UC_WEATHER_FORECAST} f JOIN picked p ON f.model = p.model AND f.issue_date = p.issue_date
          WHERE f.target_day BETWEEN DATE'{ws}' AND date_add(DATE'{ws}', 6)
          GROUP BY f.target_day
        ),
        norms AS (
          SELECT fc.target_day, avg(o.t_mean) AS t_normal, avg(o.precip) AS p_normal
          FROM fc
          CROSS JOIN (SELECT explode(sequence(1, {NORMAL_YEARS})) AS k) ks
          JOIN {UC_WEATHER_OBSERVED} o
            ON o.day BETWEEN date_sub(add_months(fc.target_day, -12 * ks.k), {NORMAL_HALF_WINDOW})
                         AND date_add(add_months(fc.target_day, -12 * ks.k), {NORMAL_HALF_WINDOW})
          GROUP BY fc.target_day
        )
        SELECT fc.target_day, fc.issue_date, coalesce(fc.t_main, fc.t_backup) AS t_mean, fc.precip,
               n.t_normal, n.p_normal
        FROM fc LEFT JOIN norms n ON n.target_day = fc.target_day
        ORDER BY fc.target_day
    """


def shape_forecast_days(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """forecast_days_sql rows -> the Insights page's 7-day strip, in °F and inches. Pure. None when
    there's no forecast for the week."""
    f = lambda c: None if c is None else round(c * 9 / 5 + 32)
    days = []
    for r in rows:
        t, tn = to_float(r["t_mean"]), to_float(r["t_normal"])
        p, pn = to_float(r["precip"]), to_float(r["p_normal"])
        day = to_date(r["target_day"])
        days.append({
            "day": iso(day), "weekday": day.strftime("%a"), "date": f"{calendar.month_abbr[day.month]} {day.day}",
            "temp_f": f(t), "normal_f": f(tn),
            "temp_vs_normal_f": None if t is None or tn is None else round((t - tn) * 9 / 5),
            "rain_in": None if p is None else round(p / 25.4, 2),
            "normal_rain_in": None if pn is None else round(pn / 25.4, 2),
        })
    if not days:
        return None
    return {"issued": iso(to_date(max(str(r["issue_date"]) for r in rows))), "days": sorted(days, key=lambda d: d["day"])}


def shape_city_weeks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """outlook_city_week rows -> the Insights chart's weekly series, oldest first. Pure."""
    out = [{
        "week": iso(to_date(r["cutoff"])),
        "actual": None if r["y"] in (None, "") else round(float(r["y"])),
        "forecast": round(float(r["forecast"])),
        "normal": round(float(r["bar_calibrated"])),
        "temp_vs_normal_f": None if r["fc_t_anom"] in (None, "") else round(float(r["fc_t_anom"]) * 9 / 5, 1),
        "is_live": as_bool(r["is_live"]),
    } for r in rows]
    return sorted(out, key=lambda w: w["week"])


# ---------------------------------------------------------------------------
# "Rolling 12-month change": the area panel's long-run trend line
# ---------------------------------------------------------------------------


def monthly_series(community_area: int, metric: str) -> tuple[list[tuple], date | None]:
    """(period, count) for every month of one metric (summed over areas for the city), and the
    shared as-of date that says whether the newest month is complete."""
    area = "" if community_area == CITY else "AND community_area = %(community_area)s"
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT period, sum(metric_count) FROM {config.TREND_METRICS_LB}
                WHERE metric = %(metric)s {area}
                GROUP BY period ORDER BY period
            """, {"metric": metric, "community_area": community_area})
            rows = cur.fetchall()
            cur.execute(f"SELECT max(as_of_date) FROM {config.TREND_ROLLING_LB}")
            as_of = cur.fetchone()[0]
    return rows, as_of


def shape_rolling_change(rows: list[tuple], as_of: date | str | None) -> dict[str, Any]:
    """Monthly (period, count) -> the 12 months ending each month against the 12 before, as a %.
    Pure. Complete months only, so the as-of month is dropped unless the data runs to its last
    day. Rolling 12-month totals carry no seasonality."""
    months = [(date.fromisoformat(iso(p)[:10]), int(float(n))) for p, n in rows]
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of[:10])
    if months and as_of and (months[-1][0].year, months[-1][0].month) == (as_of.year, as_of.month) \
            and as_of.day < calendar.monthrange(as_of.year, as_of.month)[1]:
        months = months[:-1]
    points = []
    for i in range(23, len(months)):
        cur = sum(n for _, n in months[i - 11:i + 1])
        prior = sum(n for _, n in months[i - 23:i - 11])
        points.append({"month": iso(months[i][0]), "last_12": cur, "prior_12": prior,
                       "pct": None if prior == 0 else round((cur / prior - 1) * 100, 1)})
    return {"points": points, "latest": points[-1] if points else None}


def rolling_change(community_area: int, metric: str = "crime_total") -> dict[str, Any]:
    """The area panel's rolling 12-month change for one crime metric (area 0 = All of Chicago).
    Not an agent tool."""
    if community_area != CITY:
        err = validate_community_area(community_area)
        if err:
            return {"ok": False, "error": err}
    try:
        rows, as_of = monthly_series(community_area, metric)
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}
    return {"ok": True, "community_area": community_area, "metric": metric, **shape_rolling_change(rows, as_of)}


# ---------------------------------------------------------------------------
# Events look-ahead (notebook 08b)
# ---------------------------------------------------------------------------

EVENTS_DAYS_AHEAD = 30
EVENT_BUCKET_LABELS = {("festival", "1"): "Street festival, 1 block", ("festival", "2-4"): "Street festival, 2–4 blocks",
                       ("festival", "5+"): "Street festival, 5+ blocks", ("club_night", "all"): "Club show night (6pm–3am)"}


def shape_event_lifts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """events_priors rows (notebook 08b) -> how much crime each kind of event adds around it, on
    average: its own blocks and the ring around them vs. the same place on comparable days. Pure."""
    out = []
    for key, label in EVENT_BUCKET_LABELS.items():
        r = next((r for r in rows if (r["kind"], str(r["bucket"])) == key), None)
        if r is not None:
            out.append({"label": label, "kind": key[0], "lift_pct": round((float(r["prior"]) - 1) * 100, 1),
                        "past_events": int(float(r["n_past"]))})
    return out


def shape_upcoming(rows: list[dict[str, Any]], community_area: int | None, today: date,
                   days: int = EVENTS_DAYS_AHEAD) -> dict[str, Any]:
    """Notebook 08b's events_upcoming rows -> get_upcoming_events' answer. Pure.

    Festivals are listed one by one, biggest expected effect first. Club nights are grouped by
    club: one night is about 0.1 extra crimes, and only a run of them adds up. Multi-week runs (a
    season-long market) are listed without a score.
    """
    horizon = today + timedelta(days=days)
    live = [r for r in rows if to_date(r["start_date"]) <= horizon and to_date(r["end_date"]) >= today
            and (community_area is None or (r.get("community_area") not in (None, "") and int(float(r["community_area"])) == community_area))]
    rnd = lambda v, n=1: None if v in (None, "") else round(float(v), n)
    ca = lambda r: None if r.get("community_area") in (None, "") else int(float(r["community_area"]))
    festivals = sorted((
        {"name": r["name"], "place": r.get("place"), "area_name": r.get("area_name"), "community_area": ca(r),
         "start_date": iso(to_date(r["start_date"])), "end_date": iso(to_date(r["end_date"])),
         "extra_crimes": rnd(r["extra"]), "extra_range": [rnd(r["extra_lo"]), rnd(r["extra_hi"])],
         "ratio": rnd(r["ratio"], 2), "past_editions": int(float(r["history"] or 0)),
         "street_segments": None if r.get("n_segments") in (None, "") else int(float(r["n_segments"]))}
        for r in live if r["kind"] == "festival"), key=lambda f: -(f["extra_crimes"] or 0))
    clubs: dict[str, dict[str, Any]] = {}
    for r in sorted((r for r in live if r["kind"] == "club_night"), key=lambda r: to_date(r["start_date"])):
        c = clubs.setdefault(r["place"], {"place": r["place"], "area_name": r.get("area_name"), "community_area": ca(r), "nights": 0,
                                          "next_date": iso(to_date(r["start_date"])), "extra_crimes": 0.0,
                                          "ratio": rnd(r["ratio"], 2), "next_acts": []})
        c["nights"] += 1
        c["extra_crimes"] += float(r["extra"] or 0)
        if len(c["next_acts"]) < 3 and r.get("name"):
            c["next_acts"].append(r["name"])
    club_list = sorted(({**c, "extra_crimes": round(c["extra_crimes"], 1)} for c in clubs.values()),
                       key=lambda c: -c["extra_crimes"])
    multi = [{"name": r["name"], "place": r.get("place"), "start_date": iso(to_date(r["start_date"])),
              "end_date": iso(to_date(r["end_date"]))} for r in live if r["kind"] == "multi_week"]
    return {
        "ok": True, "community_area": community_area,
        "area_name": area_name(community_area) if community_area else None,
        "window": [iso(today), iso(horizon)],
        "festivals": festivals if community_area else festivals[:10],
        "club_nights": club_list if community_area else club_list[:8],
        "multi_week_unscored": multi,
        "total_extra_crimes": round(sum(f["extra_crimes"] or 0 for f in festivals) + sum(c["extra_crimes"] for c in club_list), 1),
    }


# ---------------------------------------------------------------------------
# Wikipedia background (notebooks 02d and 03c)
# ---------------------------------------------------------------------------

WIKI_COLUMNS = ["community_area", "area_name", "section", "content", "page_url"]


def search_wiki(query: str, community_area: int | None, num_results: int) -> list[dict[str, Any]]:
    """Hybrid (embedding + keyword) search over the Wikipedia passages in UC_WIKI_INDEX.

    Keywords matter here: neighborhood, street and landmark names are exact terms that embeddings
    alone can blur. Not cached; the index answers in well under a second.

    The area filter is tried in the Standard endpoint's dict form, then the Storage-Optimized
    endpoint's SQL form, so either kind of endpoint works.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    def run(filters: str | None):
        return w.vector_search_indexes.query_index(
            index_name=UC_WIKI_INDEX, columns=WIKI_COLUMNS, query_text=query,
            query_type="HYBRID", num_results=num_results, filters_json=filters,
        )

    if community_area is None:
        resp = run(None)
    else:
        try:
            resp = run(json.dumps({"community_area": int(community_area)}))
        except Exception:
            resp = run(f"community_area = {int(community_area)}")
    columns = [c.name for c in (resp.manifest.columns or [])] if resp.manifest else []
    return shape_wiki_hits(columns, (resp.result.data_array if resp.result else None) or [])


def shape_wiki_hits(columns: list[str], data_array: list[list[Any]]) -> list[dict[str, Any]]:
    """query_index's (column names, rows) -> passages, best first. The score is the last column;
    values can come back as strings or floats, so everything casts."""
    hits = []
    for row in data_array:
        r = dict(zip(columns, row))
        score = to_float(r.get("score"))
        hits.append({
            "community_area": int(float(r["community_area"])),
            "area_name": r.get("area_name"),
            "section": r.get("section"),
            "text": passage_text(r.get("content") or ""),
            "page_url": r.get("page_url"),
            "score": None if score is None else round(score, 3),
        })
    return hits
