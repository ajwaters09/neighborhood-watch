"""The chat agent's ten tools: eight reads and two writes.

| Tool | Kind | Backed by |
|---|---|---|
| get_area_trend | read | Lakebase Synced Tables (monthly history + trailing windows) |
| get_recent_activity | read | silver crime and 311 records (Unity Catalog) |
| rank_areas | read | trailing windows, vs. normal, or per-resident level across all 77 areas |
| get_leading_indicators | read | the 311 -> crime lead-lag analysis (notebook 05) |
| get_area_summary | read | the nightly narrative (notebook 09) |
| get_next_week_outlook | read | the next-week forecast (notebook 08) |
| get_upcoming_events | read | the events look-ahead (notebook 08b) |
| search_area_background | read | Wikipedia passages in a vector search index (notebook 03c) |
| subscribe_to_area | write | Lakebase `area_subscriptions` |
| log_resident_report | write | Lakebase `resident_reports` |

They're plain functions: agent_chat.py describes them to the model (TOOL_SPECS) and dispatches its
calls, and the app's routes call several directly. The reads and row shaping they share with the
app's other views live in area_data.py.

Each returns a JSON-serializable dict, {"ok": True, ...} or {"ok": False, "error": "..."}, rather
than raising: a structured error is more useful to the model than a traceback. @_logged records
every call in Lakebase's `tool_invocations` for the usage analytics (notebook 07).
"""

from __future__ import annotations

import functools
import inspect
import json
import time
from datetime import date, timedelta
from typing import Any, Callable

from src import area_data as ad
from src import config, db_connect
from src.area_wiki import LICENSE
from src.community_areas import area_name
from src.constants import WINDOW_DAYS_OPTIONS


def _log_invocation(tool_name: str, result: dict[str, Any], user_id: Any, community_area: Any) -> None:
    """Best-effort row in `tool_invocations` for one tool call.

    Change Data Feed carries these rows to Unity Catalog, where `tool_usage_daily` (notebook 07)
    reports requests over time and success rates per tool. A logging failure must never break the
    call it's logging, so this swallows its own exceptions (and can under-report when Lakebase is
    down).
    """
    try:
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_invocations
                        (tool_name, user_id, community_area, success, error_message)
                    VALUES (%(tool_name)s, %(user_id)s, %(community_area)s, %(success)s, %(error_message)s)
                    """,
                    {
                        "tool_name": tool_name,
                        "user_id": user_id,
                        "community_area": community_area,
                        "success": bool(result.get("ok")),
                        "error_message": result.get("error"),
                    },
                )
    except Exception:
        pass


def _logged(tool_name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Log every call to the wrapped tool via `_log_invocation`.

    Arguments are bound to the tool's own signature, so `user_id` and `community_area` are found
    whether they're passed by position or keyword. Tools without a `user_id` log it as None.
    The undecorated function stays reachable as `__wrapped__`, for calls that shouldn't count as
    agent usage.
    """
    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = fn(*args, **kwargs)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            _log_invocation(
                tool_name,
                result,
                user_id=bound.arguments.get("user_id"),
                community_area=bound.arguments.get("community_area"),
            )
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


@_logged("get_area_trend")
def get_area_trend(community_area: int, months: int = 6, window_days: int = 30,
                   include_context: bool = True) -> dict[str, Any]:
    """Current trend scores and recent history for a community area.

    Reads two Lakebase Synced Tables: `area_trend_metrics_lb` (monthly history) and
    `area_trend_rolling_lb` (the last N days against the N days before, as of one date).

    Prefer `pct_change_vs_prior_window` for "better or worse right now": both sides are complete
    windows. The monthly `pct_change_vs_prior` compares a partial current month with a full one.
    The current month's `history` entry carries a straight-line projection to a full month
    (`area_data.annotate_partial_month`), so it can read "on pace for ~N" instead of a false dip.

    Args:
        community_area: Chicago community area number, 1-77.
        months: months of monthly history per metric (default 6).
        window_days: the trailing window, 30, 60 or 90 (default 30).
        include_context: also attach each metric's window vs. normal for this time of year and
            the area's `usual_level` (area_data.get_area_context). The app's area panel turns it
            off and fetches those separately, so a cold SQL warehouse never holds it up.

    Returns:
        {"ok": True, "community_area": int, "area_name": str, "window_days": int,
         "as_of_date": "YYYY-MM-DD" | None, "metrics": {
            "<metric>": {"latest_period": "YYYY-MM-DD", "latest_count": int,
                         "pct_change_vs_prior": float | None,
                         "rolling_3mo_avg": float,
                         "history": [{"period": "YYYY-MM-DD", "count": int,
                                      # only on the as-of month, if incomplete:
                                      "partial": True, "days_observed": int,
                                      "days_in_month": int, "projected_count": int}, ...],
                         "window_count": int, "prior_window_count": int,
                         "pct_change_vs_prior_window": float | None,
                         # with include_context:
                         "normal_count": float, "pct_vs_normal": float | None}
        }, "usual_level": {"crime_total": {...}, "crime_violent": {...}}}   # with include_context
        or {"ok": False, "error": str}

        The two tables are independent snapshots, so a metric can lack the other one's keys.
    """
    err = ad.validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}
    if window_days not in WINDOW_DAYS_OPTIONS:
        return {"ok": False, "error": f"window_days must be one of {WINDOW_DAYS_OPTIONS}, got {window_days}"}

    monthly_query = f"""
        WITH bounds AS (
            SELECT max(period) AS latest FROM {config.TREND_METRICS_LB}
        )
        SELECT t.period, t.metric, t.metric_count, t.pct_change_vs_prior, t.rolling_3mo_avg
        FROM {config.TREND_METRICS_LB} t, bounds
        WHERE t.community_area = %(community_area)s
          AND t.period >= bounds.latest - (CAST(%(months)s AS text) || ' months')::interval
        ORDER BY t.metric, t.period
    """
    rolling_query = f"""
        SELECT metric, window_count, prior_window_count, pct_change_vs_prior_window, as_of_date
        FROM {config.TREND_ROLLING_LB}
        WHERE community_area = %(community_area)s AND window_days = %(window_days)s
    """
    try:
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(monthly_query, {"community_area": community_area, "months": months})
                monthly_rows = cur.fetchall()
                cur.execute(rolling_query, {"community_area": community_area, "window_days": window_days})
                rolling_rows = cur.fetchall()
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}

    metrics: dict[str, dict[str, Any]] = {}
    for period, metric, count, pct_change, rolling_avg in monthly_rows:
        m = metrics.setdefault(metric, {"history": []})
        m["history"].append({"period": period.isoformat(), "count": count})
        # Rows are ordered by period, so the last one per metric is the latest.
        m["latest_period"] = period.isoformat()
        m["latest_count"] = count
        m["pct_change_vs_prior"] = pct_change
        m["rolling_3mo_avg"] = rolling_avg

    as_of = None
    for metric, window_count, prior_window_count, pct_change_window, row_as_of in rolling_rows:
        m = metrics.setdefault(metric, {"history": []})
        m["window_count"] = window_count
        m["prior_window_count"] = prior_window_count
        m["pct_change_vs_prior_window"] = pct_change_window
        as_of = as_of or row_as_of

    for m in metrics.values():
        ad.annotate_partial_month(m["history"], as_of)

    result = {
        "ok": True,
        "community_area": community_area,
        "area_name": area_name(community_area),
        "window_days": window_days,
        "as_of_date": as_of.isoformat() if as_of else None,
        "metrics": metrics,
    }
    return attach_context(result) if include_context else result


def attach_context(result: dict[str, Any]) -> dict[str, Any]:
    """Add vs-normal numbers to each metric and the area's usual level (crime_total and the
    headline metric) to a get_area_trend result. Best-effort: a miss lands in `context_error`."""
    ctx = ad.get_area_context(result["community_area"], result["window_days"])
    if not ctx["ok"]:
        result["context_error"] = ctx["error"]
        return result
    for metric, v in ctx["vs_normal"].items():
        if metric in result["metrics"]:
            result["metrics"][metric].update(normal_count=v["normal_count"], pct_vs_normal=v["pct_vs_normal"])
    result["usual_level"] = {m: ctx["usual_level"][m] for m in ("crime_total", ad.LEVEL_METRIC) if m in ctx["usual_level"]}
    if ctx["errors"]:
        result["context_error"] = "; ".join(ctx["errors"])
    return result


@_logged("get_recent_activity")
def get_recent_activity(community_area: int, days: int = 30, limit: int = 25) -> dict[str, Any]:
    """Recent crime and 311 records for a community area, from the silver tables in Unity Catalog.

    Args:
        community_area: Chicago community area number, 1-77.
        days: how many days back to look (default 30).
        limit: max combined records to return (default 25), most recent first.

    Returns:
        {"ok": True, "community_area": int, "records": [
            {"source": "crime"|"311", "timestamp": "ISO8601", ...fields...}, ...
        ]} or {"ok": False, "error": str}
    """
    err = ad.validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}

    since = (date.today() - timedelta(days=days)).isoformat()
    crime_query = f"""
        SELECT date, primary_type, description, location_description, arrest, block
        FROM {db_connect.UC_CRIMES}
        WHERE community_area = {int(community_area)} AND date >= '{since}'
        ORDER BY date DESC
        LIMIT {int(limit)}
    """
    sr_query = f"""
        SELECT created_date, sr_type, status, street_address
        FROM {db_connect.UC_311}
        WHERE community_area = {int(community_area)} AND created_date >= '{since}'
        ORDER BY created_date DESC
        LIMIT {int(limit)}
    """
    # Every interpolated value is a validated int or a date, never raw user text.

    try:
        crime_rows = ad.query_uc(crime_query)
        sr_rows = ad.query_uc(sr_query)
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}

    records = []
    for rec in crime_rows:
        records.append({
            "source": "crime",
            "timestamp": ad.iso(rec["date"]),
            "type": rec["primary_type"],
            "description": rec["description"],
            "location_description": rec["location_description"],
            "arrest": rec["arrest"],
            "block": rec["block"],
        })
    for rec in sr_rows:
        records.append({
            "source": "311",
            "timestamp": ad.iso(rec["created_date"]),
            "type": rec["sr_type"],
            "status": rec["status"],
            "address": rec["street_address"],
        })

    records.sort(key=lambda r: r["timestamp"], reverse=True)
    return {"ok": True, "community_area": community_area, "records": records[:limit]}


# ---------------------------------------------------------------------------
# Metric names
#
# Left to guess, the model tries plausible but wrong names (crime_fraud for
# crime_fraud_financial). So agent_chat puts the real list in the system prompt,
# resolve_metric() maps loose names onto it, and a miss returns the valid list
# so the model can correct itself in one step.
# ---------------------------------------------------------------------------

_METRIC_CACHE_TTL_S = 3600

_metric_cache: tuple[float, list[str]] | None = None


def list_metric_names() -> list[str]:
    """Every metric in area_trend_rolling_lb, cached for an hour (it only changes when gold reruns)."""
    global _metric_cache
    if _metric_cache and time.monotonic() - _metric_cache[0] < _METRIC_CACHE_TTL_S:
        return _metric_cache[1]
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT metric FROM {config.TREND_ROLLING_LB} ORDER BY metric")
            names = [r[0] for r in cur.fetchall()]
    _metric_cache = (time.monotonic(), names)
    return names


def _slug(text: str) -> str:
    return "_".join(w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split())


def resolve_metric(name: str, valid: list[str]) -> tuple[str | None, str | None]:
    """Map a loose metric name onto a real one: (metric, None) or (None, error listing valid names).

    Tries, in order: an exact match; the slug ("Fraud/Financial" -> "fraud_financial"); the slug
    with a crime_ or 311_ prefix; then the one valid metric containing every word given, plurals
    trimmed ("vacant buildings" -> 311_vacant_abandoned_building). A bare "total" means
    crime_total. An ambiguous partial match is reported, not guessed.
    """
    if name in valid:
        return name, None
    slug = _slug(name or "")
    candidates = [slug, f"crime_{slug}", f"311_{slug}"]
    for c in candidates:
        if c in valid:
            return c, None
    words = [w for w in slug.split("_") if w and w not in ("crime", "crimes", "311", "requests", "request")]
    if words:
        hits = [m for m in valid if all(w.rstrip("s") in m for w in words)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, f"metric {name!r} is ambiguous -- did you mean one of: {', '.join(hits)}?"
    return None, f"unknown metric {name!r}. Valid metrics: {', '.join(valid)}"


# ---------------------------------------------------------------------------
# Cross-area ranking
# ---------------------------------------------------------------------------

CONTEXT_SORTS = ("above_normal", "below_normal", "highest_usual_level", "lowest_usual_level")

RANK_SORTS = ("pct_increase", "pct_decrease", "abs_increase", "abs_decrease", "volume") + CONTEXT_SORTS


@_logged("rank_areas")
def rank_areas(
    metric: str = "crime_total",
    window_days: int = 30,
    sort_by: str = "pct_increase",
    limit: int = 10,
    min_prior_count: int = 20,
) -> dict[str, Any]:
    """Rank all 77 community areas on one metric, in one query instead of 77 get_area_trend calls.

    The change sorts read area_trend_rolling_lb. above_normal / below_normal rank the window
    against the area's normal for this time of year (area_trend_normals: steadier than the prior
    window), and highest/lowest_usual_level rank crime per 1,000 residents over the last 12
    months (area_profile). Every result carries both kinds of context when those tables exist.

    `min_prior_count` keeps small numbers out of the percentage sorts: 2 -> 4 incidents is
    "+100%" but means nothing beside 500 -> 600. Areas whose prior window (or normal) falls below
    it are left out and counted in `excluded_low_baseline`.

    Args:
        metric: e.g. "crime_total", "311_total" or a category. Loose names are resolved
            (resolve_metric): "fraud" -> crime_fraud_financial.
        window_days: 30, 60 or 90 (default 30).
        sort_by: one of RANK_SORTS. "volume" is the current-window count, highest first.
        limit: how many areas to return (default 10, max 77).
        min_prior_count: minimum prior-window count for the percentage sorts (default 20).

    Returns:
        {"ok": True, "metric": str, "window_days": int, "sort_by": str,
         "as_of_date": "YYYY-MM-DD" | None, "excluded_low_baseline": int,
         "city_total_window": int, "city_total_prior_window": int,
         "areas": [{"rank": int, "community_area": int, "area_name": str,
                    "window_count": int, "prior_window_count": int,
                    "abs_change": int, "pct_change": float | None}, ...]}
        or {"ok": False, "error": str}
    """
    if window_days not in WINDOW_DAYS_OPTIONS:
        return {"ok": False, "error": f"window_days must be one of {WINDOW_DAYS_OPTIONS}, got {window_days}"}
    if sort_by not in RANK_SORTS:
        return {"ok": False, "error": f"sort_by must be one of {RANK_SORTS}, got {sort_by}"}
    limit = max(1, min(int(limit), 77))
    try:
        metric, err = resolve_metric(metric, list_metric_names())
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}
    if err:
        return {"ok": False, "error": err}

    if sort_by in CONTEXT_SORTS:
        try:
            normals, profile = ad.normals_rows(), ad.profile_rows()
        except Exception as exc:
            return {"ok": False, "error": f"vs-normal / usual-level tables unavailable (has notebook 04 run?): {exc}"}
        return _rank_context(normals, profile, metric, window_days, sort_by, limit, min_prior_count)
    try:
        rows = _rolling_rows(metric, window_days)
    except Exception as exc:
        return {"ok": False, "error": f"query failed: {exc}"}
    if not rows:
        return {"ok": False, "error": f"no data for metric {metric!r} -- check the metric name"}
    out = _rank_rows(rows, metric, window_days, sort_by, limit, min_prior_count)
    try:
        _add_context(out["areas"], ad.normals_rows(), ad.profile_rows(), metric, window_days)
    except Exception:
        pass   # the ranking stands on its own; context is extra
    return out


def _rolling_rows(metric: str, window_days: int) -> list[tuple]:
    query = f"""
        SELECT community_area, window_count, prior_window_count, pct_change_vs_prior_window, as_of_date
        FROM {config.TREND_ROLLING_LB}
        WHERE metric = %(metric)s AND window_days = %(window_days)s
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"metric": metric, "window_days": window_days})
            return cur.fetchall()


def _add_context(areas: list[dict[str, Any]], normals_rows, profile_rows, metric: str, window_days: int) -> None:
    """Attach each ranked area's vs-normal and usual-level numbers, in place. Pure."""
    normals = ad.shape_normals(normals_rows, window_days, metric)
    profile = ad.shape_profile(profile_rows, metric)
    for a in areas:
        n, p = normals.get(a["community_area"], {}).get(metric), profile.get(a["community_area"], {}).get(metric)
        if n:
            a.update(normal_count=n["normal_count"], pct_vs_normal=n["pct_vs_normal"])
        if p:
            a.update(usual_rate_per_1k=p["rate_per_1k"], usual_level_rank=p["citywide_rank"], usual_level=p.get("label"))


def _rank_context(normals_rows, profile_rows, metric: str, window_days: int, sort_by: str, limit: int,
                  min_count: int) -> dict[str, Any]:
    """The vs-normal and usual-level rankings (CONTEXT_SORTS). Pure."""
    normals = ad.shape_normals(normals_rows, window_days, metric)
    profile = ad.shape_profile(profile_rows, metric)
    if not normals and not profile:
        return {"ok": False, "error": f"no data for metric {metric!r} -- check the metric name"}
    areas = [{"community_area": a, "area_name": area_name(a)} for a in sorted(set(normals) | set(profile))]
    _add_context(areas, normals_rows, profile_rows, metric, window_days)
    for a in areas:
        a["window_count"] = normals.get(a["community_area"], {}).get(metric, {}).get("window_count")
    excluded = 0
    if sort_by in ("above_normal", "below_normal"):
        eligible = [a for a in areas if a.get("pct_vs_normal") is not None and a.get("normal_count", 0) >= min_count]
        excluded = len(areas) - len(eligible)
        areas = sorted(eligible, key=lambda a: a["pct_vs_normal"], reverse=(sort_by == "above_normal"))
    else:
        eligible = [a for a in areas if a.get("usual_rate_per_1k") is not None]
        areas = sorted(eligible, key=lambda a: a["usual_rate_per_1k"], reverse=(sort_by == "highest_usual_level"))
    for i, a in enumerate(areas, 1):
        a["rank"] = i
    as_of = next((r["as_of_date"] for r in normals_rows if r["metric"] == metric), None)
    return {
        "ok": True, "metric": metric, "window_days": window_days, "sort_by": sort_by,
        "as_of_date": ad.iso(as_of), "excluded_low_baseline": excluded, "min_prior_count": min_count,
        "areas": areas[:limit],
    }


def _rank_rows(rows, metric, window_days, sort_by, limit, min_prior_count) -> dict[str, Any]:
    """Rank (area, window_count, prior_count, pct, as_of) rows. Pure."""
    areas = [
        {
            "community_area": int(a),
            "area_name": area_name(a),
            "window_count": int(w or 0),
            "prior_window_count": int(p or 0),
            "abs_change": int(w or 0) - int(p or 0),
            "pct_change": round(float(pct), 1) if pct is not None else None,
        }
        for a, w, p, pct, _ in rows
    ]
    excluded = 0
    if sort_by.startswith("pct_"):
        eligible = [a for a in areas if a["prior_window_count"] >= min_prior_count and a["pct_change"] is not None]
        excluded = len(areas) - len(eligible)
        areas = sorted(eligible, key=lambda a: a["pct_change"], reverse=(sort_by == "pct_increase"))
    elif sort_by.startswith("abs_"):
        areas = sorted(areas, key=lambda a: a["abs_change"], reverse=(sort_by == "abs_increase"))
    else:
        areas = sorted(areas, key=lambda a: a["window_count"], reverse=True)
    for i, a in enumerate(areas, 1):
        a["rank"] = i
    as_of = rows[0][4]
    return {
        "ok": True,
        "metric": metric,
        "window_days": window_days,
        "sort_by": sort_by,
        "as_of_date": as_of.isoformat() if hasattr(as_of, "isoformat") else as_of,
        "excluded_low_baseline": excluded,
        "min_prior_count": min_prior_count,
        "city_total_window": sum(int(r[1] or 0) for r in rows),
        "city_total_prior_window": sum(int(r[2] or 0) for r in rows),
        "areas": areas[:limit],
    }


# ---------------------------------------------------------------------------
# Read tools over the nightly UC tables (see area_data's UC section)
# ---------------------------------------------------------------------------


@_logged("get_leading_indicators")
def get_leading_indicators(community_area: int | None = None, window_days: int = 30, include_matrix: bool = False) -> dict[str, Any]:
    """Which 311 request types historically lead which crime categories, and, for an area, which
    of them are rising now.

    From notebook 05: lagged correlations of area-level 311 and crime counts after removing each
    area's typical level and each month's citywide swing (src/leading_indicators.py). A pair
    "leads" when 311 this month tracks crime 1-3 months later more strongly than the two move
    together in the same month, and more strongly than crime tracks later 311. Correlation, not
    causation.

    Args:
        community_area: optional, 1-77. Adds `watch` (leading 311 types up at least RISING_PCT
            on a prior count of at least RISING_MIN_PRIOR) and `near` (up to NEAR_LIMIT climbing
            past NEAR_PCT but not there yet).
        window_days: the trailing window for the "rising now" check (30, 60 or 90).
        include_matrix: also return every (311 type, crime, lag) row, for the Insights heatmap.

    Returns:
        {"ok": True, "pairs": [{"sr_metric", "crime_metric", "lag_months", "r",
                                "p_value", "reverse_r", "lag0_r", "n"}, ...],
         "watch": [{"sr_metric", "window_count", "prior_window_count",
                    "pct_change", "leads": [{"crime_metric", "lag_months", "r"}]}],   # if community_area
         "near": [same shape as watch],                                               # if community_area
         "matrix": [...]}                                                             # if include_matrix
        or {"ok": False, "error": str}
    """
    if community_area is not None:
        err = ad.validate_community_area(community_area)
        if err:
            return {"ok": False, "error": err}
    if window_days not in WINDOW_DAYS_OPTIONS:
        return {"ok": False, "error": f"window_days must be one of {WINDOW_DAYS_OPTIONS}, got {window_days}"}
    try:
        rows = ad.lead_lag_rows()
    except Exception as exc:
        return {"ok": False, "error": f"leading-indicator table unavailable (has notebook 05 run?): {exc}"}

    pairs = ad.summarize_leading_pairs(rows)
    out: dict[str, Any] = {"ok": True, "pairs": pairs}
    if include_matrix:
        out["matrix"] = rows

    if community_area is not None:
        sr_metrics = sorted({p["sr_metric"] for p in pairs})
        signals: dict[str, list[dict[str, Any]]] = {"watch": [], "near": []}
        if sr_metrics:
            query = f"""
                SELECT metric, window_count, prior_window_count, pct_change_vs_prior_window
                FROM {config.TREND_ROLLING_LB}
                WHERE community_area = %(community_area)s AND window_days = %(window_days)s
                  AND metric = ANY(%(metrics)s)
            """
            try:
                with db_connect.get_pg_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(query, {"community_area": community_area, "window_days": window_days, "metrics": sr_metrics})
                        current = cur.fetchall()
            except Exception as exc:
                return {"ok": False, "error": f"query failed: {exc}"}
            signals = ad.split_signals(current, pairs)
        out.update({"community_area": community_area, "area_name": area_name(community_area), "window_days": window_days, **signals})
    return out


@_logged("get_area_summary")
def get_area_summary(community_area: int) -> dict[str, Any]:
    """Plain-language summary of what's been happening in an area lately.

    Written nightly by notebook 09 (`ai_query` over the area's recent numbers), so this is a
    lookup, not a live LLM call. `context` holds the facts the narrative was written from; quote
    numbers from there.

    Returns:
        {"ok": True, "community_area": int, "area_name": str, "as_of_date": str,
         "narrative": str, "context": dict, "generated_at": str, "model": str}
        or {"ok": False, "error": str}
    """
    err = ad.validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}
    try:
        rows = ad.query_uc_cached(f"""
            SELECT community_area, as_of_date, narrative, context_json, model, generated_at
            FROM {ad.UC_NARRATIVES}
        """)
    except Exception as exc:
        return {"ok": False, "error": f"narratives table unavailable (has notebook 09 run?): {exc}"}
    row = next((r for r in rows if int(r["community_area"]) == int(community_area)), None)
    if row is None:
        return {"ok": False, "error": f"no summary yet for community area {community_area}"}
    try:
        context = json.loads(row["context_json"]) if isinstance(row["context_json"], str) else (row["context_json"] or {})
    except json.JSONDecodeError:
        context = {}
    return {
        "ok": True,
        "community_area": int(community_area),
        "area_name": area_name(community_area),
        "as_of_date": ad.iso(row["as_of_date"]),
        "narrative": row["narrative"],
        "context": context,
        "generated_at": ad.iso(row["generated_at"]),
        "model": row["model"],
    }


@_logged("get_next_week_outlook")
def get_next_week_outlook(community_area: int | None = None) -> dict[str, Any]:
    """This week's crime outlook: will an area come in above or below its normal for the time of year?

    A forecast from notebook 08 (src/outlook/), not an observed trend. The citywide weekly total
    comes from momentum, holidays and the weather forecast; each area gets that citywide call
    times its own recent momentum, and a call: "up" (P(above normal) >= 0.65), "down" (<= 0.35)
    or "unclear". Most of the swing is citywide, so neighboring areas often get the same call.
    `track_record` is the walk-forward backtest behind the calls.

    Args:
        community_area: optional, 1-77. When given, also returns `area` with its call.

    Returns:
        {"ok": True, "week_start", "week_end", "data_through",
         "citywide": {"pct_vs_normal", "drivers_pct": {"momentum", "weather", "calendar"},
                      "forecast_temp_vs_normal_f", "forecast_rain_vs_normal_in", "holidays"},
         "calls": {"up": [{"community_area", "area_name", "p_above"}], "down": [...]},
         "track_record": {"backtest_from", "backtest_to", "share_of_area_weeks_called",
                          "calls_right_pct", "calls_right_ci_pct", "base_rate_above_normal_pct"},
         "area": {"community_area", "area_name", "call", "p_above_normal", "forecast", "normal",
                  "pct_vs_normal", "citywide_part_pct", "local_part_pct"},         # if community_area
         "effects": {"per_degree_f_pct", "per_inch_rain_pct", "momentum_10pct_pct", "month_start_pct",
                     "holidays": {name: pct}}}                                      # see area_data.weather_effects
        or {"ok": False, "error": str}
    """
    if community_area is not None:
        err = ad.validate_community_area(community_area)
        if err:
            return {"ok": False, "error": err}
    try:
        summary = ad.query_uc_cached(f"SELECT * FROM {ad.UC_OUTLOOK_SUMMARY}")
        city = ad.query_uc_cached(f"SELECT * FROM {ad.UC_OUTLOOK_CITY} WHERE is_live")
        areas = ad.query_uc_cached(ad.OUTLOOK_AREAS_SQL)
    except Exception as exc:
        return {"ok": False, "error": f"outlook tables unavailable (has notebook 08 run?): {exc}"}
    if not summary or not city or not areas:
        return {"ok": False, "error": "no outlook yet (notebook 08 hasn't written one)"}
    out = ad.shape_outlook(summary[0], city[0], areas, community_area)
    try:
        out["effects"] = ad.weather_effects(ad.query_uc_cached(f"SELECT feature, per_unit FROM {ad.UC_OUTLOOK_COEFS}"))
    except Exception:
        pass   # the forecast stands on its own
    return out


@_logged("get_upcoming_events")
def get_upcoming_events(community_area: int | None = None, days: int = ad.EVENTS_DAYS_AHEAD) -> dict[str, Any]:
    """Street festivals and club nights coming up, and the extra crime they tend to bring nearby.

    From notebook 08b. Festivals are filed CDOT street permits; club nights are ticketed shows at
    20 small music clubs (pro sports are left out on purpose). `extra_crimes` is the expected
    number of extra incidents in the event's own blocks and the ring around them (about a quarter
    mile), from its own past editions (or club) pooled with the average for events like it. The
    numbers are small: street festivals run about +31% in their own cells, club nights about +11%.

    Args:
        community_area: optional, 1-77. Without it, the citywide top 10 festivals and top 8 clubs.
        days: how far ahead to look (1-90, default 30).

    Returns:
        {"ok": True, "community_area", "area_name", "window": [from, to],
         "festivals": [{"name", "place", "area_name", "start_date", "end_date", "extra_crimes",
                        "extra_range", "ratio", "past_editions", "street_segments"}],
         "club_nights": [{"place", "area_name", "nights", "next_date", "extra_crimes", "ratio", "next_acts"}],
         "multi_week_unscored": [{"name", "place", "start_date", "end_date"}],
         "total_extra_crimes"}
        or {"ok": False, "error": str}
    """
    if community_area is not None:
        err = ad.validate_community_area(community_area)
        if err:
            return {"ok": False, "error": err}
    if not 1 <= int(days) <= 90:
        return {"ok": False, "error": f"days must be between 1 and 90, got {days}"}
    try:
        rows = ad.query_uc_cached(f"""
            SELECT event_id, kind, name, place, start_date, end_date, community_area, area_name, history,
                   n_segments, ratio, extra, extra_lo, extra_hi
            FROM {ad.UC_EVENTS_UPCOMING}
        """)
    except Exception as exc:
        return {"ok": False, "error": f"events table unavailable (has notebook 08b run?): {exc}"}
    return ad.shape_upcoming(rows, community_area, date.today(), int(days))


@_logged("search_area_background")
def search_area_background(query: str, community_area: int | None = None, num_results: int = 5) -> dict[str, Any]:
    """Background on the community areas from their Wikipedia articles: history, the
    neighborhoods inside each area, landmarks, parks, schools, transit, politics.

    A hybrid (meaning plus exact-word) search over the passages notebook 03c indexed, so a
    neighborhood or landmark name finds its area. The articles' crime and population figures are
    dated; the other tools own current numbers.

    Args:
        query: what to look for, e.g. "history", "Wicker Park", "Green Line stations".
        community_area: optional, 1-77. Without it, all 77 articles are searched.
        num_results: passages to return (1-10, default 5).

    Returns:
        {"ok": True, "query", "community_area", "area_name", "source",
         "passages": [{"community_area", "area_name", "section", "text", "page_url", "score"}]}
        or {"ok": False, "error": str}
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "query must not be empty"}
    if community_area is not None:
        err = ad.validate_community_area(community_area)
        if err:
            return {"ok": False, "error": err}
    num_results = max(1, min(int(num_results), 10))
    try:
        passages = ad.search_wiki(query, community_area, num_results)
    except Exception as exc:
        return {"ok": False, "error": f"background search unavailable (has notebook 03c run?): {exc}"}
    return {
        "ok": True,
        "query": query,
        "community_area": community_area,
        "area_name": area_name(community_area) if community_area else None,
        "source": LICENSE,
        "passages": passages,
    }


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------


@_logged("subscribe_to_area")
def subscribe_to_area(user_id: int, community_area: int) -> dict[str, Any]:
    """Subscribe a user to trend updates for a community area. Idempotent.

    Args:
        user_id: an existing users.user_id.
        community_area: Chicago community area number, 1-77.

    Returns:
        {"ok": True, "subscription_id": int | None, "already_subscribed": bool}
        or {"ok": False, "error": str}
    """
    err = ad.validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}

    query = """
        INSERT INTO area_subscriptions (user_id, community_area)
        VALUES (%(user_id)s, %(community_area)s)
        ON CONFLICT (user_id, community_area) DO NOTHING
        RETURNING subscription_id
    """
    try:
        # A psycopg 3 connection commits on a clean `with` exit and rolls back on an exception.
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"user_id": user_id, "community_area": community_area})
                row = cur.fetchone()
    except Exception as exc:
        return {"ok": False, "error": f"insert failed: {exc}"}

    if row is None:
        return {"ok": True, "subscription_id": None, "already_subscribed": True}
    return {"ok": True, "subscription_id": row[0], "already_subscribed": False}


@_logged("log_resident_report")
def log_resident_report(user_id: int, community_area: int, description: str) -> dict[str, Any]:
    """File a resident concern report for a community area.

    Args:
        user_id: existing users.user_id.
        community_area: Chicago community area number, 1-77.
        description: free-text description of the concern (required, non-empty).

    Returns:
        {"ok": True, "report_id": int} or {"ok": False, "error": str}
    """
    err = ad.validate_community_area(community_area)
    if err:
        return {"ok": False, "error": err}
    if not description or not description.strip():
        return {"ok": False, "error": "description must be a non-empty string"}

    query = """
        INSERT INTO resident_reports (user_id, community_area, description)
        VALUES (%(user_id)s, %(community_area)s, %(description)s)
        RETURNING report_id
    """
    try:
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {
                    "user_id": user_id,
                    "community_area": community_area,
                    "description": description.strip(),
                })
                report_id = cur.fetchone()[0]
    except Exception as exc:
        return {"ok": False, "error": f"insert failed: {exc}"}

    return {"ok": True, "report_id": report_id}
