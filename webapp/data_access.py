"""App-only queries: everything the UI needs that isn't an agent tool.

Login and sign-up, the area panel's city and neighbor benchmarks, subscriptions, alert rules
(create, delete, clear, and evaluation against the trend table), chat-turn logging and feedback,
and the Insights page's citywide read.

Reads that need every area at once go to the Synced Tables directly rather than calling
get_area_trend 77 times. Unlike the agent tools these raise on failure; main._try turns that into
an inline error.
"""

from __future__ import annotations

import json
from typing import Any

from src import agent_tools, area_data, config, db_connect
from src.community_areas import AREA_NAMES, NEIGHBORS


def area_names() -> dict[int, str]:
    """{community_area: "Rogers Park", ...} -- see src/community_areas.py."""
    return AREA_NAMES


def find_user_by_email(email: str) -> tuple[int, str | None] | None:
    """(user_id, display_name) for an existing account, or None -- the login path."""
    query = "SELECT user_id, display_name FROM users WHERE email = %(email)s"
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"email": email.strip().lower()})
            row = cur.fetchone()
    return (row[0], row[1]) if row else None


def create_user(email: str, display_name: str | None = None) -> int:
    """Insert a new account -- the sign-up path. A duplicate email raises
    psycopg.errors.UniqueViolation, which /signup turns into "already registered, log in instead"."""
    query = """
        INSERT INTO users (email, display_name)
        VALUES (%(email)s, %(display_name)s)
        RETURNING user_id
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"email": email.strip().lower(), "display_name": display_name})
            return cur.fetchone()[0]


def list_metrics() -> list[str]:
    query = f"SELECT DISTINCT metric FROM {config.TREND_METRICS_LB} ORDER BY metric"
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def list_subscriptions(user_id: int) -> list[dict[str, Any]]:
    query = """
        SELECT community_area, created_at FROM area_subscriptions
        WHERE user_id = %(user_id)s ORDER BY community_area
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"user_id": user_id})
            return [{"community_area": a, "created_at": c} for a, c in cur.fetchall()]


def unsubscribe(user_id: int, community_area: int) -> bool:
    """Drop one subscription. True if there was one to drop. CDF records the DELETE, which
    subscription_events (notebook 06) reports as an unsubscribe."""
    query = "DELETE FROM area_subscriptions WHERE user_id = %(user_id)s AND community_area = %(community_area)s"
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"user_id": user_id, "community_area": community_area})
            return cur.rowcount > 0


# Alert rules come in two kinds (alert_rules.kind):
# - "threshold": flags when a metric's last ALERT_WINDOW_DAYS are up at least threshold_pct on the
#   window before (the My Areas form, and "Alert me if it crosses" on the Signals card).
# - "trend": "Alert me on this trend" on the Signals card. It flags on every new data pull, whatever
#   the numbers do, with how the rise has moved since the last update (still climbing, holding
#   steady, easing). threshold_pct holds the signal line it was set at, for context only.
#
# Both check new data only. `quiet_through` is the trend data's as-of date when the rule was set or
# its alert last cleared, and a rule fires only on data newer than that, so a new rule waits for
# the next pull and a cleared alert stays quiet until then. `last_pct` is the metric's % change at
# that same moment: what a trend update compares against. The as-of date is the Synced Table's.
ALERT_KINDS = ("threshold", "trend")
ALERT_WINDOW_DAYS = 30  # matches agent_tools.get_area_trend's own default
_CURRENT_AS_OF = f"(SELECT max(as_of_date) FROM {config.TREND_ROLLING_LB})"
# The metric's current % change, for a rule's own area and metric (in an UPDATE of alert_rules) or
# for the query's parameters (in the INSERT).
_PCT_SQL = """(SELECT pct_change_vs_prior_window FROM {rolling} t
               WHERE t.community_area = {area} AND t.metric = {metric} AND t.window_days = %(window_days)s)"""
_CURRENT_PCT = _PCT_SQL.format(rolling=config.TREND_ROLLING_LB, area="alert_rules.community_area",
                               metric="alert_rules.metric")
_CURRENT_PCT_NEW = _PCT_SQL.format(rolling=config.TREND_ROLLING_LB, area="%(community_area)s", metric="%(metric)s")
# A trend update within this many percentage points of the last one reads as "holding steady".
TREND_STEADY_PTS = 5.0


def list_alert_rules(user_id: int) -> list[dict[str, Any]]:
    query = """
        SELECT rule_id, community_area, metric, threshold_pct, kind, quiet_through FROM alert_rules
        WHERE user_id = %(user_id)s ORDER BY community_area, metric
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"user_id": user_id})
            return [
                {"rule_id": r, "community_area": a, "metric": m, "threshold_pct": float(t), "kind": k, "quiet_through": q}
                for r, a, m, t, k, q in cur.fetchall()
            ]


def create_alert_rule(user_id: int, community_area: int, metric: str, threshold_pct: float,
                      kind: str = "threshold") -> int:
    if kind not in ALERT_KINDS:
        raise ValueError(f"kind must be one of {ALERT_KINDS}")
    query = f"""
        INSERT INTO alert_rules (user_id, community_area, metric, threshold_pct, kind, quiet_through, last_pct)
        VALUES (%(user_id)s, %(community_area)s, %(metric)s, %(threshold_pct)s, %(kind)s,
                {_CURRENT_AS_OF}, {_CURRENT_PCT_NEW})
        RETURNING rule_id
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {
                "user_id": user_id, "community_area": community_area, "metric": metric,
                "threshold_pct": threshold_pct, "kind": kind, "window_days": ALERT_WINDOW_DAYS,
            })
            return cur.fetchone()[0]


def delete_alert_rule(rule_id: int, user_id: int) -> None:
    query = "DELETE FROM alert_rules WHERE rule_id = %(rule_id)s AND user_id = %(user_id)s"
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"rule_id": rule_id, "user_id": user_id})


def clear_alert(rule_id: int, user_id: int) -> bool:
    """Dismiss a triggered alert: the rule stays, but goes quiet until the next data pull, and a
    trend rule's next update compares against the numbers just seen."""
    query = f"""
        UPDATE alert_rules SET quiet_through = {_CURRENT_AS_OF}, last_pct = {_CURRENT_PCT}
        WHERE rule_id = %(rule_id)s AND user_id = %(user_id)s
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"rule_id": rule_id, "user_id": user_id, "window_days": ALERT_WINDOW_DAYS})
            return cur.rowcount == 1


def trend_direction(pct: float, last_pct: float | None) -> str | None:
    """How a trend rule's metric moved since its last update: "climbing", "steady" or "easing"
    (None with nothing to compare against). Pure."""
    if last_pct is None:
        return None
    if pct - last_pct >= TREND_STEADY_PTS:
        return "climbing"
    if pct - last_pct <= -TREND_STEADY_PTS:
        return "easing"
    return "steady"


def evaluate_alert_rules(user_id: int) -> list[dict[str, Any]]:
    """The user's alerts to show: threshold rules whose trailing-window % change meets the
    threshold, and every trend rule, on data newer than the rule's quiet_through (see ALERT_KINDS
    above).

    Rules are judged on one fixed window (ALERT_WINDOW_DAYS of area_trend_rolling_lb), so a
    threshold always means the same thing, and never on the month-over-month figure, which a
    partial current month skews.
    """
    query = f"""
        SELECT r.rule_id, r.community_area, r.metric, r.threshold_pct, r.kind, r.last_pct,
               t.pct_change_vs_prior_window, t.as_of_date
        FROM alert_rules r
        JOIN {config.TREND_ROLLING_LB} t
          ON t.community_area = r.community_area AND t.metric = r.metric
        WHERE r.user_id = %(user_id)s
          AND t.window_days = %(window_days)s
          AND t.pct_change_vs_prior_window IS NOT NULL
          AND (r.kind = 'trend' OR t.pct_change_vs_prior_window >= r.threshold_pct)
          AND (r.quiet_through IS NULL OR t.as_of_date > r.quiet_through)
        ORDER BY r.community_area, r.metric
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"user_id": user_id, "window_days": ALERT_WINDOW_DAYS})
            rows = cur.fetchall()
    return [shape_alert(rule_id, area, metric, threshold, kind, last, pct, as_of)
            for rule_id, area, metric, threshold, kind, last, pct, as_of in rows]


def shape_alert(rule_id, area, metric, threshold, kind, last_pct, pct, as_of) -> dict[str, Any]:
    """One evaluate_alert_rules row -> what the My Areas banner shows. Pure."""
    pct = float(pct)
    last = None if last_pct is None else float(last_pct)
    return {
        "rule_id": rule_id, "community_area": area, "metric": metric, "kind": kind,
        "threshold_pct": float(threshold), "pct_change_vs_prior_window": pct, "last_pct": last,
        "direction": trend_direction(pct, last) if kind == "trend" else None,
        "window_days": ALERT_WINDOW_DAYS, "as_of_date": as_of,
    }


SMALL_BASELINE = 10  # below this prior-window count, % change is noise -- the UI shows absolute change instead


def area_benchmarks(window_days: int) -> dict[str, dict[str, Any]]:
    """City context for every metric, so one area's numbers can be read against the rest.

    Per metric:
      - "percentiles": {community_area: 0-100} -- where each area's current-window count ranks
        among all 77 (percent_rank * 100), for the crime card's rank chip.
      - "city_window": the whole city's current-window count, for the crime-type table's
        share-of-crime bars.
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT metric, community_area, window_count,
                       percent_rank() OVER (PARTITION BY metric ORDER BY window_count) AS pr
                FROM {config.TREND_ROLLING_LB}
                WHERE window_days = %(window_days)s
            """, {"window_days": window_days})
            rows = cur.fetchall()
    return _assemble_benchmarks(rows)


def _assemble_benchmarks(rows) -> dict[str, dict[str, Any]]:
    """(metric, community_area, window_count, percent_rank) rows -> area_benchmarks' shape. Pure."""
    out: dict[str, dict[str, Any]] = {}
    for metric, area, count, pr in rows:
        b = out.setdefault(metric, {"percentiles": {}, "city_window": 0.0})
        b["percentiles"][str(int(area))] = round(float(pr) * 100)
        b["city_window"] += float(count or 0)
    return out


def neighbor_averages(community_area: int, window_days: int) -> dict[str, float]:
    """{metric: current-window count averaged over the areas bordering this one
    (community_areas.NEIGHBORS)}, for the crime card's "Adjacent area mean"."""
    areas = NEIGHBORS.get(int(community_area), [])
    if not areas:
        return {}
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT metric, avg(window_count) FROM {config.TREND_ROLLING_LB}
                WHERE community_area = ANY(%(areas)s) AND window_days = %(window_days)s
                GROUP BY metric
            """, {"areas": areas, "window_days": window_days})
            return {metric: round(float(avg), 1) for metric, avg in cur.fetchall()}


def log_chat_turn(
    chat_id: str,
    user_id: int | None,
    turn_index: int,
    active_area: int | None,
    user_message: str,
    assistant_message: str | None,
    trace: dict[str, Any],
    raw_messages: list[dict[str, Any]],
    error: str | None = None,
) -> int | None:
    """Best-effort row in chat_turns (lakebase/schema.sql) for analytics and evals.

    Returns the new turn_id (the thumbs up/down buttons need it), or None if logging failed. Like
    agent_tools._log_invocation, a logging failure never breaks the reply.
    """
    query = """
        INSERT INTO chat_turns (chat_id, user_id, turn_index, active_area, user_message,
                                assistant_message, tool_calls, raw_messages, model, latency_ms, error)
        VALUES (%(chat_id)s, %(user_id)s, %(turn_index)s, %(active_area)s, %(user_message)s,
                %(assistant_message)s, %(tool_calls)s::jsonb, %(raw_messages)s::jsonb, %(model)s,
                %(latency_ms)s, %(error)s)
        RETURNING turn_id
    """
    try:
        with db_connect.get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {
                    "chat_id": chat_id, "user_id": user_id, "turn_index": turn_index,
                    "active_area": active_area, "user_message": user_message,
                    "assistant_message": assistant_message,
                    "tool_calls": json.dumps(trace.get("tool_calls", []), default=str),
                    "raw_messages": json.dumps(raw_messages, default=str),
                    "model": trace.get("model"), "latency_ms": trace.get("latency_ms"),
                    "error": error,
                })
                return cur.fetchone()[0]
    except Exception:
        return None


def set_chat_feedback(turn_id: int, user_id: int, feedback: int) -> bool:
    """Record a thumbs up (1) / down (-1) on one of the user's own chat turns."""
    query = """
        UPDATE chat_turns SET feedback = %(feedback)s, feedback_at = now()
        WHERE turn_id = %(turn_id)s AND user_id = %(user_id)s
    """
    with db_connect.get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"turn_id": turn_id, "user_id": user_id, "feedback": feedback})
            return cur.rowcount == 1


def get_citywide_insights() -> dict[str, Any]:
    """What moves crime citywide, for the Insights view: the forecast model's weather, holiday and
    momentum effects, how its weekly forecast has tracked, and how much festivals and club nights
    add. The agent gets the effects through get_next_week_outlook instead. Each part is
    best-effort, with its own error."""
    out: dict[str, Any] = {"ok": True, "weather": None, "weather_error": None, "events": None, "events_error": None}
    # The tools' unlogged originals (__wrapped__): a page load isn't agent usage.
    unlogged = lambda tool: getattr(tool, "__wrapped__", tool)
    try:
        coefs = area_data.query_uc_cached(f"SELECT feature, per_unit FROM {area_data.UC_OUTLOOK_COEFS}")
        weeks = area_data.query_uc_cached(f"""
            SELECT cutoff, y, forecast, bar_calibrated, fc_t_anom, is_live FROM {area_data.UC_OUTLOOK_CITY}
            ORDER BY cutoff DESC LIMIT {area_data.CITY_WEEKS_SHOWN + 1}
        """)
        outlook = unlogged(agent_tools.get_next_week_outlook)()
        out["weather"] = {"effects": area_data.weather_effects(coefs), "weeks": area_data.shape_city_weeks(weeks),
                          "outlook": outlook if outlook["ok"] else None, "forecast": None}
    except Exception as exc:
        out["weather_error"] = f"outlook tables unavailable (has notebook 08 run?): {exc}"
    # The daily forecast behind this week's outlook: extra, so the section renders without it.
    if out["weather"] and out["weather"]["outlook"]:
        try:
            week_start = area_data.to_date(out["weather"]["outlook"]["week_start"])
            out["weather"]["forecast"] = area_data.shape_forecast_days(
                area_data.query_uc_cached(area_data.forecast_days_sql(week_start)))
        except Exception:
            pass
    try:
        lifts = area_data.shape_event_lifts(
            area_data.query_uc_cached(f"SELECT kind, bucket, n_past, prior FROM {area_data.UC_EVENTS_PRIORS}"))
        upcoming = unlogged(agent_tools.get_upcoming_events)()
        out["events"] = {"lifts": lifts, "upcoming": upcoming if upcoming["ok"] else None}
    except Exception as exc:
        out["events_error"] = f"events tables unavailable (has notebook 08b run?): {exc}"
    return out
