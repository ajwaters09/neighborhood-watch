"""Neighborhood Watch: the FastAPI + HTMX/Alpine app, deployed as a Databricks App (app.yaml).

Views: a /login page (email only, no passwords: demo scope) gates Explore (the map and the
selected area's panel), My Areas (subscriptions and alert rules), Insights, and a floating chat
panel. Data comes from src/agent_tools.py and src/area_data.py (shared with the agent) and
webapp/data_access.py (app-only Lakebase queries), authenticated as whatever identity runs the
process: the app's service principal when deployed, your `databricks auth login` locally.

Only what changed re-renders: HTMX swaps a template partial for each form and selector, and the
map and charts draw client-side (static/app.js) from small JSON endpoints. Every write confirms
with a toast through an `HX-Trigger` header (_trigger), including writes the chat agent makes.

Run locally against the workspace with `uvicorn webapp.main:app --reload`, or with canned data
and no Databricks at all with NW_FAKE_DATA=1 (webapp/_fake.py).
"""

from __future__ import annotations

import calendar
import json
import os
import secrets
from decimal import Decimal
from typing import Any

import psycopg
from fastapi import FastAPI, Form, Request
from markdown_it import MarkdownIt
from markupsafe import Markup
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from src import agent_chat, agent_tools, area_data
from src.constants import HOLIDAY_LABELS, RISING_PCT, WINDOW_DAYS_OPTIONS
from webapp import data_access

# NW_FAKE_DATA=1 swaps every Lakebase, Unity Catalog and LLM call for canned data
# (webapp/_fake.py), for UI work without Databricks. Never set on the deployed app.
if os.environ.get("NW_FAKE_DATA") == "1":
    from webapp import _fake

    _fake.install()

_HERE = os.path.dirname(__file__)

# Routes are plain `def`, not `async def`: every data call (psycopg, the Statement Execution API,
# the chat model) blocks, so FastAPI runs each request in its threadpool and the requests a page
# load fires together run in parallel.

app = FastAPI(title="Neighborhood Watch")
app.add_middleware(
    SessionMiddleware,
    # A random per-process fallback keeps local dev zero-config, but sessions then reset on every
    # restart. Deployments set SESSION_SECRET (app.yaml's session-secret resource).
    secret_key=os.environ.get("SESSION_SECRET", secrets.token_hex(32)),
)
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))
if os.environ.get("NW_FAKE_DATA") == "1":
    _fake.add_routes(app)   # /_fake/next-pull: simulate a nightly pull so alert rules can fire


def metric_label(metric: str) -> str:
    """'crime_motor_vehicle_theft' -> 'Motor vehicle theft' (mirrored by NW.metricLabel in app.js)."""
    for prefix, total_label in (("crime_", "All crime"), ("311_", "All 311 requests")):
        if metric.startswith(prefix):
            rest = metric[len(prefix):]
            return total_label if rest == "total" else rest.replace("_", " ").capitalize()
    return metric


templates.env.filters["metric_label"] = metric_label


def crime_phrase(metric: str) -> str:
    """'crime_property' -> 'property crime', 'crime_drugs' -> 'drug crime' -- for running text."""
    if metric == "crime_total":
        return "crime"
    word = {"drugs": "drug"}.get(metric.removeprefix("crime_"), None)
    return f"{word or metric_label(metric).lower()} crime"


templates.env.filters["crime_phrase"] = crime_phrase

# The agent answers in Markdown (bold, lists, the odd table). "html": False
# escapes any raw HTML the model emits instead of passing it through, so
# rendering its output can't inject markup into the page.
_markdown = MarkdownIt("commonmark", {"html": False, "breaks": True, "linkify": False}).enable("table")


def render_markdown(text: str | None) -> Markup:
    return Markup(_markdown.render(text or ""))


templates.env.filters["markdown"] = render_markdown


def short_date(iso: str | None) -> str:
    """'2026-10-31' -> 'Oct 31'."""
    if not iso:
        return ""
    y, m, d = (int(x) for x in str(iso)[:10].split("-"))
    return f"{calendar.month_abbr[m]} {d}"


templates.env.filters["short_date"] = short_date


def static_version() -> str:
    """Appended to /static URLs (?v=...) so browsers pick up an edited app.js/styles.css instead of a stale cached copy."""
    static_dir = os.path.join(_HERE, "static")
    return str(int(max(os.path.getmtime(os.path.join(static_dir, f)) for f in ("app.js", "styles.css"))))


templates.env.globals["static_version"] = static_version

# Chat transcripts, keyed by a random id kept in the session cookie (the transcript itself would
# overflow the ~4KB cookie). In memory, so a restart drops conversations in progress; every turn
# is also logged to Lakebase's chat_turns.
_CHAT_HISTORIES: dict[str, list[dict[str, Any]]] = {}


def _json(obj: Any) -> str:
    """json.dumps for data islands: Postgres NUMERIC comes back as Decimal, dates as date."""
    def default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default)


def _try(fn, *args, **kwargs) -> tuple[Any, str | None]:
    """Run fn, returning (result, None) or (None, str(exception)).

    data_access functions raise (the agent tools return {"ok": False, ...} instead), so routes use
    this to degrade a partial to an inline error rather than a 500.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, str(exc)


def _trigger(response: Response, toast: tuple[str, str] | None = None, **events: Any) -> Response:
    """Attach an HX-Trigger header: an optional (message, kind) toast plus any other client events.

    kind is "success" | "error" | "info" -- _toasts.html's toast store styles by it.
    """
    payload: dict[str, Any] = dict(events)
    if toast:
        payload["toast"] = {"message": toast[0], "kind": toast[1]}
    if payload:
        response.headers["HX-Trigger"] = json.dumps(payload)
    return response


def _metric_order(metrics: list[str]) -> list[str]:
    """The "_total" metric first, then the categories alphabetically by display label."""
    return sorted(metrics, key=lambda m: (not m.endswith("_total"), metric_label(m)))


def _current_user(request: Request) -> dict[str, Any] | None:
    uid = request.session.get("user_id")
    if uid is None:
        return None
    return {
        "user_id": uid,
        "email": request.session.get("email"),
        "display_name": request.session.get("display_name"),
    }


def _not_logged_in(request: Request) -> Response:
    """Where an unauthenticated request goes: HX-Redirect for HTMX calls, a plain 303 otherwise."""
    if request.headers.get("HX-Request"):
        return Response(headers={"HX-Redirect": "/login"})
    return RedirectResponse("/login", status_code=303)


def _area_name(community_area: int) -> str:
    if community_area == area_data.CITY:
        return area_data.CITY_NAME
    return data_access.area_names().get(community_area, f"Area {community_area}")


def _subscriptions_context(user: dict[str, Any]) -> dict[str, Any]:
    subs, subs_error = _try(data_access.list_subscriptions, user["user_id"])
    rules, rules_error = _try(data_access.list_alert_rules, user["user_id"])
    triggered, _ = _try(data_access.evaluate_alert_rules, user["user_id"])
    return {
        "subscriptions": subs or [], "subs_error": subs_error,
        "alert_rules": rules or [], "rules_error": rules_error,
        "triggered_alerts": triggered or [],
    }


def _form_context(request: Request, user: dict[str, Any]) -> dict[str, Any]:
    """Shared context every form-bearing partial needs: area list + metrics + subscription state."""
    area_names = data_access.area_names()
    area_options = sorted(area_names.items(), key=lambda kv: kv[1])
    metrics, metrics_error = _try(data_access.list_metrics)
    metrics = metrics or []
    ctx: dict[str, Any] = {
        "request": request,
        "user": user,
        "area_names": area_names,
        "area_options": area_options,
        "metrics": metrics,
        "crime_metrics": _metric_order([m for m in metrics if m.startswith("crime_")]),
        "sr_metrics": _metric_order([m for m in metrics if m.startswith("311_")]),
        "metrics_error": metrics_error,
    }
    ctx.update(_subscriptions_context(user))
    return ctx


def _subscribe_slot_html(request: Request, user: dict[str, Any], community_area: int | None, subscribed: bool | None = None) -> str:
    """The area panel's Subscribe button, as an out-of-band swap for #subscribe-slot."""
    if not community_area:          # None, or the citywide view (0): nothing to subscribe to
        return ""
    if subscribed is None:
        subs, _ = _try(data_access.list_subscriptions, user["user_id"])
        subscribed = any(s["community_area"] == community_area for s in subs or [])
    return templates.get_template("_subscribe_button.html").render({
        "request": request, "community_area": community_area, "subscribed": subscribed, "oob": True,
    })


def _refresh_oob_html(request: Request, user: dict[str, Any], active_area: int | None) -> str:
    """Everything a write can make stale: My Areas panel, the alert badge, and the active area's Subscribe button."""
    ctx = _form_context(request, user)
    ctx["oob"] = True
    return (
        templates.get_template("_subscriptions.html").render(ctx)
        + templates.get_template("_alert_badge.html").render(ctx)
        + _subscribe_slot_html(request, user, active_area)
    )


# ---------------------------------------------------------------------------
# Full page
# ---------------------------------------------------------------------------


@app.get("/")
def index(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    ctx = _form_context(request, user)
    chat_id = request.session.get("chat_id")
    ctx.update({
        # All of Chicago (0) until an area is picked. Client-side NW.selectArea overrides this
        # from the URL hash (#area=25) when present, so reloads keep the selected area.
        "default_area": area_data.CITY,
        "city_name": area_data.CITY_NAME,
        "window_days_options": WINDOW_DAYS_OPTIONS,
        "serving_endpoint": agent_chat.SERVING_ENDPOINT,
        "small_baseline": data_access.SMALL_BASELINE,
        "chat_messages": _CHAT_HISTORIES.get(chat_id, []) if chat_id else [],
    })
    return templates.TemplateResponse(request, "base.html", ctx)


# ---------------------------------------------------------------------------
# Sign up / log in / log out
# ---------------------------------------------------------------------------


@app.get("/login")
def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {})


def _auth_error(message: str) -> HTMLResponse:
    return HTMLResponse(f'<p class="form-error" role="alert">{message}</p>')


def _start_session(request: Request, user_id: int, email: str, display_name: str | None) -> Response:
    request.session["user_id"] = user_id
    request.session["email"] = email.strip().lower()
    request.session["display_name"] = display_name or ""
    return Response(headers={"HX-Redirect": "/"})


@app.post("/login")
def login(request: Request, email: str = Form(...)):
    found, error = _try(data_access.find_user_by_email, email)
    if error:
        return _auth_error(f"Couldn't reach the database: {error}")
    if not found:
        return _auth_error('No account for that email yet. <a href="#" @click.prevent="mode = \'signup\'">Sign up instead</a>.')
    user_id, display_name = found
    return _start_session(request, user_id, email, display_name)


@app.post("/signup")
def signup(request: Request, email: str = Form(...), display_name: str = Form("")):
    try:
        user_id = data_access.create_user(email, display_name.strip() or None)
    except psycopg.errors.UniqueViolation:
        return _auth_error("That email is already registered. <a href=\"#\" @click.prevent=\"mode = 'login'\">Log in instead</a>.")
    except Exception as exc:
        return _auth_error(f"Couldn't create the account: {exc}")
    return _start_session(request, user_id, email, display_name.strip())


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return Response(headers={"HX-Redirect": "/login"})


# ---------------------------------------------------------------------------
# Map data (JSON, consumed client-side by static/app.js)
# ---------------------------------------------------------------------------


MAP_MODES = ("level", "normal", "forecast")


@app.get("/api/map-data")
def map_data(metric: str, window_days: int = 30, mode: str = "normal"):
    """One metric for all 77 areas, for one of the map's layers:
    - level ("Crime per capita"): crime per 1,000 residents over the last 12 months, with its
      ratio to the city's rate (area_profile, via UC)
    - normal ("Trend Map"): the window vs. normal for this time of year (area_trend_normals, via UC)
    - forecast ("Outlook"): all crime only, whatever `metric` says (outlook_area_week +
      area_trend_normals, via UC)
    """
    if mode not in MAP_MODES:
        return {"ok": False, "error": f"mode must be one of {MAP_MODES}", "data": {}}
    try:
        if mode == "normal":
            snapshot = area_data.normals_snapshot(metric, window_days)
        elif mode == "level":
            snapshot = area_data.profile_snapshot(metric)
        else:
            snapshot = area_data.forecast_snapshot()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "data": {}}
    return {"ok": True, "data": {str(k): v for k, v in snapshot.items()}}


@app.get("/api/area-context")
def area_context(community_area: int, window_days: int = 30):
    """The area panel's vs-normal numbers and crime per resident, fetched after the panel renders
    (a cold SQL warehouse can take seconds; the panel shouldn't wait on it). Area 0 is the city."""
    if community_area == area_data.CITY:
        return area_data.get_city_context(window_days)
    return area_data.get_area_context(community_area, window_days)


@app.get("/api/area-rolling")
def area_rolling(community_area: int, metric: str = "crime_total"):
    """The area panel's rolling 12-month change for one crime metric. Area 0 is the city."""
    return area_data.rolling_change(community_area, metric)


def _warm_uc_cache() -> None:
    """Fill the UC read cache in the background at startup, so the first visitor's map, area
    panel and Insights view don't wait on a cold warehouse. Best-effort."""
    for fn in (area_data.normals_rows, area_data.profile_rows, area_data.lead_lag_rows,
               data_access.get_citywide_insights):
        try:
            fn()
        except Exception:
            pass


@app.on_event("startup")
def _startup() -> None:
    import threading
    threading.Thread(target=_warm_uc_cache, daemon=True).start()


# ---------------------------------------------------------------------------
# Area detail
# ---------------------------------------------------------------------------


@app.get("/area-detail")
def area_detail(request: Request, community_area: int, window_days: int = 30):
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    is_city = community_area == area_data.CITY
    # include_context=False: the vs-normal numbers and usual level come from UC, which can be
    # slow on a cold warehouse -- app.js fetches them from /api/area-context after this renders.
    # 12 months of history: the crime-type table's sparklines.
    if is_city:
        result = area_data.city_trend(months=12, window_days=window_days)
    else:
        result = agent_tools.get_area_trend(community_area, months=12, window_days=window_days, include_context=False)
    metrics = result.get("metrics", {}) if result["ok"] else {}
    subs, _ = _try(data_access.list_subscriptions, user["user_id"])
    # City ranks and shares and the adjacent area mean are context on top of the area's own
    # numbers -- if they fail, the panel still renders without them.
    benchmarks, _ = _try(data_access.area_benchmarks, window_days)
    neighbors = None if is_city else _try(data_access.neighbor_averages, community_area, window_days)[0]
    ctx = {
        "request": request,
        "community_area": community_area,
        "is_city": is_city,
        "area_name": _area_name(community_area),
        "neighbors_json": _json(neighbors or {}),
        "subscribed": any(s["community_area"] == community_area for s in subs or []),
        "error": None if result["ok"] else result["error"],
        "metrics": metrics,
        "crime_metrics": _metric_order([m for m in metrics if m.startswith("crime_")]),
        "metrics_json": _json({m: v for m, v in metrics.items() if m.startswith("crime_")}),
        # 311 feeds only the summary card, so no monthly history.
        "sr_json": _json({m: {k: v for k, v in d.items() if k != "history"} for m, d in metrics.items() if m.startswith("311_")}),
        "benchmarks_json": _json(benchmarks or {}),
        "as_of_date": result.get("as_of_date"),
        "small_baseline": data_access.SMALL_BASELINE,
        "window_days": window_days,
    }
    return templates.TemplateResponse(request, "_area_detail.html", ctx)


@app.get("/area-insight")
def area_insight(request: Request, community_area: int, window_days: int = 30):
    """The area panel's lower cards: The week ahead, Upcoming events, What's happening here, and
    Signals to watch.

    Loaded after /area-detail renders (hx-trigger="load"), because they read Unity Catalog through
    the SQL warehouse, which can take seconds on a cold start.

    Each signal gets one-click alert buttons on the 311 type itself (the early signal), never on
    the crime category it tends to precede. Rules the user already has render as "Alert set".
    """
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    is_city = community_area == area_data.CITY
    area = None if is_city else community_area
    # The narrative and the 311 signals are per area; the citywide view skips them.
    no = {"ok": False, "error": None}
    summary = no if is_city else agent_tools.get_area_summary(community_area)
    signals = no if is_city else agent_tools.get_leading_indicators(community_area, window_days=window_days)
    outlook = agent_tools.get_next_week_outlook(area)
    events = agent_tools.get_upcoming_events(area)
    # The week-ahead card leads with how the area itself has been running lately.
    context = area_data.get_city_context(30) if is_city else area_data.get_area_context(community_area, 30)
    lately = context.get("vs_normal", {}).get("crime_total") if context["ok"] else None
    rules, _ = _try(data_access.list_alert_rules, user["user_id"])
    ctx = {
        "request": request,
        "community_area": community_area,
        "is_city": is_city,
        "area_name": _area_name(community_area),
        "summary": summary if summary["ok"] else None,
        "summary_error": None if summary["ok"] else summary["error"],
        "outlook": outlook if outlook["ok"] else None,
        "outlook_error": None if outlook["ok"] else outlook["error"],
        "outlook_drivers": outlook_drivers(outlook["citywide"]) if outlook["ok"] else [],
        "lately": lately,
        "small_baseline": data_access.SMALL_BASELINE,
        "events": events if events["ok"] else None,
        "events_error": None if events["ok"] else events["error"],
        "watch": signals.get("watch", []) if signals["ok"] else [],
        "near": signals.get("near", []) if signals["ok"] else [],
        "signals_error": None if signals["ok"] else signals["error"],
        "window_days": window_days,
        "alert_pct": SIGNAL_ALERT_PCT,
        # "kind:metric" for the rules this area already has, so each button can show "Alert set".
        "existing_rules": {f'{r["kind"]}:{r["metric"]}' for r in rules or [] if r["community_area"] == community_area},
    }
    return templates.TemplateResponse(request, "_area_insight.html", ctx)


def outlook_drivers(citywide: dict[str, Any]) -> list[dict[str, Any]]:
    """What moves next week's citywide forecast, biggest first: [{"label", "detail", "pct"}], e.g.
    {"label": "Weather", "detail": "Forecast to average 4.1 °F warmer than normal.", "pct": 1.8}.
    The same drivers apply to every area. Drivers under half a percent are left out."""
    d = citywide["drivers_pct"]
    t, rain = citywide["forecast_temp_vs_normal_f"], citywide["forecast_rain_vs_normal_in"]
    if abs(t) >= 2:
        weather = f"Forecast to average {abs(t):.1f} °F {'warmer' if t > 0 else 'cooler'} than normal."
    elif abs(rain) >= 0.4:
        weather = f"{'More' if rain > 0 else 'Less'} rain than usual in the forecast ({rain:+.1f} in)."
    else:
        weather = "Forecast close to normal."
    momentum = ("Crime across the city has run above normal lately, and part of a run like that "
                "tends to carry into the next week.") if d["momentum"] > 0 else (
                "Crime across the city has run below normal lately, and part of a lull like that "
                "tends to carry into the next week.")
    holidays = [HOLIDAY_LABELS.get(h, h.replace("_", " ")) for h in citywide["holidays"]]
    calendar_row = (("Holiday", f"The week includes {' and '.join(holidays)}.") if holidays
                    else ("Calendar", "The week includes the 1st of a month."))
    rows = [{"label": "Recent citywide crime", "detail": momentum, "pct": d["momentum"]},
            {"label": "Weather", "detail": weather, "pct": d["weather"]},
            {"label": calendar_row[0], "detail": calendar_row[1], "pct": d["calendar"]}]
    return sorted((r for r in rows if abs(r["pct"]) >= 0.5), key=lambda r: -abs(r["pct"]))


# The Signals card's one-click alert threshold: the bar that makes a leading 311 type "rising",
# so a "near" signal's alert fires exactly when it would move into "watch".
SIGNAL_ALERT_PCT = RISING_PCT


@app.get("/insights")
def insights(request: Request):
    """The Insights view: which 311 request types have historically led which crime categories."""
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    result = agent_tools.get_leading_indicators(include_matrix=True)
    ctx: dict[str, Any] = {"request": request, "error": None if result["ok"] else result["error"]}
    if result["ok"]:
        ctx.update(_insights_matrix(result["matrix"]))
        ctx["pairs"] = result["pairs"]
    ctx.update(_citywide_insights_context(data_access.get_citywide_insights()))
    return templates.TemplateResponse(request, "_insights.html", ctx)


def _citywide_insights_context(ins: dict[str, Any]) -> dict[str, Any]:
    """The Insights view's weather and events sections: template values plus the chart data."""
    weather, events = ins.get("weather"), ins.get("events")
    ctx: dict[str, Any] = {"weather": weather, "weather_error": ins.get("weather_error"),
                           "events": events, "events_error": ins.get("events_error")}
    if weather:
        eff = weather["effects"]
        ctx["holidays"] = [{"label": HOLIDAY_LABELS.get(h, h.replace("_", " ")).removeprefix("the "), "pct": v}
                           for h, v in eff["holidays"].items()]
        outlook = weather.get("outlook")
        ctx["this_week_drivers"] = outlook_drivers(outlook["citywide"]) if outlook else []
        ctx["weather_json"] = _json({"weeks": weather["weeks"], "holidays": ctx["holidays"]})
    if events:
        ctx["events_json"] = _json({"lifts": events["lifts"]})
    return ctx


def _insights_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pivot indicator_lead_lag rows into the heatmap's 311 x crime grid. Each cell carries its
    strongest forward correlation at lags 1-3, plus every lag's forward and reverse r for the
    click-through chart."""
    sr_metrics = _metric_order(sorted({r["sr_metric"] for r in rows}))
    crime_metrics = _metric_order(sorted({r["crime_metric"] for r in rows}))
    cells: dict[str, dict[str, Any]] = {}
    for r in rows:
        cell = cells.setdefault(f'{r["sr_metric"]}|{r["crime_metric"]}', {
            "sr_metric": r["sr_metric"], "crime_metric": r["crime_metric"],
            "by_lag": [], "best_lag": None, "r": None, "is_leading": False,
        })
        cell["by_lag"].append({"lag": r["lag_months"], "r": r["r"], "reverse_r": r["reverse_r"], "p_value": r["p_value"]})
        if r["lag_months"] >= 1 and r["r"] is not None and (cell["r"] is None or r["r"] > cell["r"]):
            cell["r"], cell["best_lag"] = r["r"], r["lag_months"]
        cell["is_leading"] = cell["is_leading"] or r["is_leading"]
    for cell in cells.values():
        cell["by_lag"].sort(key=lambda x: x["lag"])
    return {
        "sr_metrics": sr_metrics,
        "crime_metrics": crime_metrics,
        "cells": cells,
        "cells_json": _json(cells),
        "max_r": max((abs(c["r"]) for c in cells.values() if c["r"] is not None), default=0.1) or 0.1,
    }


@app.get("/area-activity")
def area_activity(request: Request, community_area: int, days: int = 30):
    result = agent_tools.get_recent_activity(community_area, days=days, limit=50)
    ctx = {
        "request": request,
        "error": None if result["ok"] else result["error"],
        "records": result.get("records", []) if result["ok"] else [],
    }
    return templates.TemplateResponse(request, "_activity.html", ctx)


# ---------------------------------------------------------------------------
# Subscriptions & alert rules
# ---------------------------------------------------------------------------


@app.post("/subscribe")
def subscribe(request: Request, community_area: int = Form(...), active_area: int | None = Form(None)):
    """Subscribe from either the area panel or My Areas; both use hx-swap="none" and get OOB refreshes."""
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    name = _area_name(community_area)
    try:
        res = agent_tools.subscribe_to_area(user["user_id"], community_area)
        if res["ok"]:
            toast = (f"You're already subscribed to {name}.", "info") if res["already_subscribed"] else (f"Subscribed to {name}.", "success")
        else:
            toast = (f"Couldn't subscribe: {res['error']}", "error")
    except Exception as exc:
        toast = (f"Couldn't subscribe: {exc}", "error")
    html = _refresh_oob_html(request, user, active_area if active_area is not None else community_area)
    return _trigger(HTMLResponse(html), toast)


@app.delete("/subscriptions/{community_area}")
def unsubscribe(request: Request, community_area: int, active_area: int | None = None):
    """My Areas' Unsubscribe (after its confirm click); hx-swap="none" with OOB refreshes, like /subscribe."""
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    name = _area_name(community_area)
    removed, error = _try(data_access.unsubscribe, user["user_id"], community_area)
    if error:
        toast = (f"Couldn't unsubscribe: {error}", "error")
    elif removed:
        toast = (f"Unsubscribed from {name}.", "success")
    else:
        toast = (f"You weren't subscribed to {name}.", "info")
    html = _refresh_oob_html(request, user, active_area)
    return _trigger(HTMLResponse(html), toast)


def _rules_response(request: Request, user: dict[str, Any], toast: tuple[str, str]) -> Response:
    """What every alert-rule write returns: the re-rendered My Areas panel (the request's hx-target)
    plus the alert badge out of band, with the toast."""
    ctx = _form_context(request, user)
    html = templates.get_template("_subscriptions.html").render(ctx) + templates.get_template("_alert_badge.html").render({**ctx, "oob": True})
    return _trigger(HTMLResponse(html), toast)


@app.post("/alert-rules")
def add_alert_rule(
    request: Request,
    community_area: int = Form(...),
    metric: str = Form(...),
    threshold_pct: float = Form(...),
    kind: str = Form("threshold"),
):
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    what = f"{metric_label(metric).lower()} in {_area_name(community_area)}"
    try:
        data_access.create_alert_rule(user["user_id"], community_area, metric, threshold_pct, kind)
        if kind == "trend":
            toast = (f"Following {what}. You'll get an update in My Areas each time new data comes in.", "success")
        else:
            toast = (f"Alert set: {what}, up {threshold_pct:g}% or more. It's checked each time new data comes in.", "success")
    except Exception as exc:
        toast = (f"Couldn't add the alert rule: {exc}", "error")
    return _rules_response(request, user, toast)


@app.delete("/alert-rules/{rule_id}")
def delete_alert_rule(request: Request, rule_id: int):
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    try:
        data_access.delete_alert_rule(rule_id, user["user_id"])
        toast = ("Alert rule removed.", "success")
    except Exception as exc:
        toast = (f"Couldn't remove the alert rule: {exc}", "error")
    return _rules_response(request, user, toast)


@app.post("/alert-rules/{rule_id}/clear")
def clear_alert(request: Request, rule_id: int):
    """Dismiss a triggered alert from the My Areas banner. The rule stays and checks the next pull."""
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    cleared, error = _try(data_access.clear_alert, rule_id, user["user_id"])
    if error or not cleared:
        toast = (f"Couldn't clear the alert: {error or 'rule not found'}", "error")
    else:
        toast = ("Alert cleared. It'll flag again only if it's still over your threshold after the next data pull.", "success")
    return _rules_response(request, user, toast)


# ---------------------------------------------------------------------------
# Report a concern (modal form in _report.html, hx-swap="none")
# ---------------------------------------------------------------------------


@app.post("/report")
def report(request: Request, community_area: int = Form(...), description: str = Form(...)):
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    if not description.strip():
        return _trigger(Response(), ("Describe what's going on before submitting.", "error"))
    try:
        res = agent_tools.log_resident_report(user["user_id"], community_area, description)
    except Exception as exc:
        res = {"ok": False, "error": str(exc)}
    if not res["ok"]:
        return _trigger(Response(), (f"Couldn't file the report: {res['error']}", "error"))
    # report-filed tells the modal to close and reset (NW.closeReport in app.js).
    return _trigger(Response(), (f"Report filed for {_area_name(community_area)}. Thanks!", "success"), **{"report-filed": True})


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _write_tool_toasts(new_messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Toasts for the write tools the agent ran successfully this turn. The area comes from each
    call's arguments, matched to its result by tool_call_id."""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for m in new_messages:
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls[tc["id"]] = (tc["function"]["name"], args)
    toasts = []
    for m in new_messages:
        name, args = calls.get(m.get("tool_call_id"), (None, {})) if m.get("role") == "tool" else (None, {})
        if name not in ("subscribe_to_area", "log_resident_report"):
            continue
        try:
            result = json.loads(m.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if not result.get("ok"):
            continue
        area = _area_name(int(args["community_area"])) if str(args.get("community_area", "")).isdigit() else "that area"
        if name == "log_resident_report":
            toasts.append((f"The agent filed your report for {area}.", "success"))
        elif result.get("already_subscribed"):
            toasts.append((f"You're already subscribed to {area}.", "info"))
        else:
            toasts.append((f"The agent subscribed you to {area}.", "success"))
    return toasts


@app.post("/chat")
def chat(request: Request, message: str = Form(...), active_area: int | None = Form(None)):
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    chat_id = request.session.get("chat_id")
    if not chat_id:
        chat_id = secrets.token_hex(8)
        request.session["chat_id"] = chat_id
    history = _CHAT_HISTORIES.setdefault(chat_id, [])
    history.append({"role": "user", "content": message})
    trace: dict[str, Any] = {}
    error = None
    try:
        new_messages = agent_chat.run_chat(
            history, current_user_id=user["user_id"], active_area=active_area or None, trace=trace,
        )
    except Exception as exc:
        error = str(exc)
        new_messages = [{"role": "assistant", "content": f"Sorry, something went wrong: {exc}"}]
    history.extend(new_messages)

    # The answer is the turn's last assistant message: the one that called no tools. If the loop ran
    # out of tool rounds there isn't one, so fall back to the last text the model wrote.
    texts = [m for m in new_messages if m.get("role") == "assistant" and m.get("content")]
    final_answer = next((m["content"] for m in reversed(texts) if not m.get("tool_calls")),
                        texts[-1]["content"] if texts else None)
    turn_id = data_access.log_chat_turn(
        chat_id=chat_id,
        user_id=user["user_id"],
        turn_index=sum(1 for m in history if m.get("role") == "user"),
        active_area=active_area,
        user_message=message,
        assistant_message=final_answer,
        trace=trace,
        raw_messages=new_messages,
        error=error,
    )

    # A write tool may have run: refresh My Areas and the Subscribe button out of band.
    refresh_html = None
    if any(m.get("role") == "tool" for m in new_messages):
        refresh_html = _refresh_oob_html(request, user, active_area)

    response = templates.TemplateResponse(request, "_chat_turn.html", {
        "new_messages": new_messages,
        "final_answer": final_answer,
        "turn_id": turn_id,
        "refresh_html": refresh_html,
    })
    toasts = _write_tool_toasts(new_messages)
    return _trigger(response, toasts[0] if toasts else None)


@app.post("/chat/feedback/{turn_id}")
def chat_feedback(request: Request, turn_id: int, feedback: int = Form(...)):
    """Thumbs up (1) / down (-1) on an agent answer -- stored on its chat_turns row for evals."""
    user = _current_user(request)
    if not user:
        return _not_logged_in(request)
    if feedback not in (-1, 1):
        return _trigger(Response(), ("Feedback must be up or down.", "error"))
    saved, error = _try(data_access.set_chat_feedback, turn_id, user["user_id"], feedback)
    if error or not saved:
        return _trigger(Response(), (f"Couldn't save feedback: {error or 'turn not found'}", "error"))
    return _trigger(
        HTMLResponse(f'<span class="fb-done">{"👍" if feedback == 1 else "👎"} Thanks for the feedback</span>'),
        ("Feedback saved. Thanks!", "success"),
    )
