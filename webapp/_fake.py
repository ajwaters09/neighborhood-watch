"""Canned stand-ins for every Lakebase, Unity Catalog and LLM call, for UI work without Databricks.

Enabled by NW_FAKE_DATA=1 (main.py). Each call sleeps briefly so loading states show, and state
lives in process memory. The chat agent is a keyword-routed script that calls the faked tools.

The numbers are deterministic per (area, metric) and consistent with each other: the map, the
area panel, the city benchmarks, rank_areas and the rising-signal check all derive from the same
_window()/_history() values. Every 11th area gets tiny counts, to exercise the small-numbers
guard, and the lead-lag matrix has a handful of planted "leading" pairs.
"""

from __future__ import annotations

import calendar
import json
import math
import random
import statistics
import time
import zlib
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src import agent_chat, agent_tools, area_data
from src.community_areas import NEIGHBORS, POPULATION, area_name
from webapp import data_access

LATENCY_S = 0.5

# Matches notebooks/03b_crime_categories.py's category slugs.
CRIME = ["crime_total", "crime_violent", "crime_property", "crime_drugs", "crime_weapons",
         "crime_public_order", "crime_fraud_financial", "crime_other"]
SR = ["311_total", "311_graffiti", "311_streetlight_out", "311_rodent_rat", "311_illegal_dumping",
      "311_vacant_abandoned_building", "311_abandoned_vehicle"]
METRICS = CRIME + SR

# (sr_metric, crime_metric) -> (best lag, r): the planted leading patterns.
PLANTED = {
    ("311_streetlight_out", "crime_property"): (1, 0.16),
    ("311_vacant_abandoned_building", "crime_violent"): (2, 0.12),
    ("311_graffiti", "crime_property"): (1, 0.09),
    ("311_illegal_dumping", "crime_drugs"): (2, 0.08),
    ("311_abandoned_vehicle", "crime_property"): (3, 0.07),
}

AS_OF = date.today() - timedelta(days=1)
# The as-of date alert rules compare quiet_through against (data_access._CURRENT_AS_OF). It only
# moves when /_fake/next-pull simulates a nightly pull, so rules can be seen to fire; the trend
# numbers themselves stay fixed at AS_OF.
_alert_as_of = {"date": AS_OF}


def _alert_pct(area: int, metric: str) -> float | None:
    """The 30-day % change alert rules see: _window's, nudged on each simulated pull so a followed
    trend's updates can come out climbing, steady or easing."""
    pct = _window(area, metric, 30)["pct_change_vs_prior_window"]
    pulls = (_alert_as_of["date"] - AS_OF).days
    if pct is None or pulls == 0:
        return pct
    return round(pct + _rng(area, metric, "pull", pulls).uniform(-12, 12), 1)

_users: dict[str, tuple[int, str | None]] = {}
_subs: dict[int, list[dict[str, Any]]] = {}
_rules: dict[int, list[dict[str, Any]]] = {}
_chat_turns: dict[int, dict[str, Any]] = {}
_next_id = {"user": 1, "rule": 1, "report": 1, "turn": 1}


def _slow(factor: float = 1.0) -> None:
    time.sleep(LATENCY_S * factor)


def _rng(*key: Any) -> random.Random:
    # crc32 rather than hash(): stable across processes/reloads.
    return random.Random(zlib.crc32(repr(key).encode()))


def _scale(area: int, metric: str) -> float:
    tiny = area % 11 == 0 and metric != "crime_total" and metric != "311_total"
    if metric.endswith("_total"):
        base = 420
    elif metric in ("crime_property", "crime_violent", "311_streetlight_out", "311_graffiti"):
        base = 110
    else:
        base = 35
    return base * (0.06 if tiny else (0.35 + 1.3 * _rng(area, metric).random()))


def _window(area: int, metric: str, window_days: int) -> dict[str, Any]:
    prior = round(_scale(area, metric) * window_days / 30 * (0.9 + 0.2 * _rng(area, metric, "p", window_days).random()))
    pct_draw = _rng(area, metric, window_days).gauss(0.05 if metric.startswith("311_") else 0, 0.22)
    count = max(0, round(prior * (1 + pct_draw)))
    pct = None if prior == 0 else round((count - prior) / prior * 100, 1)
    return {"window_count": count, "prior_window_count": prior, "pct_change_vs_prior_window": pct}


def _periods(months: int) -> list[date]:
    d = AS_OF.replace(day=1)
    out = []
    for _ in range(months + 1):
        out.append(d)
        d = (d - timedelta(days=1)).replace(day=1)
    return list(reversed(out))


def _history(area: int, metric: str, months: int) -> list[dict[str, Any]]:
    r = _rng(area, metric, "hist")
    base = _scale(area, metric)
    hist = []
    for p in _periods(months):
        season = 1 + 0.18 * (1 if p.month in (6, 7, 8) else -0.5 if p.month in (12, 1, 2) else 0)
        count = max(0, round(base * season * (1 + 0.18 * r.uniform(-1, 1))))
        if (p.year, p.month) == (AS_OF.year, AS_OF.month):
            count = round(count * AS_OF.day / calendar.monthrange(p.year, p.month)[1])
        hist.append({"period": p.isoformat(), "count": count})
    return hist


def _normals_rows() -> list[dict[str, Any]]:
    """area_trend_normals stand-in: normal = the prior window nudged by a per-area drift, so
    vs-normal and vs-prior-window disagree sometimes, like the real thing."""
    rows = []
    for a in range(1, 78):
        for m in METRICS:
            for w in (30, 60, 90):
                cur = _window(a, m, w)
                normal = round(cur["prior_window_count"] * (1 + _rng(a, m, "normal").gauss(0.03, 0.12)), 1)
                pct = None if normal <= 0 else (cur["window_count"] / normal - 1) * 100
                rows.append({"community_area": a, "metric": m, "window_days": w, "window_count": cur["window_count"],
                             "normal_count": normal, "pct_vs_normal": pct, "as_of_date": AS_OF})
    return rows


def _profile_rows() -> list[dict[str, Any]]:
    """area_profile stand-in: a 12-month count from _scale per resident (±20% per-area drift), ranked
    across areas."""
    rows = []
    for m in METRICS:
        drift = {a: 1 + 0.4 * _rng(a, m, "drift").random() - 0.2 for a in range(1, 78)}
        annual = {a: _scale(a, m) * 12 * (3 if a in (68, 67, 26, 27, 29) else 1) * drift[a] for a in range(1, 78)}
        rates = {a: annual[a] / POPULATION[a] * 1000 for a in range(1, 78)}
        order = sorted(rates, key=rates.get)
        for a in range(1, 78):
            below = sum(1 for b in rates if rates[b] < rates[a])
            rows.append({"community_area": a, "metric": m, "population": POPULATION[a], "annual_count": annual[a],
                         "rate_per_1k": rates[a], "citywide_percentile": round(below / 76 * 100),
                         "citywide_rank": 77 - order.index(a), "years": 1, "as_of_date": AS_OF})
    return rows


FAKE_CITY_OUTLOOK = {"pct_vs_normal": 0.042, "pct_momentum": 0.025, "pct_weather": 0.018, "pct_calendar": 0.0,
                     "fc_t_anom": 2.3, "fc_p_anom_cm": -0.4, "holidays": ""}


def _outlook_areas() -> list[dict[str, Any]]:
    """outlook_area_week's live rows stand-in."""
    areas = []
    for a in range(1, 78):
        r = _rng(a, "outlook")
        normal = _window(a, "crime_total", 30)["window_count"] / 30 * 7 + 1
        local = r.uniform(-0.08, 0.08)
        forecast = normal * (1 + FAKE_CITY_OUTLOOK["pct_vs_normal"]) * (1 + local)
        # the real model's NB spread (dispersion ~0.01), by normal approximation
        p = 0.5 * (1 + math.erf((forecast - normal) / math.sqrt(2 * (forecast + 0.01 * forecast ** 2))))
        areas.append({"community_area": a, "call": "up" if p >= 0.65 else "down" if p <= 0.35 else "unclear",
                      "p_above": p, "forecast": forecast, "normal": normal, "pct_vs_normal": forecast / normal - 1,
                      "pct_citywide": FAKE_CITY_OUTLOOK["pct_vs_normal"], "pct_local": local})
    return areas


# outlook_coefficients' per_unit, from a run of src/outlook on real data.
COEFS = {"dep_7d": 0.2393, "dep_28d": 0.4543, "hol_in_recent_7d": -0.0061, "hol_new_years_day": 0.0235,
         "hol_mlk_day": -0.0214, "hol_presidents_day": -0.0245, "hol_easter": -0.0337, "hol_memorial_day": -0.0024,
         "hol_juneteenth": -0.0387, "hol_independence_day": 0.0077, "hol_labor_day": -0.0072, "hol_columbus_day": -0.0135,
         "hol_halloween": 0.0337, "hol_veterans_day": 0.0090, "hol_thanksgiving": -0.0412, "hol_christmas_eve": 0.0184,
         "hol_christmas": -0.1017, "hol_new_years_eve": 0.0321, "month_start": 0.0181, "t_anom": 0.0096,
         "p_anom_cm": -0.0035, "t_anom_recent_7d": -0.0055}
# events_priors, from offline_ml/README.md's festival size priors (run of 2026-09-26) and club nights.
PRIORS = [{"kind": "festival", "bucket": "1", "n_past": 1864, "prior": 1.09},
          {"kind": "festival", "bucket": "2-4", "n_past": 1046, "prior": 1.12},
          {"kind": "festival", "bucket": "5+", "n_past": 271, "prior": 1.30},
          {"kind": "club_night", "bucket": "all", "n_past": 14065, "prior": 1.11}]


def _city_weeks() -> list[dict[str, Any]]:
    """outlook_city_week stand-in: a year of weekly actual/forecast/normal, plus this week live."""
    monday = date.today() - timedelta(days=date.today().weekday())
    rows = []
    for k in range(53, -1, -1):
        wk = monday - timedelta(weeks=k)
        r = _rng("cityweek", wk.isoformat())
        normal = 4700 * (1 + 0.1 * math.sin((wk.timetuple().tm_yday - 100) / 365 * 2 * math.pi))
        t = r.gauss(0, 3)
        forecast = normal * (0.97 + 0.0096 * t + r.gauss(0, 0.015))
        live = k == 0
        if 0 < k < 3:
            continue          # the reporting lag: recent weeks aren't scored yet
        rows.append({"cutoff": wk, "y": None if live else round(forecast * (1 + r.gauss(0, 0.035))), "forecast": forecast,
                     "bar_calibrated": normal, "fc_t_anom": t, "is_live": live})
    return rows


def _forecast_days() -> list[dict[str, Any]]:
    """forecast_days_sql stand-in: this week's daily forecast (°C, mm) against a late-September
    normal, averaging FAKE_CITY_OUTLOOK's fc_t_anom warmer, with a couple of wet days."""
    monday = date.today() - timedelta(days=date.today().weekday())
    rows = []
    for i in range(7):
        day = monday + timedelta(days=i)
        r = _rng("forecast", day.isoformat())
        normal = 17.5 - 0.15 * i
        rain = [0.0, 0.3, 6.1, 11.4, 0.8, 0.0, 0.0][i]
        rows.append({"target_day": day, "issue_date": monday - timedelta(days=1),
                     "t_mean": normal + FAKE_CITY_OUTLOOK["fc_t_anom"] + r.uniform(-2.5, 2.5), "precip": rain,
                     "t_normal": normal, "p_normal": 2.4})
    return rows


def _lead_lag_matrix() -> list[dict[str, Any]]:
    rows = []
    for s in SR:
        for c in CRIME:
            best = PLANTED.get((s, c))
            for lag in (0, 1, 2, 3):
                rr = _rng(s, c, lag)
                noise = rr.gauss(0, 0.02)
                if best:
                    r = best[1] if lag == best[0] else best[1] * 0.45 + noise
                else:
                    r = noise + (0.03 if lag == 0 else 0)
                rows.append({
                    "sr_metric": s, "crime_metric": c, "lag_months": lag, "r": round(r, 4), "n": 6800,
                    "p_value": 1e-6 if best and lag == best[0] else 0.2,
                    "reverse_r": round(rr.gauss(0, 0.02), 4), "lag0_r": None,
                    "is_leading": bool(best and lag == best[0]),
                })
    lag0 = {(r["sr_metric"], r["crime_metric"]): r["r"] for r in rows if r["lag_months"] == 0}
    for r in rows:
        r["lag0_r"] = lag0[(r["sr_metric"], r["crime_metric"])]
    return rows


def install() -> None:
    import psycopg

    # ---------------------------------------------------------------- users

    def find_user_by_email(email):
        _slow()
        return _users.get(email.strip().lower())

    def create_user(email, display_name=None):
        _slow()
        email = email.strip().lower()
        if email in _users:
            raise psycopg.errors.UniqueViolation("duplicate key value violates unique constraint")
        uid = _next_id["user"]
        _next_id["user"] += 1
        _users[email] = (uid, display_name)
        return uid

    # ---------------------------------------------------------------- trends

    def list_metrics():
        return sorted(METRICS)

    def get_area_trend(community_area, months=6, window_days=30, include_context=True):
        _slow()
        metrics = {}
        for m in METRICS:
            hist = _history(community_area, m, months)
            area_data.annotate_partial_month(hist, AS_OF)
            metrics[m] = {
                "latest_period": hist[-1]["period"], "latest_count": hist[-1]["count"],
                "pct_change_vs_prior": None,
                "rolling_3mo_avg": sum(h["count"] for h in hist[-3:]) / 3,
                "history": hist,
                **_window(community_area, m, window_days),
            }
        result = {"ok": True, "community_area": community_area, "area_name": area_name(community_area),
                  "window_days": window_days, "as_of_date": AS_OF.isoformat(), "metrics": metrics}
        return agent_tools.attach_context(result) if include_context else result

    def area_benchmarks(window_days):
        rows = []
        for m in METRICS:
            counts = [(_window(a, m, window_days)["window_count"], a) for a in range(1, 78)]
            for count, a in counts:
                # percent_rank: share of areas strictly below (ties share the lowest rank).
                rows.append((m, a, count, sum(1 for c, _ in counts if c < count) / 76))
        return data_access._assemble_benchmarks(rows)

    def query_uc_cached(sql):
        _slow()
        if "outlook_coefficients" in sql:
            return [{"feature": f, "per_unit": v} for f, v in COEFS.items()]
        if "outlook_city_week" in sql:
            return _city_weeks()
        if "events_priors" in sql:
            return PRIORS
        if "outlook_area_week" in sql:
            return _outlook_areas()
        if "bronze_weather_forecast" in sql:
            return _forecast_days()
        raise RuntimeError(f"[fake] no stand-in for: {sql.strip()[:80]}")

    def neighbor_averages(community_area, window_days):
        _slow(0.3)
        areas = NEIGHBORS.get(int(community_area), [])
        if not areas:
            return {}
        return {m: round(statistics.mean(_window(a, m, window_days)["window_count"] for a in areas), 1) for m in METRICS}

    def city_trend(months=6, window_days=30):
        _slow()
        monthly_rows, rolling_rows_ = [], []
        for m in METRICS:
            hists = [_history(a, m, months) for a in range(1, 78)]
            for i, p in enumerate(hists[0]):
                monthly_rows.append((date.fromisoformat(p["period"]), m, sum(h[i]["count"] for h in hists)))
            ws = [_window(a, m, window_days) for a in range(1, 78)]
            rolling_rows_.append((m, sum(w["window_count"] for w in ws), sum(w["prior_window_count"] for w in ws), AS_OF))
        return area_data.shape_city_trend(monthly_rows, rolling_rows_, window_days)

    def monthly_series(community_area, metric):
        """~8 years of monthly counts: a slow decline, a 2020 dip and 2021 rebound, seasonality,
        and a per-area recent turn, through a partial as-of month."""
        _slow(0.3)
        base = (_scale(community_area, metric) if community_area else sum(_scale(a, metric) for a in range(1, 78)))
        turn = _rng(community_area, metric, "turn").uniform(-0.25, 0.25)
        rows, p = [], date(2018, 12, 1)
        while p <= AS_OF.replace(day=1):
            years = (p.year - 2018) + p.month / 12
            level = base * (1.25 - 0.05 * years) * (0.8 if p.year == 2020 and p.month >= 4 else 1.0)
            level *= 1 + turn * max(0.0, years - 6.5)
            season = 1 + 0.15 * math.sin((p.month - 4) / 12 * 2 * math.pi)
            rows.append((p, max(0, round(level * season * (1 + 0.06 * _rng(community_area, metric, p.isoformat()).gauss(0, 1))))))
            p = (p + timedelta(days=32)).replace(day=1)
        return rows, AS_OF

    def rolling_rows(metric, window_days):
        _slow()
        return [(a, w["window_count"], w["prior_window_count"], w["pct_change_vs_prior_window"], AS_OF)
                for a in range(1, 78) for w in [_window(a, metric, window_days)]]

    def normals_rows():
        _slow(2)   # a UC read: the slow path the app loads after the panel
        return _normals_rows()

    def profile_rows():
        _slow(2)
        return _profile_rows()

    def get_leading_indicators(community_area=None, window_days=30, include_matrix=False):
        _slow(1.5)
        rows = _lead_lag_matrix()
        pairs = area_data.summarize_leading_pairs(rows)
        out = {"ok": True, "pairs": pairs}
        if include_matrix:
            out["matrix"] = rows
        if community_area is not None:
            current = []
            for s in sorted({p["sr_metric"] for p in pairs}):
                w = _window(community_area, s, window_days)
                current.append((s, w["window_count"], w["prior_window_count"], w["pct_change_vs_prior_window"]))
            out.update({"community_area": community_area, "area_name": area_name(community_area),
                        "window_days": window_days, **area_data.split_signals(current, pairs)})
        return out

    def get_area_summary(community_area):
        _slow(1.5)
        crime = _window(community_area, "crime_total", 30)
        prop = _window(community_area, "crime_property", 30)
        lights = _window(community_area, "311_streetlight_out", 30)
        r = _rng(community_area, "blocks")
        blocks = [{"count": r.randint(3, 14), "block": f"0{r.randint(10, 79)}XX W {r.choice(['CHICAGO', 'MADISON', 'NORTH', 'DIVISION'])} AVE"} for _ in range(3)]
        name = area_name(community_area)
        direction = "up" if (crime["pct_change_vs_prior_window"] or 0) > 0 else "down"
        narrative = (
            f"(fake mode) Reported crime in {name} is {direction} over the last 30 days: {crime['window_count']} incidents "
            f"vs. {crime['prior_window_count']} the 30 days before. Property crime makes up the largest share "
            f"({prop['window_count']} reports), clustered around {blocks[0]['block'].title()}. Streetlight-out requests "
            f"went from {lights['prior_window_count']} to {lights['window_count']}; citywide, rises in this request type "
            "have historically come before more property crime about a month later, a pattern worth watching rather than a prediction."
        )
        return {
            "ok": True, "community_area": community_area, "area_name": name, "as_of_date": AS_OF.isoformat(),
            "narrative": narrative,
            "context": {"top_crime_blocks": blocks,
                        "top_311_types": [{"count": lights["window_count"], "sr_type": "Street Light Out Complaint"},
                                          {"count": r.randint(5, 30), "sr_type": "Graffiti Removal Request"}]},
            "generated_at": datetime.now(timezone.utc).isoformat(), "model": "fake-model",
        }

    def get_next_week_outlook(community_area=None):
        _slow(1.2)
        monday = date.today() - timedelta(days=date.today().weekday())
        city, areas = FAKE_CITY_OUTLOOK, _outlook_areas()
        summary = {"live_cutoff": monday, "live_week_end": monday + timedelta(days=6), "data_through": AS_OF - timedelta(days=8),
                   "test_first": date(2022, 1, 3), "test_last": date(2026, 8, 31), "share_called": 0.234,
                   "hit_rate": 0.747, "hit_lo": 0.706, "hit_hi": 0.779, "base_rate": 0.48}
        return area_data.shape_outlook(summary, city, areas, community_area)

    def get_upcoming_events(community_area=None, days=30):
        _slow(1.0)
        today = date.today()
        on = lambda n: (today + timedelta(days=n)).isoformat()
        rows = [   # a few recognizable ones for Lake View (6), a random handful elsewhere
            {"kind": "festival", "name": "Wrigleyville Halloween Crawl", "place": "3450-3600 N CLARK ST", "community_area": 6,
             "area_name": "Lake View", "start_date": on(12), "end_date": on(13), "history": 4, "n_segments": 4,
             "ratio": 1.53, "extra": 7.1, "extra_lo": 1.5, "extra_hi": 13.4},
            {"kind": "festival", "name": "Boo Bash", "place": "3506-3519 N CLARK ST", "community_area": 6, "area_name": "Lake View",
             "start_date": on(5), "end_date": on(5), "history": 0, "n_segments": 1, "ratio": 1.09, "extra": 0.4,
             "extra_lo": -2.3, "extra_hi": 4.1},
        ] + [{"kind": "club_night", "name": act, "place": "Vic Theatre", "community_area": 6, "area_name": "Lake View",
              "start_date": on(n), "end_date": on(n), "history": 250, "n_segments": None, "ratio": 1.11, "extra": 0.1,
              "extra_lo": 0.05, "extra_hi": 0.15}
             for n, act in [(1, "Cortex"), (3, "Lambrini Girls"), (8, "Grits & Eggs Podcast"), (15, "Wednesday")]]
        for a in range(1, 78):
            r = _rng(a, "events")
            if a != 6 and r.random() < 0.35:
                start = r.randint(1, 28)
                rows.append({"kind": "festival", "name": r.choice(["Fall Fest", "Harvest Market", "Taste of the Block", "Art Walk"]),
                             "place": f"{r.randint(1, 60) * 100}-{r.randint(61, 90) * 100} N CLARK ST", "community_area": a,
                             "area_name": area_name(a), "start_date": on(start), "end_date": on(start + r.randint(0, 1)),
                             "history": r.choice([0, 0, 2, 5]), "n_segments": r.randint(1, 6), "ratio": round(r.uniform(1.04, 1.3), 2),
                             "extra": round(r.uniform(0.05, 1.5), 1), "extra_lo": -1.0, "extra_hi": 3.0})
        return area_data.shape_upcoming(rows, community_area, today, days)

    def get_recent_activity(community_area, days=30, limit=50):
        _slow()
        r = _rng(community_area, days)
        now = datetime.now(timezone.utc)
        recs = []
        for _ in range(min(limit, 12)):
            ts = (now - timedelta(hours=r.randint(1, days * 24))).isoformat()
            if r.random() < 0.5:
                recs.append({"source": "crime", "timestamp": ts, "type": r.choice(["THEFT", "BATTERY", "NARCOTICS", "CRIMINAL DAMAGE"]),
                             "description": r.choice(["SIMPLE", "$500 AND UNDER", "TO VEHICLE"]), "arrest": r.random() < 0.2})
            else:
                recs.append({"source": "311", "timestamp": ts, "type": r.choice(["Street Light Out Complaint", "Graffiti Removal Request", "Rodent Baiting/Rat Complaint"]),
                             "address": f"{r.randint(100, 5900)} W {r.choice(['CHICAGO', 'MADISON', 'NORTH'])} AVE", "status": r.choice(["Open", "Completed"])})
        recs.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"ok": True, "community_area": community_area, "records": recs}

    def search_area_background(query, community_area=None, num_results=5):
        _slow()
        area = community_area or 24
        name = area_name(area)
        url = "https://en.wikipedia.org/wiki/" + name.replace(" ", "_") + ",_Chicago"
        sections = [("Overview", f"(fake mode) {name} is one of the 77 community areas of Chicago."),
                    ("History", f"(fake mode) {name} grew up along the streetcar lines in the late 1800s."),
                    ("Neighborhoods", f"(fake mode) {name} takes in several smaller neighborhoods.")]
        passages = [{"community_area": area, "area_name": name, "section": sec, "text": text,
                     "page_url": url, "score": round(0.9 - 0.1 * i, 3)}
                    for i, (sec, text) in enumerate(sections[:num_results])]
        return {"ok": True, "query": query, "community_area": community_area,
                "area_name": name if community_area else None, "source": "Wikipedia, CC BY-SA 4.0",
                "passages": passages}

    # ---------------------------------------------------------------- subscriptions / rules / reports

    def list_subscriptions(user_id):
        return sorted(_subs.get(user_id, []), key=lambda s: s["community_area"])

    def list_alert_rules(user_id):
        return list(_rules.get(user_id, []))

    def create_alert_rule(user_id, community_area, metric, threshold_pct, kind="threshold"):
        _slow()
        if kind not in data_access.ALERT_KINDS:
            raise ValueError(f"kind must be one of {data_access.ALERT_KINDS}")
        rid = _next_id["rule"]
        _next_id["rule"] += 1
        _rules.setdefault(user_id, []).append({"rule_id": rid, "community_area": community_area, "metric": metric,
                                               "threshold_pct": float(threshold_pct), "kind": kind,
                                               "quiet_through": _alert_as_of["date"],
                                               "last_pct": _alert_pct(community_area, metric)})
        return rid

    def delete_alert_rule(rule_id, user_id):
        _slow()
        _rules[user_id] = [r for r in _rules.get(user_id, []) if r["rule_id"] != rule_id]

    def clear_alert(rule_id, user_id):
        _slow()
        for r in _rules.get(user_id, []):
            if r["rule_id"] == rule_id:
                r["quiet_through"] = _alert_as_of["date"]
                r["last_pct"] = _alert_pct(r["community_area"], r["metric"])
                return True
        return False

    def evaluate_alert_rules(user_id):
        out = []
        for r in _rules.get(user_id, []):
            if r["quiet_through"] is not None and _alert_as_of["date"] <= r["quiet_through"]:
                continue
            pct = _alert_pct(r["community_area"], r["metric"])
            if pct is not None and (r["kind"] == "trend" or pct >= r["threshold_pct"]):
                out.append(data_access.shape_alert(r["rule_id"], r["community_area"], r["metric"], r["threshold_pct"],
                                                   r["kind"], r["last_pct"], pct, _alert_as_of["date"]))
        return out

    def subscribe_to_area(user_id, community_area):
        _slow()
        subs = _subs.setdefault(user_id, [])
        if any(s["community_area"] == community_area for s in subs):
            return {"ok": True, "subscription_id": None, "already_subscribed": True}
        subs.append({"community_area": community_area, "created_at": datetime.now()})
        return {"ok": True, "subscription_id": len(subs), "already_subscribed": False}

    def unsubscribe(user_id, community_area):
        _slow()
        subs = _subs.get(user_id, [])
        _subs[user_id] = [s for s in subs if s["community_area"] != community_area]
        return len(_subs[user_id]) < len(subs)

    def log_resident_report(user_id, community_area, description):
        _slow()
        rid = _next_id["report"]
        _next_id["report"] += 1
        return {"ok": True, "report_id": rid}

    # ---------------------------------------------------------------- chat + logging

    def log_chat_turn(chat_id, user_id, turn_index, active_area, user_message, assistant_message, trace, raw_messages, error=None):
        tid = _next_id["turn"]
        _next_id["turn"] += 1
        _chat_turns[tid] = {"chat_id": chat_id, "user_id": user_id, "turn_index": turn_index, "active_area": active_area,
                            "user_message": user_message, "assistant_message": assistant_message,
                            "tool_calls": trace.get("tool_calls"), "latency_ms": trace.get("latency_ms"), "error": error}
        print(f"[fake] chat_turns row {tid}: {json.dumps(_chat_turns[tid], default=str)[:300]}")
        return tid

    def log_invocation(tool_name, result, user_id, community_area):
        print(f"[fake] tool_invocations row: {tool_name} area={community_area} ok={bool(result.get('ok'))}")

    def set_chat_feedback(turn_id, user_id, feedback):
        turn = _chat_turns.get(turn_id)
        if not turn or turn["user_id"] != user_id:
            return False
        turn["feedback"] = feedback
        print(f"[fake] chat_turns row {turn_id} feedback={feedback}")
        return True

    def run_chat(messages, max_tool_rounds=4, current_user_id=None, active_area=None, trace=None):
        started = time.monotonic()
        time.sleep(1.2)
        text = messages[-1]["content"].lower()
        tool_log: list[dict[str, Any]] = []

        def call(name, args, fn):
            t0 = time.monotonic()
            result = fn(**args)
            tool_log.append({"name": name, "arguments": json.dumps(args), "ok": result.get("ok"),
                             "error": result.get("error"), "latency_ms": int((time.monotonic() - t0) * 1000)})
            tc = {"id": f"call_{len(tool_log)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            return result, [{"role": "assistant", "content": "", "tool_calls": [tc]},
                            {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)}]

        if "subscribe" in text:
            area = active_area or 25
            _, msgs = call("subscribe_to_area", {"user_id": current_user_id, "community_area": area}, subscribe_to_area)
            reply = f"Done. You're subscribed to {area_name(area)} (area {area})."
        elif "which area" in text or "largest" in text or "biggest" in text:
            result, msgs = call("rank_areas", {"metric": "crime_total", "sort_by": "pct_increase", "limit": 5}, agent_tools.rank_areas)
            lines = [f"{a['rank']}. {a['area_name']}: {a['prior_window_count']} → {a['window_count']} ({a['pct_change']:+.1f}%)"
                     for a in result["areas"]]
            reply = ("(fake mode) Biggest 30-day increases in reported crime:\n" + "\n".join(lines) +
                     f"\n\n{result['excluded_low_baseline']} areas with fewer than {result['min_prior_count']} incidents in the prior window were left out, since small numbers swing wildly.")
        elif "history" in text or "tell me about" in text:
            area = active_area or 24
            result, msgs = call("search_area_background", {"query": "history", "community_area": area}, search_area_background)
            p = result["passages"][0]
            reply = f"{result['passages'][1]['text']} (from [Wikipedia]({p['page_url']}))"
        else:
            area = active_area or 25
            result, msgs = call("get_area_summary", {"community_area": area}, get_area_summary)
            reply = result["narrative"]
        if trace is not None:
            trace.update({"model": "fake-model", "tool_calls": tool_log, "latency_ms": int((time.monotonic() - started) * 1000)})
        return msgs + [{"role": "assistant", "content": reply}]

    for name, fn in [("find_user_by_email", find_user_by_email), ("create_user", create_user), ("list_metrics", list_metrics),
                     ("area_benchmarks", area_benchmarks),
                     ("list_subscriptions", list_subscriptions), ("list_alert_rules", list_alert_rules),
                     ("create_alert_rule", create_alert_rule), ("delete_alert_rule", delete_alert_rule),
                     ("clear_alert", clear_alert), ("unsubscribe", unsubscribe),
                     ("evaluate_alert_rules", evaluate_alert_rules), ("log_chat_turn", log_chat_turn),
                     ("set_chat_feedback", set_chat_feedback), ("neighbor_averages", neighbor_averages)]:
        setattr(data_access, name, fn)
    for name, fn in [("monthly_series", monthly_series), ("query_uc_cached", query_uc_cached), ("city_trend", city_trend),
                     ("normals_rows", normals_rows), ("profile_rows", profile_rows)]:
        setattr(area_data, name, fn)
    for name, fn in [("list_metric_names", lambda: sorted(METRICS)), ("get_area_trend", get_area_trend), ("get_recent_activity", get_recent_activity),
                     ("_rolling_rows", rolling_rows), ("get_leading_indicators", get_leading_indicators),
                     ("get_area_summary", get_area_summary), ("get_next_week_outlook", get_next_week_outlook),
                     ("get_upcoming_events", get_upcoming_events),
                     ("search_area_background", search_area_background),
                     ("subscribe_to_area", subscribe_to_area), ("log_resident_report", log_resident_report)]:
        setattr(agent_tools, name, fn)
    agent_tools._log_invocation = log_invocation     # @_logged would otherwise write to the real Lakebase
    agent_chat.run_chat = run_chat


def add_routes(app) -> None:
    """Fake-mode-only dev routes, registered by main.py next to the real ones."""
    from fastapi.responses import RedirectResponse

    @app.get("/_fake/next-pull")
    def next_pull():
        """Simulate a nightly data pull for alert rules: every rule set or cleared before now
        gets checked again. Visit it, then open My Areas."""
        _alert_as_of["date"] += timedelta(days=1)
        return RedirectResponse("/", status_code=303)
