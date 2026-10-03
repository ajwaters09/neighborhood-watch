"""Offline tests for the area-context helpers in src/area_data.py and src/agent_tools.py: vs-normal
and usual level, the map layers, the city view, and the outlook and events shaping.

The Unity Catalog reads are stubbed; these check the pure shaping and ranking, including the
all-string values the Statement Execution API returns.

    python -m pytest tests/test_area_context.py
"""

from datetime import date

from src import agent_tools as at
from src import area_data as ad
from src.constants import level_band

# As the Statement Execution API returns them: every value a string.
NORMALS = [
    {"community_area": "14", "metric": "crime_total", "window_days": "30", "window_count": "210",
     "normal_count": "187.6", "pct_vs_normal": "11.94", "as_of_date": "2026-09-17"},
    {"community_area": "68", "metric": "crime_total", "window_days": "30", "window_count": "385",
     "normal_count": "387.3", "pct_vs_normal": "-0.59", "as_of_date": "2026-09-17"},
    {"community_area": "6", "metric": "crime_total", "window_days": "30", "window_count": "473",
     "normal_count": "516.5", "pct_vs_normal": "-8.42", "as_of_date": "2026-09-17"},
    {"community_area": "9", "metric": "crime_total", "window_days": "30", "window_count": "12",
     "normal_count": "8.0", "pct_vs_normal": "50.0", "as_of_date": "2026-09-17"},
    {"community_area": "14", "metric": "crime_total", "window_days": "60", "window_count": "400",
     "normal_count": "380.0", "pct_vs_normal": "5.3", "as_of_date": "2026-09-17"},
]
PROFILE = [
    {"community_area": a, "metric": "crime_violent", "population": pop, "annual_count": n, "rate_per_1k": r,
     "citywide_percentile": p, "citywide_rank": rank, "years": "3", "as_of_date": "2026-09-17"}
    for a, pop, n, r, p, rank in [("14", "47830", "727", "15.2", "29", "55"), ("68", "21337", "1946", "91.2", "99", "2"),
                                  ("6", "90652", "1360", "15.0", "26", "57"), ("9", "11525", "40", "3.5", "0", "77")]
]


def test_level_band_edges():
    assert level_band(99)["label"] == "among the highest" and level_band(99)["band"] == 5
    assert level_band(80)["band"] == 5
    assert level_band(79)["label"] == "above average"
    assert level_band(40)["label"] == "near the middle"
    assert level_band(0)["label"] == "among the lowest" and level_band(0)["band"] == 1
    assert level_band(None) is None


def test_shape_normals_casts_and_filters():
    out = ad.shape_normals(NORMALS, 30)
    assert set(out) == {14, 68, 6, 9}
    assert out[14]["crime_total"] == {"window_count": 210, "normal_count": 187.6, "pct_vs_normal": 11.9}
    assert ad.shape_normals(NORMALS, 60) == {14: {"crime_total": {"window_count": 400, "normal_count": 380.0, "pct_vs_normal": 5.3}}}
    assert set(ad.shape_normals(NORMALS, 30, community_area=68)) == {68}


def test_shape_profile_adds_band():
    p = ad.shape_profile(PROFILE)[68]["crime_violent"]
    assert p["rate_per_1k"] == 91.2 and p["citywide_rank"] == 2 and p["label"] == "among the highest"
    assert ad.shape_profile(PROFILE)[6]["crime_violent"]["label"] == "below average"


def test_rank_above_normal_skips_small_baselines():
    out = at._rank_context(NORMALS, PROFILE, "crime_total", 30, "above_normal", 10, 20)
    assert [a["community_area"] for a in out["areas"]] == [14, 68, 6]     # area 9's normal of 8 is too small
    assert out["excluded_low_baseline"] == 1
    assert out["areas"][0]["pct_vs_normal"] == 11.9 and out["areas"][0]["rank"] == 1
    below = at._rank_context(NORMALS, PROFILE, "crime_total", 30, "below_normal", 10, 20)
    assert below["areas"][0]["community_area"] == 6


def test_rank_usual_level():
    hi = at._rank_context(NORMALS, PROFILE, "crime_violent", 30, "highest_usual_level", 2, 20)
    assert [a["community_area"] for a in hi["areas"]] == [68, 14]
    assert hi["areas"][0]["usual_level"] == "among the highest"
    lo = at._rank_context(NORMALS, PROFILE, "crime_violent", 30, "lowest_usual_level", 1, 20)
    assert lo["areas"][0]["community_area"] == 9


def test_attach_context_merges_and_survives_a_missing_table(monkeypatch):
    base = lambda: {"ok": True, "community_area": 14, "window_days": 30,
                    "metrics": {"crime_total": {"history": [], "window_count": 210, "prior_window_count": 152}}}
    monkeypatch.setattr(ad, "normals_rows", lambda: NORMALS)
    monkeypatch.setattr(ad, "profile_rows", lambda: PROFILE)
    r = at.attach_context(base())
    assert r["metrics"]["crime_total"]["pct_vs_normal"] == 11.9
    assert r["usual_level"]["crime_violent"]["citywide_rank"] == 55
    assert "context_error" not in r

    def boom():
        raise RuntimeError("table not found")
    monkeypatch.setattr(ad, "profile_rows", boom)
    r = at.attach_context(base())
    assert r["metrics"]["crime_total"]["normal_count"] == 187.6       # the other half still lands
    assert r["usual_level"] == {} and "table not found" in r["context_error"]


def test_annotate_partial_month_uses_the_as_of_day():
    hist = [{"period": "2026-08-01", "count": 197}, {"period": "2026-09-01", "count": 110}]
    ad.annotate_partial_month(hist, date(2026, 9, 17))
    assert hist[-1]["days_observed"] == 17 and hist[-1]["projected_count"] == round(110 * 30 / 17)


# ---------------------------------------------------------------- Insights: weather + events


def test_weather_effects_in_plain_units():
    coefs = [{"feature": f, "per_unit": str(v)} for f, v in
             {"t_anom": 0.0096, "p_anom_cm": -0.0035, "dep_7d": 0.2393, "dep_28d": 0.4543, "month_start": 0.0181,
              "hol_in_recent_7d": -0.0061, "hol_christmas": -0.1017, "hol_halloween": 0.0337}.items()]
    e = ad.weather_effects(coefs)
    assert e["per_degree_f_pct"] == 0.5 and e["per_inch_rain_pct"] == -0.9     # 0.0096/°C x 5/9; -0.0035/cm x 2.54
    assert e["momentum_10pct_pct"] == 6.8          # both momentum windows 10% hot: exp(0.6936 * ln 1.1) - 1
    assert list(e["holidays"]) == ["halloween", "christmas"]      # biggest first, hol_in_recent_7d isn't a holiday
    assert e["holidays"]["christmas"] == -9.7


def test_shape_city_weeks_sorts_and_casts():
    rows = [{"cutoff": "2026-09-21", "y": None, "forecast": "4561.3", "bar_calibrated": "4861.8", "fc_t_anom": "-2.97", "is_live": "true"},
            {"cutoff": "2026-09-07", "y": "4672", "forecast": "4893.3", "bar_calibrated": "4826.1", "fc_t_anom": "2.74", "is_live": "false"}]
    w = ad.shape_city_weeks(rows)
    assert [x["week"] for x in w] == ["2026-09-07", "2026-09-21"]
    assert w[0] == {"week": "2026-09-07", "actual": 4672, "forecast": 4893, "normal": 4826, "temp_vs_normal_f": 4.9, "is_live": False}
    assert w[1]["actual"] is None and w[1]["is_live"] is True


def test_shape_forecast_days_in_imperial():
    rows = [  # all strings, °C and mm; the second day has no normal yet
        {"target_day": "2026-09-22", "issue_date": "2026-09-20", "t_mean": "20.0", "precip": "6.35",
         "t_normal": "17.5", "p_normal": "2.54"},
        {"target_day": "2026-09-21", "issue_date": "2026-09-20", "t_mean": "15.0", "precip": "0.0",
         "t_normal": None, "p_normal": None},
    ]
    out = ad.shape_forecast_days(rows)
    assert out["issued"] == "2026-09-20"
    mon, tue = out["days"]   # sorted by day
    assert (tue["weekday"], tue["date"]) == ("Tue", "Sep 22")
    assert (tue["temp_f"], tue["normal_f"], tue["temp_vs_normal_f"]) == (68, 64, 4)
    assert (tue["rain_in"], tue["normal_rain_in"]) == (0.25, 0.1)
    assert (mon["temp_f"], mon["normal_f"], mon["temp_vs_normal_f"], mon["rain_in"]) == (59, None, None, 0.0)
    assert ad.shape_forecast_days([]) is None


def test_forecast_days_sql_uses_the_day_before_the_week():
    sql = ad.forecast_days_sql(date(2026, 9, 21))
    assert "issue_date <= date_sub(DATE'2026-09-21', 1)" in sql
    assert "date_add(DATE'2026-09-21', 6)" in sql


def test_shape_event_lifts_orders_buckets():
    rows = [{"kind": "club_night", "bucket": "all", "n_past": "14065", "prior": "1.11"},
            {"kind": "festival", "bucket": "5+", "n_past": "271", "prior": "1.30"},
            {"kind": "festival", "bucket": "1", "n_past": "1864", "prior": "1.09"}]
    lifts = ad.shape_event_lifts(rows)
    assert [l["label"] for l in lifts] == ["Street festival, 1 block", "Street festival, 5+ blocks", "Club show night (6pm–3am)"]
    assert lifts[1]["lift_pct"] == 30.0 and lifts[2]["past_events"] == 14065


# ---------------------------------------------------------------- All of Chicago (area 0)


def test_shape_city_trend_sums_and_projects():
    monthly = [(date(2026, 8, 1), "crime_total", 20701), (date(2026, 9, 1), "crime_total", 11114)]
    rolling = [("crime_total", 19783, 20170, date(2026, 9, 17))]
    t = ad.shape_city_trend(monthly, rolling, 30)
    m = t["metrics"]["crime_total"]
    assert t["community_area"] == ad.CITY and t["area_name"] == "All of Chicago" and t["as_of_date"] == "2026-09-17"
    assert m["window_count"] == 19783 and m["pct_change_vs_prior_window"] == -1.9
    assert m["history"][-1]["projected_count"] == round(11114 * 30 / 17)


def test_shape_city_context_totals_and_extremes():
    c = ad.shape_city_context(NORMALS, PROFILE, 30)
    total = c["vs_normal"]["crime_total"]
    assert total["window_count"] == 210 + 385 + 473 + 12
    assert total["normal_count"] == round(187.6 + 387.3 + 516.5 + 8.0, 1)
    v = c["usual_level"]["crime_violent"]
    assert v["highest"]["area_name"] == "Englewood" and v["lowest"]["community_area"] == 9
    assert v["rate_per_1k"] == round((727 + 1946 + 1360 + 40) / (47830 + 21337 + 90652 + 11525) * 1000, 1)


# ---------------------------------------------------------------- map layers: city ratio, forecast turns


def test_city_ratio():
    snap = ad.with_city_ratio(ad.shape_profile(PROFILE, "crime_violent"), "crime_violent")
    city = (727 + 1946 + 1360 + 40) / (47830 + 21337 + 90652 + 11525) * 1000          # 23.8 per 1,000
    assert snap[68]["city_rate_per_1k"] == round(city, 1)
    assert snap[68]["ratio_to_city"] == round(91.2 / city, 2)                           # Englewood ~3.8x the city


def test_shape_forecast_map_flags_turns_only_on_confident_calls():
    rows = [{"community_area": "14", "call": "down", "p_above": "0.30", "forecast": "42", "normal": "45", "pct_vs_normal": "-0.067"},
            {"community_area": "68", "call": "unclear", "p_above": "0.40", "forecast": "80", "normal": "82", "pct_vs_normal": "-0.02"},
            {"community_area": "6", "call": "up", "p_above": "0.70", "forecast": "130", "normal": "118", "pct_vs_normal": "0.10"},
            {"community_area": "9", "call": "up", "p_above": "0.66", "forecast": "3", "normal": "2", "pct_vs_normal": "0.5"}]
    fc = ad.shape_forecast_map(rows, ad.shape_normals(NORMALS, 30, "crime_total"))
    assert fc[14]["turning"] == "easing"            # +11.9% lately, called down
    assert fc[68]["turning"] is None                 # about normal lately, no call
    assert fc[6]["turning"] == "worsening"           # -8.4% lately, called up
    assert fc[9]["turning"] is None                  # +50% lately and called up: a continuation
    assert fc[14]["pct_vs_normal"] == -6.7 and fc[14]["recent_pct_vs_normal"] == 11.9


def test_outlook_concern_marks_only_the_extremes():
    rows = [{"community_area": str(a), "call": "unclear", "p_above": str(a / 100), "forecast": "10", "normal": "10",
             "pct_vs_normal": "0"} for a in range(1, 31)]
    fc = ad.shape_forecast_map(rows, {})
    assert [a for a in fc if fc[a]["concern"] == "high"] == list(range(21, 31))    # the 10 most likely above normal
    assert [a for a in fc if fc[a]["concern"] == "low"] == list(range(1, 11))
    assert all(fc[a]["concern"] is None for a in range(11, 21))


def test_outlook_flips_dont_take_concern_slots():
    rows = [{"community_area": str(a), "call": "unclear", "p_above": str(a / 100), "forecast": "10", "normal": "10",
             "pct_vs_normal": "0"} for a in range(1, 31)]
    rows[-1]["call"] = "down"                                        # area 30: most likely above, but called down...
    normals = {30: {"crime_total": {"window_count": 20, "normal_count": 10, "pct_vs_normal": 100.0}}}   # ...after running hot
    fc = ad.shape_forecast_map(rows, normals)
    assert fc[30]["turning"] == "easing" and fc[30]["concern"] is None
    assert [a for a in fc if fc[a]["concern"] == "high"] == list(range(20, 30))                      # still 10 red


# ---------------------------------------------------------------- rolling 12-month change


def test_shape_rolling_change_needs_24_months_and_drops_the_partial_one():
    rows = [(date(2023 + (i // 12), i % 12 + 1, 1), 100 if i < 12 else 110) for i in range(25)]   # Jan 2023 .. Jan 2025
    r = ad.shape_rolling_change(rows[:24], date(2024, 12, 31))          # Dec 2024 complete: exactly one point
    assert len(r["points"]) == 1 and r["latest"] == {"month": "2024-12-01", "last_12": 1320, "prior_12": 1200, "pct": 10.0}
    r = ad.shape_rolling_change(rows, date(2025, 1, 17))                 # Jan 2025 is partial: dropped
    assert [p["month"] for p in r["points"]] == ["2024-12-01"]
    r = ad.shape_rolling_change(rows, "2025-01-31")                      # ...unless it ran to the last day
    assert r["latest"]["month"] == "2025-01-01" and r["latest"]["prior_12"] == 100 * 11 + 110
    assert ad.shape_rolling_change(rows[:10], None) == {"points": [], "latest": None}
