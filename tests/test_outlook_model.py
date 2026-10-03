"""Offline checks for the next-week outlook (src/outlook/) and src/open_meteo.py on a synthetic
city.

The synthetic city has 77 areas with their own levels and a shared yearly cycle. Crime runs
TEMP_EFFECT (log per degree C) above normal in warm weeks. Weather has an observed record back
to 1996 and a forecast archive from 2021 (observed + noise, like the real archive). So a working
pipeline should:
- recover the planted temperature effect, and find none when it's switched off,
- beat the calibrated bar citywide, and rank its models in the leaderboard accordingly,
- never let a feature see crime from after cutoff - REPORT_LAG_DAYS, or a forecast issued on
  or after the cutoff,
- take a new model spec without code changes,
- parse Open-Meteo responses and reject malformed ones.

Run: python -m pytest tests/test_outlook_model.py (requirements-dev.txt).
"""
import functools
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import open_meteo as om
from src import outlook as m

TEMP_EFFECT = 0.02
LAST_FULL = date(2025, 6, 13)


def synthetic_city(effect=TEMP_EFFECT, seed=0):
    rng = np.random.default_rng(seed)
    wx_days = pd.date_range("1996-01-01", LAST_FULL + timedelta(days=5), freq="D")
    doy = wx_days.dayofyear.to_numpy()
    anom = np.zeros(len(wx_days))
    for i in range(1, len(wx_days)):
        anom[i] = 0.8 * anom[i - 1] + rng.normal(0, 1.8)
    t = 10 - 13 * np.cos(2 * np.pi * (doy - 15) / 365.25) + anom
    precip = rng.gamma(0.4, 6, len(wx_days))
    obs = pd.DataFrame({"day": wx_days.date, "t_mean": t, "precip": precip})

    fc_rows = []
    for d, tt, pp in zip(wx_days.date, t, precip):
        if d < date(2021, 1, 1):
            continue
        for lead in range(1, 8):
            noise = rng.normal(0, 0.4 * np.sqrt(lead))
            for model in om.FORECAST_MODELS:
                fc_rows.append({"target_day": d, "issue_date": d - timedelta(days=lead), "model": model,
                                "t_mean": tt + noise, "precip": max(pp + rng.normal(0, 2), 0.0)})
    forecasts = pd.DataFrame(fc_rows)

    crime_days = pd.date_range("2001-01-01", LAST_FULL, freq="D")
    season = 1 + 0.2 * np.sin(2 * np.pi * (crime_days.dayofyear.to_numpy() - 100) / 365.25)
    weekday = np.where(crime_days.weekday >= 5, 1.1, 0.96)
    anom_c = pd.Series(anom, index=wx_days).reindex(crime_days).to_numpy()
    rate = 60 * season * weekday * np.exp(effect * anom_c)
    base = rng.lognormal(0, 0.6, 77)
    base /= base.sum()
    counts = rng.poisson(rate[:, None] * base[None, :])
    area_daily = pd.DataFrame({
        "community_area": np.tile(np.arange(1, 78), len(crime_days)),
        "day": np.repeat(crime_days.date, 77), "n": counts.ravel(),
    })
    return area_daily[area_daily["n"] > 0], obs, forecasts


@functools.cache
def _run(effect=TEMP_EFFECT):
    return m.run(*synthetic_city(effect=effect), LAST_FULL, LAST_FULL + timedelta(days=9))


def test_recovers_planted_weather_effect_and_beats_the_bar():
    out = _run()
    coefs = out["coefficients"].set_index("feature")["per_unit"]
    assert abs(coefs["t_anom"] - TEMP_EFFECT) < 0.006, coefs["t_anom"]
    s = out["summary"].iloc[0]
    assert s["city_skill"] > 0.1, s["city_skill"]
    assert s["test_weeks"] > 150
    live = out["area_weeks"][out["area_weeks"]["is_live"]]
    assert len(live) == 77 and live["p_above"].between(0, 1).all()
    assert set(live["call"]) <= {"up", "down", "unclear"}


def test_no_weather_effect_when_there_is_none():
    out = _run(0.0)
    coefs = out["coefficients"].set_index("feature")["per_unit"]
    assert abs(coefs["t_anom"]) < 0.006, coefs["t_anom"]


def test_leaderboard_and_calibration():
    out = _run()
    lb = out["leaderboard"].set_index(["level", "model"])
    s = out["summary"].iloc[0]
    # the champions' rows are the summary's numbers
    assert np.isclose(lb.loc[("city", m.CHAMPION_CITY), "skill"], s["city_skill"])
    assert np.isclose(lb.loc[("area", m.CHAMPION_AREA), "skill"], s["area_skill"])
    assert np.isclose(lb.loc[("area", m.CHAMPION_AREA), "brier_skill"], s["brier_skill"])
    assert lb.loc[("city", "seasonal_bar"), "skill"] == 0.0          # the reference scores itself 0
    # the planted effect is weather: the full model beats the ones without it
    assert lb.loc[("city", "full"), "skill"] > lb.loc[("city", "momentum_calendar"), "skill"] + 0.1
    cal = out["calibration"]
    assert cal["n"].sum() == s["scored_area_weeks"]
    assert ((cal["mean_p"] >= cal["bin_lo"]) & (cal["mean_p"] <= cal["bin_hi"])).all()
    assert (abs(cal["observed"] - cal["mean_p"]) < 0.1)[cal["n"] > 500].all()


def test_a_new_spec_is_backtested_beside_the_champions():
    extra = {"weather_only": m.CityModelSpec("weather_only", ("weather",))}
    out = m.run(*synthetic_city(), LAST_FULL, LAST_FULL + timedelta(days=9), city_models={**m.CITY_MODELS, **extra})
    lb = out["leaderboard"].set_index(["level", "model"])
    assert ("city", "weather_only") in lb.index and not lb.loc[("city", "weather_only"), "champion"]
    # adding a challenger doesn't change what's served
    pd.testing.assert_frame_equal(out["area_weeks"], _run()["area_weeks"])


def test_features_ignore_crime_after_the_lag():
    area_daily, obs, forecasts = synthetic_city()
    cutoff = date(2024, 6, 17)
    cut_day = cutoff - timedelta(days=m.REPORT_LAG_DAYS)          # the first day a forecast at cutoff can't see
    full = _rows_at(area_daily, obs, forecasts, cutoff, LAST_FULL)
    trunc = _rows_at(area_daily[area_daily["day"] < cut_day], obs, forecasts, cutoff, cut_day - timedelta(days=1))
    for col in ("bar", "dep_7d", "dep_28d"):
        assert np.isclose(full[0][col], trunc[0][col]), col
    for col in ("expected", *m.AREA_FEATURES):
        assert np.allclose(full[1][col], trunc[1][col]), col
    later = _rows_at(area_daily, obs, forecasts, cutoff + timedelta(weeks=1), LAST_FULL)
    later_trunc = _rows_at(area_daily[area_daily["day"] < cut_day], obs, forecasts, cutoff + timedelta(weeks=1),
                           cut_day - timedelta(days=1))
    assert not np.isclose(later[0]["dep_7d"], later_trunc[0]["dep_7d"])     # not vacuous


def _rows_at(area_daily, obs, forecasts, cutoff, last_full):
    grid, wide, n_real = m.day_grid(area_daily, last_full)
    city = m.city_rows(grid, wide, n_real, obs, forecasts, cutoff, cutoff).iloc[0]
    areas = m.area_rows(grid, wide, n_real, cutoff, cutoff).sort_values("community_area")
    return city, areas


def test_recent_snow_is_the_known_week_before_the_lag():
    c = date(2024, 2, 5)
    days = pd.date_range("2010-01-01", "2024-03-01").date
    obs = pd.DataFrame({"day": days, "t_mean": 0.0, "precip": 0.0, "snow": 0.0})
    L = m.REPORT_LAG_DAYS
    obs.loc[obs["day"] == c - timedelta(days=L + 1), "snow"] = 5.0        # the newest known day: counts
    obs.loc[obs["day"] == c - timedelta(days=L), "snow"] = 100.0          # not known yet at the cutoff
    obs.loc[obs["day"] == c - timedelta(days=L + 8), "snow"] = 100.0      # before the window
    got = m.weather_features(obs, pd.DataFrame(columns=["target_day", "issue_date", "model", "t_mean", "precip"]), [c])
    assert got["snow_recent_7d"].iloc[0] == 5.0
    no_snow = m.weather_features(obs.drop(columns="snow"), pd.DataFrame(columns=["target_day", "issue_date", "model",
                                                                                  "t_mean", "precip"]), [c])
    assert no_snow["snow_recent_7d"].iloc[0] == 0.0


def test_forecast_uses_only_issues_before_the_cutoff():
    c = date(2024, 6, 3)
    fc = pd.DataFrame([
        {"target_day": c, "issue_date": c - timedelta(days=1), "model": om.TEMP_MODEL, "t_mean": 10.0, "precip": None},
        {"target_day": c, "issue_date": c, "model": om.TEMP_MODEL, "t_mean": 99.0, "precip": None},
        {"target_day": c, "issue_date": c - timedelta(days=3), "model": om.PRECIP_MODEL, "t_mean": 12.0, "precip": 5.0},
        {"target_day": c + timedelta(days=1), "issue_date": c - timedelta(days=2), "model": om.PRECIP_MODEL,
         "t_mean": 13.0, "precip": 1.0},
    ])
    got = m.issued_forecast(fc, [c]).set_index("target_day")
    assert got.loc[c, "t_mean"] == 10.0                           # the same-day issue (99) is ignored
    assert got.loc[c, "precip"] == 5.0
    assert got.loc[c + timedelta(days=1), "t_mean"] == 13.0       # GFS missing -> JMA fills temperature
    assert np.isnan(got.loc[c + timedelta(days=2), "t_mean"])
    live = m.issued_forecast(fc, [c], {c: c}).set_index("target_day")
    assert live.loc[c, "t_mean"] == 99.0                          # the live week may use today's issue


def test_training_weeks_were_known_by_the_test_week():
    for test_first in (date(2024, 1, 1), date(2024, 1, 8), date(2025, 3, 3)):
        tl = m.train_last_for(test_first)
        assert tl.weekday() == 0
        assert tl + timedelta(days=7 + m.REPORT_LAG_DAYS) <= test_first
        assert tl + timedelta(days=14 + m.REPORT_LAG_DAYS) > test_first     # and it's the newest such week


def test_holidays_and_calls():
    h = m.holidays(2023, 2026).set_index(["holiday"]).groupby(level=0)["day"].apply(set)
    assert date(2025, 11, 27) in h["thanksgiving"] and date(2024, 3, 31) in h["easter"]
    assert date(2026, 5, 25) in h["memorial_day"] and date(2023, 9, 4) in h["labor_day"]
    mu = np.array([80.0, 100.0, 120.0])
    p = m.p_above(mu, np.full(3, 100.0), 0.01)
    assert (np.diff(p) > 0).all()
    assert list(m.call(np.array([0.7, 0.5, 0.3]))) == ["up", "unclear", "down"]


def test_open_meteo_parsers():
    obs = om.parse_observed({"daily": {"time": ["2024-01-01", "2024-01-02"], "temperature_2m_mean": [1.0, None],
                                       "temperature_2m_max": [3.0, None], "precipitation_sum": [0.5, None],
                                       "snowfall_sum": [0.0, None]}})
    assert len(obs) == 1 and obs[0]["day"] == date(2024, 1, 1)          # not-yet-published days are dropped
    fc = om.parse_forecast({"daily": {"time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                                      "temperature_2m_mean": [1.0, 2.0, None], "precipitation_sum": [0.0, 1.0, None]}},
                           "gfs_seamless", date(2024, 1, 2))
    assert [r["lead_days"] for r in fc] == [0]                           # past days and past-horizon nulls dropped
    hourly = {"time": [f"2024-01-01T{h:02d}:00" for h in range(24)] + ["2024-01-02T00:00"]}
    for n in om.PREVIOUS_RUN_LEADS:
        hourly[f"temperature_2m_previous_day{n}"] = [float(n)] * 25
        hourly[f"precipitation_previous_day{n}"] = [0.1] * 25
    pr = om.parse_previous_runs({"hourly": hourly}, "jma_seamless")
    assert len(pr) == 7 and {r["target_day"] for r in pr} == {date(2024, 1, 1)}     # the 1-hour day is dropped
    assert {(r["lead_days"], r["issue_date"]) for r in pr} == {(n, date(2024, 1, 1) - timedelta(days=n)) for n in range(1, 8)}
    for bad in ({}, {"daily": {"time": ["2024-01-01"]}}):
        try:
            om.parse_observed(bad)
        except ValueError:
            continue
        raise AssertionError(f"malformed response accepted: {bad}")
