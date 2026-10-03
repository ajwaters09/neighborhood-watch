"""The whole outlook in one call: backtest every model, then forecast the live week with the
champions. Notebook 08 runs this nightly; the tutorials run it locally.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.outlook import evaluation as ev
from src.outlook.features import area_rows, city_rows, day_grid, weather_features, with_forecast_weather
from src.outlook.models import area_lean, call, dispersion, drivers, fit_area, fit_city, p_above
from src.outlook.specs import (AREA_MODELS, AREA_TRAIN_FROM, CHAMPION_AREA, CHAMPION_CITY, CITY_MODELS,
                               CITY_TRAIN_FROM, REPORT_LAG_DAYS, TEST_START, AreaModelSpec, CityModelSpec)

CITY_OUT = ["cutoff", "y", "bar", "dep_7d", "dep_28d", "fc_t_anom", "fc_p_anom_cm", "holidays"]
AREA_OUT = ["community_area", "cutoff", "fold", "y", "normal", "bar_city", "forecast", "p_above", "call", "above",
            "pct_citywide", "pct_local", "pct_vs_normal", "alpha"]


def live_cutoff(last_full: date, run_date: date, city: pd.DataFrame) -> date:
    """The newest Monday, on or before the run date, whose crime features are fully known
    (every day before c - L is in the data). With fresh data it's this week's Monday."""
    latest = min(last_full + timedelta(days=REPORT_LAG_DAYS + 1), run_date)
    ok = city[(city["cutoff"] <= latest) & city["bar"].notna()]
    if ok.empty:
        raise ValueError(f"no cutoff has its crime features known by {latest}")
    return ok["cutoff"].max()


def _city_forecast(city: pd.DataFrame, rows: pd.DataFrame, train_last: date, spec: CityModelSpec,
                   fold: int) -> tuple[pd.DataFrame, object]:
    """One fold's (or the live week's) citywide forecast and drivers for `rows`."""
    model, k = fit_city(city, train_last, spec)
    out = rows[CITY_OUT].assign(fold=fold, bar_calibrated=rows["bar"] * k,
                                forecast=rows["bar"] * model.multiplier(rows), k=k)
    return pd.concat([out, drivers(model, rows)], axis=1), model


def backtest_city(city: pd.DataFrame, test: pd.DataFrame, fold_list: list[dict], spec: CityModelSpec) -> pd.DataFrame:
    """Every test week's citywide forecast from `spec`, refit each fold."""
    parts = []
    for f in fold_list:
        rows = with_forecast_weather(test[test["cutoff"].between(f["test_first"], f["test_last"])])
        parts.append(_city_forecast(city, rows, f["train_last"], spec, f["fold"])[0])
    return pd.concat(parts, ignore_index=True)


def combine(areas: pd.DataFrame, city: pd.DataFrame, spec: AreaModelSpec) -> pd.DataFrame:
    """Put each area's lean on the citywide call.

    normal = the area's bar x the city bar's fitted level; bar_city = the area's bar x the full
    citywide call; forecast = the lean x that call (or just the normal, for a spec that doesn't
    use the city call).
    """
    m = city.set_index("cutoff")
    mult = areas["cutoff"].map(m["forecast"] / m["bar"])
    k = areas["cutoff"].map(m["k"])
    out = areas.assign(normal=areas["expected"] * k, bar_city=areas["expected"] * mult,
                       forecast=areas["lean_forecast"] * mult if spec.use_city_call else areas["expected"] * k)
    out["pct_citywide"] = mult / k - 1
    out["pct_local"] = out["forecast"] / out["bar_city"] - 1
    out["pct_vs_normal"] = out["forecast"] / out["normal"] - 1
    return out


def backtest_area(areas: pd.DataFrame, city_bt: pd.DataFrame, fold_list: list[dict], spec: AreaModelSpec) -> pd.DataFrame:
    """Every test area-week's forecast from `spec` on the champion citywide call, with P(above
    normal) and a call. Each fold's dispersion comes from earlier folds' misses only, so fold 0
    gets no probabilities."""
    weeks = set(city_bt["cutoff"])
    parts = []
    for f in fold_list:
        fold_weeks = {c for c in weeks if f["test_first"] <= c <= f["test_last"]}
        a_te = areas[areas["cutoff"].isin(fold_weeks) & areas["y"].notna()].copy()
        a_te["lean_forecast"] = area_lean(fit_area(areas, f["train_last"], spec), a_te)
        parts.append(a_te.assign(fold=f["fold"]))
    bt = combine(pd.concat(parts, ignore_index=True), city_bt, spec)

    probs = []
    for f in sorted(bt["fold"].unique())[1:]:
        cur, prev = bt[bt["fold"] == f].copy(), bt[bt["fold"] < f]
        cur["alpha"] = dispersion(prev["y"].to_numpy(), prev["forecast"].to_numpy())
        probs.append(cur)
    bt = pd.concat([bt[bt["fold"] == 0].assign(alpha=np.nan), *probs], ignore_index=True)
    bt["p_above"] = np.where(bt["alpha"].notna(),
                             p_above(bt["forecast"].to_numpy(), bt["normal"].to_numpy(), bt["alpha"].fillna(1.0).to_numpy()),
                             np.nan)
    bt["call"] = np.where(bt["p_above"].notna(), call(bt["p_above"].fillna(0.5).to_numpy()), None)
    bt["above"] = (bt["y"] > bt["normal"]).astype("boolean")
    return bt


def run(area_daily: pd.DataFrame, obs: pd.DataFrame, forecasts: pd.DataFrame, last_full: date, run_date: date, *,
        city_models: dict[str, CityModelSpec] | None = None, area_models: dict[str, AreaModelSpec] | None = None,
        champion_city: str = CHAMPION_CITY, champion_area: str = CHAMPION_AREA) -> dict[str, pd.DataFrame]:
    """Backtest every model, then forecast the live week with the champions.

    Args:
        area_daily: (community_area, day, n), crime counts per area per day, from silver.
        obs: (day, t_mean, precip), observed weather, contiguous.
        forecasts: (target_day, issue_date, model, t_mean, precip), every archived forecast.
        last_full: the newest complete day of crime data.
        run_date: today. The live week uses the newest forecast issued by then. Backtest weeks
            use the one issued the day before their cutoff, which is what the archive can replay.
        city_models / area_models: the specs to backtest (default: specs.CITY_MODELS /
            AREA_MODELS). Pass extra specs to try an idea against the champions.
        champion_city / champion_area: the specs whose forecasts are served.

    Returns:
        city_weeks: per test week plus the live week: the champion's forecast and its drivers.
        area_weeks: per area per test week plus the live week: forecast, P(above normal), call.
        summary: one row: the live week's dates and the champions' backtest scores.
        coefficients: the live citywide model's effects, per unit of each feature.
        leaderboard: every model's backtest scores, champions flagged.
        calibration: P(above normal) by bin against how often areas came in above.
    """
    city_models = city_models or CITY_MODELS
    area_models = area_models or AREA_MODELS
    city_spec, area_spec = city_models[champion_city], area_models[champion_area]

    grid, wide, n_real = day_grid(area_daily, last_full)
    horizon = last_full + timedelta(days=REPORT_LAG_DAYS + 1)
    city = city_rows(grid, wide, n_real, obs, forecasts, CITY_TRAIN_FROM, horizon)
    live = live_cutoff(last_full, run_date, city)
    fresh = weather_features(obs, forecasts, [live], {live: run_date})
    if fresh["fc_t_anom"].isna().all():
        raise ValueError(f"no weather forecast for the week of {live} issued by {run_date}; has 02b run today?")
    fc_cols = ["fc_t_anom", "fc_p_anom_cm", "fc_t_days", "fc_p_days"]
    city.loc[city["cutoff"] == live, fc_cols] = fresh[fc_cols].to_numpy()
    areas = area_rows(grid, wide, n_real, AREA_TRAIN_FROM, live)

    # --- walk-forward backtest over complete test weeks with a forecast ---------------------
    test = city[(city["cutoff"] >= TEST_START) & city["y"].notna() & city["fc_t_anom"].notna() & (city["cutoff"] < live)]
    fold_list = ev.folds(sorted(test["cutoff"]))
    city_bts = {name: backtest_city(city, test, fold_list, spec) for name, spec in city_models.items()}
    city_bt = city_bts[champion_city]
    area_bts = {name: backtest_area(areas, city_bt, fold_list, spec) for name, spec in area_models.items()}
    area_bt = area_bts[champion_area]

    # --- the live week ------------------------------------------------------------------------
    train_last = ev.train_last_for(live)
    city_live, model = _city_forecast(city, with_forecast_weather(city[city["cutoff"] == live]), train_last, city_spec, -1)
    a_live = areas[areas["cutoff"] == live].copy()
    a_live["lean_forecast"] = area_lean(fit_area(areas, train_last, area_spec), a_live)
    a_live = combine(a_live.assign(fold=-1), city_live, area_spec)
    scored = area_bt[area_bt["fold"] > 0]
    a_live["alpha"] = dispersion(scored["y"].to_numpy(), scored["forecast"].to_numpy())
    a_live["p_above"] = p_above(a_live["forecast"].to_numpy(), a_live["normal"].to_numpy(), a_live["alpha"].iloc[0])
    a_live["call"] = call(a_live["p_above"].to_numpy())
    a_live["above"] = (a_live["y"] > a_live["normal"]).astype("boolean").where(a_live["y"].notna(), pd.NA)

    city_weeks = pd.concat([city_bt.assign(is_live=False), city_live.assign(is_live=True)], ignore_index=True)
    city_weeks["pct_vs_normal"] = city_weeks["forecast"] / city_weeks["bar_calibrated"] - 1
    area_weeks = pd.concat([area_bt.assign(is_live=False), a_live.assign(is_live=True)], ignore_index=True)[AREA_OUT + ["is_live"]]
    coefs = model.per_unit().rename("per_unit").reset_index().rename(columns={"index": "feature"})
    coefs["pct_per_unit"] = np.expm1(coefs["per_unit"]) * 100
    return {
        "city_weeks": city_weeks, "area_weeks": area_weeks,
        "summary": ev.summary(city_bt, area_bt, live, last_full), "coefficients": coefs,
        "leaderboard": ev.leaderboard(city_bts, area_bts, city_models, area_models, champion_city, champion_area),
        "calibration": ev.calibration(area_bt),
    }
