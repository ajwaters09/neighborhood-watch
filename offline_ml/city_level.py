"""1-week track, step 1: how much of next week's citywide crime can be called ahead of time?

    offline_ml/.venv/bin/python offline_ml/weather.py       # first: weather + holidays
    offline_ml/.venv/bin/python offline_ml/city_level.py    # ~3 s

At a 1-week horizon, a large share of a community area's miss is the whole city running hot or
cold that week. This step models only that citywide level. Its output is a multiplier on the
bar, which the local (cell) model can later apply.

The setup, per cutoff Monday c. The target is all crime in [c, c+7). Crime is only usable from
days before c - REPORT_LAG_DAYS, matching the live feed's lag.
- **The bar:** the city's trailing-year daily rate x a daily seasonal factor, summed over the
  7 days. The factor is the median of the same-weekday ratio (day / its trailing-year rate) at
  52, 104 ... 260 weeks back, +-1 week. That's 15 values, so one odd year or one holiday in the
  references barely moves it. Day of week is built in, because every reference is the same
  weekday.
- **Momentum:** how far the last 7 and 28 *known* days ran from their own bar. With the lag,
  the newest known week ends 9 days before the cutoff.
- **Calendar:** one flag per holiday falling in the target week, plus the 1st of a month. The
  crime file stamps "sometime this month" records on the 1st, about 2% of a week's crime.
- **Weather:** the target week's temperature and precipitation anomalies against a 10-year
  normal, plus the observed temperature anomaly over the momentum window. That last one lets the
  model tell a hot week from a real upturn.

**Training on observed weather, testing on forecasts.** Every model learns the weather effect
from observed weather (ERA5, back to 2007). A test week gets the forecast that was issued the
day before its cutoff (lead 1-7), from Open-Meteo's forecast archive. The `full_observed` variant
swaps in the weather that actually happened, as an upper bound on what better forecasts could
buy.

Walk-forward: quarterly folds from TEST_START. A fold trains only on cutoffs whose target week
was already *known* at its first test cutoff, which is 3 weeks back with the lag. Training
starts in 2007 and skips COVID (cutoffs 2020-03-09 .. 2021-07-04), whose swings don't repeat.

The payoff check: each community area's weekly bar is scaled by the city multiplier and scored
at area x week, the grain the app would show. `oracle_city` scales by the actual citywide
outcome, which is the most any citywide model could win there.

Outputs (offline_data/model/h1/):
    city_predictions.parquet   per test cutoff: target, bar, every variant, features, fold
    city_summary.csv           citywide and area x week scores per variant
    city_marginal.csv          what each piece adds over the one before, with CIs
    city_coefficients.csv      final-fold coefficients as % effect per unit
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from backtest import poisson_deviance, skill_ci
from config import CLEAN, FOLD_WEEKS, REPORT_LAG_DAYS, SEASONAL_YEARS, TEST_START, WEATHER, WEEK_MODEL
from weather import daily_normals, issued_forecast

L = REPORT_LAG_DAYS
DAY0 = date(2001, 1, 1)                 # first day of the crime file (a Monday)
TRAIN_FROM = date(2007, 1, 8)           # the seasonal factor needs 1 + 5 years (+1 week) of history
COVID = (date(2020, 3, 9), date(2021, 7, 5))
TRAILING_DAYS = 364
MIN_FORECAST_DAYS = 3
SEASON_SHIFTS = [364 * k - 7 * j for k in range(1, SEASONAL_YEARS + 1) for j in (-1, 0, 1)]

MOMENTUM = ["dep_7d", "dep_28d", "hol_in_recent_7d"]
WEATHER_COLS = ["t_anom", "p_anom_cm", "t_anom_recent_7d"]
VARIANTS = {                            # name -> (feature groups, weather at test time)
    "bar_calibrated": ([], None),
    "momentum": ([MOMENTUM], None),
    "momentum_calendar": ([MOMENTUM, "calendar"], None),
    "full_forecast": ([MOMENTUM, "calendar", WEATHER_COLS], "forecast"),
    "full_observed": ([MOMENTUM, "calendar", WEATHER_COLS], "observed"),
    "no_momentum": (["calendar", WEATHER_COLS], "forecast"),
    "no_calendar": ([MOMENTUM, WEATHER_COLS], "forecast"),
}
REF = "bar_calibrated"                  # the bar with its level fitted: the fair reference
HEADLINE = "full_forecast"
STEPS = [                               # (piece, variant with it, variant before it)
    ("momentum", "momentum", REF),
    ("calendar", "momentum_calendar", "momentum"),
    ("weather forecast", "full_forecast", "momentum_calendar"),
    ("perfect weather instead", "full_observed", "full_forecast"),
]
# A light ridge on standardized features. Near-duplicate flags (Christmas Eve and Christmas share
# a week in 6 years out of 7) and flags not yet seen in training (Juneteenth before 2022) would
# otherwise make the fit singular. Well-identified effects shrink by ~0.1%.
GLM_ALPHA = 1e-3


# ---------------------------------------------------------------------------------------------
# Daily counts and the bar
# ---------------------------------------------------------------------------------------------

def load_daily() -> tuple[pl.DataFrame, pl.DataFrame, date]:
    """Contiguous daily counts citywide and per community area, through the last full day.

    `n` counts everything and feeds the target. `known` leaves out crimes whose case was opened
    in a later calendar year than they occurred (they weren't in the feed in time), and feeds the
    bar and momentum. It's the same rule panel.py uses.
    """
    crimes = pl.read_parquet(CLEAN / "crimes.parquet", columns=["occurred_at", "case_year", "community_area"])
    last_full = crimes["occurred_at"].max().date() - timedelta(days=1)    # the export's last day is partial
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    per_day = crimes.with_columns(pl.col("occurred_at").dt.date().alias("day"), known.alias("_k")).filter(
        pl.col("day").is_between(DAY0, last_full))
    days = pl.date_range(DAY0, last_full, "1d", eager=True).to_frame("day")
    count = lambda df, keys: df.group_by(keys).agg(pl.len().alias("n"), pl.col("_k").sum().alias("known"))
    city = days.join(count(per_day, "day"), on="day", how="left").fill_null(0)
    areas = per_day.filter(pl.col("community_area").is_not_null())
    area = (days.join(areas.select("community_area").unique(), how="cross")
            .join(count(areas, ["community_area", "day"]), on=["community_area", "day"], how="left")
            .fill_null(0).sort("community_area", "day"))
    return city, area, last_full


def daily_season(city: pl.DataFrame) -> pl.DataFrame:
    """day -> seasonal factor s (expected day count / trailing-year daily rate). Every input is
    at least 357 days old."""
    trailing = pl.col("known").rolling_mean(TRAILING_DAYS).shift(1)
    return (
        city.sort("day").with_columns((pl.col("known") / trailing).alias("_ratio"))
        .with_columns(pl.concat_list([pl.col("_ratio").shift(s) for s in SEASON_SHIFTS]).list.median().alias("s"))
        .select("day", "s")
    )


def lagged_frame(daily: pl.DataFrame, season: pl.DataFrame, over: str | None = None) -> pl.DataFrame:
    """Per day c: target, bar and momentum for a cutoff on c. Rows for non-Mondays are dropped
    later. `daily` must be contiguous per `over` group, so a shift of k rows is k days."""
    o = (lambda e: e.over(over)) if over else (lambda e: e)
    rate = pl.col("known").rolling_mean(TRAILING_DAYS).shift(1)      # mean over [d - 364, d)
    df = daily.join(season, on="day").sort(*([over] if over else []), "day").with_columns(o(rate).alias("_rate"))
    window = lambda col, n, shift: o(pl.col(col).rolling_sum(n).shift(shift))
    return df.with_columns(
        o(pl.col("n").rolling_sum(7).shift(-6)).alias("y"),                        # [c, c+7)
        (o(pl.col("_rate").shift(L)) * o(pl.col("s").rolling_sum(7).shift(-6))).alias("bar"),
        # momentum windows end at c - L. Each is judged against its own bar, anchored on the
        # trailing year before the window starts.
        *((window("known", n, L + 1) / (o(pl.col("_rate").shift(L + n)) * window("s", n, L + 1))).log().alias(name)
          for n, name in ((7, "dep_7d"), (28, "dep_28d"))),
    ).drop("_rate")


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

def weekly_weather() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per cutoff: observed and forecast anomalies for the target week, and the observed
    temperature anomaly over the momentum window [c - L - 7, c - L)."""
    obs = pl.read_parquet(WEATHER / "observed_daily.parquet")
    norm = daily_normals(obs)
    daily = obs.join(norm, on="day").sort("day").with_columns(
        (pl.col("t_mean") - pl.col("t_mean_normal")).alias("_ta"), (pl.col("precip") - pl.col("precip_normal")).alias("_pa"))
    observed = daily.select(
        pl.col("day").alias("cutoff"),
        pl.col("_ta").rolling_mean(7).shift(-6).alias("t_anom"),
        (pl.col("_pa").rolling_sum(7).shift(-6) / 10).alias("p_anom_cm"),
        pl.col("_ta").rolling_mean(7).shift(L + 1).alias("t_anom_recent_7d"),
    )
    # A cutoff's forecast: day c + i at lead i + 1, i.e. all issued the day before the cutoff.
    # Where the archive lacks a day, that day counts as normal weather, which is what a
    # forecaster without one would assume. Two test weeks (2026-04-13 and 04-20) lack 4 days in
    # both models. A week needs at least MIN_FORECAST_DAYS real days.
    fc = (
        issued_forecast(pl.read_parquet(WEATHER / "forecast_daily.parquet"))
        .with_columns((pl.col("day") - pl.duration(days=pl.col("lead").cast(pl.Int64) - 1)).cast(pl.Date).alias("cutoff"))
        .join(norm, on="day")
        .group_by("cutoff")
        .agg(((pl.col("t_mean") - pl.col("t_mean_normal")).sum() / 7).alias("t_anom"),
             ((pl.col("precip") - pl.col("precip_normal")).sum() / 10).alias("p_anom_cm"),
             pl.col("t_mean").is_not_null().sum().alias("_nt"), pl.col("precip").is_not_null().sum().alias("_np"))
        .filter((pl.col("_nt") >= MIN_FORECAST_DAYS) & (pl.col("_np") >= MIN_FORECAST_DAYS))
        .with_columns((pl.min_horizontal("_nt", "_np") < 7).alias("fc_partial"))
        .drop("_nt", "_np")
    )
    return observed, fc


def calendar(cutoffs: pl.Series) -> tuple[pl.DataFrame, list[str]]:
    """Holiday flags for the target week, the 1st-of-month flag, and holidays in the momentum
    window (a holiday there drags `dep_7d` for a reason that won't repeat)."""
    hol = pl.read_parquet(WEATHER / "holidays.parquet")
    names = sorted(hol["holiday"].unique())
    in_week = (hol.with_columns(pl.col("day").dt.truncate("1w").alias("cutoff"))
               .with_columns(pl.lit(1).alias("v")).pivot("holiday", index="cutoff", values="v", aggregate_function="max"))
    out = cutoffs.to_frame("cutoff").join(in_week, on="cutoff", how="left").with_columns(
        pl.col(names).fill_null(0).cast(pl.Float64).name.prefix("hol_"),
        ((pl.col("cutoff").dt.month() != (pl.col("cutoff") + pl.duration(days=6)).dt.month())
         | (pl.col("cutoff").dt.day() == 1)).cast(pl.Float64).alias("month_start"),
    ).select("cutoff", *(f"hol_{n}" for n in names), "month_start")
    hol_days = set(hol["day"].to_list())
    recent = [sum((c - timedelta(days=L + 1 + i)) in hol_days for i in range(7)) for c in cutoffs]
    out = out.with_columns(pl.Series("hol_in_recent_7d", recent, dtype=pl.Float64))
    return out, [f"hol_{n}" for n in names] + ["month_start"]


def build(city: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    season = daily_season(city)
    rows = lagged_frame(city, season).filter(pl.col("day").dt.weekday() == 1).rename({"day": "cutoff"})
    cal, cal_cols = calendar(rows["cutoff"])
    observed, fc = weekly_weather()
    rows = (rows.join(cal, on="cutoff")
            .join(observed, on="cutoff", how="left")
            .join(fc.rename({"t_anom": "fc_t_anom", "p_anom_cm": "fc_p_anom_cm"}), on="cutoff", how="left"))
    return rows.select("cutoff", "y", "bar", *MOMENTUM, *cal_cols, *WEATHER_COLS, "fc_t_anom", "fc_p_anom_cm", "fc_partial"), cal_cols


# ---------------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------------

def make_folds(test_cutoffs: list[date]) -> list[dict]:
    """Quarterly folds. A training target [c, c+7) must be in the feed by the fold's first test
    cutoff t, so c + 7 <= t - L, which is 3 Mondays back with L = 8."""
    folds = []
    for i in range(0, len(test_cutoffs), FOLD_WEEKS):
        block = test_cutoffs[i:i + FOLD_WEEKS]
        latest_ok = block[0] - timedelta(days=L + 7)
        train_last = latest_ok - timedelta(days=latest_ok.weekday())
        folds.append({"fold": len(folds), "train_last": train_last, "test_first": block[0], "test_last": block[-1]})
    return folds


def fit_predict(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Poisson GLM on y / bar with weight bar (the same as a log(bar) offset). Returns
    predictions and the coefficients per unit of each feature."""
    ytr, btr = train["y"].to_numpy(), train["bar"].to_numpy()
    if not cols:
        k = ytr.sum() / btr.sum()
        return test["bar"].to_numpy() * k, None
    x = train.select(cols).to_numpy()
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    model = PoissonRegressor(alpha=GLM_ALPHA, solver="newton-cholesky", max_iter=500)
    model.fit((x - mu) / sd, ytr / btr, sample_weight=btr)
    pred = test["bar"].to_numpy() * model.predict((test.select(cols).to_numpy() - mu) / sd)
    return pred, model.coef_ / sd


def effect(col: str, b: float) -> dict:
    """A coefficient as the % change in next week's crime for a readable step of its feature."""
    unit, step = {"dep_7d": ("recent 7 days ran 10% above their bar", np.log(1.1)),
                  "dep_28d": ("recent 28 days ran 10% above their bar", np.log(1.1)),
                  "t_anom": ("+1 C warmer than normal (target week)", 1.0),
                  "t_anom_recent_7d": ("+1 C warmer than normal (momentum window)", 1.0),
                  "p_anom_cm": ("+1 cm more rain than normal", 1.0),
                  "hol_in_recent_7d": ("one holiday in the momentum window", 1.0)}.get(
        col, ("the target week has it", 1.0))
    return {"step": unit, "pct_effect": 100 * float(np.expm1(b * step))}


def walk_forward(rows: pl.DataFrame, cal_cols: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    usable = rows.drop_nulls(["y", "bar", *MOMENTUM, *WEATHER_COLS]).filter(pl.col("cutoff") >= TRAIN_FROM)
    test = usable.filter(pl.col("cutoff") >= TEST_START)
    assert test["fc_t_anom"].null_count() == 0 and test["fc_p_anom_cm"].null_count() == 0, "forecast missing for a test week"
    train_pool = usable.filter(~pl.col("cutoff").is_between(*COVID, closed="left"))
    folds = make_folds(sorted(test["cutoff"].to_list()))

    parts, coefs = [], []
    for f in folds:
        tr = train_pool.filter(pl.col("cutoff") <= f["train_last"])
        te = test.filter(pl.col("cutoff").is_between(f["test_first"], f["test_last"]))
        assert tr["cutoff"].max() + timedelta(days=7 + L) <= te["cutoff"].min()
        out = te.select("cutoff", "y", "bar").with_columns(pl.lit(f["fold"]).alias("fold"))
        for name, (groups, test_weather) in VARIANTS.items():
            cols = [c for g in groups for c in (cal_cols if g == "calendar" else g)]
            te_x = te
            if test_weather == "forecast":
                te_x = te.with_columns(pl.col("fc_t_anom").alias("t_anom"), pl.col("fc_p_anom_cm").alias("p_anom_cm"))
            pred, coef = fit_predict(tr, te_x, cols)
            out = out.with_columns(pl.Series(name, pred))
            if f is folds[-1] and name == HEADLINE:
                coefs = [{"feature": c, "coef": float(b), **effect(c, float(b))} for c, b in zip(cols, coef)]
        parts.append(out)
    preds = pl.concat(parts).with_columns(pl.col("bar").alias("bar_raw")).join(
        test.select("cutoff", *MOMENTUM, *WEATHER_COLS, "fc_t_anom", "fc_p_anom_cm", "fc_partial"), on="cutoff")
    return preds, pl.DataFrame(coefs)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def area_rows(area: pl.DataFrame, city: pl.DataFrame, preds: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """Community area x test week. Each area's bar is its own trailing-year rate x the city's
    seasonal factor. Each city variant scales it by that variant's multiplier (variant / bar)."""
    season = daily_season(city)
    a = (lagged_frame(area, season, over="community_area")
         .filter(pl.col("day").dt.weekday() == 1).rename({"day": "cutoff"})
         .select("community_area", "cutoff", pl.col("y").alias("y_area"), pl.col("bar").alias("bar_area"))
         .join(preds.select("cutoff", "fold", "y", "bar", *models), on="cutoff"))
    return a.select(
        "community_area", "cutoff", "fold", pl.col("y_area").alias("y"),
        *((pl.col("bar_area") * pl.col(m) / pl.col("bar")).alias(m) for m in models),
        (pl.col("bar_area") * pl.col("y") / pl.col("bar")).alias("oracle_city"),
    )


def summarize(preds: pl.DataFrame, areas: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    out = []
    for grain, df, ms in (("city", preds, models), ("area_week", areas, [*models, "oracle_city"])):
        ref_dev = df.select(poisson_deviance("y", REF).mean()).item()
        for m in ms:
            skill, lo, hi = skill_ci(df, m, REF)
            row = {"grain": grain, "model": m, "deviance": df.select(poisson_deviance("y", m).mean()).item(),
                   "skill_vs_ref": skill, "ci_lo": lo, "ci_hi": hi}
            if grain == "city":
                dep = np.log(df["y"].to_numpy() / df[REF].to_numpy())
                miss = np.log(df["y"].to_numpy() / df[m].to_numpy())
                row |= {"mape": float(np.mean(np.abs(df[m].to_numpy() / df["y"].to_numpy() - 1))),
                        "swing_explained": 1 - np.var(miss) / np.var(dep)}
            out.append(row)
        # context: the bar's own error, and how much of it is pure chance at this grain
        if grain == "area_week":
            mu = df[REF].to_numpy()
            y_sim = np.random.default_rng(0).poisson(mu, size=(10, mu.size)).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                floor = float(np.mean(2 * (np.where(y_sim > 0, y_sim * np.log(y_sim / mu), 0) - (y_sim - mu))))
            out.append({"grain": grain, "model": "poisson_noise_floor", "deviance": floor,
                        "skill_vs_ref": 1 - floor / ref_dev})
    return pl.DataFrame(out)


def marginal(preds: pl.DataFrame, areas: pl.DataFrame) -> pl.DataFrame:
    """What each piece adds on top of the variant before it, as deviance skill with a 95% CI
    from resampling whole folds. It's paired: both sides are scored on the same weeks."""
    out = []
    for piece, with_it, before in STEPS:
        row = {"piece": piece, "vs": before}
        for grain, df in (("city", preds), ("area_week", areas)):
            skill, lo, hi = skill_ci(df, with_it, before)
            row |= {f"{grain}": skill, f"{grain}_lo": lo, f"{grain}_hi": hi}
        out.append(row)
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------------------------

def check_against_raw(city: pl.DataFrame, rows: pl.DataFrame) -> None:
    """Recompute target, bar level and 7-day momentum for a few cutoffs with explicit date
    bounds, straight from the daily counts."""
    season = daily_season(city)
    s = dict(season.iter_rows())
    n = dict(city.select("day", "n").iter_rows())
    k = dict(city.select("day", "known").iter_rows())
    span = lambda d, a, b, src: sum(src[d + timedelta(days=i)] for i in range(a, b))
    for c, y, bar, dep7 in rows.filter(pl.col("cutoff") >= TEST_START).sample(5, seed=3).select(
            "cutoff", "y", "bar", "dep_7d").iter_rows():
        assert y == span(c, 0, 7, n)
        rate = span(c, -L - TRAILING_DAYS, -L, k) / TRAILING_DAYS
        assert np.isclose(bar, rate * span(c, 0, 7, s)), (c, bar)
        rate_m = span(c, -L - 7 - TRAILING_DAYS, -L - 7, k) / TRAILING_DAYS
        assert np.isclose(dep7, np.log(span(c, -L - 7, -L, k) / (rate_m * span(c, -L - 7, -L, s)))), (c, dep7)


def check_future_deleted(city: pl.DataFrame, rows: pl.DataFrame, cut: date = date(2024, 6, 3)) -> None:
    """Delete all crime from `cut` on and rebuild. Every crime-side input for a cutoff c only
    uses days before c - L, so cutoffs up to cut + L must come out identical."""
    city_cut = city.filter(pl.col("day") < cut)
    again = lagged_frame(city_cut, daily_season(city_cut)).rename({"day": "cutoff"}).filter(
        pl.col("cutoff") <= cut + timedelta(days=L))
    cols = ["dep_7d", "dep_28d"]
    a = rows.select("cutoff", *cols).join(again.select("cutoff", *cols), on="cutoff", suffix="_cut")
    # The bar's level must match too; its seasonal factor needs the target week's own days but
    # only from 357+ days earlier, so only the target's own count (y) may differ.
    bars = rows.select("cutoff", "bar").join(again.select("cutoff", pl.col("bar").alias("bar_cut")), on="cutoff")
    assert a.height > 1000 and bars.drop_nulls().height > 1000
    for c in cols:
        assert a.filter(~pl.col(c).is_close(pl.col(f"{c}_cut")) & pl.col(c).is_not_null()).height == 0, c
    assert bars.drop_nulls().filter(~pl.col("bar").is_close(pl.col("bar_cut"))).height == 0, "bar"


# ---------------------------------------------------------------------------------------------

def main() -> None:
    WEEK_MODEL.mkdir(parents=True, exist_ok=True)
    city, area, last_full = load_daily()
    rows, cal_cols = build(city)
    check_against_raw(city, rows)
    check_future_deleted(city, rows)

    preds, coefs = walk_forward(rows, cal_cols)
    models = ["bar_raw", *VARIANTS]
    areas = area_rows(area, city, preds, models)
    summary = summarize(preds, areas, models)
    steps = marginal(preds, areas)

    preds.write_parquet(WEEK_MODEL / "city_predictions.parquet")
    summary.write_csv(WEEK_MODEL / "city_summary.csv")
    steps.write_csv(WEEK_MODEL / "city_marginal.csv")
    coefs.write_csv(WEEK_MODEL / "city_coefficients.csv")

    by_year = (preds.group_by(pl.col("cutoff").dt.year().alias("year"))
               .agg(*(poisson_deviance("y", m).sum().alias(m) for m in (REF, "momentum_calendar", HEADLINE, "full_observed")))
               .with_columns((1 - pl.col(m) / pl.col(REF)).alias(m) for m in ("momentum_calendar", HEADLINE, "full_observed"))
               .drop(REF).sort("year"))
    print(f"data through {last_full}; test cutoffs {preds['cutoff'].min()} .. {preds['cutoff'].max()} "
          f"({preds.height} weeks, {preds['fold'].n_unique()} folds, {preds['fc_partial'].sum()} with a partial forecast); "
          f"leak checks passed")
    print(f"citywide weekly crime: mean {preds['y'].mean():,.0f}, sd of log departure from the calibrated bar "
          f"{np.log(preds['y'] / preds[REF]).std():.3f}\n")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True):
        print(summary)
        print("\nwhat each piece adds (deviance skill over the variant before it, 95% CI by fold):")
        print(steps)
        print("\ncitywide skill vs calibrated bar, by year:")
        print(by_year)
        print(f"\n{HEADLINE}, final-fold coefficients (% change in next week's crime):")
        print(coefs.sort(pl.col("pct_effect").abs(), descending=True).select("feature", "step", "pct_effect"))


if __name__ == "__main__":
    main()
