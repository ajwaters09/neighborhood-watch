"""Model rows for the outlook: one per cutoff Monday (citywide) or per area and cutoff.

Every feature at cutoff c uses crime from before c - REPORT_LAG_DAYS only, matching what the
live feed holds, and weather forecasts issued before c. tests/test_outlook_model.py checks both.

To add a citywide feature: compute it in a builder below (or a new one added to CITY_BUILDERS),
then list it in a FeatureGroup in specs.py. To add an area feature: add a formula to
AREA_FEATURES and name it in an AreaModelSpec.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.constants import NORMAL_HALF_WINDOW, NORMAL_YEARS
from src.open_meteo import PRECIP_MODEL, TEMP_MODEL
from src.outlook.specs import (FEATURE_GROUPS, HOLIDAY_NAMES, MIN_FORECAST_DAYS, PAD_DAYS, REPORT_LAG_DAYS,
                               SEASON_SHIFTS, TRAILING_DAYS)

# ---------------------------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of the month; n = -1 is the last one."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l + 90) // 2530
    month = (h + l - 7 * m + 90) // 25
    return date(year, month, (h + l - 7 * m + 33 * month + 19) % 32)


def holidays(first_year: int, last_year: int) -> pd.DataFrame:
    """Federal holidays on their actual dates (not the observed Friday/Monday), plus Easter,
    Halloween, Christmas Eve and New Year's Eve."""
    rows = []
    for y in range(first_year, last_year + 1):
        days = {
            "new_years_day": date(y, 1, 1), "mlk_day": _nth_weekday(y, 1, 0, 3),
            "presidents_day": _nth_weekday(y, 2, 0, 3), "easter": _easter(y),
            "memorial_day": _nth_weekday(y, 5, 0, -1), "independence_day": date(y, 7, 4),
            "labor_day": _nth_weekday(y, 9, 0, 1), "columbus_day": _nth_weekday(y, 10, 0, 2),
            "halloween": date(y, 10, 31), "veterans_day": date(y, 11, 11),
            "thanksgiving": _nth_weekday(y, 11, 3, 4), "christmas_eve": date(y, 12, 24),
            "christmas": date(y, 12, 25), "new_years_eve": date(y, 12, 31),
        }
        if y >= 2021:                       # a federal holiday since 2021
            days["juneteenth"] = date(y, 6, 19)
        rows += [{"day": d, "holiday": name} for name, d in days.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Window sums over a contiguous day grid
# ---------------------------------------------------------------------------------------------

def _csum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])


def _wsum(cs: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Sum over grid days [start, end) for each pair; NaN where the window leaves the grid."""
    ok = (start >= 0) & (end <= len(cs) - 1)
    s, e = np.clip(start, 0, len(cs) - 1), np.clip(end, 0, len(cs) - 1)
    out = cs[e] - cs[s]
    return np.where(ok.reshape((-1,) + (1,) * (out.ndim - 1)), out, np.nan)


def daily_season(city: np.ndarray) -> np.ndarray:
    """Per grid day, expected count / trailing-year daily rate: the median of the same
    weekday's ratio at 52, 104 ... 260 weeks back, +-1 week (15 values, so one odd year or a
    holiday in the references barely moves it). Every input is at least 357 days old."""
    t = len(city)
    cs = _csum(city.astype(float))
    idx = np.arange(t)
    rate = _wsum(cs, idx - TRAILING_DAYS, idx) / TRAILING_DAYS
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = city / rate
    stack = np.full((len(SEASON_SHIFTS), t), np.nan)
    for k, sh in enumerate(SEASON_SHIFTS):
        stack[k, sh:] = ratio[:t - sh]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns before 5 years of history
        return np.nanmedian(stack, axis=0)


def unit_windows(counts: np.ndarray, season: np.ndarray, ci: np.ndarray, n_real: int) -> dict[str, np.ndarray]:
    """For cutoff grid indices `ci` and every unit (column of `counts`): the target, the bar
    and the momentum windows. Arrays are (len(ci), units).

    - y: all crime in [c, c+7); NaN where that week runs past the data (`n_real` grid days)
    - bar: the unit's rate over [c-L-364, c-L) x the season summed over the target week
    - dep_Nd: log(crime in [c-L-N, c-L) / that window's own bar)
    - c_28d / c_364d: raw counts in the windows ending c-L
    """
    L = REPORT_LAG_DAYS
    cs = _csum(counts.astype(float))
    # The season is NaN for the first 5 years (no references yet). A plain cumulative sum would
    # carry that NaN forward forever, so sum with NaNs zeroed and mark windows that touch one.
    css, cns = _csum(np.nan_to_num(season)), _csum(np.isnan(season).astype(float))
    col = lambda a: a[:, None]
    rate_at = lambda j: _wsum(cs, j - TRAILING_DAYS, j) / TRAILING_DAYS
    s_sum = lambda a, b: col(np.where(_wsum(cns, a, b) == 0, _wsum(css, a, b), np.nan))
    y = _wsum(cs, ci, ci + 7)
    y[ci + 7 > n_real] = np.nan
    out = {"y": y, "bar": rate_at(ci - L) * s_sum(ci, ci + 7),
           "c_28d": _wsum(cs, ci - L - 28, ci - L), "c_364d": _wsum(cs, ci - L - TRAILING_DAYS, ci - L)}
    with np.errstate(divide="ignore", invalid="ignore"):
        for n in (7, 28):
            out[f"dep_{n}d"] = np.log(_wsum(cs, ci - L - n, ci - L) / (rate_at(ci - L - n) * s_sum(ci - L - n, ci - L)))
    return out


# ---------------------------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------------------------

def daily_normals(obs: pd.DataFrame, until: date) -> pd.DataFrame:
    """Each day's normal temperature and precipitation, through `until`: the mean over the
    same date +-7 days in each of the previous 10 years (observed days only, so a normal never
    includes its own day). Days whose windows aren't fully observed get NaN."""
    obs = obs.sort_values("day")
    grid = pd.date_range(obs["day"].min(), max(pd.Timestamp(until), pd.Timestamp(obs["day"].max())), freq="D")
    o = obs.set_index(pd.to_datetime(obs["day"]))[["t_mean", "precip"]].reindex(grid)
    if o.loc[: pd.Timestamp(obs["day"].max()), "t_mean"].isna().any():
        raise ValueError("observed weather has gaps; 02b re-pulls a trailing window, so check its runs")
    out = {"day": grid.date}
    win = 2 * NORMAL_HALF_WINDOW + 1
    idx = np.arange(len(grid))
    for col, name in (("t_mean", "t_normal"), ("precip", "p_normal")):
        x = o[col].to_numpy()
        cs, cn = _csum(np.nan_to_num(x)), _csum((~np.isnan(x)).astype(float))
        total, ok = np.zeros(len(grid)), np.ones(len(grid), dtype=bool)
        for k in range(1, NORMAL_YEARS + 1):
            c = idx - round(365.25 * k)
            lo, hi = c - NORMAL_HALF_WINDOW, c + NORMAL_HALF_WINDOW + 1
            s, n = _wsum(cs, lo, hi), _wsum(cn, lo, hi)
            ok &= n == win
            total += np.nan_to_num(s) / win
        out[name] = np.where(ok, total / NORMAL_YEARS, np.nan)
    return pd.DataFrame(out)


def issued_forecast(forecasts: pd.DataFrame, cutoffs: list[date], limits: dict[date, date] | None = None) -> pd.DataFrame:
    """Per cutoff c: the forecast for each day of [c, c+7) as it stood the day before the
    cutoff (the newest issue_date <= c - 1), or as of `limits[c]` where given.

    Temperature is TEMP_MODEL's, falling back to PRECIP_MODEL's where it's missing.
    Precipitation is PRECIP_MODEL's. Returns long rows (cutoff, target_day, t_mean, precip),
    with NaN where no forecast existed.
    """
    limits = limits or {}
    left = pd.DataFrame([(c, c + timedelta(days=i), limits.get(c, c - timedelta(days=1))) for c in cutoffs for i in range(7)],
                        columns=["cutoff", "target_day", "limit"])
    left["limit"] = pd.to_datetime(left["limit"])
    left = left.sort_values("limit")

    def latest(model: str, col: str) -> pd.Series:
        r = forecasts.loc[(forecasts["model"] == model) & forecasts[col].notna(), ["target_day", "issue_date", col]].copy()
        r["issue_date"] = pd.to_datetime(r["issue_date"])
        m = pd.merge_asof(left, r.sort_values("issue_date"), left_on="limit", right_on="issue_date",
                          by="target_day", direction="backward")
        return m.set_index(["cutoff", "target_day"])[col]

    t = latest(TEMP_MODEL, "t_mean").combine_first(latest(PRECIP_MODEL, "t_mean"))
    p = latest(PRECIP_MODEL, "precip")
    return pd.DataFrame({"t_mean": t, "precip": p}).reset_index()


def weather_features(obs: pd.DataFrame, forecasts: pd.DataFrame, cutoffs: list[date],
                     limits: dict[date, date] | None = None) -> pd.DataFrame:
    """Per cutoff c:
    - observed t_anom / p_anom_cm over the target week (training only)
    - fc_t_anom / fc_p_anom_cm: the same from the forecast issued the day before (what test
      and live weeks use). Missing forecast days count as normal weather; a week needs
      MIN_FORECAST_DAYS of each.
    - t_anom_recent_7d: the observed temperature anomaly over [c-L-7, c-L)
    - snow_recent_7d: observed snowfall (cm) over the same days; 0 when `obs` has no `snow`
    """
    L = REPORT_LAG_DAYS
    norms = daily_normals(obs, max(cutoffs) + timedelta(days=7))
    daily = norms.merge(obs[["day", "t_mean", "precip"]], on="day", how="left")
    daily["ta"], daily["pa"] = daily["t_mean"] - daily["t_normal"], daily["precip"] - daily["p_normal"]
    daily["snow"] = daily["day"].map(obs.set_index("day")["snow"]) if "snow" in obs else 0.0
    by_day = daily.set_index("day")

    def span(c: date, a: int, b: int, col: str) -> np.ndarray:
        return by_day[col].reindex([c + timedelta(days=i) for i in range(a, b)]).to_numpy()

    rows = []
    for c in cutoffs:
        ta, pa, rec = span(c, 0, 7, "ta"), span(c, 0, 7, "pa"), span(c, -L - 7, -L, "ta")
        snow = span(c, -L - 7, -L, "snow").astype(float)
        rows.append({"cutoff": c, "t_anom": ta.mean() if not np.isnan(ta).any() else np.nan,
                     "p_anom_cm": pa.sum() / 10 if not np.isnan(pa).any() else np.nan,
                     "t_anom_recent_7d": rec.mean() if not np.isnan(rec).any() else np.nan,
                     "snow_recent_7d": snow.sum() if not np.isnan(snow).any() else np.nan})
    out = pd.DataFrame(rows)

    fc = issued_forecast(forecasts, cutoffs, limits).merge(norms, left_on="target_day", right_on="day", how="left")
    fc["ta"], fc["pa"] = fc["t_mean"] - fc["t_normal"], fc["precip"] - fc["p_normal"]
    agg = fc.groupby("cutoff").agg(fc_t_sum=("ta", "sum"), fc_t_days=("ta", "count"),
                                   fc_p_sum=("pa", "sum"), fc_p_days=("pa", "count")).reset_index()
    enough = (agg["fc_t_days"] >= MIN_FORECAST_DAYS) & (agg["fc_p_days"] >= MIN_FORECAST_DAYS)
    agg["fc_t_anom"] = np.where(enough, agg["fc_t_sum"] / 7, np.nan)
    agg["fc_p_anom_cm"] = np.where(enough, agg["fc_p_sum"] / 10, np.nan)
    return out.merge(agg[["cutoff", "fc_t_anom", "fc_p_anom_cm", "fc_t_days", "fc_p_days"]], on="cutoff", how="left")


# ---------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------

def day_grid(area_daily: pd.DataFrame, last_full: date) -> tuple[pd.DatetimeIndex, pd.DataFrame, int]:
    """Dense day x area counts from the first day through `last_full`, padded PAD_DAYS past it
    with zeros (only the live week's bar reads the pad, via the season, never its counts)."""
    real = pd.date_range(area_daily["day"].min(), last_full, freq="D")
    grid = pd.date_range(real[0], real[-1] + timedelta(days=PAD_DAYS), freq="D")
    wide = (area_daily.pivot_table(index="day", columns="community_area", values="n", aggfunc="sum")
            .reindex(columns=range(1, 78)))
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(grid).fillna(0.0)
    return grid, wide, len(real)


def mondays(grid: pd.DatetimeIndex, first: date, last: date) -> np.ndarray:
    d = grid.date
    return np.flatnonzero((grid.weekday == 0) & (d >= first) & (d <= last))


@dataclass
class CityContext:
    """What a citywide feature builder can read, for the cutoffs being built."""

    cutoffs: list[date]
    windows: dict[str, np.ndarray]      # unit_windows on the citywide series, one column
    obs: pd.DataFrame
    forecasts: pd.DataFrame


def _momentum_columns(ctx: CityContext) -> pd.DataFrame:
    return pd.DataFrame({"cutoff": ctx.cutoffs, "dep_7d": ctx.windows["dep_7d"][:, 0],
                         "dep_28d": ctx.windows["dep_28d"][:, 0]})


def _calendar_columns(ctx: CityContext) -> pd.DataFrame:
    """Holiday flags for the target week, the 1st of the month, holidays in the momentum
    window, and `holidays`, the week's holiday names for display."""
    L, cutoffs = REPORT_LAG_DAYS, ctx.cutoffs
    rows = pd.DataFrame({"cutoff": cutoffs})
    hol = holidays(cutoffs[0].year - 1, cutoffs[-1].year + 1)
    hol["week"] = [d - timedelta(days=d.weekday()) for d in hol["day"]]
    for h in HOLIDAY_NAMES:
        weeks = set(hol.loc[hol["holiday"] == h, "week"])
        rows[f"hol_{h}"] = [float(c in weeks) for c in cutoffs]
    rows["month_start"] = [float(c.month != (c + timedelta(days=6)).month or c.day == 1) for c in cutoffs]
    hol_days = set(hol["day"])
    rows["hol_in_recent_7d"] = [float(sum((c - timedelta(days=L + 1 + i)) in hol_days for i in range(7))) for c in cutoffs]
    rows["holidays"] = [", ".join(h for h in HOLIDAY_NAMES if r[f"hol_{h}"]) for _, r in rows.iterrows()]
    return rows


def _weather_columns(ctx: CityContext) -> pd.DataFrame:
    return weather_features(ctx.obs, ctx.forecasts, ctx.cutoffs)


CITY_BUILDERS: list[Callable[[CityContext], pd.DataFrame]] = [_momentum_columns, _calendar_columns, _weather_columns]


def city_rows(grid, wide, n_real, obs, forecasts, first: date, last: date) -> pd.DataFrame:
    """One row per cutoff Monday in [first, last]: the target y, the bar, and every builder's
    columns."""
    city = wide.sum(axis=1).to_numpy()
    season = daily_season(city)
    ci = mondays(grid, first, last)
    w = unit_windows(city[:, None], season, ci, n_real)
    cutoffs = [d for d in grid.date[ci]]
    rows = pd.DataFrame({"cutoff": cutoffs, "y": w["y"][:, 0], "bar": w["bar"][:, 0]})
    ctx = CityContext(cutoffs, w, obs, forecasts)
    for build in CITY_BUILDERS:
        rows = rows.merge(build(ctx), on="cutoff", how="left")
    missing = {c for g in FEATURE_GROUPS.values() for c in g.columns} - set(rows.columns)
    if missing:
        raise ValueError(f"feature groups name columns no builder makes: {sorted(missing)}")
    return rows


def with_forecast_weather(df: pd.DataFrame) -> pd.DataFrame:
    """The rows as they look at forecast time: every at_forecast_time swap applied."""
    swaps = {col: df[fc] for g in FEATURE_GROUPS.values() for col, fc in g.at_forecast_time.items()}
    return df.assign(**swaps)


# Area features, from the area's own unit_windows. Each is a log ratio, so 0 means "on trend".
AREA_FEATURES: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    # The last 28 known days against the trailing-year pace, +1 smoothed for small areas.
    "c_accel_28d": lambda w: np.log((w["c_28d"] + 1) / (w["c_364d"] * 28 / TRAILING_DAYS + 1)),
}


def area_rows(grid, wide, n_real, first: date, last: date) -> pd.DataFrame:
    """One row per (area, cutoff Monday): the target y, the area's bar (`expected`), and every
    AREA_FEATURES column. Areas share the citywide season."""
    city = wide.sum(axis=1).to_numpy()
    season = daily_season(city)
    ci = mondays(grid, first, last)
    w = unit_windows(wide.to_numpy(), season, ci, n_real)
    areas = wide.columns.to_numpy()
    out = pd.DataFrame({
        "community_area": np.tile(areas, len(ci)), "cutoff": np.repeat(grid.date[ci], len(areas)),
        "y": w["y"].ravel(), "expected": w["bar"].ravel(),
    })
    for name, formula in AREA_FEATURES.items():
        out[name] = formula(w).ravel()
    return out
