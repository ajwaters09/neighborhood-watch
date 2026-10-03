"""The outlook's knobs: data constants, feature groups, model specs, and which models ship.

Everything a model experiment changes lives here as data. The pipeline backtests every spec in
CITY_MODELS and AREA_MODELS on the same walk-forward folds, writes their scores to the
leaderboard, and serves the CHAMPION_* ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# --- Data -------------------------------------------------------------------------------------

REPORT_LAG_DAYS = 8          # the live crime feed's newest complete day is ~9 days back
TRAILING_DAYS = 364          # an area's "rate" is its daily mean over this many days
SEASONAL_YEARS = 5           # the daily seasonal factor looks back this many years
SEASON_SHIFTS = [364 * k - 7 * j for k in range(1, SEASONAL_YEARS + 1) for j in (-1, 0, 1)]
PAD_DAYS = 21                # the day grid runs past the data so the live week's bar can be built
MIN_FORECAST_DAYS = 3        # a week needs this many forecast days; missing ones count as normal

# --- Backtest ---------------------------------------------------------------------------------

CITY_TRAIN_FROM = date(2007, 1, 8)    # the seasonal factor needs 1 + 5 years (+1 week) of history
AREA_TRAIN_FROM = date(2020, 3, 2)
COVID = (date(2020, 3, 9), date(2021, 7, 5))   # skipped in citywide training: its swings don't repeat
TEST_START = date(2022, 1, 3)
FOLD_WEEKS = 13              # refit every quarter as the test period walks forward

# --- Calls ------------------------------------------------------------------------------------

UP, DOWN = 0.65, 0.35        # P(above normal) at or past these is an "up" / "down" call

# --- Features ---------------------------------------------------------------------------------

HOLIDAY_NAMES = [
    "new_years_day", "mlk_day", "presidents_day", "easter", "memorial_day", "juneteenth", "independence_day",
    "labor_day", "columbus_day", "halloween", "veterans_day", "thanksgiving", "christmas_eve", "christmas",
    "new_years_eve",
]


@dataclass(frozen=True)
class FeatureGroup:
    """A set of citywide features that enter a model together and get one driver in the output.

    columns: the model inputs, in fit order. features.city_rows must produce each of them.
    required: columns that must be known for a week to be used in training. Every model trains
        on the same weeks, so the union over all groups applies to all of them.
    at_forecast_time: column -> the column that replaces it when forecasting. Weather trains on
        what was observed and forecasts from what the forecast said.
    """

    name: str
    columns: tuple[str, ...]
    required: tuple[str, ...] = ()
    at_forecast_time: dict[str, str] = field(default_factory=dict)
    description: str = ""


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "momentum": FeatureGroup(
        "momentum", ("dep_7d", "dep_28d", "hol_in_recent_7d"), required=("dep_7d", "dep_28d"),
        description="The city's last 7 and 28 known days against their own seasonal bar, and holidays in "
                    "the last 7 (which depress them)."),
    "calendar": FeatureGroup(
        "calendar", tuple(f"hol_{h}" for h in HOLIDAY_NAMES) + ("month_start",),
        description="A flag per holiday in the target week, and whether it includes the 1st of a month."),
    "weather": FeatureGroup(
        "weather", ("t_anom", "p_anom_cm", "t_anom_recent_7d", "snow_recent_7d"),
        required=("t_anom", "p_anom_cm", "t_anom_recent_7d"),
        at_forecast_time={"t_anom": "fc_t_anom", "p_anom_cm": "fc_p_anom_cm"},
        description="The target week's temperature and rain against normal (observed in training, "
                    "forecast when forecasting), and the last 7 known days' temperature against normal "
                    "and snowfall, so a lull the weather caused isn't carried forward as a trend."),
}

# --- Models -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CityModelSpec:
    """A citywide model: the seasonal bar times a Poisson GLM on `groups` (no groups: the
    calibrated bar alone). `alpha` is the ridge penalty on standardized features."""

    name: str
    groups: tuple[str, ...]
    alpha: float = 1e-3
    description: str = ""


@dataclass(frozen=True)
class AreaModelSpec:
    """An area model. With `use_city_call`, the forecast is the area's bar x the champion
    citywide call x a lean fit on `features` (no features: no lean). Without it, the forecast is
    the area's normal: the bar every other area model is scored against."""

    name: str
    features: tuple[str, ...] = ()
    use_city_call: bool = True
    alpha: float = 1e-4
    description: str = ""


# A light ridge for the city GLM. Christmas Eve and Christmas share a week 6 years in 7, and
# Juneteenth has no training weeks before 2022; unpenalized, those make the fit singular.
CITY_MODELS: dict[str, CityModelSpec] = {
    "seasonal_bar": CityModelSpec(
        "seasonal_bar", (),
        description="Trailing-year rate x a daily seasonal factor, scaled to its fitted level. The baseline."),
    "momentum": CityModelSpec("momentum", ("momentum",), description="The bar x recent momentum."),
    "momentum_calendar": CityModelSpec(
        "momentum_calendar", ("momentum", "calendar"), description="Adds holidays and the 1st of the month."),
    "full": CityModelSpec(
        "full", ("momentum", "calendar", "weather"), description="Adds the week's weather forecast."),
}

AREA_MODELS: dict[str, AreaModelSpec] = {
    "normal": AreaModelSpec(
        "normal", use_city_call=False,
        description="The area's own bar at the city's fitted level: what 'normal' means."),
    "citywide_only": AreaModelSpec(
        "citywide_only", description="The area's bar x the citywide call, with no local lean."),
    "momentum_lean": AreaModelSpec(
        "momentum_lean", ("c_accel_28d",),
        description="Adds the area's last 28 known days against its trailing year."),
}

CHAMPION_CITY = "full"
CHAMPION_AREA = "momentum_lean"
