"""Next-week crime outlook for Chicago's 77 community areas: will each area come in above or
below its normal for this time of year?

The model (`run`, in pipeline.py):
- **Citywide:** next week's total = the city's seasonal bar x a Poisson GLM on top.
  - The bar is the trailing-year daily rate x a daily seasonal factor: the median of the same
    weekday's ratio 1-5 years back, +-1 week.
  - The GLM's features come in groups (specs.FEATURE_GROUPS): momentum (the last 7 and 28 known
    days against their own bar), calendar (holiday weeks, the 1st of the month) and weather (the
    target week's forecast temperature and rain against normal).
- **Areas:** each area's own bar x the citywide call x a one-feature lean: its last 28 known
  days against its trailing year. Training pins each week's expected counts to the actual
  citywide total, so the lean is purely local.
- **Calls:** P(next week > the area's normal) from a negative binomial around the forecast, its
  dispersion fit on earlier folds' misses. Up at P >= 0.65, down at P <= 0.35, else unclear.

**The reporting lag is built in.** Crime is only usable from days before cutoff -
REPORT_LAG_DAYS, as in the live feed, and a training week must have been in the feed by the
test week's cutoff too. Backtest weeks use the weather forecast issued the day before.

**Every run backtests every model** in specs.CITY_MODELS and specs.AREA_MODELS on the same
quarterly walk-forward folds (2022 on), so the scores the app quotes are always current, and the
leaderboard shows what each piece of the model adds.

To try an idea:
1. Add a feature: compute it in features.py and list it in a FeatureGroup (citywide) or
   AREA_FEATURES (per area).
2. Add a model that uses it: a CityModelSpec or AreaModelSpec in specs.py, or pass extra specs
   to `run(..., city_models=...)` without touching the defaults.
3. Compare: the leaderboard scores it beside the champions on the same weeks.
4. Promote it: point CHAMPION_CITY / CHAMPION_AREA at it.

Pure numpy, pandas, scipy and scikit-learn, so it runs anywhere: tests/test_outlook_model.py
checks it on a synthetic city, and tutorials/05 runs it on real data. report.py draws the
backtest, the calibration and the leaderboard.
"""

from src.outlook.evaluation import calibration, deviance, folds, leaderboard, train_last_for
from src.outlook.features import (AREA_FEATURES, area_rows, city_rows, day_grid, daily_normals, holidays,
                                  issued_forecast, weather_features)
from src.outlook.models import OffsetGLM, call, dispersion, p_above
from src.outlook.pipeline import run
from src.outlook.specs import (AREA_MODELS, CHAMPION_AREA, CHAMPION_CITY, CITY_MODELS, DOWN, FEATURE_GROUPS,
                               REPORT_LAG_DAYS, UP, AreaModelSpec, CityModelSpec, FeatureGroup)

__all__ = [
    "AREA_FEATURES", "AREA_MODELS", "AreaModelSpec", "CHAMPION_AREA", "CHAMPION_CITY", "CITY_MODELS", "CityModelSpec",
    "DOWN", "FEATURE_GROUPS", "FeatureGroup", "OffsetGLM", "REPORT_LAG_DAYS", "UP", "area_rows", "calibration", "call",
    "city_rows", "daily_normals", "day_grid", "deviance", "dispersion", "folds", "holidays", "issued_forecast",
    "leaderboard", "p_above", "run", "train_last_for", "weather_features",
]
