"""Paths and modeling constants shared by the offline_ml steps."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "offline_data" / "clean"     # preprocess.py output
PANEL = ROOT / "offline_data" / "panel"     # panel.py output
MODEL = ROOT / "offline_data" / "model"     # backtest outputs
EVENTS = ROOT / "offline_data" / "events"   # events_fetch.py (raw/) and events_build.py output

CELL_RES = 8            # model grain: ~0.7 km^2 cells, ~5 crimes per cell-week
REPORT_RES = 7          # coarser grain the early-warning calls are also judged at
HORIZON_WEEKS = 4       # forecast the next 4 weeks of crime from each weekly cutoff

# Weekly counts start early enough that the seasonal factor can look back five years
# from the first training cutoff. Weeks start on Monday.
GRID_START = date(2014, 1, 6)

# The cells to model are chosen from this window only, so the choice can't peek at the
# test period. 12/yr (about one crime a month) keeps 770 cells holding 99.7% of 2022+ crime.
UNIVERSE_WINDOW = (date(2019, 3, 4), date(2022, 1, 3))
MIN_CRIMES_PER_YEAR = 12

FIRST_CUTOFF = date(2020, 3, 2)   # 311 starts 2019-03; its features need a year of history
TEST_START = date(2022, 1, 3)     # cutoffs before this are training-only
FOLD_WEEKS = 13                   # refit every quarter as the test period walks forward
SEASONAL_YEARS = 5                # the seasonal factor is a median over this many prior years

# --- 1-week track (weather.py, city_level.py) ---------------------------------------------------
WEATHER = ROOT / "offline_data" / "weather"   # weather.py output
WEEK_MODEL = MODEL / "h1"                     # 1-week track outputs, kept apart from the 4-week ones

# The live crime feed runs about 8 days behind. On 2026-09-26 its newest complete day was 09-17,
# with 09-18 part-loaded. So a forecast made at cutoff Monday c only sees crime from days before
# c - REPORT_LAG_DAYS, and the week just before the cutoff is never known in time.
REPORT_LAG_DAYS = 8
# Open-Meteo's archive of past forecasts, so the 2022+ test period is scored on forecasts that
# were really issued. Over 2022+ weeks (weather.py prints this), GFS tracks the weekly
# temperature anomaly far better (r = 0.94 vs JMA's 0.76), but its archived precipitation only
# starts in 2024. JMA's goes back to 2021 and is the better of the two anyway (0.44 vs 0.39).
# JMA also fills the few days GFS temperature is missing.
TEMP_FORECAST_MODEL = "gfs_seamless"
PRECIP_FORECAST_MODEL = "jma_seamless"

# Event history starts with the weekly grid. CDOT's current permit system also starts in
# 2014, and pulling club setlists back further would spend setlist.fm quota for nothing.
EVENTS_START = date(2014, 1, 1)
