# Research sandbox

Where the app's models were chosen: local experiments on the full raw crime and 311 exports,
with walk-forward backtests, leakage checks and placebos, before anything was built on
Databricks. It starts from the raw files and its own cleaning rules rather than the pipeline's,
and works on H3 cells as well as community areas.

Three tracks, each with a walkthrough notebook that tells the story with charts from the saved
outputs:

| Track | Question | Answer | What shipped |
|---|---|---|---|
| [4 weeks, H3 cells](#4-week-track-does-311-help-predict-crime) ([walkthrough](walkthrough.ipynb)) | Does 311 improve a crime forecast? | No: recent crime already carries whatever 311 knows | 311 became a descriptive lead-lag signal (`notebooks/05`) |
| [1 week, community areas](#1-week-track) ([walkthrough](week_walkthrough.ipynb)) | Will an area run above its normal next week? | Yes, mostly through the citywide call: weather forecast, momentum, holidays | the next-week outlook (`src/outlook/`, `notebooks/08`) |
| [Events](#events) ([walkthrough](events_walkthrough.ipynb)) | Do festivals and club shows bring crime nearby? | Festivals +31% in their own cells, club nights +11%: small in absolute terms | the events look-ahead (`src/events_model.py`, `notebooks/08b`) |

## Setup and run

```bash
python3 -m venv offline_ml/.venv
offline_ml/.venv/bin/pip install -r offline_ml/requirements.txt
offline_ml/.venv/bin/python offline_ml/preprocess.py     # ~15 s on an M-series laptop
```

Inputs go in `offline_data/`, which is gitignored:
- `crimes_bulk_export.csv`: the portal's full crimes export.
- `part-*.parquet`: the 311 export, with all-string columns.

Outputs land in `offline_data/clean/`:

| File | Rows | What |
|---|---|---|
| `crimes.parquet` | 8,638,566 | One row per crime record, 2001-01-01 to 2026-09-12, sorted by `occurred_at` |
| `sr311.parquet` | 6,836,544 | One row per place-based 311 request, 2019-03-01 to 2026-09-22, sorted by `created_at` |
| `sr_type_catalog.csv` | 108 | Per-type volume, first/last seen, duplicate/cancel/resident shares, median days to close |
| `cleaning_log.json` | | Row counts at every drop or fix step |

Load them with `pl.read_parquet(...)` or `pd.read_parquet(...)`. Timestamps are naive
Chicago local time.

## H3

- `h3_r9` is the base cell: ~0.1 km², ~174 m edge. The crime file has 5,380 distinct r9 cells.
  - Crime coordinates are snapped to the block (a median 16 m spread within a block), so a
    finer resolution would just be slicing up noise.
- `h3_r8` (~0.7 km², 907 cells) and `h3_r7` (~5 km², 158 cells) are precomputed parents.
- Cells are stored as `Int64`, the same encoding as Databricks' `h3_longlatash3`, so
  joins carry over.
  - `h3.int_to_str(x)` gives the familiar hex string.
- `community_area` is kept as given, for a future rollup. It isn't a clean container for
  cells, though: only 70% of r8 cells have ≥90% of their records in a single community area.

## Crime cleaning

| Issue found | Handling |
|---|---|
| 149 rows geocoded to Missouri (lat 36.6) | Coordinates nulled, then filled like other missing ones |
| 100,195 rows with no coordinates (1.2%), concentrated in fraud, theft and narcotics | Filled with the median point of other crimes on the same `block` (needs ≥3 geocoded crimes on that block): 95,963 filled, `geo_source = 'block_median'`. The remaining 4,232 are dropped |
| (check on the fill) | 50k held-out geocoded rows filled this way land a median 14 m from their real point (p95 96 m). 91% fall in the same r9 cell |
| 613k rows with no community area, mostly 2001–02 | Kept. H3 doesn't need it. Community area `0` is set to null |
| Timestamps at 00:00/00:01 are the "time unknown" convention (268k rows) | `time_unknown` flag. Don't trust hour-of-day on these rows |
| The 1st of the month at 00:00/00:01 has ~2× a normal day's count (34k rows) | `date_is_month_placeholder` flag. The real date is "sometime that month" |
| Some crimes are reported long after they occur | The case-number prefix encodes the year the case was opened (G\* = 2001, HH = 2002 … JK = 2026), learned from the data. `report_lag_min_days` is a **lower bound** on reporting lag. 126,602 rows (1.5%) have one > 0, with a median of 110 days |
| Homicides use a separate ID range (< 100k), one record per victim | `is_homicide_victim_record`. Multi-victim incidents share a `case_number` |
| `CRIM SEXUAL ASSAULT` renamed `CRIMINAL SEXUAL ASSAULT` in 2019 | Merged into the new name |
| 218 location descriptions, many spelling variants of the same place | Normalized to 143, plus a 12-value `location_group` (street, residential, commercial, parking, school, vehicle, transit, institutional, park_open_space, vacant, airport, other/unknown) |
| No standard severity grouping | `ucr_class` from the FBI code: violent (01A, 02, 03, 04A, 04B), property (05, 06, 07, 09), other |

## 311 cleaning

| Issue found | Handling |
|---|---|
| `311 INFORMATION ONLY CALL` (5.16M) is geocoded to the 311 call center. `Aircraft Noise Complaint` (2.59M) sits on a single O'Hare address | Both types dropped. They make up 53% of raw rows and aren't place-based events |
| Jan–Feb 2019 are a partial migration: Jan has ~30% and Feb ~65% of later years' volume | Trimmed to `created_at >= 2019-03-01` (58k rows) |
| Real request types occasionally parked on those same two placeholder addresses | Dropped (990 rows) |
| No coordinates | Dropped (13k rows) |
| Socrata's own `duplicate` flag (repeat reports of the same issue) | **Kept** as `is_duplicate`, with `parent_sr_number`. Repeat reporting may itself be signal, so leave that choice to the model |
| `origin` has 46 raw values | `origin_group`: resident (84%), city_internal (10%, e.g. crews' mass entry of graffiti), elected_official (7%). Resident reports and city sweeps probably mean different things |
| 9 rows closed before they were created | `closed_at` nulled. `days_to_close` is null for open requests |
| Some types start, stop, or are seasonal | See `sr_type_catalog.csv`. `looks_discontinued` means nothing was seen in the final year |

## 4-week track: does 311 help predict crime?

The question: **is a cell about to run above its own normal?** Forecast all crime (enforcement-driven
types included) for the next 4 weeks, per r8 cell and weekly cutoff. The headline experiment is
whether 311 improves that beyond crime's own history.

```bash
offline_ml/.venv/bin/python offline_ml/panel.py      # step 1: cells + weekly counts   -> offline_data/panel/
offline_ml/.venv/bin/python offline_ml/backtest.py   # step 2: baselines + scoring     -> offline_data/model/
offline_ml/.venv/bin/python offline_ml/features.py   # step 3: features + leak checks  -> offline_data/model/features.parquet
offline_ml/.venv/bin/python offline_ml/models.py     # step 4: crime-only models (~2 min) -> offline_data/model/step4_*
offline_ml/.venv/bin/python offline_ml/ablation.py   # step 5: the 311 experiment (~14 min) -> offline_data/model/step5_*
offline_ml/.venv/bin/python offline_ml/early_warning.py  # step 6: calls + live forecast  -> offline_data/model/step6_*, live_forecast_*
```

**Start with [`walkthrough.ipynb`](walkthrough.ipynb).** It tells the whole story with charts, and
it runs top to bottom in about 5 seconds from the saved outputs. It even runs the leakage
checks live. Open it in VS Code or Jupyter with the `offline_ml/.venv` interpreter.

The tree model is scikit-learn's `HistGradientBoostingRegressor`, not LightGBM. LightGBM's macOS
wheel needs Homebrew's `libomp`, which isn't installed. `brew install libomp` would let
LightGBM drop in.

Shared constants (grain, horizon, test start, fold length) live in `config.py`.

**Step 1, `panel.py`:**
- **Cells:** 770 r8 cells averaging at least 12 crimes a year, chosen using 2019-03 → 2021 data
  only. They hold 99.7% of crime since 2022.
- **Weekly grid:** every cell × every Monday-start week from 2014-01 to 2026-08-31. The last,
  partial export week is dropped. Weeks with no crime are explicit zeros.
- **`known_*` counts:** these leave out crimes whose case was opened in a later calendar year
  than they occurred (1.8%). Those reports weren't known in time to be used, so features and
  baselines use `known_*` while the target uses everything.
- **311:** counts per cell × week × type, split into resident-reported and duplicate
  requests, stored sparse.

**Step 2, `backtest.py`:**
- **Rows:** one per (cell, cutoff Monday). History comes strictly from before the cutoff, and
  the target is all crime in the 4 weeks from it.
- **Test period:** cutoffs from 2022-01-03 to 2026-08-10, split into 19 quarterly folds. A
  fold may only train on cutoffs whose target window ends before the fold starts.
- **Baselines:** the bar is `seasonal_trailing` — the cell's last 52 weeks, scaled to 4 weeks,
  × a citywide seasonal factor (the median of the same calendar position's ratio over the 5
  prior years).

| r8, 2022–26 test | Poisson deviance | AUC "above normal" | top-5% capture |
|---|---|---|---|
| **seasonal_trailing (bar)** | **1.81** | 0.50 (by construction) | 19% |
| trailing (no season) | 1.88 | 0.50 | 19% |
| last 4 weeks | 2.72 | 0.56 | 19% |
| same 4 weeks last year | 3.50 | 0.50 | 18% |
| Poisson noise floor | 1.03 | | |

What the table says:
- **"Where" is already solved by the baseline.** Every baseline puts about 19% of crime in the
  top 5% of cells.
- **The bar is a real bar.** It beats both raw momentum and last year on deviance.
- **There is room above it.** The noise floor suggests up to about 43% of the bar's deviance
  could be won at most (an upper bound, since crime is burstier than Poisson).
  - Getting the citywide total exactly right would win 7.7%.
  - Blending in 20% momentum already wins 3.4%.
- **At r7, momentum ranks above-normal areas with AUC 0.60,** up from 0.56 at r8, as the noise
  argument predicted.

**Step 3, `features.py`:** 261,800 rows (770 cells × 340 weekly cutoffs, 2020-03 → 2026-08).

- **Crime-side features (30):**
  - the cell's crime over the last 1, 4, 13 and 52 weeks, plus the year before that
  - the same 4 weeks last year
  - violent and property splits
  - acceleration ratios (the recent rate vs the 52-week rate)
  - the cell's mix of residential, street, commercial and domestic crime
  - ring-1 and ring-2 neighbors' counts and acceleration
  - citywide acceleration, the seasonal factor, and week of year
- **311 features (66):**
  - totals over 1, 4, 13 and 52 weeks, with acceleration
  - the resident and duplicate shares
  - the **backlog open at the cutoff**, rebuilt from timestamps, since `status` describes today
  - the backlog open 4+ weeks
  - the 13-week average days to close
  - a 4-week count and an acceleration ratio for each of 27 types (the top 25 by pre-2022
    volume, plus vacant building and vacant lot)
  - ring-1 neighbors' 311
- `FEATURE_GROUPS` maps column prefixes to crime vs 311 for the ablation.
- **Leakage checks run on every build:**
  1. Recompute 5 features for random rows straight from the event tables.
  2. Delete everything from 2024-06-03 on, rebuild, and require all 97 features to be
     identical for every earlier row.
- **First look, training rows only, comparing cells within the same week:**
  - Crime momentum tracks next-month departures from expected best: the cell's own r = 0.16,
    neighbors' 0.14.
  - Last year's rise predicts a partial fall back (r = −0.12).
  - The best 311 features are much weaker: resident share 0.06, open backlog 0.04.
  - The step-5 ablation decides whether 311 adds anything the crime features don't already
    carry.

**Step 4, `models.py`: the bar sets the citywide total, the models spread it across cells.**

- **Training:** each cell's expected count is pinned to the week's actual citywide total, so a
  model learns only which cells run hot or cold relative to the rest of the city.
- **Forecast:** expected × m(x), rescaled back to the bar's citywide total.
- **Local features only:** the models use the 25 `crime` features. The 5 citywide/calendar `t_`
  features are left out.
- **Why this design:** the first version let the models set the citywide level too.
  - The tree model learned citywide swings that didn't repeat. It forecast Q1 2025 at 0.77×
    the bar when the city came in at 0.94×, for −84% skill that quarter.
  - Dropping week-of-year didn't fix that. Dropping the citywide features did.
  - Splitting the skill into parts showed the local part was steady every year, while the
    citywide part carried all the swings.
- **Both models fit y / expected with weight = expected.** That's equivalent to a log(expected)
  offset.
- **The trees pick their size on a time-ordered holdout:** the newest training quarter, behind
  a 4-week gap.

| Crime-only, 2022–26 test | Deviance skill vs bar (95% CI, by quarter) | AUC above normal |
|---|---|---|
| **Poisson GLM, r8** | **+3.6% (+3.1% to +4.1%)** | 0.571 |
| Poisson GLM, summed to r7 | +4.9% (+3.3% to +6.5%) | 0.580 |
| Gradient-boosted trees, r8 | +2.8% (+2.2% to +3.4%) | 0.571 |
| Gradient-boosted trees, summed to r7 | +3.7% (+2.3% to +5.3%) | 0.579 |

What the table says:
- **The gain is small but reliable.** It's positive in every test year (GLM r8: +2.5% in 2022,
  +3.8–4.5% after).
- **The linear model beats the trees.** At this signal-to-noise, interactions don't pay.
- **What drives the GLM** (final-fold coefficients):
  - cells with a higher expected count come in lower (−14% per SD): a regression-to-the-mean
    correction to the trailing average
  - the last 4 weeks push the forecast up (+7% per SD)
  - neighbors' last 4 weeks push it up too (+3.5%)
  - last year's rise predicts a partial fall back (−3.4%)
- **The trees lean on the same signals:** 4-week acceleration, expected level, and
  year-over-year change.
- **At r7, the models rank above-normal areas worse than raw momentum** (AUC 0.58 vs 0.60).
  Their shrinkage is tuned to noisy r8 cells, and it's too cautious once cells are summed. Step 6
  compares both.

**Step 5, `ablation.py`: 311 adds nothing measurable.**

| Lift over the same model without 311, r8 (95% CI by quarter) | GLM | Trees |
|---|---|---|
| + real 311 (66 features) | −0.89% (−1.28% to −0.49%) | +0.01% (−0.13% to +0.15%) |
| + shuffled 311 (placebo) | −0.53% | +0.07% |
| real vs shuffled | −0.36% (−0.71% to −0.04%) | −0.05% (−0.32% to +0.26%) |

- **The placebo** shuffles each week's 311 features across cells. That keeps citywide patterns and
  breaks only the link to place.
- **The GLM overfits the 66 noisy features.** Real 311 even does worse than shuffled, which suggests
  the 311–crime relationship drifts over time.
- **The trees learn to ignore 311.**
- **No sub-group helps** (totals, backlog, per-type, neighbors), and none of the 27 request types
  helps on its own. The best is +0.03%, with a CI including 0.
- **Power:** the CIs are about ±0.4% wide, so an effect a seventh the size of crime history's own
  +3.6% would have shown up.
- **Conclusion:** at r8 × 4 weeks, recent crime already carries whatever 311 knows about next month.
  311 might still matter at a finer place-and-moment scale, e.g. a streetlight event study.

**Step 6, `early_warning.py`: calibrated calls.**

- **Probabilities:** P(next 4 weeks > expected) comes from a negative binomial around the forecast.
  Its dispersion is fit only on earlier folds' out-of-sample misses.
- **Calls:** up at P ≥ 0.65, down at P ≤ 0.35, otherwise unclear.
- **Which r7 forecast to use:**

  | Option | Skill | AUC | Brier |
  |---|---|---|---|
  | r8 GLM summed | +4.9% | 0.580 | **0.243** |
  | GLM fit on r7 directly | +5.7% | 0.574 | 0.244 |
  | one-feature momentum GLM | +5.3% | **0.602** | 0.245 |

  - Momentum ranks best.
  - The summed r8 model has the best Brier score (ranking and calibration together), so it drives
    the live calls.
- **How good are the calls?** At r7, 7.7% of area-weeks get one. Up calls are right 68% of the
  time and down calls 67%, against a 53% base rate. Brier skill over climatology is 4%.
- **Calibration:** reliability is close across the 0.3–0.7 range where almost all forecasts fall.
  Observed rates run 1–3 points above predicted.
A refined noise estimate for the walkthrough simulates Poisson draws instead of using a 1/λ
approximation. In the 2022+ test years, about 66% of an r8 cell's 4-week departures from normal
are chance, about 52% for r7 areas, and about 28% for r7 × 13 weeks. The rough version quoted
while planning said ~77% for r8 × 4 weeks.

## Caveats for modeling

- The newest week or so of crime is incomplete (it's reported with a lag). The crime file
  ends 2026-09-12, while 311 runs to 2026-09-22.
- Crime before 2019-03 has no 311 counterpart. Use it for crime-only baselines and
  seasonality.
- Every training cutoff before 2022 falls in the COVID era, so the first test folds learn from
  an unusual regime. Read step 4's results by year.
- H3 cells cover the city unevenly. Some r9 cells are mostly water, rail yards or O'Hare.
  A modeling panel will need its own decision about which cells to keep, and it has to
  zero-fill cell × period combinations with no events.

## 1-week track

The question: **will a community area run above its own normal *next week*?** The 4-week track
above lets the bar set the citywide total. At one week, though, the whole city running hot or cold
is a large share of every area's miss:
- At area × week, the citywide level is 19% of the bar's deviance.
- At r8 × 4 weeks it's 6–8%.

So this track models the citywide level first, then the local split, then turns both into calls for
the 77 community areas.

```bash
offline_ml/.venv/bin/python offline_ml/weather.py      # step 1: weather history, archived forecasts, holidays (~20 s first run)
offline_ml/.venv/bin/python offline_ml/city_level.py   # step 2: citywide 1-week model + payoff at area x week (~3 s)
offline_ml/.venv/bin/python offline_ml/local_week.py   # step 3: which cells run hot or cold, + the fresher-311 test (~5 min)
offline_ml/.venv/bin/python offline_ml/area_week.py    # step 4: community-area forecasts + up/down calls (~5 s)
```

Outputs go to `offline_data/weather/` and `offline_data/model/h1/`.

**Start with [`week_walkthrough.ipynb`](week_walkthrough.ipynb).** It tells the whole story with 8 charts, from why "up
vs last week" is the wrong target to a map of one week's calls against what happened. It runs top to bottom in about
10 seconds from the saved outputs, including every leakage check for steps 2 and 3.

**The reporting lag is part of the design.** The live feed's newest complete day runs about 9
days back. So a forecast made on cutoff Monday c only sees crime before c − 8 days
(`REPORT_LAG_DAYS`), and the week right before the cutoff is never known in time.
- The features respect this, and so do the folds: a fold trains only on target weeks that were
  already in the feed at its first test cutoff, which is 3 weeks back.
- A day first appears about 98% complete and fills in to about 99.5% over two weeks. The backtest
  uses final counts, so live momentum reads ~1.5% low. That lowers live forecasts by roughly 0.5%.

**Step 1, `weather.py`.** Open-Meteo, no key, one point near the middle of the city:
- **Observed:** ERA5 reanalysis, daily, from 1996. It trains the weather effect and gives the
  normals: the same date ±7 days over the 10 prior years.
- **Forecasts:** the Previous Runs API keeps what each forecast said 1–7 days ahead. That lets
  every test week use the forecast issued the day before its cutoff, not the weather that happened.
  Coverage by model:

  | Model | Archive | Weekly temp anomaly r, 2022+ | Weekly precip anomaly r |
  |---|---|---|---|
  | GFS (used for temperature) | temperature from 2021-03, precipitation only from 2024-01 | **0.94** | 0.39 (2024+) |
  | JMA (used for precipitation, and fills GFS gaps) | both from 2021-01 | 0.76 | **0.44** |

  - Correlations are on anomalies. Raw weekly temperature correlates ~0.99 with anything that knows
    the seasons.
  - ECMWF and GEM only start in 2024.
  - Two test weeks (2026-04-13 and 04-20) lack 4 forecast days in both models. The missing days
    count as normal weather.
- **Holidays:** computed from their rules. That's the federal holidays on their actual dates, plus
  Easter, Halloween, Christmas Eve and New Year's Eve.

**Step 2, `city_level.py`.** Per cutoff Monday, the target is all crime in the next 7 days.
- **The bar:** the city's trailing-year daily rate × a daily seasonal factor. The factor is the
  median of the same weekday's ratio 1–5 years back, ±1 week (15 values). Day of week is built in.
- **Momentum:** how far the last 7 and 28 known days ran from their own bar.
- **Calendar:** holiday flags for the target week, plus a 1st-of-month flag. "Sometime this month"
  records are stamped on the 1st, and the model puts that at +1.9%, matching the ~2% measured
  directly.
- **Weather:** the target week's temperature and rain anomalies, plus the observed temperature
  anomaly over the momentum window, so a hot week isn't mistaken for an upturn.
- **Model:** a Poisson GLM on top of the bar, trained on observed weather 2007+ (COVID skipped)
  and tested on forecasts. It's walked forward quarterly over 244 weeks, 2022–2026-08.
- **Leakage checks run on every build:**
  1. Recompute the target, bar and momentum from raw daily counts.
  2. Delete everything from 2024-06-03 on, and require identical features for cutoffs up to 8
     days later.

Skill is deviance skill over the calibrated bar (the bar with its level fitted), with a 95% CI
from resampling whole quarters. "Area × week" scales each community area's own bar by the city
multiplier.

| 2022–26 test | Citywide weekly total | Area × week |
|---|---|---|
| **Momentum + calendar + weather forecast** | **+61% (+40 to +74%)**, weekly error 6.2% → 4.1% | **+11.8% (+5.4 to +18.7%)** |
| Same, with perfect weather | +62% | +11.9% |
| Ceiling: knowing the actual citywide total | | +19.2% |
| Pure Poisson chance, which no model can win | | 50% of the bar's deviance |

What each piece adds over the one before (paired, same weeks):

| Piece | Citywide | Area × week |
|---|---|---|
| Momentum | +41% (+21 to +53%) | +7.8% (+2.8 to +13.2%) |
| Calendar | +5% (−4 to +13%) | +0.6% (−0.4 to +1.9%) |
| **Weather forecast** | **+32% (+15 to +45%)** | **+3.7% (+1.3 to +6.1%)** |
| Perfect weather instead of the forecast | +2% (−11 to +15%) | +0.2% (−0.9 to +1.3%) |

What it says:
- **Weather is a go.** It's the second-biggest piece and clearly nonzero. The 1–7 day forecast
  captures essentially all of what perfect knowledge of the weather would add.
- **Size of the effects:** +1.0% crime per °C above normal in the target week, and −0.4% per extra
  cm of rain. A warm spell in the momentum window is discounted at −0.6% per °C, so it isn't read as
  a trend.
- **Momentum is the biggest piece.** A recent 28-day window running 10% above its bar carries
  +4.5% into next week, and the last 7 days add +2.5%. Partly this is the trailing-year bar lagging
  real multi-month trends. The 2022 rebound, for instance, gets +82% citywide skill.
- **The calendar barely registers overall,** though individual weeks move:
  - Christmas week nets about −3%, while its Eve and Day flags split into offsetting coefficients.
  - Thanksgiving −3%, Easter −4%, New Year's Eve +7%, Halloween +4%.
- **Every test year is positive citywide:** +22% in 2024 (the weakest) to +82% in 2022.
- **In context:** the city model alone wins about a quarter of the non-chance deviance at area ×
  week. The 4-week crime-only local models won +3.6% at r8. Skills are relative to each grain's own
  bar, so this isn't a like-for-like comparison, but it shows where the 1-week signal lives.

**Step 3, `local_week.py`: which cells run hot or cold next week, relative to the rest of the city?**

It uses the same split as the 4-week track's step 4, and reuses `models.py`'s GLM, trees and
walk-forward unchanged.
- **The reference to beat is `bar_city`:** each r8 cell's bar (its trailing-year rate × the city's
  daily season) × step 2's weekly multiplier. It already carries the citywide call, so any skill
  over it is purely local.
- **Features are rolling windows over daily counts,** ending 8 days before the cutoff. The weekly grid
  can't express the lag, because the last usable day is a Saturday. There are 30 crime-side
  features:
  - the cell's last 7 / 28 / 91 / 364 known days, and the year before
  - violent and property splits, and acceleration ratios
  - the location mix
  - ring-1 and ring-2 neighbors
  - the cell's own seasonal bump at this time of year, 1 and 2 years back
- **311 gets a new test.** The 311 feed is close to real time (on 2026-09-26 it already held that
  morning's requests), so its windows end 1 day before the cutoff and cover the week crime can't see.
  The `*_311` variants add 8 such features.
- **Leakage checks run on every build:**
  1. A recount from the event tables.
  2. A rebuild from the data as it stood at 2024-06-17, with crime deleted from exactly cutoff − 8
     days and 311 from cutoff − 1. All 39 feature columns and the bar must match, and a week later
     they must differ.
- **Floor:** the bar is floored at 0.125 a week. 14 cells have stretches with no crime in the
  trailing year.

| 2022–26 test, 187,880 cell-weeks | Skill over `bar_city` (95% CI) | Total skill over the calibrated bar | AUC above normal | Share of `bar_city`'s deviance that's chance |
|---|---|---|---|---|
| GLM, r8 | **+1.3% (+1.1 to +1.5%)** | +3.0% | 0.532 | 78% |
| trees, r8 | +1.1% (+0.8 to +1.3%) | +2.8% | 0.536 | |
| GLM, summed to r7 | **+2.4% (+1.4 to +3.3%)** | +9.9% | 0.549 | 65% |
| trees, summed to r7 | +2.0% (+1.4 to +2.7%) | +9.5% | 0.550 | |

What it says:
- **The local model adds a small, steady gain on top of the city call.** It's positive every
  year at both grains. At r7, the citywide call contributes 7.6% of the total 9.9%.
- **It's smaller than the 4-week track's +3.6%** because a week is noisier: 78% of an r8 cell's
  miss is chance, against 66% over 4 weeks.
- **The GLM beats the trees again.** Its main levers match step 4:
  - regression to the mean: cells with a higher bar come in lower, −21% per SD
  - the last 28 days push the forecast up, +11% per SD
  - neighbors' recent crime pushes it up a little
  - last year's rise predicts a partial fall back
- **Fresher 311 adds nothing:** −0.01% (CI −0.06 to +0.04%) with the GLM, and −0.02% with the trees.
  The CI is tight enough to rule out anything bigger than ~0.05%. Seeing the week crime can't
  doesn't make 311 useful either.
- **Raw 28-day momentum ranks above-normal areas slightly better than the GLM,** 0.564 vs 0.549 at
  r7 (`local_ranking.csv`). The GLM is better at forecast size, which deviance rewards. The 4-week
  track showed the same split, so step 4's up/down calls should compare both.
- **The biggest cell-week spikes are Lollapalooza.** The 2021 and 2022 festival weeks each bring
  ~300 crimes to that cell, against a bar of ~20–36. They're mostly pickpocketing and thefts on
  Grant Park property along the 300 block of E Randolph. A known-event feature from
  the events work could catch weeks like these. Only 4 test cell-weeks are that extreme, so it
  barely moves the scores.

**Step 4, `area_week.py`: next week's forecast and up/down call for each community area.**

- **An area's normal** is its own calibrated bar (trailing-year rate × the city's daily season × the
  city bar's fitted level), built from the area's own crime records.
- **Every option starts from `bar_city`:** the normal × step 2's citywide call. They differ only in
  the local tilt:
  - `cells_summed`: step 3's cell forecasts, split into areas by each cell's 2016–2021 crime share.
  - `cells_tilt`: the area's own bar_city × those cells' hot/cold lean.
  - `native_glm`: a GLM fit on the 77 areas directly.
  - `momentum`: a one-feature area GLM, using the last 28 known days vs the trailing year.
- **Probabilities and calls** follow the 4-week `early_warning.py`, whose helpers are reused:
  - an NB around the forecast, with its dispersion from earlier folds only
  - up at P ≥ 0.65, down at P ≤ 0.35
- **Two kinds of call:**
  - vs normal: will the area beat its own normal?
  - relative: will it beat bar_city, i.e. outpace what the citywide call implies?
- **Hit-rate CIs resample whole quarters,** since calls in the same week share one citywide call.

2022–26 test: 77 areas × 244 weeks, 62 crimes per area-week on average. 48.4% of area-weeks came
in above normal.

| Option | Skill over normal | Skill over bar_city | AUC above normal |
|---|---|---|---|
| bar_city (citywide call only) | +11.8% | — | 0.501 |
| cells_summed | +13.2% | +1.6% (+0.2 to +2.9%) | 0.570 |
| cells_tilt | +13.9% | +2.4% (+1.0 to +3.7%) | 0.575 |
| native_glm | **+14.5%** | **+3.2% (+2.1 to +4.3%)** | 0.577 |
| momentum | +14.4% | +3.0% (+2.2 to +3.9%) | **0.580** |

Calls, folds 1–18:

| Calls vs | Option | Brier skill | Area-weeks called | Right (95% CI) | Weeks with any call |
|---|---|---|---|---|---|
| normal | bar_city | 7.9% | 18.6% | 74.6% (69–79%) | 35% |
| normal | **momentum** (best Brier) | **9.4%** | **23.4%** | **74.5% (70–78%)** | 90%, median 6 a week |
| normal | native_glm | 9.2% | 25.2% | 73.8% (69–77%) | 96% |
| relative | momentum | 2.2% | 1.4% | 73.7% (69–79%) | 62%, ~1 a week |
| relative | native_glm | 2.1% | 2.0% | 65.2% (60–70%) | 78% |

What it says:
- **Area calls are much better than the 4-week track's r7 calls.** There, 7.7% of area-weeks got a
  call, right 67–68% of the time against a 53% base rate, with 4% Brier skill. Here, 23% get a call,
  right 75% against 48%, with 9.4% Brier skill.
- **Most of that is the citywide call.** bar_city alone makes calls on 19% of area-weeks at the same
  accuracy.
  - For momentum's calls, the median citywide part is 8.5%, and the local part 1.9%.
  - Calls cluster: the citywide call alone has a call in only 35% of weeks, and some weeks it calls
    all 77 areas.
  - Calls in the same week stand or fall together. In the last test week (Aug 31), the citywide call
    was +5.6% and 4 of the 5 top up-calls missed.
- **The local tilt does add something.**
  - Deviance: +3.0–3.2% over bar_city.
  - Calls: more of them (23% vs 19% of area-weeks) and spread across 90% of weeks, at the same
    accuracy.
  - On its own, as relative calls, it's right 74% of the time but only calls ~1 area a week, mostly
    "down."
- **Areas don't need the cell model.** The area-fit GLM and one-feature momentum match or beat
  the cell forecasts rolled up.
  - Splitting cells across areas costs accuracy: `cells_summed` is worst, and `cells_tilt`, which
    only borrows their lean, does better.
  - The cell model stays useful for a hex map layer.
- **Momentum drives the shipped calls.** It has the best Brier and AUC, is tied on deviance, and is
  one feature. So a call's local reason can be as plain as "its last 4 weeks ran X% above normal."
- **Calibration:** in the 0.3–0.6 bins, predicted runs ~4 points above observed. Above 0.65,
  observed runs 1–5 points above predicted, so up calls are, if anything, conservative.

`area_test_probs.parquet` is the table the app would read, one row per area × week:
- the normal and the forecast
- P(above) and the call
- `pct_citywide` and `pct_local`, which give the "why"
- the area name and the outcome

**In the app:** the citywide model and the `momentum` area option became the next-week outlook
(`src/outlook/`, run nightly by `notebooks/08`), which reproduces this backtest on these exports.

## Events

The question: **do events move crime and 311 in the cells around them?** If they clearly do,
upcoming events could be flagged ahead of time. Major pro sports are out of scope, since
everyone already plans around those. The focus is city-permitted events and shows at smaller
music clubs.

```bash
offline_ml/.venv/bin/python offline_ml/events_fetch.py permits        # ~1 min, no key
offline_ml/.venv/bin/python offline_ml/events_fetch.py setlistfm      # re-run daily until every club is done
offline_ml/.venv/bin/python offline_ml/events_fetch.py ticketmaster   # upcoming-events snapshot
offline_ml/.venv/bin/python offline_ml/events_build.py                # -> offline_data/events/
offline_ml/.venv/bin/python offline_ml/events_impact.py               # permits + clubs, ~15 s
offline_ml/.venv/bin/python offline_ml/events_lookahead.py            # upcoming festivals + club nights
offline_ml/.venv/bin/python offline_ml/events_backtest.py             # replays 2017-2025
```

**Start with [`events_walkthrough.ipynb`](events_walkthrough.ipynb).** It tells the whole story with
charts and runs top to bottom in about 10 seconds from the saved outputs, including three checks run live:
- the pooled ratios re-derive from the per-event table;
- null dates behave like chance;
- scoring "as of" a date can't see past it.

Keys go in `offline_ml/.env` (gitignored): `SETLISTFM_API_KEY`, `TICKETMASTER_API_KEY`.

| Source | What it gives | Limits |
|---|---|---|
| CDOT street-use permits (`pubx-yq2d`) | Festivals, block parties, parades, runs, rallies, 2014+, with a point per closed street segment | Whole days only, no start times |
| Park District event permits (`pk66-w54g`) | Events in parks, with a size level (6 = 10,000+) | One row per facility per day, and dates include setup and teardown |
| Park polygons (`ejsh-fztr`) | Park number → r9 cells | 45 permit runs name a park with no polygon |
| setlist.fm | Club shows since 2014, one setlist per act | 1,440 requests/day. Partial coverage: Schubas has a logged show on ~25% of days but books most nights |
| Ticketmaster Discovery | Upcoming events with start times | No history at all. Some venue coordinates are wrong (it puts the Empty Bottle in New York) |

Rules in `events_build.py`, each from profiling the pulls:
- **Status:** CDOT `Complete` → held, `Cancelled`/`Denied` dropped. Park `Approved`/`Completed`/
  `Issued` → held. Anything still ahead of today and not rejected → planned. Past applications
  that never completed are dropped.
- **Street Closure permits are dropped.** They're bridge work, sign dedications and graduations,
  with a median span of 8 days.
- **Merging CDOT applications:** festivals, parades, runs and rallies with the same name and
  dates become one event. Block parties never merge, because they're nearly all named "Block Party".
- **Park events** are runs of consecutive days with the same park, name and category.
- **`long_run` = more than 4 days.** That span is a reservation window, not crowd days. The Park
  District holds Grant Park for 27 days around Lollapalooza's 4, while CDOT's street permits for
  it cover exactly Jul 31–Aug 3. Skip long runs when measuring event-day effects.
  Big park festivals still get their true days from their CDOT permits.
- **Club shows:** one event per club per date. The act with the longest logged set is taken as
  the headliner, and 58% of setlists are placeholders with no songs. A club is only included once
  its whole history is fetched, so a half-pulled club can't make its older years look show-free.
- **Coordinates:** club coordinates come from the Census geocoder on the street address,
  because setlist.fm only knows the city.

Outputs (`offline_data/events/`), from an early build with 3 of the 20 clubs fetched:

| Category | Held | Planned | long_run | Median r9 cells |
|---|---|---|---|---|
| cdot block_party | 47,723 | 363 | 1 | 1 |
| cdot festival | 4,373 | 102 | 550 | 1 |
| cdot parade | 3,025 | 18 | 1 | 1 |
| cdot athletic | 1,369 | 36 | 7 | 1 |
| cdot assembly | 554 | 4 | 4 | 1 |
| park_event | 8,581 | 198 | 313 | 1 |
| park_athletic | 2,450 | 0 | 71 | 18 |
| park_corporate (ends 2019) | 1,391 | 0 | 66 | 1 |
| park_commemorative | 471 | 0 | 0 | 8 |
| club_show | 2,580 | 2 | 0 | 1 |

- `events.parquet`: one row per event. It holds source, category, name, venue, dates,
  `long_run`, status, the size proxies (`size_level`, `n_segments`, `full_closure`,
  `n_facilities`, `n_acts`), the centroid and its r9 cell.
- `event_cells.parquet`: every r9 cell an event touches.

Caveats:
- **Big parks dilute.** A permit names the park, not where in it the event is, so a run through
  Lincoln Park covers every cell of the park. `n_cells` shows how spread out each event is.
- **The same festival can appear twice**, once from its street permit and once from its park
  permit, and Lollapalooza's two CDOT applications carry different names. That doesn't matter
  when measuring cell-days (was any event there?), but it inflates event counts.
- **Unlogged club shows** mean the no-show days used for comparison include some real shows.
  That pulls any measured effect toward zero.

### Impact of permitted events, `events_impact.py permits` (~10 s)

**Design:** observed vs. expected per event.
- **Where:** an event's own r9 cells (ring 0), plus the cells 1 and 2 steps out.
- **Expected:** the same cell on the same weekday 1–4 weeks either side. Days with an event
  within one cell are skipped, and a cell-day needs at least 4 clean control days. Permit
  events also rule out the day either side; club shows rule out only their own night.
- **Citywide scaling:** that baseline is scaled by how the whole city's count on the day
  compared with the same control days. This absorbs holidays, weather and trend.
- **Pooling:** per category, ratio = Σ observed / Σ expected, with a 90% interval from a
  Poisson bootstrap over events.
- **Checks that should read ~1.0:** the day before, the day after, and a placebo date 5 weeks later.
- **Coverage:** crime runs 2014 → 2026-09-05, and 311 (resident-reported only) runs
  2019-03 → 2026-09-15. Club shows are a separate run, below.

Crime, observed/expected [90% interval]:

| | own cells, event days | 1 cell out | 2 cells out | day before | placebo +5 wk | extra crimes per event (own + ring 1) |
|---|---|---|---|---|---|---|
| festival (2,684) | **1.31** [1.24–1.39] | 1.11 [1.08–1.14] | 1.03 | 1.02 | 1.03 | +0.48 |
| festival, 5+ street segments (219) | **1.52** [1.34–1.73] | 1.21 [1.12–1.29] | 0.98 | 0.96 | 1.03 | +2.47 |
| block party (35,798) | **1.06** [1.04–1.09] | 0.99 | 1.00 | 1.03 | 0.99 | ~0 |
| parade (2,515) | 0.96 [0.89–1.02] | 1.01 | 1.04 | 0.98 | 1.00 | ~0 |
| park_event (3,271) | 1.07 [0.99–1.16] | 1.04 [1.01–1.07] | 1.00 | 0.98 | 1.02 | +0.07 |

What it says:
- **Street festivals raise crime where they are.** The effect passes every check:
  - It's near 1.0 the day before, the day after and on the placebo date.
  - It fades with distance (+31% → +11% → +3%).
  - It grows with size: +52% for festivals closing 5+ segments.
  - It lands in daytime hours: +41% between 10am and 10pm, against +17% overnight.
  - By type: battery +52%, theft +39%, motor vehicle theft +26%.
- **The absolute numbers are modest.** A typical festival brings about half an extra crime over
  its run, across its cells and the ring around them. One closing 5+ street segments brings about
  2.5.
- **Block parties are +6% in their own cells**, with battery +14% and assault +18% but robbery
  −16%. That's about one extra crime per 100 block parties. The ring around them doesn't move.
- **Parades don't change crime overall**, but police-found offenses (narcotics, weapons, public
  peace, liquor law) rise 42%. That's the deployment, not the crowd.
- **Park District events show a small lift one cell out** (+4%, 1.01–1.07); their own cells are
  inconclusive. The biggest park festivals are long runs and get excluded, and their crowd days
  are covered by their CDOT festival permits instead. Park polygons also spread each event over
  the whole park.
- **311 is at most a whisper.** Resident requests are +8% in festival cells (1.02–1.14, placebo
  0.99). Parking complaints at full-closure festivals (+64%) are the closest thing to a lead, and
  even that interval (0.98–2.24) includes no change.

Caveats:
- **Many comparisons.** The summary has ~1,600 intervals, so about 10% exclude 1.0 by chance.
  Trust patterns that also pass the placebo, timing and distance checks, which the festival
  and block-party crime results do. Don't trust single cells in the table.
- **Intervals on tiny counts aren't reliable.** With 20+ expected incidents, 9.1% of null-date
  intervals (placebo, day before, day after) exclude 1.0, right at chance. Across all of them it's
  17.4%, because a group with zero observed gets a zero-width interval. Below ~20 expected, read a
  ratio as a hint.
- **Police presence and recorded crime are tangled.** More officers at an event can mean more
  thefts reported on the spot, not more thefts committed.
- **Pooled estimates aren't independent.** A cell-day shared by two events (a festival's street
  and park permits) counts toward each of them.

Outputs in `offline_data/events/impact/`:
- `oe_by_event.parquet`: observed and expected per event × ring × timing × outcome.
- `summary.csv`: every pooled ratio.

### Impact of club shows, `events_impact.py clubs` (~5 s)

Same design as the permit run, with three changes:
- **Night window.** setlist.fm has show dates but no times, so crime counts from 6pm to 3am,
  keyed to the show's date. Crimes with an unknown time are dropped, since they're logged at
  midnight and would land inside the night.
- **Control nights** are the same club's nights without a logged show, on the same weekday
  1–4 weeks either side. The busiest clubs log a show on a third of nights, so a one-day
  buffer around shows would leave almost nothing to compare against.
- **Coverage:** 20 clubs, 14,065 show nights since 2014. 9,280 of them have 4+ clean control
  nights and are scored.

Crime on show nights, observed/expected [90% interval]:

| | club's cell | 1 cell out | 2 cells out | day before | placebo +5 wk |
|---|---|---|---|---|---|
| all show nights | **1.11** [1.06–1.17] | 1.04 [1.01–1.07] | 1.03 [1.01–1.05] | 0.98 | 1.01 |
| 6pm–midnight | 1.14 [1.07–1.20] | 1.05 | 1.04 | 0.97 | 1.01 |
| midnight–3am | 1.03 [0.94–1.13] | 1.01 | 1.02 | 1.01 | 1.01 |
| theft | 1.20 [1.11–1.30] | 1.06 | 1.02 | 0.97 | 0.99 |
| battery | 1.12 [0.99–1.24] | 1.03 | 1.02 | 1.04 | 1.06 |
| assault | 1.35 [1.05–1.68] | 1.13 | 1.03 | 0.89 | 0.82 |

What it says:
- **Club shows add a little crime right around the club, before midnight.** The effect passes
  the placebo and the day-before check, and fades with distance.
- **The absolute numbers are tiny.** A show adds about 0.05 crimes across its cell and the two
  rings around it: about one extra crime per 18 logged shows.
- **The extra is mostly theft, plus simple battery and assault.** In raw show-night vs.
  other-night counts (unmatched, so only the mix is meaningful), the biggest excesses are simple
  battery, theft (including pickpocketing and theft from buildings), and damage to or theft of
  vehicles.
- **Size proxies don't separate anything.** Headliner-plus-openers nights (1.07) look like
  single-act nights (1.15). Fri–Sat (1.07) looks like Sun–Thu (1.14).
- **Per-club numbers are noisy:**
  - Clearly above 1: Vic 1.17 [1.02–1.34], Bottom Lounge 1.31 [1.08–1.56], Lincoln Hall 1.36
    [1.10–1.65], Thalia Hall 1.26 [1.04–1.52].
  - Cobra Lounge is 2.38 [1.81–3.18], mostly theft and criminal damage on the street. That
    looks like car break-ins during shows, though its placebo also runs a bit high (1.25).
  - Sleeping Village is *lower* on show nights: 0.65 [0.48–0.82].
  - Concord is 1.10 [0.89–1.34] once its address is corrected to 2047 N Milwaukee.
  - Beyond noise, clubs differ less than these spreads suggest: the look-ahead estimates a
    between-club spread of only ~0.10.
- **311:** resident requests overall don't move (0.98). Business complaints double (2.05
  [1.43–2.89], placebo 0.88). That's plausibly noise and crowd complaints about venues, but the
  counts are small.

Caveats:
- **Unlogged shows** sit among the control nights, which pulls every ratio toward 1.0. The true
  effect is likely somewhat larger.
- **Scored nights skew toward weeknights.** Only 52% of Fri–Sat show nights have enough clean
  control nights, against 79% of weeknights. Park West, Bottom Lounge, Cobra Lounge and Thalia
  Hall are the least covered.

Outputs: `offline_data/events/impact/oe_by_event_clubs.parquet` and `summary_clubs.csv`.

### Look-ahead: upcoming festivals and club nights, `events_lookahead.py` (~5 s)

This ranks upcoming street festivals (CDOT permits already filed) and club nights (the latest
Ticketmaster snapshot plus any setlist.fm listings) by the extra crime they're likely to bring
to their own cells and the ring around them. It writes `festivals_<date>.csv` and
`club_nights_<date>.csv` to `offline_data/events/lookahead/`.

How each event is scored:
- **Baseline:** crimes normally expected there over the event's dates. That's whole days for
  festivals and 6pm–3am for club nights: each cell's mean on the same weekday within ±4 weeks of
  the same date in each of the last 3 years, skipping days with an event nearby.
- **Ratio:** the event's own track record pooled with a prior (Gamma-Poisson).
  - Festivals: past editions (same name once years are dropped, within 1.5 km), with the average
    lift for festivals its size as the prior.
  - Club nights: the club's past show nights, with the all-club average as the prior.
- **How much weight history gets** is estimated each run, by method of moments on how much the
  groups truly differ beyond Poisson noise:
  - Festivals with 2+ past editions vary by ~0.45 around their size average, so the prior
    counts for only ~4–5 expected crimes. A festival with a few editions on file mostly speaks
    for itself. Pub crawls are the ones that stand out.
  - Clubs vary by only ~0.10, so the prior counts for ~100 expected crimes, and a club's own
    record barely moves the all-club average (1.05 over the club's cell and ring).
- **Scored "as of" a date.** Only events and crime before that date are used, which is what lets
  `events_backtest.py` replay past years through the same code.
  - A past event counts as history only once the 4 weeks after it are known too. Its measured O/E
    compares it with the same weekday up to 4 weeks later.
  - Without that rule, a festival held just before the cutoff would bring in crime from after it.
  - The notebook checks this live: it deletes every crime from 2025 on, rebuilds, and requires 2025's
    scores to come out identical.
- **Extra crimes** = baseline × (ratio − 1), with a 90% interval from the ratio's posterior.

Festival size priors (run of 2026-09-26):

| Size | Past festivals | Crime lift | Violent lift |
|---|---|---|---|
| 1 street segment | 1,864 | 1.09 | 1.03 |
| 2–4 segments | 1,046 | 1.12 | 1.13 |
| 5+ segments | 271 | 1.30 | 1.26 |

**Festivals:** 89 scored, about 48 extra crimes expected across all of them.

| # | Date | Festival | Area | Past editions | Ratio | Extra crimes [90%] |
|---|---|---|---|---|---|---|
| 1 | 2027-03-13 | The Shamrock Crawl | Lake View | 3 | 2.40 | 10.3 [6.9–13.9] |
| 2 | 10-31 | Wrigleyville Halloween Crawl | Lake View | 4 | 1.53 | 7.1 [1.5–13.4] |
| 3 | 12-12 | TBOX (bar crawl) | Lake View | 6 | 1.95 | 5.2 [2.4–8.4] |
| 4 | 10-02 | Apple Fest | Lincoln Square | 8 | 1.63 | 2.4 [0.3–4.9] |
| 5 | 12-31 | Wrigleyville NYE Crawl | Lake View | 4 | 1.46 | 2.2 [0.1–4.6] |
| 6 | 10-03 | Chicago Vintage Fest Pilsen | Lower West Side | 3 | 1.44 | 2.1 [−0.6–5.4] |

**Club nights:** 353 upcoming nights at 14 clubs through May 2027, about 14 extra crimes in
total.

| Club | Nights | Club's ratio [90%] | Extra per night | Extra total |
|---|---|---|---|---|
| Bottom Lounge | 35 | 1.22 [1.14–1.30] | 0.20 | 7.0 |
| Vic Theatre | 57 | 1.11 [1.05–1.18] | 0.09 | 5.3 |
| Joe's on Weed St | 16 | 1.11 [1.00–1.22] | 0.10 | 1.6 |
| Subterranean | 22 | 1.06 [1.00–1.12] | 0.05 | 1.1 |
| Empty Bottle | 64 | 1.03 [0.96–1.11] | 0.01 | 0.9 |

- **What drives the festival ranking:** a festival's own record, now that crime is counted
  properly. Wrigleyville pub crawls top the list. The Shamrock Crawl has run 2.4× its baseline
  over 3 past editions. After that come busy areas with big closures.
- **Club nights matter mainly in bulk.** Each night is 0.2 extra crimes at most, but the Bottom
  Lounge's 35 upcoming nights add up to ~7.
- **Individual events are uncertain.** Nearly every festival's interval includes zero.
- **The lists are only as complete as the filings and ticketing.**
  - Festivals are mostly fall 2026, and 55 of the 89 are still "Application in Review".
  - Six clubs don't sell through Ticketmaster (Sleeping Village, Martyrs', Cobra Lounge, the
    Hideout, the Promontory, and the closed Double Door), so they barely appear.
- **Multi-week runs are listed but not scored:** Christkindlmarket, ZooLights, Winterland at
  Gallagher Way and 8 others. Their permits cover a season, and their effect was never measured.

### Backtest of the festival look-ahead, `events_backtest.py` (~3 s)

Each year 2017–2025 is replayed as of January 1st through the same scoring code, using only
festivals and crime from before then. What actually happened is each festival's observed crime
(own cells + ring) minus `events_impact.py`'s controlled expected count. 2,212 festival-years
have a measured outcome, with 1,105 actual extra crimes in total.

| Ranking (within each year) | Spearman vs. actual | Top 20% captures | Top 20% O/E | Rest O/E |
|---|---|---|---|---|
| **full (what ships)** | 0.075 | **74%** | 1.27 | 1.09 |
| no history (size prior only) | 0.054 | 69% | 1.22 | 1.12 |
| baseline only | 0.042 | 67% | 1.20 | 1.14 |
| size only | 0.082 | 62% | 1.28 | 1.11 |

Paired bootstrap over festivals, capture difference [90%]:
- full − baseline only: +7.4% [−0.4% to +16.6%]
- full − no history: +5.0% [−2.1% to +12.8%]

Ties (size only has three distinct values a year) are broken by event id, and rows are sorted
before the bootstrap, so reruns match exactly.

What it says:
- **The totals are about right.** It predicted 1,217 extra crimes over the 9 years, and 1,105
  happened. Single years swing a lot (2017: 128 predicted, 40 happened; 2021: 67 predicted, 163
  happened).
- **The top decile holds most of the effect, a little over-predicted.** It predicts 4.2 extra
  crimes per festival, and 3.1 happened. That's 62% of all the extra crime.
- **Below the top decile, the order is weak.** Deciles 1–9 average between −0.07 and 0.65 actual
  extra crimes, with no clear trend.
- **A festival's own history helps a little.** The full model captures the most: +5% over size
  priors alone, and +7% over baseline alone. Both intervals still reach zero.
- **In practice:** use the list as a short filter. Pub crawls and big closures on busy corridors
  are where the extra incidents land. Don't read meaning into the order below the top 10–20%.

Output: `offline_data/events/lookahead/backtest.csv`, one row per festival-year.
