# Modeling

Three analytical pieces sit behind the app:

| Piece | Question | Code | Notebook | Tutorial |
|---|---|---|---|---|
| [Next-week outlook](#next-week-outlook) | Will each area come in above or below its normal next week? | [`src/outlook/`](../src/outlook/) | `08` | [05](../tutorials/05_next_week_outlook.ipynb) |
| [Events look-ahead](#events-look-ahead) | How much extra crime will an upcoming festival or club night bring? | [`src/events_model.py`](../src/events_model.py) | `08b` | [06](../tutorials/06_events_lookahead.ipynb) |
| [Leading indicators](#leading-indicators) | Do 311 requests rise before crime does? | [`src/leading_indicators.py`](../src/leading_indicators.py) | `05` | [04](../tutorials/04_leading_indicators.ipynb) |

Each is plain numpy/pandas/scipy/scikit-learn, tested locally on synthetic data with a planted
answer and run nightly on Databricks. The research that chose these designs, including what was
tried and dropped, is in [`offline_ml/`](../offline_ml/README.md).

## Next-week outlook

![Outlook dashboard](images/outlook-dashboard.png)

*`report.dashboard()` on the real data through 2026-09-11. Notebook 08 draws the same figure every
night and logs it to MLflow.*

### The target and the baseline

The target is all crime in a community area over the week starting each Monday. "Normal" is the
**seasonal bar**: the area's daily rate over the trailing 364 days × a daily seasonal factor, the
median of the same weekday's ratio to its trailing-year rate 1-5 years back, ±1 week (15 values,
so one odd year or a holiday barely moves it). Every model is scored against it.

Asking "above or below last week?" was rejected early: most week-to-week change is noise and
seasonality. "Above or below normal for this time of year" is both more stable and more useful.

### The model

1. **Citywide:** next week's total = the city's bar × a Poisson GLM (fit as y/bar with weight
   bar, which equals a log-bar offset). Feature groups, in `specs.FEATURE_GROUPS`:
   - *momentum:* the city's last 7 and 28 *known* days against their own bar (log ratios), and
     holidays in the last 7;
   - *calendar:* a flag for each of 15 holidays falling in the target week, and the 1st of a
     month;
   - *weather:* the week's temperature and rain against normal, from the forecast issued before
     the week (observed weather in training), and the last 7 known days' temperature anomaly and
     snowfall, so a lull the weather caused isn't carried forward as a trend.
2. **Per area:** the area's own bar × the citywide multiplier × a one-feature local lean, the
   area's last 28 known days against its trailing year. The lean is trained with each week's
   expected counts pinned to the actual citywide total, so it learns only who runs hot or cold
   relative to the rest of the city.
3. **Calls:** P(next week > the area's normal) from a negative binomial around the forecast, its
   dispersion fit only on earlier folds' out-of-sample misses. **Up** at P ≥ 0.65, **down** at
   P ≤ 0.35, otherwise unclear. Each call splits into a citywide part (shared by every area) and
   a local part, and the app shows both.

### Honest evaluation

- **The reporting lag is built in.** The live crime feed's newest complete day is about 9 days
  back, so features at cutoff *c* use only crime before *c* − 8 days, and a training week must
  already have been in the feed at the test week's cutoff (`evaluation.train_last_for`).
  `tests/test_outlook_model.py` rebuilds features from truncated data and requires them to match.
- **Weather uses forecasts that were really issued.** Open-Meteo's Previous Runs archive gives
  each backtest week the forecast from the day before its cutoff, not the weather that happened.
- **Walk-forward folds.** Test weeks from 2022 on, refit every quarter (19 folds, 244 weeks).
  COVID weeks are left out of citywide training; their swings don't repeat.
- **Every run re-scores the backtest**, so the track record the app quotes is always current.

### Results

Backtest, 244 weeks (2022-01-03 to 2026-08-31), all models on the same weeks:

| Level | Model | Features | Deviance skill | Brier skill | Called | Right | AUC |
|---|---|---|---|---|---|---|---|
| city | seasonal_bar | (none) | 0 (reference) | | | | |
| city | momentum | momentum | +43.1% | | | | |
| city | momentum_calendar | + calendar | +45.8% | | | | |
| city | **full** (champion) | + weather | **+63.9%** | | | | |
| area | normal | (none) | 0 (reference) | +1.2% | 0% | | 0.498 |
| area | citywide_only | the city call alone | +11.6% | +7.7% | 19% | 74% | 0.646 |
| area | **momentum_lean** (champion) | + c_accel_28d | **+14.0%** | **+9.3%** | **24%** | **75%** | **0.664** |

- **Citywide:** the mean weekly error falls from 6.0% (the bar) to 3.8%. Weather is the biggest
  single gain: about +1% crime per °C (+0.5% per °F) above normal. Holiday weeks move the total
  from about −10% (Christmas) to +3% (Halloween, New Year's Eve).
- **Areas:** calls on 24% of 17,787 area-weeks, right 75% of the time (95% CI 71-78%,
  resampling whole folds, since a week's calls stand or fall together) against a 48% base rate.
  Most of that skill is the citywide call; the local lean adds more calls at the same accuracy.
- **Calibration:** between 0.3 and 0.6, where most forecasts fall, observed rates run 3-4 points
  below predicted; above 0.6 they run at or above it, so up calls are if anything conservative.

### Recent snow: a refinement found with the leaderboard

Snowfall in the 7 known days before the cutoff was tested as a challenger the way
[tutorial 05](../tutorials/05_next_week_outlook.ipynb) shows, and added to the weather group: +0.8
points of citywide skill (63.2% → 63.9%), positive in every test year (+0.4 to +2.0 points), with
area calls and Brier skill holding or improving. Its coefficient is positive: snow suppresses crime
in the momentum window, and without it the model carried that lull into the next week.

### What was tried and dropped

From the research in `offline_ml/` (each with walk-forward backtests and leakage checks):
- **311 as a crime predictor.** 66 311 features (totals, backlog, per type, neighbors) added
  nothing to a crime forecast at H3 cell × 4 weeks (−0.89% for the GLM, +0.01% for trees, CIs
  about ±0.4%), and fresher 311 covering the week crime can't see added nothing at 1 week either.
  311's role in the app is the descriptive lead-lag analysis below, not the forecast.
- **Gradient-boosted trees** were consistently a little worse than the GLM at this
  signal-to-noise, and learned citywide swings that didn't repeat.
- **H3 cell models rolled up to areas** matched but didn't beat the one-feature area lean, which
  is simpler and explains itself ("its last 4 weeks ran X% above normal").

### Extending it

Everything an experiment changes is data in [`src/outlook/specs.py`](../src/outlook/specs.py), and
every run backtests all registered models on the same folds:

```python
from src import outlook

# 1. a new citywide feature: compute it in a builder in features.py, list it in a group
# 2. a challenger that uses it, without touching the defaults
extra = {"weather_only": outlook.CityModelSpec("weather_only", ("weather",))}
out = outlook.run(area_daily, obs, forecasts, last_full, run_date,
                  city_models={**outlook.CITY_MODELS, **extra})
out["leaderboard"]          # scored beside the champions on the same weeks

# 3. promote it: point specs.CHAMPION_CITY (or CHAMPION_AREA) at it
```

Area features work the same way through `features.AREA_FEATURES` and `AreaModelSpec`. A new
feature group gets its own driver column (`pct_<group>`) automatically; the app shows the
existing three. [Tutorial 05](../tutorials/05_next_week_outlook.ipynb) walks through adding a
feature end to end.

### Showcasing it

- `report.model_card(out)`: a Markdown summary of the live week, the backtest and the leaderboard.
- `report.dashboard(out)`: the figure above. The individual charts (`plot_backtest`,
  `plot_leaderboard`, `plot_calibration`, `plot_drivers`, `plot_effects`, `plot_area_map`) work
  on their own too.
- Notebook 08 writes `outlook_leaderboard` and `outlook_calibration` to Unity Catalog and logs
  each run's settings, scores, model card and charts to an MLflow experiment, so the backtest can
  be tracked over time.

### Known limitations

- The newest crime days arrive about 98% complete, so live forecasts read about 0.5% low.
- Weather is one point near the city's middle: a citywide signal, not a local one.
- "Below normal" weeks have been common since 2024, as crime has run under the prior year's
  pace. The model isn't biased low; its backtest forecasts run 1-3% high.

## Events look-ahead

Upcoming street festivals (from filed CDOT permits) and club nights (Ticketmaster and
setlist.fm at 20 small music clubs) get an expected number of extra crimes in their own H3 cells
and the ring around them. Pro sports are out of scope on purpose: everyone plans around them.

**Measuring past events.** For each past event, observed vs. expected crime, where expected is
the same cells on the same weekday 1-4 weeks either side (skipping "busy" days with any event
nearby), scaled by how the whole city did that day against those control days. That absorbs
holidays, weather and trend. Checks that should read ~1.0 do: the day before, the day after, and
a placebo date 5 weeks later.

| Event type | Own cells | Ring | Extra crimes per event |
|---|---|---|---|
| Street festival | +31% | +11% | ~0.5 |
| Festival closing 5+ street segments | +52% | +21% | ~2.5 |
| Club show night (6 pm-3 am) | +11% | +4% | ~0.05 |
| Block party | +6% | ~0 | ~0.01 |

**Scoring upcoming events.** Extra crime = baseline × (ratio − 1). The baseline is the crime
normally expected in those cells on those dates (each cell's mean on the same weekday ±4 weeks
in each of the last 3 years). The ratio is a Gamma-Poisson posterior: the event's own history
(past editions of the same festival within 1.5 km, or the club's past nights) pooled with a prior
(festivals of its size, or all clubs). How much weight the prior gets comes from how much events
truly differ beyond Poisson noise, estimated each run: festivals differ a lot (pub crawls stand
out), so a few editions mostly speak for themselves; clubs barely differ.

**Backtest.** Replaying 2017-2025 as of each January 1, the top 20% of the ranking captured 74% of
the extra crime (69% for size priors alone), and total predicted extra crime was close to actual
(1,217 vs. 1,105 over nine years). Below the top decile the order is weak, so the app treats the
list as a short filter for busy corridors, not an alarm.

## Leading indicators

Do 311 requests for physical disorder (streetlights out, vacant buildings, dumping, graffiti)
rise before crime does? Per metric, monthly log counts per area are demeaned two ways (each area's
own level, each month's citywide swing), then a 311 type's residual in month t−k is correlated
with a crime category's in month t, pooled across areas, for k = 0-3. The same is done with crime
leading 311. A pair is flagged as leading at its best lag k ≥ 1 when the correlation is
Bonferroni-significant across every test, at least 0.05, and stronger than both the same-month
and the reverse correlation. A shuffled-area placebo should find nothing, and does.

This is descriptive, and the app says so: correlation, not causation, a citywide pattern rather
than a claim about any one area, and p-values that are optimistic because months within an area
aren't independent. In the app, leading 311 types that are rising in an area become "signals to
watch", each with a one-click alert.
