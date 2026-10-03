"""1-week track, step 4: next week's forecast and up/down call for each community area.

    offline_ml/.venv/bin/python offline_ml/local_week.py   # first: cell forecasts (step 3)
    offline_ml/.venv/bin/python offline_ml/area_week.py    # ~5 s

**An area's normal** is its own calibrated bar: the area's trailing-year rate x the city's daily
season x the city bar's fitted level. It's built from the area's own crime records, the numbers
the app shows. "Above normal" means next week's count beats it.

Every forecast starts from `bar_city`: that normal x step 2's citywide call for the week. The
options differ only in the local tilt on top:
- `bar_city`: no tilt. How far the citywide call gets on its own.
- `cells_summed`: step 3's cell GLM forecasts, split across areas by each cell's share of crime
  records in each area (2016 through 2021, before any test week), then summed.
- `cells_tilt`: bar_city x the same cells' tilt (their summed forecast / their summed
  bar_city). The level comes from the area's own history and only the hot/cold lean from the
  cells. This sidesteps the split: only 70% of r8 cells keep 90%+ of their crime in one area.
- `native_glm`: a GLM fit directly on the 77 areas, with area-level crime features.
- `momentum`: a one-feature GLM on areas, using the last 28 known days against the trailing-year
  rate. Plain momentum out-ranked the models at r7 in both tracks.

The area models reuse models.py's `walk_forward` (training pinned to each week's actual citywide
total). Their forecasts are then scaled by the same citywide call.

**Probabilities and calls** follow early_warning.py:
- P(next week > normal) comes from a negative binomial around the forecast.
- Its dispersion is fit only on earlier folds' out-of-sample misses, so fold 0 gets no calls.
- Up at P >= 0.65, down at P <= 0.35.
- Brier skill is measured against climatology: the above-normal rate of the earlier folds.

Every call is split into a citywide part (the week's call vs normal) and a local part (the tilt),
which is what a UI would show as the reason.

Two kinds of call are scored:
- **Vs normal:** will the area beat its own normal?
- **Relative:** will it beat bar_city, i.e. outpace what the citywide call implies for it?

Hit rates get CIs from resampling whole folds, because calls made in the same week share one
citywide call and stand or fall together.

Outputs (offline_data/model/h1/):
    area_comparison.csv       deviance skill and AUC per option, with CIs
    area_calls.csv            Brier scores, share called and hit rates per option, for calls vs
                              normal and relative calls vs bar_city
    area_reliability.csv      predicted vs observed above-normal rate by probability bin
    area_test_probs.parquet   per area x test week, best option: normal, forecast, P(above),
                              call, outcome, citywide and local parts
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from backtest import BAR, auc_ci, skill_ci
from city_level import daily_season, lagged_frame, load_daily
from config import CELL_RES, CLEAN, EVENTS, FIRST_CUTOFF, PANEL, REPORT_LAG_DAYS, TEST_START, WEEK_MODEL
from early_warning import call_metrics, reliability, with_probabilities
from features import log_ratio
from local_week import ago, win
from models import fit_glm, walk_forward

CELL = f"h3_r{CELL_RES}"
AREA = "community_area"
SHARE_FROM = date(2016, 1, 4)    # cell -> area shares use 2016-2021 crime, all before the test period
NORMAL, REF = "normal", "bar_city"
OPTIONS = ["bar_city", "cells_summed", "cells_tilt", "native_glm", "momentum"]
AREA_FEATURES = ["c_7d", "c_28d", "c_91d", "c_364d", "c_prev_364d", "c_accel_7d", "c_accel_28d", "c_accel_91d",
                 "c_yoy", "c_season_1y", "c_log_expected"]


# ---------------------------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------------------------

def area_rows(area_daily: pl.DataFrame, season: pl.DataFrame) -> pl.DataFrame:
    """Per area and cutoff Monday: the target, the area's bar (trailing-year rate x city season)
    and area-level crime features. All crime features end REPORT_LAG_DAYS before the cutoff."""
    w = lambda col, n: win(col, n, over=AREA)
    df = lagged_frame(area_daily, season, over=AREA).with_columns(
        w("known", 7).alias("c_7d"), w("known", 28).alias("c_28d"), w("known", 91).alias("c_91d"),
        w("known", 364).alias("c_364d"),
        ago("known", 364, 364 + REPORT_LAG_DAYS + 1, over=AREA).alias("c_prev_364d"),
        ago("known", 35, 343, over=AREA).alias("_ly_35d"),
        ago("known", 364, 182, over=AREA).alias("_ly_year"),
    ).filter((pl.col("day").dt.weekday() == 1) & (pl.col("day") >= FIRST_CUTOFF))
    return df.select(
        AREA, pl.col("day").alias("cutoff"), "y", pl.col("bar").alias("expected"),
        "c_7d", "c_28d", "c_91d", "c_364d", "c_prev_364d",
        log_ratio(pl.col("c_7d"), pl.col("c_364d") * 7 / 364).alias("c_accel_7d"),
        log_ratio(pl.col("c_28d"), pl.col("c_364d") * 28 / 364).alias("c_accel_28d"),
        log_ratio(pl.col("c_91d"), pl.col("c_364d") * 91 / 364).alias("c_accel_91d"),
        log_ratio(pl.col("c_364d"), pl.col("c_prev_364d")).alias("c_yoy"),
        log_ratio(pl.col("_ly_35d"), pl.col("_ly_year") * 35 / 364).alias("c_season_1y"),
        pl.col("bar").log().alias("c_log_expected"),
    ).sort("cutoff", AREA)


def cell_area_shares() -> pl.DataFrame:
    """Each modeled cell's share of crime records per community area, SHARE_FROM to TEST_START."""
    cells = pl.read_parquet(PANEL / "cells.parquet").select(CELL)
    crimes = pl.read_parquet(CLEAN / "crimes.parquet", columns=["occurred_at", AREA, CELL])
    shares = (
        crimes.join(cells, on=CELL, how="semi")
        .filter(pl.col(AREA).is_not_null()
                & pl.col("occurred_at").dt.date().is_between(SHARE_FROM, TEST_START, closed="left"))
        .group_by(CELL, AREA).len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(CELL)).alias("share"))
        .select(CELL, AREA, "share")
    )
    assert shares[CELL].n_unique() == cells.height, "a modeled cell has no crime with an area"
    return shares


def cells_to_areas(shares: pl.DataFrame) -> pl.DataFrame:
    """Step 3's cell forecasts (already scaled by the citywide call), split into areas."""
    cells = pl.read_parquet(WEEK_MODEL / "local_predictions.parquet").select(CELL, "cutoff", "glm_crime", "bar_city")
    return (
        cells.join(shares, on=CELL)
        .group_by(AREA, "cutoff")
        .agg((pl.col("glm_crime") * pl.col("share")).sum().alias("cells_summed"),
             (pl.col("bar_city") * pl.col("share")).sum().alias("_cells_bar_city"))
    )


def city_call() -> pl.DataFrame:
    """Per test week: `_k` is the city bar's fitted level and `_m` step 2's full citywide call,
    each as a multiple of the raw bar."""
    return pl.read_parquet(WEEK_MODEL / "city_predictions.parquet").select(
        "cutoff", pl.col("fold").alias("_city_fold"), (pl.col("bar_calibrated") / pl.col("bar_raw")).alias("_k"),
        (pl.col("full_forecast") / pl.col("bar_raw")).alias("_m"))


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def compare(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("y") > pl.col(NORMAL)).alias("above"))
    out = []
    for m in OPTIONS:
        s_n, lo_n, hi_n = skill_ci(df, m, NORMAL)
        s_c, lo_c, hi_c = skill_ci(df, m, REF)
        a, alo, ahi = auc_ci(df.with_columns((pl.col(m) / pl.col(NORMAL)).alias("_s")), "_s")
        out.append({"option": m, "skill_vs_normal": s_n, "lo": lo_n, "hi": hi_n,
                    "skill_vs_bar_city": s_c, "lo_c": lo_c, "hi_c": hi_c, "auc": a, "auc_lo": alo, "auc_hi": ahi})
    return pl.DataFrame(out)


def hit_rate_ci(probs: pl.DataFrame, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Share of up/down calls that came true, with a 95% CI from resampling whole folds. Calls
    in the same week share one citywide call and stand or fall together, so rows can't be
    treated as independent."""
    per = (probs.filter(pl.col("call") != "unclear")
           .group_by("fold").agg(((pl.col("call") == "up") == pl.col("above")).sum().alias("right"), pl.len().alias("n")))
    r, n = per["right"].to_numpy(), per["n"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(r), size=(reps, len(r)))
    boot = r[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(r.sum() / n.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def calls_per_week(probs: pl.DataFrame) -> dict:
    """Calls cluster in weeks with a strong citywide call. How spread out are they?"""
    wk = probs.group_by("cutoff").agg((pl.col("call") == "up").sum().alias("up"), (pl.col("call") == "down").sum().alias("down"))
    n = wk.height
    return {"weeks": n, "weeks_with_a_call": wk.filter((pl.col("up") + pl.col("down")) > 0).height / n,
            "median_calls_per_week": float((wk["up"] + wk["down"]).median()),
            "max_calls_in_a_week": int((wk["up"] + wk["down"]).max()),
            "weeks_with_both_directions": wk.filter((pl.col("up") > 0) & (pl.col("down") > 0)).height / n}


def area_names() -> pl.DataFrame:
    path = EVENTS / "raw" / "community_areas.parquet"   # fetched by events_fetch.py permits
    if not path.exists():
        return pl.DataFrame(schema={AREA: pl.Int64, "area_name": pl.String})
    return pl.read_parquet(path).select(pl.col("area_numbe").cast(pl.Int64).alias(AREA),
                                        pl.col("community").str.to_titlecase().alias("area_name"))


# ---------------------------------------------------------------------------------------------

def main() -> None:
    city, area_daily, _ = load_daily()
    season = daily_season(city)
    rows = area_rows(area_daily, season)
    train_rows = rows.filter(pl.col("y").is_not_null())
    assert train_rows["expected"].min() > 0

    native, _ = walk_forward(train_rows, AREA_FEATURES, fit_glm, "native_glm", id_cols=(AREA,))
    momentum, _ = walk_forward(train_rows, ["c_accel_28d"], fit_glm, "momentum", id_cols=(AREA,))
    shares = cell_area_shares()
    keys = [AREA, "cutoff"]

    df = (
        native.select(*keys, "fold", "y", "expected", "native_glm")
        .join(momentum.select(*keys, "momentum"), on=keys)
        .join(city_call(), on="cutoff")
        .join(cells_to_areas(shares), on=keys)
        .with_columns(
            (pl.col("expected") * pl.col("_k")).alias(NORMAL),
            (pl.col("expected") * pl.col("_m")).alias("bar_city"),
            (pl.col("native_glm") * pl.col("_m")).alias("native_glm"),
            (pl.col("momentum") * pl.col("_m")).alias("momentum"),
        )
        .with_columns((pl.col("bar_city") * pl.col("cells_summed") / pl.col("_cells_bar_city")).alias("cells_tilt"))
        .sort("cutoff", AREA)
    )
    assert df.height == 77 * df["cutoff"].n_unique() and df.null_count().sum_horizontal().item() == 0
    assert (df["fold"] == df["_city_fold"]).all(), "step 2 and the area models disagree on folds"

    comparison = compare(df)
    comparison.write_csv(WEEK_MODEL / "area_comparison.csv")

    # Two kinds of call. "Normal": will the area beat its own normal? "Relative": will it beat
    # bar_city, i.e. outpace what the citywide call implies for it? Relative calls isolate what
    # the local tilt can call on its own.
    metrics, rel, all_probs = [], [], {}
    for kind, target in (("normal", NORMAL), ("relative", REF)):
        base = df.with_columns(pl.col(target).alias(BAR))    # early_warning's helpers read the threshold from BAR
        for m in OPTIONS:
            if kind == "relative" and m == REF:
                continue
            probs = with_probabilities(base, m)
            hit, hit_lo, hit_hi = hit_rate_ci(probs)
            metrics.append({"call_vs": kind, **call_metrics(probs, m, base), "hit_rate": hit, "hit_lo": hit_lo,
                            "hit_hi": hit_hi, **calls_per_week(probs), "alpha_last": probs["alpha"][-1]})
            if kind == "normal":
                all_probs[m] = probs
                rel.append(reliability(probs, m))
    metrics, rel = pl.DataFrame(metrics), pl.concat(rel)
    metrics.write_csv(WEEK_MODEL / "area_calls.csv")
    rel.write_csv(WEEK_MODEL / "area_reliability.csv")

    best = metrics.filter(pl.col("call_vs") == "normal").sort("brier")["forecast"][0]
    out = (
        all_probs[best]
        .select(AREA, "cutoff", "fold", "y", pl.col(BAR).alias(NORMAL), pl.col(best).alias("forecast"), "p_above", "call",
                "above", (pl.col("_m") / pl.col("_k") - 1).alias("pct_citywide"),
                (pl.col(best) / pl.col("bar_city") - 1).alias("pct_local"))
        .with_columns((pl.col("forecast") / pl.col(NORMAL) - 1).alias("pct_vs_normal"), pl.lit(best).alias("option"))
        .with_columns(pl.col(AREA).cast(pl.Int64))
        .join(area_names(), on=AREA, how="left")
    )
    out.write_parquet(WEEK_MODEL / "area_test_probs.parquet")

    base_rate = df.select((pl.col("y") > pl.col(NORMAL)).mean()).item()
    called = out.filter(pl.col("call") != "unclear")
    right = called.select(((pl.col("call") == "up") == pl.col("above")).mean()).item()
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=220, float_precision=3, tbl_hide_dataframe_shape=True):
        print(f"77 community areas x {df['cutoff'].n_unique()} test weeks ({df['cutoff'].min()} .. {df['cutoff'].max()}); "
              f"mean {df['y'].mean():.0f} crimes per area-week; {base_rate:.1%} of area-weeks came in above normal\n")
        print("forecast options (deviance skill over the area's normal and over bar_city; within-week AUC):")
        print(comparison)
        print("\ncalls (folds 1-18; up = P >= 0.65, down = P <= 0.35; hit-rate CI by fold):")
        print(metrics.select("call_vs", "forecast", "brier", "brier_skill", "share_called", "hit_rate", "hit_lo", "hit_hi",
                             "up_hit_rate", "down_hit_rate", "weeks_with_a_call", "median_calls_per_week", "max_calls_in_a_week"))
        print(f"\nreliability, {best} (best Brier):")
        print(rel.filter(pl.col("forecast") == best).drop("forecast"))
        print(f"\n{best}: {called.height:,} calls, {right:.1%} right; where calls come from "
              f"(median |citywide part| {called['pct_citywide'].abs().median():.1%}, "
              f"|local part| {called['pct_local'].abs().median():.1%})")
        print("\nlatest test week:")
        last = out.filter(pl.col("cutoff") == out["cutoff"].max()).sort("p_above", descending=True)
        print(pl.concat([last.head(5), last.tail(3)]).select(
            AREA, "area_name", NORMAL, "forecast", "pct_citywide", "pct_local", "p_above", "call", "y"))


if __name__ == "__main__":
    main()
