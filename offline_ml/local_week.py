"""1-week track, step 3: which cells run hot or cold next week, relative to the rest of the city?

    offline_ml/.venv/bin/python offline_ml/city_level.py    # first: the citywide multiplier
    offline_ml/.venv/bin/python offline_ml/local_week.py    # ~5 min

The same split as the 4-week track's step 4. city_level.py sets next week's citywide total, and
the models here decide how it spreads across the 770 r8 cells:
- **The cell's bar:** its trailing-year daily rate x the city's daily seasonal factor over the
  target week (city_level's construction, per cell).
- **The reference to beat, `bar_city`:** that bar x city_level's multiplier for the week
  (full_forecast / its bar). It already carries the citywide call, so any skill over it is local.
- **Training** pins each week's expected counts to the actual citywide total, so the models learn
  only which cells run hot or cold relative to the rest. models.py's `walk_forward`, `fit_glm`
  and `fit_gbm` are reused unchanged. Its folds leave a 4-week gap before each test quarter, one
  week more than the reporting lag strictly needs.

**Everything respects the reporting lag.** Crime features only use days before c -
REPORT_LAG_DAYS. The weekly grid can't express that (the last usable day is a Saturday), so the
features are rolling windows over daily counts.

**311 gets a new role.** The 311 feed is close to real time: on 2026-09-26 it already held
requests from that morning. So 311 can see the gap week that crime can't, and its windows end
SR_LAG_DAYS before the cutoff. The 4-week ablation found 311 adds nothing when both feeds are
equally current. `*_311` variants test whether that changes when 311 is 7 days fresher.

Outputs (offline_data/model/h1/):
    local_features.parquet      per (cell, cutoff): target, bar, every feature
    local_predictions.parquet   per test (cell, cutoff): target, bar, bar_city, every model
    local_summary.csv           scores at r8 and r7, overall and by year
    local_ranking.csv           ranking above-normal: GLM vs raw momentum
    local_glm_coefficients.csv  final-fold GLM coefficients (% per SD)
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from h3.api import basic_int as h3

from backtest import auc_ci, noise_floor, poisson_deviance, skill_ci
from city_level import daily_season, lagged_frame, load_daily
from config import CELL_RES, CLEAN, FIRST_CUTOFF, PANEL, REPORT_LAG_DAYS, REPORT_RES, TEST_START, WEEK_MODEL
from features import log_ratio
from models import fit_gbm, fit_glm, walk_forward

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"
L = REPORT_LAG_DAYS
SR_LAG_DAYS = 1                 # 311 windows end the day before the cutoff (the feed is ~real time)
GRID_FROM = date(2017, 1, 2)    # far enough back for 2-year-old seasonal windows at FIRST_CUTOFF
LEAK_CHECK_CUTOFF = date(2024, 6, 17)
# The 4-week track's floor of 0.5, per week. 14 cells have stretches with no crime in the trailing
# year, and a zero forecast makes Poisson deviance infinite whenever one happens.
EXPECTED_FLOOR = 0.125

REF = "bar_city"
MODELS = {                      # name -> (fit, feature prefixes)
    "glm_crime": (fit_glm, ("c_", "n1_", "n2_")),
    "gbm_crime": (fit_gbm, ("c_", "n1_", "n2_")),
    "glm_crime_311": (fit_glm, ("c_", "n1_", "n2_", "s_", "sn1_")),
    "gbm_crime_311": (fit_gbm, ("c_", "n1_", "n2_", "s_", "sn1_")),
}


# ---------------------------------------------------------------------------------------------
# Daily cell counts
# ---------------------------------------------------------------------------------------------

def load_events() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cells = pl.read_parquet(PANEL / "cells.parquet").select(CELL, PARENT)
    crimes = pl.read_parquet(CLEAN / "crimes.parquet", columns=[
        "occurred_at", "case_year", "ucr_class", "location_group", "domestic", CELL]).join(cells, on=CELL, how="semi")
    sr = pl.read_parquet(CLEAN / "sr311.parquet", columns=["created_at", "origin_group", CELL]).join(cells, on=CELL, how="semi")
    return cells, crimes, sr


def daily_cells(cells: pl.DataFrame, crimes: pl.DataFrame, sr: pl.DataFrame, last_day: date) -> pl.DataFrame:
    """Dense cell x day counts from GRID_FROM to `last_day`. `n` counts every crime (the target).
    The `known*` columns leave out crimes whose case was opened in a later calendar year, as
    panel.py does. `sr*` count 311 requests by the day they were created."""
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    lg = pl.col("location_group")
    c = (
        crimes.with_columns(pl.col("occurred_at").dt.date().alias("day"))
        .filter(pl.col("day").is_between(GRID_FROM, last_day))
        .group_by(CELL, "day")
        .agg(pl.len().alias("n"), known.sum().alias("known"),
             (known & (pl.col("ucr_class") == "violent")).sum().alias("known_violent"),
             (known & (pl.col("ucr_class") == "property")).sum().alias("known_property"),
             (known & (lg == "residential")).sum().alias("_res"), (known & (lg == "street")).sum().alias("_street"),
             (known & (lg == "commercial")).sum().alias("_comm"), (known & pl.col("domestic")).sum().alias("_dom"))
    )
    s = (
        sr.with_columns(pl.col("created_at").dt.date().alias("day"))
        .filter(pl.col("day").is_between(GRID_FROM, last_day))
        .group_by(CELL, "day")
        .agg(pl.len().alias("sr"), (pl.col("origin_group") == "resident").sum().alias("sr_resident"))
    )
    days = pl.date_range(GRID_FROM, last_day, "1d", eager=True).to_frame("day")
    grid = cells.join(days, how="cross")
    out = grid.join(c, on=[CELL, "day"], how="left").join(s, on=[CELL, "day"], how="left")
    counts = [x for x in out.columns if x not in (CELL, PARENT, "day")]
    return out.with_columns(pl.col(counts).fill_null(0).cast(pl.Int32)).sort(CELL, "day")


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

def win(col: str, days: int, lag: int = L, over: str = CELL) -> pl.Expr:
    """Sum of `col` over the `days` days ending just before cutoff - lag, per `over` group."""
    return pl.col(col).rolling_sum(days).shift(lag + 1).over(over)


def ago(col: str, days: int, end_days_before: int, over: str = CELL) -> pl.Expr:
    """Sum of `col` over the `days` days ending `end_days_before` days before the cutoff."""
    return pl.col(col).rolling_sum(days).shift(end_days_before).over(over)


def cell_block(daily: pl.DataFrame, season: pl.DataFrame) -> pl.DataFrame:
    """Per cell and cutoff Monday: target, bar, and the cell's own crime and 311 history."""
    df = lagged_frame(daily, season, over=CELL).with_columns(
        win("known", 7).alias("c_7d"),
        win("known", 28).alias("c_28d"),
        win("known", 91).alias("c_91d"),
        win("known", 364).alias("c_364d"),
        ago("known", 364, L + 1 + 364).alias("c_prev_364d"),
        win("known_violent", 91).alias("c_violent_91d"),
        win("known_violent", 364).alias("c_violent_364d"),
        win("known_property", 91).alias("c_property_91d"),
        win("known_property", 364).alias("c_property_364d"),
        *(win(src, 364).alias(f"_{src}_364d") for src in ("_res", "_street", "_comm", "_dom")),
        # The cell's own seasonal bump at this time of year: the 5 weeks around the target week,
        # 1 and 2 years back, against the year centred on them. The city's shape is in the bar
        # already; this is how much more (or less) seasonal the cell is.
        ago("known", 35, 343).alias("_ly_35d"),
        ago("known", 364, 182).alias("_ly_year"),
        ago("known", 35, 343 + 364).alias("_ly2_35d"),
        ago("known", 364, 182 + 364).alias("_ly2_year"),
        # 311, ending SR_LAG_DAYS before the cutoff: the 7-day window is the week crime can't see
        win("sr", 7, SR_LAG_DAYS).alias("s_7d"),
        win("sr", 28, SR_LAG_DAYS).alias("s_28d"),
        win("sr", 364, SR_LAG_DAYS).alias("s_364d"),
        win("sr_resident", 28, SR_LAG_DAYS).alias("_sr_res_28d"),
    ).filter((pl.col("day").dt.weekday() == 1) & (pl.col("day") >= FIRST_CUTOFF))
    share = lambda c: pl.when(pl.col("c_364d") > 0).then(pl.col(c) / pl.col("c_364d")).otherwise(0.0)
    return df.select(
        CELL, PARENT, pl.col("day").alias("cutoff"), "y",
        pl.col("bar").clip(lower_bound=EXPECTED_FLOOR).alias("expected"),
        pl.col("bar").clip(lower_bound=EXPECTED_FLOOR).log().alias("c_log_expected"),
        "c_7d", "c_28d", "c_91d", "c_364d", "c_prev_364d",
        "c_violent_91d", "c_violent_364d", "c_property_91d", "c_property_364d",
        log_ratio(pl.col("c_7d"), pl.col("c_364d") * 7 / 364).alias("c_accel_7d"),
        log_ratio(pl.col("c_28d"), pl.col("c_364d") * 28 / 364).alias("c_accel_28d"),
        log_ratio(pl.col("c_91d"), pl.col("c_364d") * 91 / 364).alias("c_accel_91d"),
        log_ratio(pl.col("c_364d"), pl.col("c_prev_364d")).alias("c_yoy"),
        log_ratio(pl.col("c_violent_91d"), pl.col("c_violent_364d") * 91 / 364).alias("c_violent_accel_91d"),
        log_ratio(pl.col("_ly_35d"), pl.col("_ly_year") * 35 / 364).alias("c_season_1y"),
        log_ratio(pl.col("_ly2_35d"), pl.col("_ly2_year") * 35 / 364).alias("c_season_2y"),
        share("__res_364d").alias("c_share_residential"),
        share("__street_364d").alias("c_share_street"),
        share("__comm_364d").alias("c_share_commercial"),
        share("__dom_364d").alias("c_share_domestic"),
        "s_7d", "s_28d", "s_364d",
        log_ratio(pl.col("s_7d"), pl.col("s_364d") * 7 / 364).alias("s_accel_7d"),
        log_ratio(pl.col("s_28d"), pl.col("s_364d") * 28 / 364).alias("s_accel_28d"),
        pl.when(pl.col("s_28d") > 0).then(pl.col("_sr_res_28d") / pl.col("s_28d")).otherwise(0.0).alias("s_resident_share_28d"),
    )


def neighbor_block(own: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    """Sums over the ring-1 (6) and ring-2 (12) neighbors that are modeled cells."""
    ids = set(cells[CELL].to_list())
    pairs = pl.DataFrame(
        [(c, n, k) for c in ids for k in (1, 2) for n in h3.grid_ring(c, k) if n in ids],
        schema={CELL: pl.Int64, "nbr": pl.Int64, "ring": pl.Int32}, orient="row",
    )
    cols = ["c_7d", "c_28d", "c_364d", "s_7d", "s_364d"]
    sums = (
        pairs.join(own.select(pl.col(CELL).alias("nbr"), "cutoff", *cols), on="nbr")
        .group_by(CELL, "cutoff", "ring").agg(pl.col(cols).sum())
        .pivot(on="ring", index=[CELL, "cutoff"], values=cols)
    )
    rn = lambda col, k: f"{col}_{k}"
    out = own.select(CELL, "cutoff").join(sums, on=[CELL, "cutoff"], how="left").fill_null(0)
    return out.select(
        CELL, "cutoff",
        *(pl.col(rn(c, k)).alias(f"n{k}_{c}") for k in (1, 2) for c in ("c_7d", "c_28d", "c_364d")),
        *(log_ratio(pl.col(rn("c_28d", k)), pl.col(rn("c_364d", k)) * 28 / 364).alias(f"n{k}_c_accel_28d") for k in (1, 2)),
        log_ratio(pl.col(rn("c_7d", 1)), pl.col(rn("c_364d", 1)) * 7 / 364).alias("n1_c_accel_7d"),
        pl.col(rn("s_7d", 1)).alias("sn1_7d"),
        log_ratio(pl.col(rn("s_7d", 1)), pl.col(rn("s_364d", 1)) * 7 / 364).alias("sn1_accel_7d"),
    )


def build_features(cells: pl.DataFrame, crimes: pl.DataFrame, sr: pl.DataFrame,
                   season: pl.DataFrame, last_day: date) -> pl.DataFrame:
    own = cell_block(daily_cells(cells, crimes, sr, last_day), season)
    return own.join(neighbor_block(own, cells), on=[CELL, "cutoff"], how="left").sort("cutoff", CELL)


def feature_columns(df: pl.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]


# ---------------------------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------------------------

def check_truncation(full: pl.DataFrame, cells, crimes, sr, season, last_day: date) -> None:
    """Rebuild from the data as it stood at LEAK_CHECK_CUTOFF: crime only before cutoff - L, 311
    only before cutoff - SR_LAG_DAYS. Every feature and the bar must match for every cutoff up to
    it. The check is tight: cutoff - L is exactly the first deleted crime day. It's also not
    vacuous, since a week later the features do change."""
    t = LEAK_CHECK_CUTOFF
    crime_end, sr_end = (pl.lit(t - timedelta(days=d)).cast(pl.Datetime("us")) for d in (L, SR_LAG_DAYS))
    again = build_features(cells, crimes.filter(pl.col("occurred_at") < crime_end),
                           sr.filter(pl.col("created_at") < sr_end), season, last_day)
    cols = ["expected", *feature_columns(full, ("c_", "n1_", "n2_", "s_", "sn1_"))]
    pick = lambda df, when: df.filter(pl.col("cutoff") <= when).select(CELL, "cutoff", *cols).sort(CELL, "cutoff")
    a, b = pick(full, t), pick(again, t)
    assert a.height == b.height > 100_000
    bad = [c for c in cols if not a[c].fill_nan(None).equals(b[c].fill_nan(None))]
    assert not bad, f"features changed when the future was deleted: {bad}"
    nxt = lambda df: df.filter(pl.col("cutoff") == t + timedelta(weeks=1)).sort(CELL)
    assert not nxt(full)["c_7d"].equals(nxt(again)["c_7d"]) and not nxt(full)["s_7d"].equals(nxt(again)["s_7d"])
    print(f"  truncation check: {len(cols)} columns identical for {a.height:,} rows up to {t} when crime from "
          f"{t - timedelta(days=L)} and 311 from {t - timedelta(days=SR_LAG_DAYS)} on are deleted")


def check_direct(full: pl.DataFrame, cells: pl.DataFrame, crimes: pl.DataFrame, sr: pl.DataFrame) -> None:
    """Recompute a few features from the event tables with explicit timestamp bounds."""
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    ids = set(cells[CELL].to_list())
    ts = lambda d: pl.lit(d).cast(pl.Datetime("us"))
    for row in full.filter(pl.col("cutoff") >= TEST_START).sample(8, seed=4).iter_rows(named=True):
        cell, c = row[CELL], row["cutoff"]
        crime_win = lambda n: pl.col("occurred_at").is_between(ts(c - timedelta(days=L + n)), ts(c - timedelta(days=L)), closed="left")
        sr_win = lambda n: pl.col("created_at").is_between(
            ts(c - timedelta(days=SR_LAG_DAYS + n)), ts(c - timedelta(days=SR_LAG_DAYS)), closed="left")
        nbrs = [n for n in h3.grid_ring(cell, 1) if n in ids]
        got = {
            "c_28d": crimes.filter((pl.col(CELL) == cell) & crime_win(28) & known).height,
            "n1_c_7d": crimes.filter(pl.col(CELL).is_in(nbrs) & crime_win(7) & known).height,
            "s_7d": sr.filter((pl.col(CELL) == cell) & sr_win(7)).height,
            "y": crimes.filter((pl.col(CELL) == cell) & pl.col("occurred_at").is_between(
                ts(c), ts(c + timedelta(days=7)), closed="left")).height,
        }
        for k, v in got.items():
            assert row[k] == v, (cell, c, k, row[k], v)
    print(f"  direct check: {len(got)} columns match a from-scratch recount for 8 random test rows")


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def with_city(preds: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """Scale the cell bar and every model by city_level's call for that week.

    `bar_calibrated` is the cell bar x the city bar's fitted level (no weekly call). `bar_city`
    adds the weekly citywide call. The models were rescaled to the bar's total within each
    cutoff, so multiplying by one number per week keeps their local split intact.
    """
    city = pl.read_parquet(WEEK_MODEL / "city_predictions.parquet").select(
        "cutoff", (pl.col("full_forecast") / pl.col("bar_raw")).alias("_m"),
        (pl.col("bar_calibrated") / pl.col("bar_raw")).alias("_k"))
    return preds.join(city, on="cutoff").select(
        CELL, PARENT, "cutoff", "fold", "y",
        (pl.col("expected") * pl.col("_k")).alias("bar_calibrated"),
        (pl.col("expected") * pl.col("_m")).alias(REF),
        *((pl.col(m) * pl.col("_m")).alias(m) for m in models),
    )


def score(preds: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """At r8 and summed to r7, overall and by year:
    - skill over bar_city (deviance), with a 95% CI from resampling whole folds
    - skill over bar_calibrated, which also credits the citywide call
    - within-week AUC for "above normal" (actual > bar_calibrated), scored by model / bar
    - the share of bar_city's deviance that is pure Poisson chance
    """
    rolled = preds.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", "bar_calibrated", REF, *models).sum())
    out = []
    for grain, df in ((f"r{CELL_RES}", preds), (f"r{REPORT_RES}", rolled)):
        df = df.with_columns((pl.col("y") > pl.col("bar_calibrated")).alias("above"),
                             pl.col("cutoff").dt.year().cast(pl.String).alias("year"))
        for year, part in [("all", df), *((y, df.filter(pl.col("year") == y)) for y in sorted(df["year"].unique()))]:
            ref_dev = part.select(poisson_deviance("y", REF).mean()).item()
            floor = noise_floor(part[REF].to_numpy())
            for m in [REF, *models]:
                s, lo, hi = skill_ci(part, m, REF) if year == "all" else (
                    1 - part.select(poisson_deviance("y", m).mean()).item() / ref_dev, None, None)
                auc = auc_ci(part.with_columns((pl.col(m) / pl.col("bar_calibrated")).alias("_s")), "_s")
                out.append({
                    "grain": grain, "year": year, "model": m, "skill_vs_bar_city": s, "ci_lo": lo, "ci_hi": hi,
                    "skill_vs_bar_calibrated": 1 - part.select(poisson_deviance("y", m).mean()).item()
                    / part.select(poisson_deviance("y", "bar_calibrated").mean()).item(),
                    "auc_above_normal": auc[0], "chance_share": floor / ref_dev,
                })
    return pl.DataFrame(out)


def momentum_ranking(preds: pl.DataFrame, full: pl.DataFrame) -> pl.DataFrame:
    """Within-week AUC for "above normal" when ranking by raw momentum (the last 28 or 7 known
    days against the trailing-year rate), next to the GLM. It's ranking only, with no forecast
    to score. In the 4-week track, momentum out-ranked the models at r7, so step 4's calls
    should consider it."""
    d = preds.join(full.select(CELL, "cutoff", "c_7d", "c_28d", "c_364d"), on=[CELL, "cutoff"])
    rolled = d.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", "bar_calibrated", "glm_crime", "c_7d", "c_28d", "c_364d").sum())
    out = []
    for grain, df in ((f"r{CELL_RES}", d), (f"r{REPORT_RES}", rolled)):
        df = df.with_columns(
            (pl.col("y") > pl.col("bar_calibrated")).alias("above"),
            (pl.col("glm_crime") / pl.col("bar_calibrated")).alias("glm_crime"),
            log_ratio(pl.col("c_28d"), pl.col("c_364d") * 28 / 364).alias("momentum_28d"),
            log_ratio(pl.col("c_7d"), pl.col("c_364d") * 7 / 364).alias("momentum_7d"))
        for score_col in ("glm_crime", "momentum_28d", "momentum_7d"):
            auc, lo, hi = auc_ci(df, score_col)
            out.append({"grain": grain, "ranked_by": score_col, "auc_above_normal": auc, "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------------------------

def main() -> None:
    WEEK_MODEL.mkdir(parents=True, exist_ok=True)
    city, _, last_day = load_daily()
    season = daily_season(city)
    cells, crimes, sr = load_events()

    t0 = time.time()
    full = build_features(cells, crimes, sr, season, last_day)
    full.write_parquet(WEEK_MODEL / "local_features.parquet")
    print(f"{full.height:,} rows: {cells.height} cells x {full['cutoff'].n_unique()} cutoffs "
          f"({full['cutoff'].min()} .. {full['cutoff'].max()}), built in {time.time() - t0:.0f}s")
    print("leakage checks:")
    check_direct(full, cells, crimes, sr)
    check_truncation(full, cells, crimes, sr, season, last_day)

    rows = full.filter(pl.col("y").is_not_null())
    preds, glm_coefs = None, None
    for name, (fit, prefixes) in MODELS.items():
        cols = feature_columns(rows, prefixes)
        t0 = time.time()
        p, infos = walk_forward(rows, cols, fit, name)
        print(f"  {name}: {len(cols)} features, {len(infos)} folds, {time.time() - t0:.0f}s")
        preds = p if preds is None else preds.join(p.select(CELL, "cutoff", name), on=[CELL, "cutoff"])
        if name == "glm_crime":
            glm_coefs = infos[-1]["coefs"]

    models = list(MODELS)
    preds = with_city(preds, models)
    preds.write_parquet(WEEK_MODEL / "local_predictions.parquet")
    summary = score(preds, models)
    summary.write_csv(WEEK_MODEL / "local_summary.csv")
    ranking = momentum_ranking(preds, full)
    ranking.write_csv(WEEK_MODEL / "local_ranking.csv")
    coef_table = pl.DataFrame({"feature": list(glm_coefs), "coef": list(glm_coefs.values())}).with_columns(
        ((pl.col("coef").exp() - 1) * 100).alias("pct_per_sd")).sort(pl.col("coef").abs(), descending=True)
    coef_table.write_csv(WEEK_MODEL / "local_glm_coefficients.csv")

    rolled = preds.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", *models).sum())
    print(f"\ntest {preds['cutoff'].min()} .. {preds['cutoff'].max()}, {preds.height:,} cell-weeks, "
          f"mean {preds['y'].mean():.1f} crimes per cell-week")
    print("what 7-days-fresher 311 adds (paired, same folds):")
    for base in ("glm", "gbm"):
        for grain, df in ((f"r{CELL_RES}", preds), (f"r{REPORT_RES}", rolled)):
            s, lo, hi = skill_ci(df, f"{base}_crime_311", f"{base}_crime")
            print(f"  {base} at {grain}: {s:+.2%} (95% CI {lo:+.2%} to {hi:+.2%})")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True):
        print()
        print(summary.filter(pl.col("year") == "all").drop("year"))
        print("\nranking above-normal cells/areas, GLM vs raw momentum (within-week AUC, 95% CI by fold):")
        print(ranking)
        print("\nby year, skill over bar_city:")
        print(summary.filter((pl.col("year") != "all") & (pl.col("model") != REF))
              .pivot(on="model", index=["grain", "year"], values="skill_vs_bar_city"))
        print("\nglm_crime, final fold: effect on the forecast of a 1-SD higher feature, top 12:")
        print(coef_table.head(12).select("feature", "pct_per_sd"))


if __name__ == "__main__":
    main()
