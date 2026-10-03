"""Step 2: walk-forward backtest scaffolding, baseline forecasts, and scoring.

    offline_ml/.venv/bin/python offline_ml/backtest.py

Every row is a (cell, cutoff) pair: history strictly before the cutoff Monday, target =
all crimes in the HORIZON_WEEKS starting at it. The later model steps reuse `cutoff_frame`,
`make_folds` and `score` from here, so baselines and models are always judged on the same
rows the same way.

Outputs (offline_data/model/):
    baseline_predictions.parquet   per (cell, test cutoff): target and every baseline
    baseline_summary.csv           the scores, overall and by year
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from config import (
    CELL_RES, FIRST_CUTOFF, FOLD_WEEKS, HORIZON_WEEKS, MODEL, PANEL, REPORT_RES, SEASONAL_YEARS, TEST_START,
)

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"
H = HORIZON_WEEKS
PRED_FLOOR = 0.5            # a 0 forecast makes Poisson deviance infinite whenever a crime happens
TOP_SHARE = 0.05            # "where" check: crime captured by the top 5% of cells

BAR = "seasonal_trailing"   # the baseline every model has to beat
BASELINES = {
    BAR: "cell's last 52 weeks, scaled to 4 weeks, x citywide seasonal factor",
    "trailing": "cell's last 52 weeks, scaled to 4 weeks",
    "last_4_weeks": "cell's last 4 weeks (momentum)",
    "same_weeks_last_year": "cell's same 4 weeks a year earlier",
}


# ---------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------

def load_weekly() -> pl.DataFrame:
    return pl.read_parquet(PANEL / "crime_weekly.parquet")


def seasonal_factor(weekly: pl.DataFrame) -> pl.DataFrame:
    """Citywide ratio of the coming 4 weeks to an average 4 weeks, per cutoff week.

    For each of the previous SEASONAL_YEARS years, take the same calendar position and
    divide that year's next-4-week total by its trailing-year 4-week average. The factor is
    the median of those ratios, so one odd year (2020) can't drag it far. Every input sits
    at least 48 weeks before the cutoff, so nothing here comes from the forecast window.
    """
    city = weekly.group_by("week").agg(pl.col("known_n").sum()).sort("week")
    ratio = city.with_columns(
        (pl.col("known_n").rolling_sum(H).shift(-(H - 1))
         / (pl.col("known_n").rolling_sum(52).shift(1) * H / 52)).alias("ratio")
    )
    past = [pl.col("ratio").shift(52 * k).alias(f"r{k}") for k in range(1, SEASONAL_YEARS + 1)]
    return (
        ratio.with_columns(past)
        .with_columns(pl.concat_list([f"r{k}" for k in range(1, SEASONAL_YEARS + 1)]).list.median().alias("season"))
        .select(pl.col("week").alias("cutoff"), "season")
        .drop_nulls()
    )


def cutoff_frame(weekly: pl.DataFrame, require_target: bool = True) -> pl.DataFrame:
    """One row per (cell, cutoff week) with the target and the history the baselines need.

    The weekly grid is dense and sorted by cell then week, so a row's `shift`/`rolling` over
    the cell looks back (or forward) exactly that many weeks. With `require_target=False`,
    cutoffs whose 4 weeks aren't over yet are kept with a null `y`.
    """
    per_cell = lambda e: e.over(CELL)
    rows = weekly.sort(CELL, "week").with_columns(
        per_cell(pl.col("crime_n").rolling_sum(H).shift(-(H - 1))).alias("y"),
        per_cell(pl.col("known_n").rolling_sum(52).shift(1)).alias("hist_52w"),
        per_cell(pl.col("known_n").rolling_sum(H).shift(1)).alias("hist_4w"),
        per_cell(pl.col("known_n").rolling_sum(H).shift(52 - H + 1)).alias("hist_same_weeks_last_year"),
    )
    return (
        rows.rename({"week": "cutoff"})
        .join(seasonal_factor(weekly), on="cutoff", how="inner")
        .filter((pl.col("y").is_not_null() | (not require_target)) & pl.col("hist_52w").is_not_null())
        .select(CELL, PARENT, "cutoff", "y", "hist_52w", "hist_4w", "hist_same_weeks_last_year", "season")
    )


def add_baselines(rows: pl.DataFrame) -> pl.DataFrame:
    trailing = pl.col("hist_52w") * H / 52
    return rows.with_columns(
        (trailing * pl.col("season")).alias(BAR),
        trailing.alias("trailing"),
        pl.col("hist_4w").cast(pl.Float64).alias("last_4_weeks"),
        pl.col("hist_same_weeks_last_year").cast(pl.Float64).alias("same_weeks_last_year"),
    ).with_columns(pl.col(m).clip(lower_bound=PRED_FLOOR) for m in BASELINES)


def make_folds(cutoffs: list[date]) -> list[dict]:
    """Walk-forward folds: each tests FOLD_WEEKS consecutive cutoffs and may train only on
    cutoffs whose whole target window ends before the first test cutoff."""
    test = sorted(c for c in cutoffs if c >= TEST_START)
    folds = []
    for i in range(0, len(test), FOLD_WEEKS):
        block = test[i:i + FOLD_WEEKS]
        train_last = block[0] - timedelta(weeks=H)
        folds.append({"fold": len(folds), "train_first": FIRST_CUTOFF, "train_last": train_last,
                      "test_first": block[0], "test_last": block[-1]})
        assert train_last + timedelta(weeks=H) <= block[0]
    return folds


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------

def poisson_deviance(y: str, mu: str) -> pl.Expr:
    y_, mu_ = pl.col(y), pl.col(mu)
    return 2 * (pl.when(y_ > 0).then(y_ * (y_ / mu_).log()).otherwise(0.0) - (y_ - mu_))


def mean_auc(df: pl.DataFrame, label: str, score: str, by: str) -> float:
    """AUC computed within each `by` group (Mann-Whitney on ranks), then averaged.

    Within a cutoff, it asks: of the cells that ended above their normal, how often did the
    forecast rank them above cells that didn't? A score that's constant within a cutoff
    (like the bar itself) gets exactly 0.5.
    """
    g = (
        df.with_columns(pl.col(score).rank("average").over(by).alias("_r"))
        .group_by(by)
        .agg(pl.col(label).sum().alias("pos"), pl.len().alias("n"), pl.col("_r").filter(pl.col(label)).sum().alias("rsum"))
        .filter((pl.col("pos") > 0) & (pl.col("pos") < pl.col("n")))
        .with_columns(((pl.col("rsum") - pl.col("pos") * (pl.col("pos") + 1) / 2)
                       / (pl.col("pos") * (pl.col("n") - pl.col("pos")))).alias("auc"))
    )
    return g["auc"].mean()


def top_capture(df: pl.DataFrame, score: str) -> float:
    """Share of crime falling in the top TOP_SHARE of cells by `score`, averaged over cutoffs.
    H3 cells are equal-area, so this is also the share of crime in 5% of the city's area."""
    k = math.ceil(df.group_by("cutoff").len()["len"].max() * TOP_SHARE)
    g = df.with_columns(pl.col(score).rank("ordinal", descending=True).over("cutoff").alias("_rk")).group_by("cutoff").agg(
        (pl.col("y").filter(pl.col("_rk") <= k).sum() / pl.col("y").sum()).alias("cap"))
    return g["cap"].mean()


def noise_floor(mu: np.ndarray, draws: int = 20, seed: int = 0) -> float:
    """Mean deviance a forecast would get if it were exactly the true mean and crime were
    Poisson around it. Real crime is burstier than Poisson, so the true floor is higher and
    `deviance - floor` is an upper bound on what any model could still win."""
    rng = np.random.default_rng(seed)
    y = rng.poisson(mu, size=(draws, mu.size)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def skill_ci(rows: pl.DataFrame, model: str, ref: str, block: str = "fold",
             reps: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Deviance skill of `model` over `ref` (1 - D_model / D_ref) with a bootstrap CI.

    Resamples whole blocks (quarterly folds by default), not rows: rows in the same quarter
    share overlapping 4-week targets and citywide shocks, so treating them as independent
    would make the interval far too narrow.
    """
    per = rows.group_by(block).agg(
        poisson_deviance("y", model).sum().alias("m"), poisson_deviance("y", ref).sum().alias("r"))
    m, r = per["m"].to_numpy(), per["r"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(m), size=(reps, len(m)))
    boot = 1 - m[idx].sum(axis=1) / r[idx].sum(axis=1)
    tail = (1 - level) / 2
    return float(1 - m.sum() / r.sum()), float(np.quantile(boot, tail)), float(np.quantile(boot, 1 - tail))


def auc_ci(rows: pl.DataFrame, score_col: str, label: str = "above", block: str = "fold",
           reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean within-cutoff AUC with a 95% CI from resampling whole blocks of cutoffs."""
    per_cutoff = (
        rows.with_columns(pl.col(score_col).rank("average").over("cutoff").alias("_r"))
        .group_by("cutoff", block)
        .agg(pl.col(label).sum().alias("pos"), pl.len().alias("n"), pl.col("_r").filter(pl.col(label)).sum().alias("rsum"))
        .filter((pl.col("pos") > 0) & (pl.col("pos") < pl.col("n")))
        .with_columns(((pl.col("rsum") - pl.col("pos") * (pl.col("pos") + 1) / 2)
                       / (pl.col("pos") * (pl.col("n") - pl.col("pos")))).alias("auc"))
    )
    per = per_cutoff.group_by(block).agg(pl.col("auc").sum().alias("s"), pl.len().alias("k"))
    s, k = per["s"].to_numpy(), per["k"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    boot = s[idx].sum(axis=1) / k[idx].sum(axis=1)
    return float(s.sum() / k.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def score(rows: pl.DataFrame, models: list[str], ref: str = BAR) -> pl.DataFrame:
    """Scores at the model grain (r8) and summed to REPORT_RES, overall and by year.

    - deviance / skill: Poisson deviance, and 1 - deviance / ref's deviance
    - auc_above_normal: within-cutoff AUC for "the next 4 weeks come in above the bar's
      expectation", scored by forecast / expectation
    - top5_capture: share of crime in the top 5% of cells (model grain only)
    """
    rolled = rows.group_by(PARENT, "cutoff").agg(pl.col("y", *models).sum())
    out = []
    for grain, df in ((f"r{CELL_RES}", rows), (f"r{REPORT_RES}", rolled)):
        df = df.with_columns((pl.col("y") > pl.col(BAR)).alias("above"), pl.col("cutoff").dt.year().alias("year"))
        for year, part in [("all", df), *((y, df.filter(pl.col("year") == y)) for y in sorted(df["year"].unique()))]:
            dev = {m: part.select(poisson_deviance("y", m).mean()).item() for m in models}
            floor = noise_floor(part[ref].to_numpy())
            for m in models:
                out.append({
                    "grain": grain, "year": str(year), "model": m, "rows": part.height,
                    "deviance": dev[m], "skill_vs_bar": 1 - dev[m] / dev[ref], "noise_floor": floor,
                    "auc_above_normal": mean_auc(part.with_columns((pl.col(m) / pl.col(BAR)).alias("_s")), "above", "_s", "cutoff"),
                    "top5_capture": top_capture(part, m) if grain == f"r{CELL_RES}" else None,
                })
    return pl.DataFrame(out)


def main() -> None:
    MODEL.mkdir(exist_ok=True)
    weekly = load_weekly()
    rows = add_baselines(cutoff_frame(weekly))
    folds = make_folds(rows["cutoff"].unique().to_list())
    test = rows.filter(pl.col("cutoff") >= TEST_START)

    # Leak check: recompute target and history for a few (cell, cutoff) pairs straight from
    # the weekly grid with explicit date bounds.
    for cell, cutoff, y, hist in test.sample(5, seed=1).select(CELL, "cutoff", "y", "hist_52w").iter_rows():
        wk = weekly.filter(pl.col(CELL) == cell)
        target = wk.filter(pl.col("week").is_between(cutoff, cutoff + timedelta(weeks=H), closed="left"))["crime_n"].sum()
        past = wk.filter(pl.col("week").is_between(cutoff - timedelta(weeks=52), cutoff, closed="left"))["known_n"].sum()
        assert (target, past) == (y, hist), (cell, cutoff, target, y, past, hist)

    test.write_parquet(MODEL / "baseline_predictions.parquet")
    summary = score(test, list(BASELINES))
    summary.write_csv(MODEL / "baseline_summary.csv")

    city = test.group_by("cutoff").agg(pl.col("y", BAR, "trailing").sum())
    city_err = lambda m: (city[m] / city["y"] - 1).abs().mean()
    print(f"test cutoffs {test['cutoff'].min()} .. {test['cutoff'].max()} ({test['cutoff'].n_unique()} weeks, "
          f"{len(folds)} quarterly folds), {test.height:,} cell-cutoff rows, mean target {test['y'].mean():.1f}")
    print(f"citywide 4-week total, mean abs error: seasonal_trailing {city_err(BAR):.1%}, trailing {city_err('trailing'):.1%}\n")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True):
        print(summary.filter(pl.col("year") == "all").drop("year", "rows"))
        print("\nby year, r8 (bar vs momentum):")
        print(summary.filter((pl.col("grain") == f"r{CELL_RES}") & (pl.col("year") != "all")
                             & pl.col("model").is_in([BAR, "last_4_weeks"]))
              .select("year", "model", "deviance", "noise_floor", "skill_vs_bar", "auc_above_normal", "top5_capture"))


if __name__ == "__main__":
    main()
