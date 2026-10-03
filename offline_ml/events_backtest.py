"""Backtest the festival look-ahead: would it have ranked past years' festivals well?

    offline_ml/.venv/bin/python offline_ml/events_backtest.py

Each year from 2017 to 2025 is replayed as of January 1st. Every festival held that year is
scored by events_lookahead.festival_scores using only festivals and crime from before then.
The scores are compared with what happened: observed crime in the festival's cells and the
ring around them, minus the expected count events_impact.py built from the weeks around it
(its controlled counterfactual). That difference is each festival's actual extra crime.

Rankings compared, each within its year:
    full           baseline x (shrunk ratio - 1), what events_lookahead.py ships
    no_history     baseline x (size prior - 1), dropping each festival's own past editions
    baseline_only  baseline alone: how much crime the area normally has on those dates
    size_only      the size prior alone (street segments)

Output: offline_data/events/lookahead/backtest.csv, one row per festival-year, plus printed metrics.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from config import EVENTS
from events_impact import busy_cell_days
from events_lookahead import OUT, event_oe, festival_scores, load_counts

YEARS = range(2017, 2026)      # three years of festival history before the first test year
TOP_SHARE = 0.2
BOOTSTRAP = 2000
SEED = 0
MODELS = ["full", "no_history", "baseline_only", "size_only"]


def replay(events, cells, oe, busy, counts) -> pl.DataFrame:
    festivals = events.filter((pl.col("status") == "held") & (pl.col("category") == "festival") & ~pl.col("long_run"))
    parts = []
    for year in YEARS:
        known_until = date(year, 1, 1) - timedelta(days=1)
        targets = festivals.filter(pl.col("start_date").dt.year() == year)
        scored, cv, _ = festival_scores(targets, events, cells, oe, busy, counts, known_until)
        parts.append(scored.select(
            "event_id", "name", "n_segments", "past_editions", "baseline_crime", "prior_crime", "ratio_crime",
            year=pl.lit(year), cv=pl.lit(cv),
            full=pl.col("extra_crime"),
            no_history=pl.col("baseline_crime") * (pl.col("prior_crime") - 1),
            baseline_only=pl.col("baseline_crime"),
            size_only=pl.col("prior_crime")))
        print(f"{year}: {targets.height} festivals scored as of {known_until}, cv {cv:.3f}")
    actual = event_oe(oe, festivals).select("event_id", actual=pl.col("O_crime") - pl.col("E_crime"),
                                            actual_O=pl.col("O_crime"), actual_E=pl.col("E_crime"))
    # festivals events_impact.py couldn't score (too few clean control days) have no actual.
    # Sorted, because join order isn't guaranteed and the seeded bootstrap weights go by row.
    return pl.concat(parts).join(actual, on="event_id").sort("year", "event_id")


def top_mask(df: pl.DataFrame, model: str) -> np.ndarray:
    """True for each year's top TOP_SHARE by the model's score. size_only has three distinct values
    a year, so ties are broken by event id: fixed, and unrelated to the outcome. (Polars' seeded
    random rank isn't reproducible inside .over().)"""
    r = (df.with_row_index("i").sort("event_id")
         .with_columns(r=pl.col(model).rank(method="ordinal", descending=True).over("year"), n=pl.len().over("year"))
         .sort("i"))
    return (r["r"] <= (r["n"] * TOP_SHARE).ceil()).to_numpy()


def metrics(df: pl.DataFrame) -> None:
    actual = df["actual"].to_numpy()
    O, E = df["actual_O"].to_numpy(), df["actual_E"].to_numpy()
    tops = {m: top_mask(df, m) for m in MODELS}
    print(f"\n{df.height:,} festival-years with a measured outcome; actual extra crimes in total: {actual.sum():.1f}")
    print(f"{'ranking':15} {'spearman':>9} {'top 20% capture':>16} {'top 20% O/E':>12} {'rest O/E':>9}")
    for m in MODELS:
        t = tops[m]
        rho = spearmanr(df[m].to_numpy(), actual).statistic
        print(f"{m:15} {rho:9.3f} {actual[t].sum() / actual.sum():16.1%} {O[t].sum() / E[t].sum():12.3f} "
              f"{O[~t].sum() / E[~t].sum():9.3f}")

    # paired bootstrap over festivals: does the full ranking capture more than baseline alone?
    rng = np.random.default_rng(SEED)
    w = rng.poisson(1.0, size=(BOOTSTRAP, len(actual))).astype(np.float64)
    tot = w @ actual

    def capture(m):
        return (w @ (actual * tops[m])) / tot

    for a, b in [("full", "baseline_only"), ("full", "no_history"), ("no_history", "baseline_only")]:
        d = capture(a) - capture(b)
        print(f"capture {a} - {b}: {d.mean():+.1%} [90%: {np.quantile(d, 0.05):+.1%} to {np.quantile(d, 0.95):+.1%}]")

    print("\ncalibration, predicted vs actual extra crimes:")
    cal = (df.group_by("year").agg(n=pl.len(), predicted=pl.sum("full"), no_history=pl.sum("no_history"),
                                   actual=pl.sum("actual")).sort("year"))
    print(cal.with_columns(pl.col("predicted", "no_history", "actual").round(1)))

    print("\nby predicted decile (full ranking, pooled across years):")
    dec = (df.with_columns(decile=(pl.col("full").rank(method="ordinal") * 10 / (pl.len() + 1)).floor().cast(pl.Int8) + 1)
           .group_by("decile").agg(n=pl.len(), predicted=pl.mean("full"), actual=pl.mean("actual"),
                                   actual_ratio=pl.sum("actual_O") / pl.sum("actual_E"))
           .sort("decile"))
    print(dec.with_columns(pl.col("predicted", "actual").round(3), pl.col("actual_ratio").round(3)))


def main() -> None:
    events = pl.read_parquet(EVENTS / "events.parquet")
    cells = pl.read_parquet(EVENTS / "event_cells.parquet")
    oe = pl.read_parquet(EVENTS / "impact" / "oe_by_event.parquet")
    _, busy = busy_cell_days(events, cells)
    counts, _ = load_counts(night=False)

    df = replay(events, cells, oe, busy, counts)
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "backtest.csv")
    pl.Config.set_tbl_rows(20), pl.Config.set_tbl_width_chars(200)
    metrics(df)


if __name__ == "__main__":
    main()
