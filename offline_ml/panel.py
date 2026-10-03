"""Step 1: choose the H3 cells to model and count events per cell per week.

    offline_ml/.venv/bin/python offline_ml/panel.py

Outputs (offline_data/panel/):
    cells.parquet         the modeled r8 cells, with their r7 parent and a few descriptors
    crime_weekly.parquet  dense cell x week grid of crime counts (zero-filled)
    sr_weekly.parquet     sparse cell x week x 311-type counts
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from h3.api import basic_int as h3

from config import (
    CELL_RES, CLEAN, GRID_START, MIN_CRIMES_PER_YEAR, PANEL, REPORT_RES, UNIVERSE_WINDOW,
)

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"


def last_complete_week(crimes: pl.DataFrame) -> date:
    """Monday of the newest week with every day present.

    The export's final day is a partial pull (11 crimes on 2026-09-12 vs ~650 a day before),
    so the last full day is the one before it.
    """
    last_full_day = crimes["occurred_at"].max().date() - timedelta(days=1)
    d = last_full_day - timedelta(days=6)
    return d - timedelta(days=d.weekday())


def week_of(col: str) -> pl.Expr:
    return pl.col(col).dt.truncate("1w").dt.date().alias("week")


def choose_cells(crimes: pl.DataFrame, sr: pl.DataFrame) -> pl.DataFrame:
    start, end = UNIVERSE_WINDOW
    years = (end - start).days / 365.25
    busy = (
        crimes.filter(pl.col("occurred_at").dt.date().is_between(start, end, closed="left"))
        .group_by(CELL).len()
        .with_columns((pl.col("len") / years).round(1).alias("crimes_per_year_pre"))
        .filter(pl.col("crimes_per_year_pre") >= MIN_CRIMES_PER_YEAR)
        .drop("len")
    )
    # Descriptive only (maps, labels in the notebook); never a model input, so using the
    # whole date range here is fine.
    area = (
        pl.concat([crimes.select(CELL, "community_area"), sr.select(CELL, "community_area")])
        .drop_nulls().group_by(CELL).agg(pl.col("community_area").mode().first())
    )
    ids = busy[CELL].to_list()
    centers = [h3.cell_to_latlng(c) for c in ids]
    return (
        busy.with_columns(
            pl.Series(PARENT, [h3.cell_to_parent(c, REPORT_RES) for c in ids], dtype=pl.Int64),
            pl.Series("lat", [p[0] for p in centers]),
            pl.Series("lon", [p[1] for p in centers]),
        )
        .join(area, on=CELL, how="left")
        .sort(CELL)
    )


def crime_weekly(crimes: pl.DataFrame, cells: pl.DataFrame, weeks: pl.Series) -> pl.DataFrame:
    """Dense cell x week counts. `crime_*` counts everything and feeds the target; `known_*`
    leaves out crimes whose case was opened in a later calendar year than they occurred
    (reported too late to have been known at the time) and feeds features and baselines."""
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    violent, prop = pl.col("ucr_class") == "violent", pl.col("ucr_class") == "property"
    counts = (
        crimes.join(cells.select(CELL), on=CELL, how="semi")
        .with_columns(week_of("occurred_at"))
        .filter(pl.col("week").is_between(weeks.min(), weeks.max()))
        .group_by(CELL, "week")
        .agg(
            pl.len().alias("crime_n"),
            violent.sum().alias("violent_n"),
            prop.sum().alias("property_n"),
            known.sum().alias("known_n"),
            (known & violent).sum().alias("known_violent_n"),
            (known & prop).sum().alias("known_property_n"),
        )
    )
    grid = cells.select(CELL, PARENT).join(weeks.to_frame("week"), how="cross")
    count_cols = [c for c in counts.columns if c.endswith("_n")]
    return (
        grid.join(counts, on=[CELL, "week"], how="left")
        .with_columns(pl.col(c).fill_null(0).cast(pl.Int32) for c in count_cols)
        .sort(CELL, "week")
    )


def sr_weekly(sr: pl.DataFrame, cells: pl.DataFrame, weeks: pl.Series) -> pl.DataFrame:
    """Sparse cell x week x type counts; features densify only the types they use."""
    return (
        sr.join(cells.select(CELL), on=CELL, how="semi")
        .with_columns(week_of("created_at"), pl.col("sr_type_slug").cast(pl.String))
        .filter(pl.col("week").is_between(weeks.min(), weeks.max()))
        .group_by(CELL, "week", "sr_type_slug")
        .agg(
            pl.len().cast(pl.Int32).alias("n"),
            (pl.col("origin_group") == "resident").sum().cast(pl.Int32).alias("n_resident"),
            pl.col("is_duplicate").sum().cast(pl.Int32).alias("n_duplicate"),
        )
        .sort(CELL, "week", "sr_type_slug")
    )


def main() -> None:
    PANEL.mkdir(exist_ok=True)
    crimes = pl.read_parquet(CLEAN / "crimes.parquet",
                             columns=["occurred_at", "case_year", "ucr_class", "community_area", CELL])
    sr = pl.read_parquet(CLEAN / "sr311.parquet",
                         columns=["created_at", "sr_type_slug", "origin_group", "is_duplicate", "community_area", CELL])

    last_week = last_complete_week(crimes)
    weeks = pl.date_range(GRID_START, last_week, "1w", eager=True)
    cells = choose_cells(crimes, sr)
    cw = crime_weekly(crimes, cells, weeks)
    sw = sr_weekly(sr, cells, weeks)

    # Self-checks: the grid is complete and every in-range crime in a kept cell is counted once.
    assert cw.height == cells.height * weeks.len()
    in_range = crimes.join(cells.select(CELL), on=CELL, how="semi").filter(
        pl.col("occurred_at").dt.date().is_between(weeks.min(), last_week + timedelta(days=6)))
    assert cw["crime_n"].sum() == in_range.height, (cw["crime_n"].sum(), in_range.height)
    assert cw.null_count().sum_horizontal().item() == 0

    cells.write_parquet(PANEL / "cells.parquet")
    cw.write_parquet(PANEL / "crime_weekly.parquet")
    sw.write_parquet(PANEL / "sr_weekly.parquet")

    all_recent = crimes.filter(
        pl.col("occurred_at").dt.date().is_between(UNIVERSE_WINDOW[1], last_week + timedelta(days=6))).height
    kept_recent = cw.filter(pl.col("week") >= UNIVERSE_WINDOW[1])["crime_n"].sum()
    print(f"cells: {cells.height} r{CELL_RES} in {cells[PARENT].n_unique()} r{REPORT_RES} parents; "
          f"they hold {kept_recent / all_recent:.2%} of crime since {UNIVERSE_WINDOW[1]}")
    print(f"weeks: {weeks.min()} .. {last_week} ({weeks.len()}); crime grid {cw.height:,} rows, "
          f"{(cw['crime_n'] == 0).mean():.0%} zero; known share {cw['known_n'].sum() / cw['crime_n'].sum():.2%}")
    print(f"311: {sw.height:,} cell-week-type rows, {sw['n'].sum():,} requests, "
          f"{sw['sr_type_slug'].n_unique()} types")


if __name__ == "__main__":
    main()
