"""Step 3: features for every (cell, cutoff) row, plus leakage checks.

    offline_ml/.venv/bin/python offline_ml/features.py

Output: offline_data/model/features.parquet. It has one row per (r8 cell, weekly cutoff)
from FIRST_CUTOFF on, with:
- the target `y` (null where the 4 weeks aren't over yet)
- `expected`, the bar's forecast, which the models use as their offset
- every feature

FEATURE_GROUPS splits the columns into crime-side and 311, which is what the step-5
ablation toggles.

Every feature is built from events strictly before the cutoff Monday. Two checks at the end
enforce that:
- **Truncation:** rebuild everything from data with the future deleted and require identical
  features.
- **Direct:** recompute a few features for random rows straight from the event tables.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
from h3.api import basic_int as h3

import panel
from backtest import BAR, add_baselines, cutoff_frame
from config import CELL_RES, CLEAN, FIRST_CUTOFF, MODEL, PANEL, REPORT_RES, TEST_START, UNIVERSE_WINDOW

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"

SR_TOP_TYPES = 25           # busiest 311 types (pre-2022 volume) get their own features
# The 311 counterpart of crime's "vacant" location group, too low-volume to make the top 25.
SR_EXTRA_TYPES = ["vacant_abandoned_building_complaint", "clean_vacant_lot_request"]
STALE_OPEN_WEEKS = 52       # a request "open" longer than this is a dead record, not live disorder
CLOSE_DAYS_CAP = 90         # one years-open request shouldn't swamp a cell's average close time
LEAK_CHECK_AT = date(2024, 6, 3)

FEATURE_GROUPS = {
    "crime": ("c_", "n1_", "n2_"),   # the cell's own crime history and its neighbors'
    "311": ("s_", "sn1_"),           # the cell's 311 activity and its ring-1 neighbors'
    # Same value for every cell in a week. The local models (step 4) leave these out, because
    # the bar sets the citywide level. They're kept for a possible citywide model.
    "citywide": ("t_",),
}


def feature_columns(df: pl.DataFrame, groups: tuple[str, ...] = tuple(FEATURE_GROUPS)) -> list[str]:
    prefixes = tuple(p for g in groups for p in FEATURE_GROUPS[g])
    return [c for c in df.columns if c.startswith(prefixes)]


def log_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """log((num + 1) / (den + 1)): 0 when the recent rate matches the longer one."""
    return ((num + 1) / (den + 1)).log()


def past(col: str, weeks: int, skip: int = 0) -> pl.Expr:
    """Sum of `col` over the `weeks` weeks ending `skip` weeks before the row's week, per cell."""
    return pl.col(col).rolling_sum(weeks).shift(1 + skip).over(CELL)


def pick_sr_types(sw: pl.DataFrame) -> list[str]:
    start, end = UNIVERSE_WINDOW
    top = (
        sw.filter(pl.col("week").is_between(start, end, closed="left"))
        .group_by("sr_type_slug").agg(pl.col("n").sum())
        .sort("n", "sr_type_slug", descending=[True, False])
        .head(SR_TOP_TYPES)["sr_type_slug"].to_list()
    )
    return top + [t for t in SR_EXTRA_TYPES if t not in top]


# ---------------------------------------------------------------------------------------------
# Feature blocks. Each returns one row per (cell, week), where "week" is the cutoff Monday.
# ---------------------------------------------------------------------------------------------

def crime_block(cw: pl.DataFrame, crimes: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    mix = (
        crimes.filter(known).join(grid.select(CELL).unique(), on=CELL, how="semi")
        .with_columns(panel.week_of("occurred_at"))
        .group_by(CELL, "week")
        .agg(
            (pl.col("location_group") == "residential").sum().alias("_res"),
            (pl.col("location_group") == "street").sum().alias("_street"),
            (pl.col("location_group") == "commercial").sum().alias("_comm"),
            pl.col("domestic").sum().alias("_dom"),
        )
    )
    df = (
        cw.join(mix, on=[CELL, "week"], how="left")
        .with_columns(pl.col("_res", "_street", "_comm", "_dom").fill_null(0))
        .sort(CELL, "week")
        .with_columns(
            past("known_n", 1).alias("c_1w"),
            past("known_n", 4).alias("c_4w"),
            past("known_n", 13).alias("c_13w"),
            past("known_n", 52).alias("c_52w"),
            past("known_n", 52, skip=52).alias("c_prev_52w"),
            past("known_n", 4, skip=48).alias("c_same_weeks_last_year"),
            past("known_violent_n", 13).alias("c_violent_13w"),
            past("known_violent_n", 52).alias("c_violent_52w"),
            past("known_property_n", 13).alias("c_property_13w"),
            past("known_property_n", 52).alias("c_property_52w"),
            *(past(src, 52).alias(f"_{name}_52w") for src, name in
              (("_res", "res"), ("_street", "street"), ("_comm", "comm"), ("_dom", "dom"))),
        )
    )
    share = lambda c: pl.when(pl.col("c_52w") > 0).then(pl.col(c) / pl.col("c_52w")).otherwise(0.0)
    return df.select(
        CELL, "week", "c_1w", "c_4w", "c_13w", "c_52w", "c_prev_52w", "c_same_weeks_last_year",
        "c_violent_13w", "c_violent_52w", "c_property_13w", "c_property_52w",
        log_ratio(pl.col("c_4w"), pl.col("c_52w") * 4 / 52).alias("c_accel_4w"),
        log_ratio(pl.col("c_13w"), pl.col("c_52w") * 13 / 52).alias("c_accel_13w"),
        log_ratio(pl.col("c_52w"), pl.col("c_prev_52w")).alias("c_yoy"),
        log_ratio(pl.col("c_violent_13w"), pl.col("c_violent_52w") * 13 / 52).alias("c_violent_accel_13w"),
        share("_res_52w").alias("c_share_residential"),
        share("_street_52w").alias("c_share_street"),
        share("_comm_52w").alias("c_share_commercial"),
        share("_dom_52w").alias("c_share_domestic"),
    )


def sr_block(sw: pl.DataFrame, sr: pl.DataFrame, grid: pl.DataFrame, types: list[str]) -> pl.DataFrame:
    keys = [CELL, "week"]
    per_type = (
        sw.filter(pl.col("sr_type_slug").is_in(types))
        .pivot(on="sr_type_slug", index=keys, values="n")
    )
    totals = sw.group_by(keys).agg(pl.col("n", "n_resident", "n_duplicate").sum())

    # Backlog: a request is open at cutoff t if it was created before t and closed at or
    # after t. `status` and `closed_at` describe today, so rebuild the state at each cutoff
    # from timestamps: +1 at the first cutoff after creation, -1 at the first cutoff after
    # closing (or STALE_OPEN_WEEKS later).
    start = pl.col("created_at").dt.truncate("1w").dt.date() + pl.duration(weeks=1)
    stale = start + pl.duration(weeks=STALE_OPEN_WEEKS)
    closed = pl.col("closed_at").dt.truncate("1w").dt.date() + pl.duration(weeks=1)
    spans = sr.select(CELL, start.alias("s"),
                      pl.when(pl.col("closed_at").is_null()).then(stale).otherwise(pl.min_horizontal(closed, stale)).alias("e"))
    old = spans.with_columns((pl.col("s") + pl.duration(weeks=4)).alias("s")).filter(pl.col("s") < pl.col("e"))

    def open_delta(sp: pl.DataFrame, name: str) -> pl.DataFrame:
        return pl.concat([
            sp.select(CELL, pl.col("s").alias("week"), pl.lit(1).alias(name)),
            sp.select(CELL, pl.col("e").alias("week"), pl.lit(-1).alias(name)),
        ]).group_by(keys).agg(pl.col(name).sum())

    closes = (
        sr.filter(pl.col("closed_at").is_not_null())
        .select(CELL, pl.col("closed_at").dt.truncate("1w").dt.date().alias("week"),
                pl.col("days_to_close").clip(upper_bound=CLOSE_DAYS_CAP))
        .group_by(keys).agg(pl.col("days_to_close").sum().alias("_close_days"), pl.len().alias("_closes"))
    )

    df = grid.select(keys)
    for part in (per_type, totals, open_delta(spans, "_open_d"), open_delta(old, "_old_d"), closes):
        df = df.join(part, on=keys, how="left")
    df = df.with_columns(pl.exclude(keys).fill_null(0)).sort(keys).with_columns(
        pl.col("_open_d").cum_sum().over(CELL).alias("s_open"),
        pl.col("_old_d").cum_sum().over(CELL).alias("s_open_4w_plus"),
        past("n", 1).alias("s_total_1w"),
        past("n", 4).alias("s_total_4w"),
        past("n", 13).alias("s_total_13w"),
        past("n", 52).alias("s_total_52w"),
        past("n_resident", 13).alias("_res_13w"),
        past("n_duplicate", 13).alias("_dup_13w"),
        past("_close_days", 13).alias("_cd_13w"),
        past("_closes", 13).alias("_cl_13w"),
        *(past(t, 4).alias(f"s_{t}_4w") for t in types),
        *(past(t, 52).alias(f"_{t}_52w") for t in types),
    )
    ratio = lambda a, b: pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b))
    return df.select(
        CELL, "week", "s_total_1w", "s_total_4w", "s_total_13w", "s_total_52w",
        log_ratio(pl.col("s_total_4w"), pl.col("s_total_52w") * 4 / 52).alias("s_total_accel_4w"),
        ratio("_res_13w", "s_total_13w").fill_null(0.0).alias("s_resident_share_13w"),
        ratio("_dup_13w", "s_total_13w").fill_null(0.0).alias("s_duplicate_share_13w"),
        "s_open", "s_open_4w_plus",
        ratio("_cd_13w", "_cl_13w").alias("s_close_days_13w"),   # null when nothing closed; models handle it
        *(pl.col(f"s_{t}_4w") for t in types),
        *(log_ratio(pl.col(f"s_{t}_4w"), pl.col(f"_{t}_52w") * 4 / 52).alias(f"s_{t}_accel_4w") for t in types),
    )


def neighbor_block(frame: pl.DataFrame, cells: pl.DataFrame) -> pl.DataFrame:
    """Sums over the ring-1 (6) and ring-2 (12) neighbors that are modeled cells."""
    ids = set(cells[CELL].to_list())
    pairs = pl.DataFrame(
        [(c, n, k) for c in ids for k in (1, 2) for n in h3.grid_ring(c, k) if n in ids],
        schema={CELL: pl.Int64, "nbr": pl.Int64, "ring": pl.Int32}, orient="row",
    )
    cols = ["c_4w", "c_52w", "s_total_4w", "s_total_52w"]
    sums = (
        pairs.join(frame.select(pl.col(CELL).alias("nbr"), "week", *cols), on="nbr")
        .group_by(CELL, "week", "ring").agg(pl.col(cols).sum())
        .pivot(on="ring", index=[CELL, "week"], values=cols)
    )
    # pivot names columns "<col>_<ring>"
    rn = lambda col, k: f"{col}_{k}"
    out = frame.select(CELL, "week").join(sums, on=[CELL, "week"], how="left").fill_null(0)
    return out.select(
        CELL, "week",
        *(pl.col(rn(c, k)).alias(f"n{k}_{c}") for k in (1, 2) for c in ("c_4w", "c_52w")),
        *(log_ratio(pl.col(rn("c_4w", k)), pl.col(rn("c_52w", k)) * 4 / 52).alias(f"n{k}_c_accel_4w") for k in (1, 2)),
        pl.col(rn("s_total_4w", 1)).alias("sn1_total_4w"),
        log_ratio(pl.col(rn("s_total_4w", 1)), pl.col(rn("s_total_52w", 1)) * 4 / 52).alias("sn1_total_accel_4w"),
    )


def time_block(cw: pl.DataFrame) -> pl.DataFrame:
    city = cw.group_by("week").agg(pl.col("known_n").sum()).sort("week")
    back = lambda n, skip=0: pl.col("known_n").rolling_sum(n).shift(1 + skip)
    return city.select(
        "week",
        pl.col("week").dt.week().alias("t_week_of_year"),
        log_ratio(back(4), back(52) * 4 / 52).alias("t_city_accel_4w"),
        log_ratio(back(13), back(52) * 13 / 52).alias("t_city_accel_13w"),
        log_ratio(back(52), back(52, skip=52)).alias("t_city_yoy"),
    )


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

def build_features(crimes: pl.DataFrame, sr: pl.DataFrame, cells: pl.DataFrame, weeks: pl.Series,
                   types: list[str]) -> pl.DataFrame:
    cw = panel.crime_weekly(crimes, cells, weeks)
    sw = panel.sr_weekly(sr, cells, weeks)
    grid = cw.select(CELL, "week")

    rows = add_baselines(cutoff_frame(cw, require_target=False)).select(
        CELL, PARENT, pl.col("cutoff").alias("week"), "y", pl.col(BAR).alias("expected"), pl.col("season").alias("t_season"))
    rows = rows.with_columns(pl.col("expected").log().alias("c_log_expected"))

    crime = crime_block(cw, crimes, grid)
    srf = sr_block(sw, sr, grid, types)
    nb = neighbor_block(crime.join(srf.select(CELL, "week", "s_total_4w", "s_total_52w"), on=[CELL, "week"]), cells)

    out = (
        rows.filter(pl.col("week") >= FIRST_CUTOFF)
        .join(crime, on=[CELL, "week"], how="left")
        .join(nb, on=[CELL, "week"], how="left")
        .join(time_block(cw), on="week", how="left")
        .join(srf, on=[CELL, "week"], how="left")
        .rename({"week": "cutoff"})
        .sort("cutoff", CELL)
    )
    return out.select(CELL, PARENT, "cutoff", "y", "expected", *feature_columns(out))


def load_events() -> tuple[pl.DataFrame, pl.DataFrame]:
    crimes = pl.read_parquet(CLEAN / "crimes.parquet", columns=[
        "occurred_at", "case_year", "ucr_class", "location_group", "domestic", "community_area", CELL])
    sr = pl.read_parquet(CLEAN / "sr311.parquet", columns=[
        "created_at", "closed_at", "days_to_close", "sr_type_slug", "origin_group", "is_duplicate", "community_area", CELL])
    return crimes, sr


def truncate(crimes: pl.DataFrame, sr: pl.DataFrame, t: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The data as it would have looked at the start of Monday `t`."""
    ts = pl.lit(t).cast(pl.Datetime("us"))
    sr_t = sr.filter(pl.col("created_at") < ts).with_columns(
        pl.when(pl.col("closed_at") < ts).then(pl.col(c)).alias(c) for c in ("closed_at", "days_to_close"))
    return crimes.filter(pl.col("occurred_at") < ts), sr_t


def check_truncation(full: pl.DataFrame, crimes, sr, cells, weeks, types) -> None:
    c_t, s_t = truncate(crimes, sr, LEAK_CHECK_AT)
    w_t = weeks.filter(weeks < LEAK_CHECK_AT)
    trunc = build_features(c_t, s_t, cells, w_t, types).filter(pl.col("cutoff") < LEAK_CHECK_AT)
    cols = feature_columns(full) + ["expected"]
    a = full.filter(pl.col("cutoff") < LEAK_CHECK_AT).select(CELL, "cutoff", *cols).sort(CELL, "cutoff")
    b = trunc.select(CELL, "cutoff", *cols).sort(CELL, "cutoff")
    assert a.height == b.height, (a.height, b.height)
    bad = [c for c in cols if not a[c].fill_nan(None).equals(b[c].fill_nan(None))]
    assert not bad, f"features changed when the future was deleted: {bad}"
    # ...and the check isn't vacuous: the targets near the cut do depend on the deleted data.
    assert trunc.filter(pl.col("cutoff") > LEAK_CHECK_AT - timedelta(weeks=4))["y"].null_count() > 0
    print(f"  truncation check: all {len(cols)} features identical for {a.height:,} rows before {LEAK_CHECK_AT} "
          f"when everything from {LEAK_CHECK_AT} on is deleted")


def check_direct(full: pl.DataFrame, crimes: pl.DataFrame, sr: pl.DataFrame, cells: pl.DataFrame) -> None:
    """Recompute a few features from the event tables with explicit timestamp bounds."""
    known = pl.col("case_year").is_null() | (pl.col("case_year") <= pl.col("occurred_at").dt.year())
    ids = set(cells[CELL].to_list())
    sample = full.filter(pl.col("cutoff") >= FIRST_CUTOFF + timedelta(weeks=60)).sample(8, seed=3)
    for row in sample.iter_rows(named=True):
        cell, t = row[CELL], row["cutoff"]
        ts, ts4 = (pl.lit(d).cast(pl.Datetime("us")) for d in (t, t - timedelta(weeks=4)))
        win4 = lambda col: pl.col(col).is_between(ts4, ts, closed="left")
        nbrs = [n for n in h3.grid_ring(cell, 1) if n in ids]
        got = {
            "c_4w": crimes.filter((pl.col(CELL) == cell) & win4("occurred_at") & known).height,
            "n1_c_4w": crimes.filter(pl.col(CELL).is_in(nbrs) & win4("occurred_at") & known).height,
            "s_total_4w": sr.filter((pl.col(CELL) == cell) & win4("created_at")).height,
            "s_graffiti_removal_request_4w": sr.filter(
                (pl.col(CELL) == cell) & win4("created_at") & (pl.col("sr_type_slug") == "graffiti_removal_request")).height,
            "s_open": sr.filter(
                (pl.col(CELL) == cell) & (pl.col("created_at") < ts)
                & (pl.col("closed_at").is_null() | (pl.col("closed_at") >= ts))
                & (pl.col("created_at").dt.truncate("1w").dt.date() + pl.duration(weeks=STALE_OPEN_WEEKS + 1) > pl.lit(t))
            ).height,
        }
        for k, v in got.items():
            assert row[k] == v, (cell, t, k, row[k], v)
    print(f"  direct check: {len(got)} features match a from-scratch recount for {sample.height} random rows")


def main() -> None:
    crimes, sr = load_events()
    cells = pl.read_parquet(PANEL / "cells.parquet")
    weeks = pl.read_parquet(PANEL / "crime_weekly.parquet")["week"].unique().sort()
    types = pick_sr_types(pl.read_parquet(PANEL / "sr_weekly.parquet"))

    full = build_features(crimes, sr, cells, weeks, types)
    full.write_parquet(MODEL / "features.parquet")

    crime_cols, sr_cols, city_cols = (feature_columns(full, (g,)) for g in ("crime", "311", "citywide"))
    print(f"{full.height:,} rows: {cells.height} cells x {full['cutoff'].n_unique()} cutoffs "
          f"({full['cutoff'].min()} .. {full['cutoff'].max()}); {full['y'].null_count():,} without a target yet")
    print(f"{len(crime_cols)} local crime features, {len(sr_cols)} 311 features ({len(types)} types), "
          f"{len(city_cols)} citywide features")
    nulls = {c: n for c, n in full.select(feature_columns(full)).null_count().row(0, named=True).items() if n}
    print(f"columns with nulls: {nulls or 'none'}")

    print("leakage checks:")
    check_direct(full, crimes, sr, cells)
    check_truncation(full, crimes, sr, cells, weeks, types)

    # A first, univariate look (training cutoffs only): which features track a cell running
    # above or below its expected level over the next 4 weeks, compared with other cells in the
    # same week? Demeaning by cutoff strips out citywide swings (COVID, a citywide 311
    # campaign) that would otherwise make any time-trending feature look predictive.
    train = full.filter((pl.col("cutoff") < TEST_START) & pl.col("y").is_not_null())
    local = lambda e: e - e.mean().over("cutoff")
    dev = local(((pl.col("y") + 1) / (pl.col("expected") + 1)).log())
    corr = pl.DataFrame(
        [(c, "311" if c in sr_cols else "crime", train.select(pl.corr(local(pl.col(c).fill_null(strategy="mean")), dev)).item())
         for c in crime_cols + sr_cols],
        schema=["feature", "group", "r"], orient="row",
    ).with_columns(pl.col("r").abs().alias("abs_r")).sort("abs_r", descending=True)
    with pl.Config(tbl_rows=20, float_precision=3, tbl_hide_dataframe_shape=True, fmt_str_lengths=45):
        print("\nstrongest within-week correlations with log(actual / expected), training rows only:")
        print(corr.head(12).drop("abs_r"))
        print("best 311 features:")
        print(corr.filter(pl.col("group") == "311").head(8).drop("abs_r"))


if __name__ == "__main__":
    main()
