"""Step 6: calibrated early-warning calls at the reporting grain, plus a live forecast.

    offline_ml/.venv/bin/python offline_ml/early_warning.py      # ~10 s; run after ablation.py

1. **Which forecast should area-level (r7) calls come from?** On the same test cutoffs:
   - `summed_r8`: the r8 GLM summed to r7 ("model small, report big")
   - `native_r7`: the same crime features, with the GLM fit directly on r7 areas
   - `momentum_r7`: a one-feature GLM at r7, using 4-week acceleration only

   The r8 GLM includes 311 only if step 5's lift interval cleared 0.
2. **Probabilities.** A forecast becomes P(next 4 weeks > expected) through a negative binomial
   centered on it. Its dispersion comes only from earlier folds' out-of-sample misses, so
   fold 0 has no probabilities.
   - Scored by the Brier score against climatology (the prior folds' above-normal rate) and a
     reliability table.
   - Calls: up (P >= 0.65), down (P <= 0.35), otherwise unclear.
3. **Live forecast** as of the newest cutoff: refit on every row whose target window ended
   before it, then forecast every cell and area.

Outputs (offline_data/model/):
    step6_r7_comparison.csv    deviance skill and AUC (with CIs) for the three r7 options
    step6_calls.csv            Brier scores and call hit rates per grain/option
    step6_reliability.csv      predicted vs observed above-normal rate, by probability bin
    step6_test_probs.parquet   per test row and grain: forecast, P(above), call, outcome
    live_forecast_r8.parquet   per cell: expected, forecast, P(above), call
    live_forecast_r7.parquet   per area: the same
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
from h3.api import basic_int as h3
from scipy.stats import nbinom

from backtest import BAR, auc_ci, skill_ci
from config import CELL_RES, HORIZON_WEEKS, MODEL, PANEL, REPORT_RES
from features import feature_columns, log_ratio
from models import fit_glm, load_rows, to_city_total, walk_forward

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"
UP, DOWN = 0.65, 0.35
R7_COUNTS = ["c_1w", "c_4w", "c_13w", "c_52w", "c_prev_52w", "c_same_weeks_last_year",
             "c_violent_13w", "c_violent_52w", "c_property_13w", "c_property_52w"]
R7_RATIOS = ["c_accel_4w", "c_accel_13w", "c_yoy", "c_log_expected"]


def r8_groups() -> tuple[str, ...]:
    """Crime features, plus 311 only if step 5's GLM lift interval at r8 was above 0."""
    lift = pl.read_csv(MODEL / "step5_lift.csv").filter(pl.col("comparison") == "glm: + 311")
    return ("crime", "311") if lift["lo_r8"].item() > 0 else ("crime",)


def to_r7(rows: pl.DataFrame) -> pl.DataFrame:
    """Sum cells into their r7 area and recompute the ratio features at that grain. An area's
    target stays null while its 4-week window is still open (a plain sum would make it 0)."""
    return (
        rows.group_by(PARENT, "cutoff")
        .agg(
            pl.when(pl.col("y").is_null().any()).then(None).otherwise(pl.col("y").sum()).alias("y"),
            pl.col("expected").sum(),
            pl.col(R7_COUNTS).sum(),
        )
        .with_columns(
            log_ratio(pl.col("c_4w"), pl.col("c_52w") * 4 / 52).alias("c_accel_4w"),
            log_ratio(pl.col("c_13w"), pl.col("c_52w") * 13 / 52).alias("c_accel_13w"),
            log_ratio(pl.col("c_52w"), pl.col("c_prev_52w")).alias("c_yoy"),
            pl.col("expected").log().alias("c_log_expected"),
        )
        .sort("cutoff", PARENT)
    )


# ---------------------------------------------------------------------------------------------
# Probabilities and calls
# ---------------------------------------------------------------------------------------------

def dispersion(df: pl.DataFrame, model: str) -> float:
    """NB2 dispersion alpha (Var = mu + alpha * mu^2), by method of moments on out-of-sample rows."""
    y, mu = df["y"].to_numpy(), df[model].to_numpy()
    return max(float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum()), 1e-4)


def p_above(mu: np.ndarray, expected: np.ndarray, alpha: float) -> np.ndarray:
    """P(Y > expected) for Y ~ NB(mean mu, dispersion alpha)."""
    k = 1 / alpha
    return nbinom.sf(np.floor(expected), k, k / (k + mu))


def call(p: pl.Expr) -> pl.Expr:
    return pl.when(p >= UP).then(pl.lit("up")).when(p <= DOWN).then(pl.lit("down")).otherwise(pl.lit("unclear"))


def with_probabilities(preds: pl.DataFrame, model: str) -> pl.DataFrame:
    """Add P(above) to each fold, using a dispersion fit on the folds before it only."""
    parts = []
    for f in sorted(preds["fold"].unique())[1:]:
        cur = preds.filter(pl.col("fold") == f)
        alpha = dispersion(preds.filter(pl.col("fold") < f), model)
        parts.append(cur.with_columns(pl.Series("p_above", p_above(cur[model].to_numpy(), cur[BAR].to_numpy(), alpha)),
                                      pl.lit(alpha).alias("alpha")))
    return pl.concat(parts).with_columns((pl.col("y") > pl.col(BAR)).alias("above"), call(pl.col("p_above")).alias("call"))


def call_metrics(probs: pl.DataFrame, label: str, preds: pl.DataFrame) -> dict:
    # Climatology: each fold's forecast is the above-normal rate seen in the folds before it.
    rates = preds.with_columns((pl.col("y") > pl.col(BAR)).alias("above")).group_by("fold").agg(
        pl.col("above").sum().alias("a"), pl.len().alias("n")).sort("fold").with_columns(
        (pl.col("a").cum_sum().shift(1) / pl.col("n").cum_sum().shift(1)).alias("clim"))
    probs = probs.join(rates.select("fold", "clim"), on="fold")
    a = pl.col("above").cast(pl.Float64)
    m = probs.select(
        ((pl.col("p_above") - a) ** 2).mean().alias("brier"),
        ((pl.col("clim") - a) ** 2).mean().alias("brier_climatology"),
        (pl.col("call") != "unclear").mean().alias("share_called"),
        pl.col("above").filter(pl.col("call") == "up").mean().alias("up_hit_rate"),
        (~pl.col("above")).filter(pl.col("call") == "down").mean().alias("down_hit_rate"),
        (pl.col("call") == "up").sum().alias("n_up"), (pl.col("call") == "down").sum().alias("n_down"),
    ).row(0, named=True)
    return {"forecast": label, **m, "brier_skill": 1 - m["brier"] / m["brier_climatology"]}


def reliability(probs: pl.DataFrame, label: str) -> pl.DataFrame:
    return (
        probs.with_columns((pl.col("p_above") * 10).floor().clip(0, 9).alias("bin"))
        .group_by("bin").agg(pl.col("p_above").mean().alias("predicted"), pl.col("above").mean().alias("observed"), pl.len().alias("n"))
        .with_columns(pl.lit(label).alias("forecast")).sort("bin")
    )


# ---------------------------------------------------------------------------------------------
# Live forecast
# ---------------------------------------------------------------------------------------------

def live_forecast(rows: pl.DataFrame, cols: list[str], id_cols: tuple[str, ...]) -> pl.DataFrame:
    """Forecast the newest cutoff, training on every cutoff whose 4 weeks ended before it."""
    live = rows["cutoff"].max()
    train = rows.filter((pl.col("cutoff") <= live - timedelta(weeks=HORIZON_WEEKS)) & pl.col("y").is_not_null())
    test = rows.filter(pl.col("cutoff") == live)
    pred, _ = fit_glm(to_city_total(train, "expected", "y"), test, cols)
    out = test.select(*id_cols, "cutoff", "expected").with_columns(pl.Series("forecast", pred))
    return to_city_total(out, "forecast", "expected")


def label_live(df: pl.DataFrame, alpha: float, id_col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.Series("p_above", p_above(df["forecast"].to_numpy(), df["expected"].to_numpy(), alpha)),
        (pl.col("forecast") / pl.col("expected") - 1).alias("pct_vs_expected"),
        pl.col(id_col).map_elements(h3.int_to_str, return_dtype=pl.String).alias("h3_id"),
    ).with_columns(call(pl.col("p_above")).alias("call")).sort("p_above", descending=True)


def main() -> None:
    rows_all = pl.read_parquet(MODEL / "features.parquet")
    rows = load_rows()
    groups = r8_groups()
    cols8 = feature_columns(rows, groups)

    # --- r8 forecasts (step-5 GLM) and the three r7 options --------------------------------
    r8, _ = walk_forward(rows, cols8, fit_glm, "glm")
    r8 = r8.rename({"expected": BAR})
    summed = r8.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", BAR, "glm").sum()).rename({"glm": "summed_r8"})
    r7 = to_r7(rows)
    assert r7["y"].null_count() == 0
    native, _ = walk_forward(r7, R7_COUNTS + R7_RATIOS, fit_glm, "native_r7", id_cols=(PARENT,))
    momentum, _ = walk_forward(r7, ["c_accel_4w"], fit_glm, "momentum_r7", id_cols=(PARENT,))
    k7 = [PARENT, "cutoff"]
    area = summed.join(native.select(*k7, "native_r7"), on=k7).join(momentum.select(*k7, "momentum_r7"), on=k7)

    comparison = []
    options = ("summed_r8", "native_r7", "momentum_r7")
    for m in options:
        s, lo, hi = skill_ci(area, m, BAR)
        a, alo, ahi = auc_ci(area.with_columns((pl.col("y") > pl.col(BAR)).alias("above"), (pl.col(m) / pl.col(BAR)).alias("_s")), "_s")
        comparison.append({"forecast": m, "skill": s, "skill_lo": lo, "skill_hi": hi, "auc": a, "auc_lo": alo, "auc_hi": ahi})
    comparison = pl.DataFrame(comparison)
    comparison.write_csv(MODEL / "step6_r7_comparison.csv")

    # --- Probabilities, calls, reliability ------------------------------------------------
    metrics, rel, test_probs = [], [], []
    for grain, df, m in [(f"r{CELL_RES}", r8, "glm"), *((f"r{REPORT_RES}", area, o) for o in options)]:
        probs = with_probabilities(df, m)
        label = f"{grain} {m}"
        metrics.append({"grain": grain, **call_metrics(probs, label, df), "alpha_last": probs["alpha"][-1]})
        rel.append(reliability(probs, label))
        test_probs.append(probs.select("cutoff", "fold", "y", BAR, pl.col(m).alias("forecast"), "p_above", "call", "above",
                                       pl.lit(label).alias("forecast_name"),
                                       pl.col(CELL if grain.endswith(str(CELL_RES)) else PARENT).alias("h3")))
    metrics, rel = pl.DataFrame(metrics), pl.concat(rel)
    metrics.write_csv(MODEL / "step6_calls.csv")
    rel.write_csv(MODEL / "step6_reliability.csv")
    pl.concat(test_probs).write_parquet(MODEL / "step6_test_probs.parquet")

    # --- Live forecast --------------------------------------------------------------------
    # The r7 calls come from whichever option had the best Brier score (ranking and calibration together).
    brier7 = metrics.filter(pl.col("grain") == f"r{REPORT_RES}").sort("brier")["forecast"][0].split(" ")[1]
    cells = pl.read_parquet(PANEL / "cells.parquet").select(CELL, "community_area", "lat", "lon")
    live8 = live_forecast(rows_all, cols8, (CELL, PARENT))
    live8 = label_live(live8, dispersion(r8, "glm"), CELL).join(cells, on=CELL, how="left")
    if brier7 == "summed_r8":
        live7 = live8.group_by(PARENT, "cutoff").agg(pl.col("expected", "forecast").sum())
    else:
        feats7 = R7_COUNTS + R7_RATIOS if brier7 == "native_r7" else ["c_accel_4w"]
        live7 = live_forecast(to_r7(rows_all), feats7, (PARENT,))
    area_ca = cells.join(rows_all.select(CELL, PARENT).unique(), on=CELL).group_by(PARENT).agg(
        pl.col("community_area").mode().first(), pl.len().alias("n_cells"))
    live7 = label_live(live7, dispersion(area, brier7), PARENT).join(area_ca, on=PARENT, how="left")
    live8.write_parquet(MODEL / "live_forecast_r8.parquet")
    live7.write_parquet(MODEL / "live_forecast_r7.parquet")

    with pl.Config(tbl_rows=30, tbl_width_chars=220, float_precision=3, tbl_hide_dataframe_shape=True):
        print(f"r8 model uses feature groups {groups} ({len(cols8)} features)\n")
        print("r7 options, 2022-26 test:")
        print(comparison)
        print("\nprobabilities and calls (folds 1-18; up = P >= 0.65, down = P <= 0.35):")
        print(metrics.drop("grain"))
        print(f"\nreliability, r8 and r7 {brier7}:")
        print(rel.filter(pl.col("forecast").is_in([f"r{CELL_RES} glm", f"r{REPORT_RES} {brier7}"]))
              .pivot(on="forecast", index="bin", values=["predicted", "observed", "n"]))
        live = live8["cutoff"][0]
        print(f"\nlive forecast as of {live} (next 4 weeks), r7 from {brier7}: "
              f"{(live7['call'] == 'up').sum()} areas called up, {(live7['call'] == 'down').sum()} down, "
              f"{(live7['call'] == 'unclear').sum()} unclear")
        print(live7.head(8).select("h3_id", "community_area", "n_cells", "expected", "forecast", "pct_vs_expected", "p_above", "call"))


if __name__ == "__main__":
    main()
