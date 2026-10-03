"""Step 5: does 311 improve the forecast beyond crime's own history?

    offline_ml/.venv/bin/python offline_ml/ablation.py      # ~8 min, mostly the tree models

Every comparison uses identical folds and rows. The *lift* is the share of the crime-only
model's deviance that the 311 version removes (1 - D_with_311 / D_crime_only). Its CI comes
from resampling whole quarterly folds.

- **Headline:** crime-only vs crime + 311, for the GLM and the trees.
- **Placebo:** crime + 311, but with each week's 311 features shuffled across cells. This
  keeps every citywide pattern in 311 and breaks only the link to the cell. Real local signal
  shows up as lift over the placebo. If the placebo itself "lifts", something other than
  local 311 information is being learned.
- **Which parts:** GLM runs that add one 311 sub-group, or one request type, at a time. With
  27 types, a few 95% intervals will miss 0 by chance, so types also get a
  Bonferroni-adjusted interval.

Outputs (offline_data/model/):
    step5_predictions.parquet   per test (cell, cutoff): target, bar, all six headline models
    step5_lift.csv              every comparison: lift and CI at r8 and r7, AUCs
    step5_type_lift.csv         GLM lift from each 311 request type on its own
    step5_glm_311_coefs.csv     final-fold GLM coefficients on the 311 features
"""

from __future__ import annotations

import re
import time

import polars as pl

from backtest import BAR, auc_ci, poisson_deviance, skill_ci
from config import CELL_RES, MODEL, REPORT_RES
from features import feature_columns
from models import fit_gbm, fit_glm, load_rows, walk_forward

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"
KEYS = [CELL, "cutoff"]


def shuffle_within_cutoff(rows: pl.DataFrame, cols: list[str], seed: int = 0) -> pl.DataFrame:
    """Give each row another same-week row's `cols`, as one block so they stay consistent."""
    base = rows.sort("cutoff", CELL)
    donor = base.select("cutoff", *cols).sample(fraction=1.0, shuffle=True, seed=seed).sort("cutoff", maintain_order=True)
    return base.drop(cols).hstack(donor.drop("cutoff")).select(rows.columns)


def sr_subgroups(sr_cols: list[str], types: list[str]) -> dict[str, list[str]]:
    per_type = {c for t in types for c in (f"s_{t}_4w", f"s_{t}_accel_4w")}
    return {
        "totals": [c for c in sr_cols if c.startswith("s_total") or c.endswith("_share_13w")],
        "backlog": [c for c in sr_cols if c in ("s_open", "s_open_4w_plus", "s_close_days_13w")],
        "types": [c for c in sr_cols if c in per_type],
        "neighbors": [c for c in sr_cols if c.startswith("sn1_")],
    }


def compare(preds: pl.DataFrame, model: str, ref: str, label: str) -> dict:
    """Lift of `model` over `ref` at both grains, plus both models' AUCs at both grains."""
    rolled = preds.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", BAR, model, ref).sum())
    out = {"comparison": label, "model": model, "ref": ref}
    for grain, df in ((f"r{CELL_RES}", preds), (f"r{REPORT_RES}", rolled)):
        s, lo, hi = skill_ci(df, model, ref)
        df = df.with_columns((pl.col("y") > pl.col(BAR)).alias("above"),
                             *((pl.col(m) / pl.col(BAR)).alias(f"_{m}") for m in (model, ref)))
        out |= {f"lift_{grain}": s, f"lo_{grain}": lo, f"hi_{grain}": hi,
                f"auc_{grain}": auc_ci(df, f"_{model}")[0], f"auc_ref_{grain}": auc_ci(df, f"_{ref}")[0]}
    return out


def run(rows: pl.DataFrame, cols: list[str], fit, name: str) -> pl.DataFrame:
    p, _ = walk_forward(rows, cols, fit, name)
    return p.rename({"expected": BAR})


def main() -> None:
    t0 = time.time()
    rows = load_rows()
    crime, sr = feature_columns(rows, ("crime",)), feature_columns(rows, ("311",))
    placebo_rows = shuffle_within_cutoff(rows, sr)

    # --- Headline + placebo, both model families ------------------------------------------
    runs = {}
    for tag, fit in (("glm", fit_glm), ("gbm", fit_gbm)):
        runs[f"{tag}_crime"] = run(rows, crime, fit, f"{tag}_crime")
        runs[f"{tag}_full"] = run(rows, crime + sr, fit, f"{tag}_full")
        runs[f"{tag}_placebo"] = run(placebo_rows, crime + sr, fit, f"{tag}_placebo")
        print(f"  {tag}: 3 walk-forwards done ({time.time() - t0:.0f}s)")
    preds = runs["glm_crime"]
    for name, p in runs.items():
        if name != "glm_crime":
            preds = preds.join(p.select(*KEYS, name), on=KEYS)
    preds.write_parquet(MODEL / "step5_predictions.parquet")

    results = []
    for tag in ("glm", "gbm"):
        results += [
            compare(preds, f"{tag}_full", f"{tag}_crime", f"{tag}: + 311"),
            compare(preds, f"{tag}_placebo", f"{tag}_crime", f"{tag}: + shuffled 311 (placebo)"),
            compare(preds, f"{tag}_full", f"{tag}_placebo", f"{tag}: real vs shuffled 311"),
        ]

    # --- Which parts of 311 carry it (GLM only; each run takes seconds) --------------------
    types = sorted({m.group(1) for c in sr if (m := re.fullmatch(r"s_(.+)_accel_4w", c)) and m.group(1) != "total"})
    for group, cols in sr_subgroups(sr, types).items():
        p = run(rows, crime + cols, fit_glm, f"glm_{group}").join(preds.select(*KEYS, "glm_crime"), on=KEYS)
        results.append(compare(p, f"glm_{group}", "glm_crime", f"glm: + 311 {group} only ({len(cols)} features)"))
    lift = pl.DataFrame(results)
    lift.write_csv(MODEL / "step5_lift.csv")

    bonf = 1 - 0.05 / len(types)
    type_rows = []
    for t in types:
        name = f"glm_type_{t}"
        p = run(rows, crime + [f"s_{t}_4w", f"s_{t}_accel_4w"], fit_glm, name).join(preds.select(*KEYS, "glm_crime"), on=KEYS)
        s, lo, hi = skill_ci(p, name, "glm_crime")
        _, blo, bhi = skill_ci(p, name, "glm_crime", level=bonf)
        type_rows.append({"sr_type": t, "lift": s, "lo": lo, "hi": hi, "bonferroni_lo": blo, "bonferroni_hi": bhi})
    type_lift = pl.DataFrame(type_rows).sort("lift", descending=True)
    type_lift.write_csv(MODEL / "step5_type_lift.csv")

    # Direction of the 311 effects in the full GLM, final fold.
    _, infos = walk_forward(rows, crime + sr, fit_glm, "glm_full")
    coefs = infos[-1]["coefs"]
    coef_table = (pl.DataFrame({"feature": list(coefs), "coef": list(coefs.values())})
                  .filter(pl.col("feature").str.starts_with("s"))
                  .with_columns(((pl.col("coef").exp() - 1) * 100).alias("pct_per_sd"))
                  .sort(pl.col("coef").abs(), descending=True))
    coef_table.write_csv(MODEL / "step5_glm_311_coefs.csv")

    by_year = (preds.group_by(pl.col("cutoff").dt.year().alias("year"))
               .agg(*((1 - poisson_deviance("y", m).sum() / poisson_deviance("y", f"{m[:3]}_crime").sum()).alias(m)
                      for m in ("glm_full", "gbm_full", "glm_placebo")))
               .sort("year"))

    print(f"\n{len(crime)} crime features, {len(sr)} 311 features; total {time.time() - t0:.0f}s\n")
    with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=4, tbl_hide_dataframe_shape=True, fmt_str_lengths=50):
        print(lift.select("comparison", "lift_r8", "lo_r8", "hi_r8", "lift_r7", "lo_r7", "hi_r7", "auc_r8", "auc_ref_r8", "auc_r7", "auc_ref_r7"))
        print("\nlift over crime-only by year (r8):")
        print(by_year)
        print(f"\nGLM lift from each 311 type alone (Bonferroni level {bonf:.4f}), top 10:")
        print(type_lift.head(10))
        print("\nfull GLM, final fold: 311 effects per SD, top 10:")
        print(coef_table.head(10).select("feature", "pct_per_sd"))


if __name__ == "__main__":
    main()
