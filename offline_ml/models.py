"""Step 4: models on top of the bar, walked forward through the test folds.

    offline_ml/.venv/bin/python offline_ml/models.py

The bar sets the citywide total. The models decide how that total spreads across cells.
- **Training:** each cell's expected count is rescaled so the week's cells sum to what the
  city actually had. The models then learn only which cells run hot or cold *relative to the
  rest of the city*, and early stopping judges exactly that.
- **Forecast:** expected × m(x), rescaled so the cells again sum to the bar's citywide total.

Why not let the models set the citywide level too? It's a time series with only ~5 years of
history. Tried that way, the tree model learned citywide swings that didn't repeat. It
forecast Q1 2025 at 0.77× the bar when the city came in at 0.94× (-84% skill that quarter).
The local parts of both models were steady every year. Details are in README.md.

The two models:
- `glm`: Poisson regression with an L2 penalty, on transformed features
- `gbm`: gradient-boosted trees with a Poisson loss (scikit-learn's HistGradientBoosting)

scikit-learn's Poisson models have no offset argument, so both fit y / expected with
sample_weight = expected. The Poisson deviance of y against expected × m equals expected ×
the deviance of y / expected against m, so this is the same fit as a log(expected) offset.

Step 4 uses the local crime features. Step 5 reuses `walk_forward` with 311 added.

Outputs (offline_data/model/):
    step4_predictions.parquet   per test (cell, cutoff): target, bar, both models, fold
    step4_summary.csv           scores overall and by year, at r8 and r7
    step4_glm_coefficients.csv  final-fold coefficients (per SD of each transformed feature)
    step4_gbm_importance.csv    final-fold permutation importance, as % of test deviance
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Callable

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_poisson_deviance

from backtest import BAR, make_folds, score, skill_ci
from config import CELL_RES, FIRST_CUTOFF, FOLD_WEEKS, HORIZON_WEEKS, MODEL, REPORT_RES
from features import feature_columns
from sklearn.linear_model import PoissonRegressor

CELL = f"h3_r{CELL_RES}"
PARENT = f"h3_r{REPORT_RES}"

GLM_ALPHA = 1e-4            # light L2: the lookback windows overlap, so their logs are collinear
GBM_PARAMS = dict(
    loss="poisson", learning_rate=0.05, max_iter=1000, max_leaf_nodes=31,
    min_samples_leaf=200,   # the target is noisy, so leaves need many rows to mean anything
    l2_regularization=1.0, early_stopping=True, n_iter_no_change=30, random_state=0,
)
VALID_WEEKS = FOLD_WEEKS    # early stopping holds out the newest quarter of each fold's training cutoffs

Fit = Callable[[pl.DataFrame, pl.DataFrame, list[str]], tuple[np.ndarray, dict]]


def load_rows() -> pl.DataFrame:
    return pl.read_parquet(MODEL / "features.parquet").filter(pl.col("y").is_not_null())


def to_city_total(df: pl.DataFrame, col: str, total: str) -> pl.DataFrame:
    """Rescale `col` within each cutoff so the cells sum to `total`'s citywide sum."""
    return df.with_columns((pl.col(col) * pl.col(total).sum().over("cutoff") / pl.col(col).sum().over("cutoff")).alias(col))


def ratio_target(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    e = df["expected"].to_numpy()
    return df["y"].to_numpy() / e, e


# ---------------------------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------------------------

def glm_design(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Put features on a scale where a linear effect on log(rate) is plausible: log1p for
    counts, sin/cos for week of year, and ratios (already logs) and shares as they are."""
    out = []
    for c in cols:
        if c == "t_week_of_year":
            angle = 2 * np.pi * pl.col(c) / 52.18
            out += [angle.sin().alias("t_week_sin"), angle.cos().alias("t_week_cos")]
        elif df.schema[c].is_integer():
            out.append(pl.col(c).cast(pl.Float64).log1p().alias(c))
        else:
            out.append(pl.col(c).cast(pl.Float64).alias(c))
    return df.select(out)


def fit_glm(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    xtr, xte = glm_design(train, cols), glm_design(test, cols)
    med = xtr.median().row(0, named=True)
    xtr, xte = (x.with_columns(pl.col(c).fill_null(med[c]) for c in x.columns) for x in (xtr, xte))
    mu, sd = xtr.mean().row(0, named=True), xtr.std().row(0, named=True)
    z = lambda x: x.select(((pl.col(c) - mu[c]) / (sd[c] or 1.0)).alias(c) for c in x.columns).to_numpy()
    r, e = ratio_target(train)
    model = PoissonRegressor(alpha=GLM_ALPHA, solver="newton-cholesky", max_iter=300)
    model.fit(z(xtr), r, sample_weight=e)
    return test["expected"].to_numpy() * model.predict(z(xte)), {"coefs": dict(zip(xtr.columns, model.coef_))}


# ---------------------------------------------------------------------------------------------
# Gradient-boosted trees
# ---------------------------------------------------------------------------------------------

def gbm_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(pl.col(cols).cast(pl.Float64)).to_numpy()


def fit_gbm(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    """Pick the number of trees on a time-ordered holdout, then refit on everything.

    The holdout is the newest quarter of training cutoffs, with a horizon-length gap so no
    holdout target overlaps a fitting row. The refit uses all training rows with that many
    trees.
    """
    last = train["cutoff"].max()
    val_start = last - timedelta(weeks=VALID_WEEKS - 1)
    fit_part = train.filter(pl.col("cutoff") <= val_start - timedelta(weeks=HORIZON_WEEKS))
    val_part = train.filter(pl.col("cutoff") >= val_start)
    r, e = ratio_target(fit_part)
    rv, ev = ratio_target(val_part)
    probe = HistGradientBoostingRegressor(**GBM_PARAMS)
    probe.fit(gbm_matrix(fit_part, cols), r, sample_weight=e,
              X_val=gbm_matrix(val_part, cols), y_val=rv, sample_weight_val=ev)
    n_trees = max(probe.n_iter_ - GBM_PARAMS["n_iter_no_change"], 10)

    r, e = ratio_target(train)
    model = HistGradientBoostingRegressor(**{**GBM_PARAMS, "early_stopping": False, "max_iter": n_trees})
    model.fit(gbm_matrix(train, cols), r, sample_weight=e)
    return test["expected"].to_numpy() * model.predict(gbm_matrix(test, cols)), {"model": model, "n_trees": n_trees}


# ---------------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------------

def walk_forward(rows: pl.DataFrame, cols: list[str], fit: Fit, name: str,
                 id_cols: tuple[str, ...] = (CELL, PARENT)) -> tuple[pl.DataFrame, list[dict]]:
    """Refit once per fold and predict that fold's test cutoffs.

    Each fold trains on every earlier cutoff whose target is complete. It returns the test
    rows (`id_cols` plus target) with a `name` column, plus each fold's fit info.
    """
    parts, infos = [], []
    for f in make_folds(rows["cutoff"].unique().to_list()):
        train = rows.filter(pl.col("cutoff").is_between(FIRST_CUTOFF, f["train_last"]))
        test = rows.filter(pl.col("cutoff").is_between(f["test_first"], f["test_last"]))
        assert train["cutoff"].max() + timedelta(weeks=HORIZON_WEEKS) <= test["cutoff"].min()
        # Training targets are known, so the expected counts can be pinned to each week's
        # actual citywide total. The model then only learns local departures.
        pred, info = fit(to_city_total(train, "expected", "y"), test, cols)
        out = test.select(*id_cols, "cutoff", "y", "expected").with_columns(
            pl.lit(f["fold"]).alias("fold"), pl.Series(name, pred))
        parts.append(to_city_total(out, name, "expected"))
        infos.append({**info, **f, "train_rows": train.height})
    return pl.concat(parts), infos


def gbm_importance(rows: pl.DataFrame, info: dict, cols: list[str]) -> pl.DataFrame:
    """How much the final fold's test deviance rises when each feature is shuffled.

    The expected counts are pinned to the actual citywide totals, so this measures the local
    fit the model was trained for.
    """
    test = to_city_total(rows.filter(pl.col("cutoff").is_between(info["test_first"], info["test_last"])), "expected", "y")
    r, e = ratio_target(test)
    x = gbm_matrix(test, cols)
    base = mean_poisson_deviance(r, info["model"].predict(x), sample_weight=e)
    imp = permutation_importance(info["model"], x, r, sample_weight=e, scoring="neg_mean_poisson_deviance",
                                 n_repeats=5, random_state=0)
    return pl.DataFrame({"feature": cols, "pct_deviance_increase": 100 * imp.importances_mean / base,
                         "sd": 100 * imp.importances_std / base}).sort("pct_deviance_increase", descending=True)


def main() -> None:
    rows = load_rows()
    cols = feature_columns(rows, ("crime",))
    t0 = time.time()
    glm_pred, glm_info = walk_forward(rows, cols, fit_glm, "glm_crime")
    t1 = time.time()
    gbm_pred, gbm_info = walk_forward(rows, cols, fit_gbm, "gbm_crime")
    t2 = time.time()

    keys = [CELL, "cutoff"]
    preds = glm_pred.join(gbm_pred.select(*keys, "gbm_crime"), on=keys).rename({"expected": BAR})
    preds.write_parquet(MODEL / "step4_predictions.parquet")
    models = [BAR, "glm_crime", "gbm_crime"]
    summary = score(preds, models)
    summary.write_csv(MODEL / "step4_summary.csv")

    coefs = glm_info[-1]["coefs"]
    coef_table = pl.DataFrame({"feature": list(coefs), "coef": list(coefs.values())}).with_columns(
        ((pl.col("coef").exp() - 1) * 100).alias("pct_per_sd")).sort(pl.col("coef").abs(), descending=True)
    coef_table.write_csv(MODEL / "step4_glm_coefficients.csv")
    importance = gbm_importance(rows, gbm_info[-1], cols)
    importance.write_csv(MODEL / "step4_gbm_importance.csv")

    print(f"{len(cols)} local crime features; {len(gbm_info)} folds; glm {t1 - t0:.0f}s, gbm {t2 - t1:.0f}s "
          f"(trees per fold: {min(i['n_trees'] for i in gbm_info)}-{max(i['n_trees'] for i in gbm_info)})\n")
    rolled = preds.group_by(PARENT, "cutoff", "fold").agg(pl.col("y", *models).sum())
    for m in ("glm_crime", "gbm_crime"):
        for grain, df in ((f"r{CELL_RES}", preds), (f"r{REPORT_RES}", rolled)):
            s, lo, hi = skill_ci(df, m, BAR)
            print(f"{m} vs bar at {grain}: deviance skill {s:+.1%} (95% CI {lo:+.1%} to {hi:+.1%})")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True):
        print()
        print(summary.filter(pl.col("year") == "all").drop("year", "rows", "noise_floor"))
        print("\nby year, r8:")
        print(summary.filter((pl.col("grain") == f"r{CELL_RES}") & (pl.col("year") != "all") & (pl.col("model") != BAR))
              .select("year", "model", "skill_vs_bar", "auc_above_normal")
              .pivot(on="model", index="year", values=["skill_vs_bar", "auc_above_normal"]))
        print("\nglm, final fold: effect on the forecast of a 1-SD higher feature, top 10:")
        print(coef_table.head(10).select("feature", "pct_per_sd"))
        print("\ngbm, final fold: % rise in test deviance when shuffled, top 10:")
        print(importance.head(10))


if __name__ == "__main__":
    main()
