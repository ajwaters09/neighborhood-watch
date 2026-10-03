"""Walk-forward folds and the scores the backtest reports.

Each test quarter is forecast by models trained only on weeks whose outcome was already in the
feed at the quarter's first cutoff, so the scores are what the live model would have done.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.outlook.specs import FOLD_WEEKS, REPORT_LAG_DAYS


def train_last_for(test_first: date) -> date:
    """The newest cutoff whose target week [c, c+7) was already in the feed at `test_first`:
    c + 7 <= test_first - L. That's the Monday 3 weeks back, with L = 8."""
    latest = test_first - timedelta(days=REPORT_LAG_DAYS + 7)
    return latest - timedelta(days=latest.weekday())


def folds(test_cutoffs: list[date]) -> list[dict]:
    """Consecutive FOLD_WEEKS-week blocks of test cutoffs, each with the last week it may train on."""
    out = []
    for i in range(0, len(test_cutoffs), FOLD_WEEKS):
        block = test_cutoffs[i:i + FOLD_WEEKS]
        out.append({"fold": len(out), "test_first": block[0], "test_last": block[-1], "train_last": train_last_for(block[0])})
    return out


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance: the loss the GLMs minimize, and the basis of every skill score here."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))))


def skill(y: np.ndarray, forecast: np.ndarray, reference: np.ndarray) -> float:
    """1 - deviance(forecast) / deviance(reference): the share of the reference's miss removed."""
    return 1 - deviance(y, forecast) / deviance(y, reference)


# ---------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------

def city_scores(city_bt: pd.DataFrame) -> dict[str, float]:
    """A citywide backtest's scores against the calibrated bar."""
    y = city_bt["y"].to_numpy()
    return {
        "city_skill": skill(y, city_bt["forecast"].to_numpy(), city_bt["bar_calibrated"].to_numpy()),
        "city_error_bar": float((city_bt["bar_calibrated"] / city_bt["y"] - 1).abs().mean()),
        "city_error_model": float((city_bt["forecast"] / city_bt["y"] - 1).abs().mean()),
    }


def _climatology(area_bt: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    """For each scored row: the above-normal rate over every fold before its own (fold 0 included)."""
    g = area_bt.groupby("fold")["above"]
    return scored["fold"].map(g.sum().cumsum().shift(1) / g.size().cumsum().shift(1))


def area_scores(area_bt: pd.DataFrame, reps: int = 2000) -> dict[str, float]:
    """An area backtest's scores, over every fold after the first (fold 0 has no earlier misses
    to set the calls' dispersion from).

    Hit-rate CIs resample whole folds: calls in the same week share one citywide call and stand
    or fall together.
    """
    s = area_bt[area_bt["fold"] > 0]
    called = s[s["call"] != "unclear"]
    right = (called["call"] == "up") == called["above"]
    hit_lo = hit_hi = float("nan")
    if len(called) and reps:
        per = called.assign(right=right).groupby("fold").agg(r=("right", "sum"), n=("right", "size"))
        idx = np.random.default_rng(0).integers(0, len(per), size=(reps, len(per)))
        boot = per["r"].to_numpy()[idx].sum(axis=1) / per["n"].to_numpy()[idx].sum(axis=1)
        hit_lo, hit_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    a = s["above"].astype(float)
    brier, brier_clim = float(((s["p_above"] - a) ** 2).mean()), float(((_climatology(area_bt, s) - a) ** 2).mean())
    return {
        "area_skill": skill(s["y"].to_numpy(), s["forecast"].to_numpy(), s["normal"].to_numpy()),
        "base_rate": float(s["above"].mean()), "share_called": len(called) / len(s),
        "hit_rate": float(right.mean()) if len(called) else float("nan"), "hit_lo": hit_lo, "hit_hi": hit_hi,
        "brier_skill": 1 - brier / brier_clim, "scored_area_weeks": len(s),
    }


def auc(p: pd.Series, outcome: pd.Series) -> float:
    """How well P(above) ranks the weeks that came in above normal: 0.5 is chance, 1 is perfect.
    The rank-sum form of the ROC area, so ties count half."""
    ok = p.notna() & outcome.notna()
    p, y = p[ok].to_numpy(), outcome[ok].astype(bool).to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summary(city_bt: pd.DataFrame, area_bt: pd.DataFrame, live: date, last_full: date) -> pd.DataFrame:
    """One row: the live week's dates and the champion's backtest scores, for the app."""
    return pd.DataFrame([{
        "live_cutoff": live, "live_week_end": live + timedelta(days=6), "data_through": last_full,
        "test_first": city_bt["cutoff"].min(), "test_last": city_bt["cutoff"].max(), "test_weeks": len(city_bt),
        **city_scores(city_bt), **area_scores(area_bt),
    }])


def leaderboard(city_bts: dict[str, pd.DataFrame], area_bts: dict[str, pd.DataFrame], city_specs: dict,
                area_specs: dict, champion_city: str, champion_area: str) -> pd.DataFrame:
    """One row per model, scored on the same test weeks.

    City models: deviance skill over the calibrated bar, and mean absolute weekly error. Area
    models (all on the champion citywide call): deviance skill over each area's normal, Brier
    skill over climatology, how often they make a call and how often it's right, and AUC.
    """
    rows = []
    for name, bt in city_bts.items():
        sc = city_scores(bt)
        spec = city_specs[name]
        rows.append({"level": "city", "model": name, "champion": name == champion_city,
                     "features": ", ".join(spec.groups) or "(none)", "description": spec.description,
                     "skill": sc["city_skill"], "weekly_error": sc["city_error_model"]})
    for name, bt in area_bts.items():
        sc = area_scores(bt, reps=0)
        spec = area_specs[name]
        scored = bt[bt["fold"] > 0]
        rows.append({"level": "area", "model": name, "champion": name == champion_area,
                     "features": ", ".join(spec.features) or "(none)", "description": spec.description,
                     "skill": sc["area_skill"], "brier_skill": sc["brier_skill"],
                     "share_called": sc["share_called"], "hit_rate": sc["hit_rate"],
                     "auc": auc(scored["p_above"], scored["above"])})
    return pd.DataFrame(rows)


def calibration(area_bt: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability of P(above normal): per probability bin, the mean prediction against how often
    areas really came in above normal. A calibrated model sits on the diagonal."""
    s = area_bt[(area_bt["fold"] > 0) & area_bt["p_above"].notna()]
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(s["p_above"], edges[1:-1]), 0, bins - 1)
    g = s.assign(bin=b, above=s["above"].astype(float)).groupby("bin")
    out = g.agg(mean_p=("p_above", "mean"), observed=("above", "mean"), n=("above", "size")).reset_index()
    out["bin_lo"], out["bin_hi"] = edges[out["bin"]], edges[out["bin"] + 1]
    return out[["bin_lo", "bin_hi", "mean_p", "observed", "n"]]
