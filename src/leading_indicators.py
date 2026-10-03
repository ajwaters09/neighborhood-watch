"""Does 311 activity lead crime? The lead-lag method behind notebook 05.

1. Monthly counts per (community area, month) for every 311 and crime metric, y = log(1 + n).
2. Two-way demeaning per metric: subtract each area's own mean (big areas have more of
   everything) and each month's citywide mean (seasons and citywide trends move both). What's
   left is "unusually high or low for this area, this month".
3. For lags k = 0-3 months, correlate a 311 metric's residual in month t-k with a crime metric's
   residual in month t, pooled across areas and months. Then the same with crime leading 311.
4. A pair "leads" at its best lag k >= 1 when that correlation is significant after a Bonferroni
   correction across every test, at least MIN_R, stronger than the same-month correlation, and
   stronger than the reverse direction.

Notebook 05 does steps 1-3 in Spark and step 4 with `flag_leading`. The pandas versions of steps
1-3 here are the reference the tests and tutorials/04 run.

Caveats: correlation, not causation, and pooled citywide rather than per area. Months within an
area aren't independent, so the p-values are optimistic; Bonferroni, MIN_R and the reverse test
are there to keep that from flagging noise, and a shuffled-area placebo (`placebo`) should
find nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

LAGS = (0, 1, 2, 3)
ALPHA = 0.01      # family-wise, Bonferroni-corrected across all pairs x lags x directions
MIN_R = 0.05      # the smallest correlation worth surfacing, whatever its p-value


def residuals(monthly: pd.DataFrame) -> pd.DataFrame:
    """(community_area, period, metric, metric_count) -> (community_area, period, metric, resid):
    log1p counts with each area's mean and each month's mean removed, per metric."""
    d = monthly.assign(y=np.log1p(monthly["metric_count"].astype(float)))
    by = lambda *keys: d.groupby(list(keys))["y"].transform("mean")
    d["resid"] = d["y"] - by("metric", "community_area") - by("metric", "period") + by("metric")
    return d[["community_area", "period", "metric", "resid"]]


def _shift_months(period: pd.Series, k: int) -> pd.Series:
    return (pd.to_datetime(period) + pd.DateOffset(months=k)).dt.date


def lagged_correlations(leader: pd.DataFrame, follower: pd.DataFrame, k: int) -> pd.DataFrame:
    """corr(leader[area, t-k], follower[area, t]) per (sr_metric, crime_metric), pooled over areas
    and months. Both inputs are `residuals` output, one side 311 metrics and the other crime;
    which one leads is the caller's choice."""
    lead = leader.assign(period=_shift_months(leader["period"], k))
    j = lead.merge(follower, on=["community_area", "period"], suffixes=("_lead", "_follow"))
    sr_first = j["metric_lead"].str.startswith("311_").all()
    j["sr_metric"] = j["metric_lead"] if sr_first else j["metric_follow"]
    j["crime_metric"] = j["metric_follow"] if sr_first else j["metric_lead"]
    out = (j.groupby(["sr_metric", "crime_metric"])
           .apply(lambda g: pd.Series({"r": g["resid_lead"].corr(g["resid_follow"]), "n": len(g)}), include_groups=False)
           .reset_index())
    out["n"] = out["n"].astype(int)
    return out.assign(lag_months=k)


def p_value(r: float | None, n: int | None) -> float | None:
    """Two-sided p-value for a Pearson correlation: t = r sqrt((n-2) / (1-r^2)), with a normal
    approximation (n is in the thousands)."""
    if r is None or pd.isna(r) or n is None or n < 3 or abs(r) >= 1:
        return None
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return math.erfc(abs(t) / math.sqrt(2))


def flag_leading(forward: pd.DataFrame, reverse: pd.DataFrame, alpha: float = ALPHA, min_r: float = MIN_R) -> pd.DataFrame:
    """Forward (311 leads) and reverse (crime leads) correlations -> the indicator_lead_lag rows:
    every (sr_metric, crime_metric, lag) with r, n, p_value, reverse_r, lag0_r and is_leading."""
    rev = reverse.rename(columns={"r": "reverse_r"})[["sr_metric", "crime_metric", "lag_months", "reverse_r"]]
    df = forward.merge(rev, on=["sr_metric", "crime_metric", "lag_months"], how="left")
    df["p_value"] = [p_value(r, n) for r, n in zip(df["r"], df["n"])]
    alpha_adj = alpha / (len(df) * 2)          # forward + reverse
    lag0 = df[df["lag_months"] == 0].set_index(["sr_metric", "crime_metric"])["r"]
    df["lag0_r"] = [lag0.get((s, c)) for s, c in zip(df["sr_metric"], df["crime_metric"])]
    df["is_leading"] = False
    for _, g in df[df["lag_months"] >= 1].groupby(["sr_metric", "crime_metric"]):
        best = g.sort_values("r", ascending=False).iloc[0]
        if (pd.notna(best["p_value"]) and best["p_value"] < alpha_adj and best["r"] >= min_r
                and (pd.isna(best["lag0_r"]) or best["r"] > best["lag0_r"])
                and (pd.isna(best["reverse_r"]) or best["r"] > best["reverse_r"])):
            df.loc[best.name, "is_leading"] = True
    return df[["sr_metric", "crime_metric", "lag_months", "r", "n", "p_value", "reverse_r", "lag0_r", "is_leading"]]


def lead_lag(monthly: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """The whole analysis on (community_area, period, metric, metric_count) rows with 311_* and
    crime_* metrics, complete months only."""
    resid = residuals(monthly)
    sr, crime = resid[resid["metric"].str.startswith("311_")], resid[resid["metric"].str.startswith("crime_")]
    forward = pd.concat([lagged_correlations(sr, crime, k) for k in lags], ignore_index=True)
    reverse = pd.concat([lagged_correlations(crime, sr, k) for k in lags], ignore_index=True)
    return flag_leading(forward, reverse)


def placebo(monthly: pd.DataFrame, k: int = 1, seed: int = 42) -> pd.DataFrame:
    """The lag-k forward correlations with each area's 311 history paired with a different,
    random area's crime. If the method works, every |r| here is near 0."""
    resid = residuals(monthly)
    sr, crime = resid[resid["metric"].str.startswith("311_")], resid[resid["metric"].str.startswith("crime_")]
    areas = sorted(sr["community_area"].unique())
    shuffled = list(np.random.default_rng(seed).permutation(areas))
    sr = sr.assign(community_area=sr["community_area"].map(dict(zip(areas, shuffled))))
    return lagged_correlations(sr, crime, k)
