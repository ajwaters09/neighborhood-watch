"""The outlook's models: a Poisson GLM on top of the seasonal bar (citywide), a one-feature lean
(per area), and the negative binomial that turns a forecast into a call.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

from src.outlook.specs import (AREA_TRAIN_FROM, CITY_TRAIN_FROM, COVID, DOWN, FEATURE_GROUPS, UP, AreaModelSpec,
                               CityModelSpec)


class OffsetGLM:
    """Poisson regression on y / expected with weight = expected: the same fit as a
    log(expected) offset, since scikit-learn has no offset argument. Features are
    standardized (nulls filled with the training median) so the ridge treats them alike."""

    def __init__(self, cols: list[str], alpha: float):
        self.cols, self.alpha = cols, alpha

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[self.cols].astype(float).fillna(self.med) - self.mu) / self.sd).to_numpy()

    def fit(self, df: pd.DataFrame, expected: np.ndarray) -> "OffsetGLM":
        x = df[self.cols].astype(float)
        self.med = x.median()
        x = x.fillna(self.med)
        self.mu, self.sd = x.mean(), x.std().replace(0, 1.0).fillna(1.0)
        self.model = PoissonRegressor(alpha=self.alpha, solver="newton-cholesky", max_iter=500)
        self.model.fit(self._x(df), df["y"].to_numpy() / expected, sample_weight=expected)
        return self

    def multiplier(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._x(df))

    def per_unit(self) -> pd.Series:
        """Coefficients per unit of each raw feature (log scale)."""
        return pd.Series(self.model.coef_, index=self.cols) / self.sd


class FittedLevel:
    """The model with no features: a constant multiplier, the bar's fitted level k."""

    def __init__(self, k: float):
        self.k, self.cols = k, []

    def multiplier(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.k)

    def per_unit(self) -> pd.Series:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------------------------
# Citywide
# ---------------------------------------------------------------------------------------------

def city_columns(spec: CityModelSpec) -> list[str]:
    return [c for g in spec.groups for c in FEATURE_GROUPS[g].columns]


def city_training_rows(rows: pd.DataFrame, train_last: date) -> pd.DataFrame:
    """Weeks from CITY_TRAIN_FROM through train_last, minus COVID, with every group's required
    columns known. The same weeks for every model, so their scores compare."""
    covid = rows["cutoff"].between(COVID[0], COVID[1] - timedelta(days=1))
    required = ["y", "bar"] + [c for g in FEATURE_GROUPS.values() for c in g.required]
    return rows[(rows["cutoff"] >= CITY_TRAIN_FROM) & (rows["cutoff"] <= train_last) & ~covid].dropna(subset=required)


def fit_city(rows: pd.DataFrame, train_last: date, spec: CityModelSpec) -> tuple[OffsetGLM | FittedLevel, float]:
    """The fitted model and k, the bar's fitted level (sum y / sum bar over training weeks)."""
    tr = city_training_rows(rows, train_last)
    k = tr["y"].sum() / tr["bar"].sum()
    cols = city_columns(spec)
    if not cols:
        return FittedLevel(k), k
    return OffsetGLM(cols, spec.alpha).fit(tr, tr["bar"].to_numpy()), k


def drivers(model: OffsetGLM | FittedLevel, df: pd.DataFrame) -> pd.DataFrame:
    """Each feature group's effect on the forecast, as a % change against a quiet week (on
    trend, normal weather, no holiday): exp(sum of beta x over the group) - 1. Groups the model
    doesn't use come out 0."""
    b = model.per_unit()
    out = {}
    for name, group in FEATURE_GROUPS.items():
        cols = [c for c in group.columns if c in b.index]
        x = df[cols].astype(float).fillna(0.0).to_numpy() if cols else np.zeros((len(df), 0))
        out[f"pct_{name}"] = np.expm1(x @ b[cols].to_numpy())
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------------------------

def fit_area(rows: pd.DataFrame, train_last: date, spec: AreaModelSpec) -> OffsetGLM | None:
    """The area lean, trained with each week's expected counts pinned to the week's actual
    total across areas, so it learns only who runs hot or cold relative to the rest. None for a
    spec with no features."""
    if not spec.features:
        return None
    tr = rows[(rows["cutoff"] >= AREA_TRAIN_FROM) & (rows["cutoff"] <= train_last)].dropna(subset=["y", "expected"])
    tr = tr[tr["expected"] > 0]
    pinned = tr["expected"] * tr.groupby("cutoff")["y"].transform("sum") / tr.groupby("cutoff")["expected"].transform("sum")
    return OffsetGLM(list(spec.features), spec.alpha).fit(tr, pinned.to_numpy())


def area_lean(model: OffsetGLM | None, df: pd.DataFrame) -> np.ndarray:
    """expected x lean, rescaled so each week's areas sum to their summed bar again. With no
    model, just `expected`."""
    if model is None:
        return df["expected"].to_numpy()
    raw = df["expected"].to_numpy() * model.multiplier(df)
    s = pd.Series(raw, index=df.index)
    return (s * df.groupby("cutoff")["expected"].transform("sum") / s.groupby(df["cutoff"]).transform("sum")).to_numpy()


# ---------------------------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------------------------

def dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """NB2 dispersion alpha (Var = mu + alpha mu^2), by method of moments on out-of-sample rows."""
    return max(float((((y - mu) ** 2) - mu).sum() / (mu ** 2).sum()), 1e-4)


def p_above(mu: np.ndarray, threshold: np.ndarray, alpha: float) -> np.ndarray:
    """P(Y > threshold) for Y ~ NB(mean mu, dispersion alpha)."""
    k = 1 / alpha
    return nbinom.sf(np.floor(threshold), k, k / (k + mu))


def call(p: np.ndarray) -> np.ndarray:
    return np.where(p >= UP, "up", np.where(p <= DOWN, "down", "unclear"))
