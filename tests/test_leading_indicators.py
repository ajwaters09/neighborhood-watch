"""Offline checks for the 311 -> crime lead-lag method (src/leading_indicators.py).

A synthetic panel of 77 areas x 72 months where one 311 type drives one crime category a month
later, on top of area sizes and a shared seasonal swing (which the demeaning must remove):
- the planted pair is flagged, at lag 1, and the unrelated pairs aren't
- the shuffled-area placebo finds nothing
- residuals sum to zero per area and per month

    python -m pytest tests/test_leading_indicators.py
"""

import numpy as np
import pandas as pd

from src import leading_indicators as li


def panel(seed=0, effect=0.4):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2019-01-01", periods=72, freq="MS").date
    size = rng.lognormal(3, 0.5, 77)
    season = 1 + 0.3 * np.sin(np.arange(72) * 2 * np.pi / 12)
    shock = rng.normal(0, 0.3, (77, 72))                       # the 311 type's own area-month swings
    rows = []
    for a in range(77):
        for t, m in enumerate(months):
            base = size[a] * season[t]
            lead = shock[a, t - 1] if t else 0.0
            rows += [
                (a + 1, m, "311_streetlight_out", rng.poisson(base * np.exp(shock[a, t]))),
                (a + 1, m, "311_graffiti", rng.poisson(base)),
                (a + 1, m, "crime_property", rng.poisson(base * 2 * np.exp(effect * lead))),
                (a + 1, m, "crime_violent", rng.poisson(base)),
            ]
    return pd.DataFrame(rows, columns=["community_area", "period", "metric", "metric_count"])


def test_planted_lead_is_flagged_and_nothing_else():
    out = li.lead_lag(panel())
    flagged = out[out["is_leading"]]
    assert set(zip(flagged["sr_metric"], flagged["crime_metric"], flagged["lag_months"])) == {
        ("311_streetlight_out", "crime_property", 1)}
    row = flagged.iloc[0]
    assert row["r"] > row["lag0_r"] and row["r"] > row["reverse_r"]
    assert len(out) == 2 * 2 * len(li.LAGS)


def test_placebo_and_residuals():
    monthly = panel()
    assert li.placebo(monthly)["r"].abs().max() < 0.05
    r = li.residuals(monthly)
    assert r.groupby(["metric", "community_area"])["resid"].sum().abs().max() < 1e-9
    assert r.groupby(["metric", "period"])["resid"].sum().abs().max() < 1e-9


def test_p_value_edges():
    assert li.p_value(None, 100) is None and li.p_value(0.5, 2) is None and li.p_value(1.0, 100) is None
    assert li.p_value(0.0, 1000) == 1.0
    assert li.p_value(0.1, 5000) < 1e-10
