"""The alert-rule shaping in webapp/data_access.py: how a followed trend's update reads against the
last one, and the row the My Areas banner shows. Offline; nothing connects.

    python -m pytest tests/test_alerts.py
"""

from decimal import Decimal

from webapp import data_access as da


def test_trend_direction_against_the_last_update():
    assert da.trend_direction(30.0, 20.0) == "climbing"
    assert da.trend_direction(10.0, 20.0) == "easing"
    assert da.trend_direction(22.0, 20.0) == "steady"
    assert da.trend_direction(20.0 + da.TREND_STEADY_PTS, 20.0) == "climbing"   # the edge counts as a move
    assert da.trend_direction(-4.0, 20.0) == "easing"                         # a rise turned into a drop
    assert da.trend_direction(15.0, None) is None


def test_shape_alert_casts_and_only_trend_rules_get_a_direction():
    trend = da.shape_alert(7, 6, "311_graffiti", Decimal("15"), "trend", Decimal("28.8"), Decimal("21.5"), "2026-09-26")
    assert trend["pct_change_vs_prior_window"] == 21.5 and trend["last_pct"] == 28.8
    assert trend["direction"] == "easing" and trend["window_days"] == da.ALERT_WINDOW_DAYS
    threshold = da.shape_alert(8, 6, "crime_total", 20, "threshold", 10, 25, "2026-09-26")
    assert threshold["direction"] is None and threshold["threshold_pct"] == 20.0
