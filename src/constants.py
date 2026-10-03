"""Values the pipeline notebooks and the app must agree on, in one place. No third-party imports,
so it loads anywhere (the deployed app included).
"""

from __future__ import annotations

from typing import Any

# The trailing windows notebook 04 computes (area_trend_rolling, area_trend_normals), which the
# app and the agent can ask for.
WINDOW_DAYS_OPTIONS = (30, 60, 90)

# area_profile's per-resident span: the last 12 months, what an area is like now. The table's
# `span` column leaves room for others.
PROFILE_SPAN = "12m"
PROFILE_YEARS = 1

# A 311 category counts as "rising" in an area when its trailing-window count is up at least
# RISING_PCT on a prior count of at least RISING_MIN_PRIOR. Used by the Signals card, the agent's
# get_leading_indicators, the one-click alert threshold, and notebook 09's narratives.
RISING_PCT = 15.0
RISING_MIN_PRIOR = 10

# The 311 request types tracked as physical-disorder signals: sr_type (exact strings, from the
# portal's "311 Service Requests - Request Types" reference, dgc7-2pdf) -> metric name. Notebook 04
# builds the 311_* metrics from these; add or remove types here.
SR_INDICATOR_MAP = {
    "Graffiti Removal Request": "311_graffiti",
    "Street Light Out Complaint": "311_streetlight_out",
    "Alley Light Out Complaint": "311_alley_light_out",
    "Abandoned Vehicle Complaint": "311_abandoned_vehicle",
    "Building Violation": "311_building_violation",
    "Vacant/Abandoned Building Complaint": "311_vacant_abandoned_building",
    "Sanitation Code Violation": "311_sanitation_violation",
    "Fly Dumping Complaint": "311_illegal_dumping",
    "Rodent Baiting/Rat Complaint": "311_rodent_rat",
    "Clean Vacant Lot Request": "311_vacant_lot",
}
# Types the portal warns carry placeholder addresses: left out of 311_total too.
SR_EXCLUDE_FROM_TOTAL = ("311 INFORMATION ONLY CALL", "Aircraft Noise Complaint")

# The crime categories by primary_type alone: notebook 03b's no-AI fallback (`method` = rules),
# and the approximation the tutorials use. Anything unlisted is "other". The pipeline's own map
# classifies (primary_type, description) pairs with ai_classify.
CRIME_CATEGORY_RULES = {
    "violent": ("HOMICIDE", "ASSAULT", "BATTERY", "ROBBERY", "CRIMINAL SEXUAL ASSAULT", "CRIM SEXUAL ASSAULT",
                "SEX OFFENSE", "KIDNAPPING", "HUMAN TRAFFICKING", "STALKING", "INTIMIDATION", "DOMESTIC VIOLENCE",
                "OFFENSE INVOLVING CHILDREN"),
    "property": ("THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE", "ARSON", "CRIMINAL TRESPASS"),
    "drugs": ("NARCOTICS", "OTHER NARCOTIC VIOLATION"),
    "weapons": ("WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION"),
    "public_order": ("PUBLIC PEACE VIOLATION", "LIQUOR LAW VIOLATION", "GAMBLING", "PROSTITUTION", "OBSCENITY",
                     "PUBLIC INDECENCY", "INTERFERENCE WITH PUBLIC OFFICER"),
    "fraud_financial": ("DECEPTIVE PRACTICE",),
}

# Daily weather normals: the mean over the same date +-NORMAL_HALF_WINDOW days in each of the
# previous NORMAL_YEARS years. The outlook (src/outlook/) measures forecast anomalies against
# these, and the Insights page's 7-day forecast shows the same normals.
NORMAL_YEARS, NORMAL_HALF_WINDOW = 10, 7

# Display names for the outlook's holiday features (src/outlook/specs.HOLIDAY_NAMES).
HOLIDAY_LABELS = {
    "new_years_day": "New Year's Day", "mlk_day": "MLK Day", "presidents_day": "Presidents' Day", "easter": "Easter",
    "memorial_day": "Memorial Day", "juneteenth": "Juneteenth", "independence_day": "the Fourth of July",
    "labor_day": "Labor Day", "columbus_day": "Columbus Day", "halloween": "Halloween", "veterans_day": "Veterans Day",
    "thanksgiving": "Thanksgiving", "christmas_eve": "Christmas Eve", "christmas": "Christmas",
    "new_years_eve": "New Year's Eve",
}

# Citywide percentile of an area's per-resident rate -> words, highest band first. Used by the
# app's usual-level badge and notebook 09's narratives.
LEVEL_BANDS = ((80, "among the highest"), (60, "above average"), (40, "near the middle"),
               (20, "below average"), (0, "among the lowest"))


def level_band(percentile: float | None) -> dict[str, Any] | None:
    """Citywide percentile -> {"band": 1-5, "label": "among the highest"}; 5 = highest."""
    if percentile is None:
        return None
    for i, (floor, label) in enumerate(LEVEL_BANDS):
        if percentile >= floor:
            return {"band": len(LEVEL_BANDS) - i, "label": label}
    return None
