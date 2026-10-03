# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 08 - Next-week outlook: will each community area run above its normal this week?
# MAGIC
# MAGIC Forecasts all crime per community area for the week starting each Monday, and calls it
# MAGIC **up**, **down** or **unclear** against the area's normal for this time of year
# MAGIC (`src/outlook/`; the method is in its `__init__` and in `docs/modeling.md`):
# MAGIC - **Citywide:** the seasonal bar x a Poisson GLM on momentum, holidays and next week's
# MAGIC   weather forecast.
# MAGIC - **Per area:** the area's own bar x that citywide call x a one-feature momentum lean.
# MAGIC - **Calls:** a negative binomial around the forecast. Up at P(above normal) >= 0.65, down
# MAGIC   at <= 0.35.
# MAGIC
# MAGIC Every run re-does the whole walk-forward backtest (quarterly refits from 2022, seconds) for
# MAGIC every model in `src/outlook/specs.py`, then forecasts the live week with the champions. So the
# MAGIC track record the app quotes is always current, and the leaderboard shows what each piece of
# MAGIC the model adds. The crime feed runs about 9 days behind, and the features only use crime from
# MAGIC before cutoff - 8 days.
# MAGIC
# MAGIC | Table | Rows |
# MAGIC |---|---|
# MAGIC | `outlook_city_week` | per test week + the live week: citywide forecast vs. normal, and its drivers |
# MAGIC | `outlook_area_week` | per area x week: normal, forecast, P(above), call, and its citywide and local parts |
# MAGIC | `outlook_summary` | one row: the live week and the champions' track record, for the app |
# MAGIC | `outlook_coefficients` | the live citywide model's effects, per unit of each feature |
# MAGIC | `outlook_leaderboard` | every model's backtest scores on the same weeks, champions flagged |
# MAGIC | `outlook_calibration` | P(above normal) by bin against how often areas came in above |
# MAGIC
# MAGIC Each run's parameters, scores and charts are also logged to MLflow (`log_to_mlflow`).
# MAGIC
# MAGIC Reads `silver_crimes`, `bronze_weather_observed` and `bronze_weather_forecast`. Runs nightly
# MAGIC after `03` and `02b`.

# COMMAND ----------

# MAGIC %pip install "scikit-learn>=1.3" "scipy>=1.10" "matplotlib>=3.7"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src import outlook
from src.community_areas import AREA_NAMES
from src.outlook import report

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.dropdown("log_to_mlflow", "true", ["true", "false"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LOG_TO_MLFLOW = dbutils.widgets.get("log_to_mlflow") == "true"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

RUN_DATE = datetime.now(ZoneInfo("America/Chicago")).date()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inputs
# MAGIC
# MAGIC Daily crime counts per area are ~700k rows, so Spark aggregates silver and only the counts
# MAGIC come to the driver. The newest day in the feed is usually part-loaded, so the model's
# MAGIC `last_full` day is the one before it.

# COMMAND ----------

area_daily = spark.sql(f"""
    SELECT community_area, to_date(date) AS day, count(*) AS n
    FROM {T('silver_crimes')}
    WHERE date >= '2001-01-01' AND community_area BETWEEN 1 AND 77
    GROUP BY 1, 2
""").toPandas()

obs = spark.sql(f"SELECT day, t_mean, precip, snow FROM {T('bronze_weather_observed')} ORDER BY day").toPandas()
forecasts = spark.sql(f"SELECT target_day, issue_date, model, t_mean, precip FROM {T('bronze_weather_forecast')}").toPandas()

# Arrow can hand DATE columns back as timestamps; the model works in plain dates.
as_date = lambda s: pd.to_datetime(s).dt.date
area_daily["day"] = as_date(area_daily["day"])
obs["day"] = as_date(obs["day"])
forecasts["target_day"], forecasts["issue_date"] = as_date(forecasts["target_day"]), as_date(forecasts["issue_date"])
last_full = area_daily["day"].max() - timedelta(days=1)

print(f"crime: {len(area_daily):,} area-days, {area_daily['day'].min()} .. {last_full} (run date {RUN_DATE})")
print(f"weather: observed {obs['day'].min()} .. {obs['day'].max()}; {len(forecasts):,} forecast rows, "
      f"newest issue {forecasts['issue_date'].max()}")
lag = (RUN_DATE - last_full).days
if lag > 12:
    print(f"WARNING: crime data ends {lag} days before today (the feed usually runs ~9 behind). "
          "Has 02_bronze_incremental been running? The live week will be older than this week.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backtest every model, forecast the live week

# COMMAND ----------

out = outlook.run(area_daily, obs, forecasts, last_full, RUN_DATE)
summary = out["summary"].iloc[0]
print(f"live week: {summary['live_cutoff']} .. {summary['live_week_end']}")
print(f"backtest {summary['test_first']} .. {summary['test_last']} ({summary['test_weeks']} weeks):")
print(f"  citywide: deviance skill {summary['city_skill']:+.1%}, weekly error {summary['city_error_bar']:.1%} (bar) -> "
      f"{summary['city_error_model']:.1%} (model)")
print(f"  areas: {summary['share_called']:.1%} of area-weeks called, {summary['hit_rate']:.1%} right "
      f"(95% CI {summary['hit_lo']:.0%}-{summary['hit_hi']:.0%}) vs a {summary['base_rate']:.0%} base rate; "
      f"Brier skill {summary['brier_skill']:+.1%}")

# The research backtest landed at ~23% called / ~75% right. Far off means the inputs changed.
if not (0.60 <= summary["hit_rate"] <= 0.90 and 0.05 <= summary["share_called"] <= 0.50):
    print("WARNING: backtest far from the research result (~23% called, ~75% right) -- check the inputs before trusting calls.")
display(out["leaderboard"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write

# COMMAND ----------

def save(df: pd.DataFrame, name: str) -> None:
    df = df.copy()
    for c in df.columns:          # nullable booleans (the live week's unknown outcome) -> plain objects
        if str(df[c].dtype) == "boolean":
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    (spark.createDataFrame(df).write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(T(name)))


city_weeks = out["city_weeks"].assign(as_of=RUN_DATE)
area_weeks = out["area_weeks"].assign(
    as_of=RUN_DATE, area_name=out["area_weeks"]["community_area"].map(AREA_NAMES))
area_weeks["community_area"] = area_weeks["community_area"].astype(int)
save(city_weeks, "outlook_city_week")
save(area_weeks, "outlook_area_week")
save(out["summary"].assign(as_of=RUN_DATE), "outlook_summary")
save(out["coefficients"].assign(as_of=RUN_DATE), "outlook_coefficients")
save(out["leaderboard"].assign(as_of=RUN_DATE), "outlook_leaderboard")
save(out["calibration"].assign(as_of=RUN_DATE), "outlook_calibration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The live week

# COMMAND ----------

live_city = city_weeks[city_weeks["is_live"]].iloc[0]
print(f"citywide: {live_city['pct_vs_normal']:+.1%} vs normal "
      f"(momentum {live_city['pct_momentum']:+.1%}, weather {live_city['pct_weather']:+.1%}, "
      f"holidays {live_city['pct_calendar']:+.1%}; forecast {live_city['fc_t_anom'] * 9 / 5:+.1f} °F vs normal)")
live_areas = area_weeks[area_weeks["is_live"]]
print(live_areas["call"].value_counts().to_dict())
display(spark.table(T("outlook_area_week")).where("is_live").orderBy("p_above", ascending=False)
        .select("community_area", "area_name", "normal", "forecast", "p_above", "call", "pct_citywide", "pct_local"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model card and charts

# COMMAND ----------

card = report.model_card(out)
fig = report.dashboard(out)
displayHTML(f"<pre style='white-space: pre-wrap'>{card}</pre>")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Track the run in MLflow
# MAGIC
# MAGIC The model refits every night by design, so what's worth tracking is each run's settings,
# MAGIC scores and charts. Comparing runs over time shows whether the backtest is drifting. A logging
# MAGIC failure never fails the job.

# COMMAND ----------

if LOG_TO_MLFLOW:
    try:
        import mlflow

        from src.outlook import specs

        user = spark.sql("SELECT current_user()").first()[0]
        mlflow.set_experiment(f"/Users/{user}/neighborhood-watch-outlook")
        with mlflow.start_run(run_name=f"outlook {RUN_DATE}"):
            mlflow.log_params({
                "live_week": str(summary["live_cutoff"]), "data_through": str(summary["data_through"]),
                "champion_city": specs.CHAMPION_CITY, "champion_area": specs.CHAMPION_AREA,
                "city_features": ", ".join(specs.CITY_MODELS[specs.CHAMPION_CITY].groups),
                "area_features": ", ".join(specs.AREA_MODELS[specs.CHAMPION_AREA].features),
                "up": specs.UP, "down": specs.DOWN, "report_lag_days": specs.REPORT_LAG_DAYS,
                "test_start": str(specs.TEST_START), "fold_weeks": specs.FOLD_WEEKS,
            })
            mlflow.log_metrics({k: float(summary[k]) for k in (
                "city_skill", "city_error_bar", "city_error_model", "area_skill", "base_rate", "share_called",
                "hit_rate", "hit_lo", "hit_hi", "brier_skill")})
            for r in out["leaderboard"].itertuples():
                mlflow.log_metric(f"skill_{r.level}_{r.model}", float(r.skill))
            mlflow.log_text(card, "model_card.md")
            mlflow.log_table(out["leaderboard"], "leaderboard.json")
            mlflow.log_table(out["calibration"], "calibration.json")
            mlflow.log_figure(fig, "outlook.png")
        print("logged to MLflow")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
