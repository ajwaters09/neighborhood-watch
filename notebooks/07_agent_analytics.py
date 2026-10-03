# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 - Analytics: app activity, agent tool usage and chat quality
# MAGIC
# MAGIC Views over the Lakebase CDF history tables (see `06`), for dashboards and evals:
# MAGIC
# MAGIC | View | Grain | Answers |
# MAGIC |---|---|---|
# MAGIC | `subscription_events_daily` | day x event type x area | subscriptions, unsubscriptions and reports over time |
# MAGIC | `tool_invocation_events` | one agent tool call | every call, reads and failures included |
# MAGIC | `tool_usage_daily` | day x tool | requests and success rate per tool |
# MAGIC | `chat_turns_current` | one chat turn | the question, the tool trajectory, the answer, latency, feedback |
# MAGIC | `chat_turns_daily` | day | turns, conversations, tool use, error rate, latency percentiles, thumbs up |
# MAGIC | `chat_eval_set` | one chat turn | an evaluation-ready shape: request, tool sequence, response, user label |
# MAGIC
# MAGIC `tool_invocations` is written by `@_logged` on every agent tool (`src/agent_tools.py`), and
# MAGIC `chat_turns` by the app's `/chat` route, both best-effort. Run once after `06`; they're views.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

EVENTS = f"{CATALOG}.{SCHEMA}.subscription_events"
EVENTS_DAILY = f"{CATALOG}.{SCHEMA}.subscription_events_daily"

INVOCATIONS_HISTORY = f"{CATALOG}.{SCHEMA}.lb_tool_invocations_history"
INVOCATION_EVENTS = f"{CATALOG}.{SCHEMA}.tool_invocation_events"
TOOL_USAGE_DAILY = f"{CATALOG}.{SCHEMA}.tool_usage_daily"

CHAT_HISTORY = f"{CATALOG}.{SCHEMA}.lb_chat_turns_history"
CHAT_TURNS = f"{CATALOG}.{SCHEMA}.chat_turns_current"
CHAT_DAILY = f"{CATALOG}.{SCHEMA}.chat_turns_daily"
CHAT_EVAL_SET = f"{CATALOG}.{SCHEMA}.chat_eval_set"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. App activity: `subscription_events_daily`

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {EVENTS_DAILY} AS
    SELECT
        DATE(event_timestamp) AS event_date,
        event_type,
        community_area,
        COUNT(*) AS event_count
    FROM {EVENTS}
    GROUP BY DATE(event_timestamp), event_type, community_area
""")

print(f"Created view {EVENTS_DAILY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Agent tool usage: `tool_invocation_events` and `tool_usage_daily`
# MAGIC
# MAGIC `subscription_events` only sees successful writes. `tool_invocations` sees every tool call,
# MAGIC reads and failures included. It's insert-only, so only insert rows are kept.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {INVOCATION_EVENTS} AS
    SELECT
        invocation_id,
        _timestamp AS event_timestamp,
        tool_name,
        user_id,
        community_area,
        success,
        error_message
    FROM {INVOCATIONS_HISTORY}
    WHERE _pg_change_type = 'insert'
""")

print(f"Created view {INVOCATION_EVENTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC Requests over time are `SUM(request_count)` by `event_date`; success rates are per `tool_name`.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {TOOL_USAGE_DAILY} AS
    SELECT
        DATE(event_timestamp) AS event_date,
        tool_name,
        COUNT(*) AS request_count,
        SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failure_count,
        ROUND(
            SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*) * 100, 1
        ) AS success_rate_pct
    FROM {INVOCATION_EVENTS}
    GROUP BY DATE(event_timestamp), tool_name
""")

print(f"Created view {TOOL_USAGE_DAILY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chat turns: `chat_turns_current`, `chat_turns_daily`, `chat_eval_set`
# MAGIC
# MAGIC `chat_turns` keeps the whole conversational unit: the question, every tool call with its
# MAGIC arguments, outcome and latency, the answer, and a thumbs up or down. Feedback updates a row
# MAGIC after it's written, so `chat_turns_current` keeps each turn's latest image. `tool_calls` is
# MAGIC JSONB in Postgres and lands in the history table as a JSON string.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {CHAT_TURNS} AS
    SELECT * EXCEPT (rn) FROM (
        SELECT
            turn_id, chat_id, user_id, turn_index, active_area,
            user_message, assistant_message,
            from_json(CAST(tool_calls AS STRING),
                      'ARRAY<STRUCT<name: STRING, arguments: STRING, ok: BOOLEAN, error: STRING, latency_ms: INT>>') AS tool_calls,
            model, latency_ms, error, feedback, feedback_at, created_at,
            ROW_NUMBER() OVER (PARTITION BY turn_id ORDER BY _timestamp DESC) AS rn
        FROM {CHAT_HISTORY}
        WHERE _pg_change_type IN ('insert', 'update_postimage')
    ) WHERE rn = 1
""")

spark.sql(f"""
    CREATE OR REPLACE VIEW {CHAT_DAILY} AS
    SELECT
        DATE(created_at) AS event_date,
        COUNT(*) AS turns,
        COUNT(DISTINCT chat_id) AS conversations,
        COUNT(DISTINCT user_id) AS users,
        ROUND(AVG(CASE WHEN size(tool_calls) > 0 THEN 1 ELSE 0 END) * 100, 1) AS pct_turns_using_tools,
        ROUND(AVG(size(tool_calls)), 2) AS avg_tool_calls_per_turn,
        ROUND(AVG(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) * 100, 1) AS error_rate_pct,
        percentile_approx(latency_ms, 0.5) AS p50_latency_ms,
        percentile_approx(latency_ms, 0.95) AS p95_latency_ms,
        COUNT(feedback) AS rated_turns,
        ROUND(AVG(CASE WHEN feedback = 1 THEN 1.0 WHEN feedback = -1 THEN 0.0 END) * 100, 1) AS thumbs_up_pct
    FROM {CHAT_TURNS}
    GROUP BY DATE(created_at)
""")

# One row per turn, ready for offline evals (mlflow.genai.evaluate or an LLM judge): the input,
# the tool trajectory, the output, and the user's own label where they gave one.
spark.sql(f"""
    CREATE OR REPLACE VIEW {CHAT_EVAL_SET} AS
    SELECT
        turn_id,
        created_at,
        active_area,
        user_message AS request,
        transform(tool_calls, t -> t.name) AS tool_sequence,
        tool_calls,
        assistant_message AS response,
        CASE feedback WHEN 1 THEN 'good' WHEN -1 THEN 'bad' END AS user_label,
        error
    FROM {CHAT_TURNS}
""")

print(f"Created views {CHAT_TURNS}, {CHAT_DAILY}, {CHAT_EVAL_SET}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC After some app use, these should have rows. An empty tool-usage view means no usage yet, or
# MAGIC that `tool_invocations` isn't reaching Unity Catalog through CDF.

# COMMAND ----------

display(spark.table(EVENTS_DAILY).orderBy("event_date", ascending=False))

# COMMAND ----------

display(spark.table(TOOL_USAGE_DAILY).orderBy("event_date", ascending=False))

# COMMAND ----------

display(spark.table(CHAT_DAILY).orderBy("event_date", ascending=False))

# COMMAND ----------

# Thumbs-down answers first: the quickest way to find what the agent gets wrong.
display(spark.sql(f"SELECT * FROM {CHAT_EVAL_SET} ORDER BY user_label ASC NULLS LAST, created_at DESC"))
