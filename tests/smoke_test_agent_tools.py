# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Smoke test: the agent tools against live Lakebase and Unity Catalog
# MAGIC
# MAGIC Not a pipeline step. Run it in the workspace after changing the tools or the connection
# MAGIC code (`src/db_connect.py`), to catch wiring problems before they reach the app. It needs no
# MAGIC secrets: the SDK authenticates as this notebook's user. It creates a throwaway user and
# MAGIC removes everything it wrote at the end.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install psycopg[binary]>=3.1.0 databricks-sdk>=0.118.0 -q

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src import agent_tools, db_connect

print("Imports OK. Proceeding to connection tests.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: the raw connection
# MAGIC
# MAGIC Separates connection problems from tool problems: if this fails, look at
# MAGIC `db_connect.get_pg_connection()`, not the tools.

# COMMAND ----------

with db_connect.get_pg_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM users")
        print(f"Connected. users table has {cur.fetchone()[0]} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: seed a throwaway test user
# MAGIC
# MAGIC No tool creates users (the app's sign-up page does), so the write tools need one. The last
# MAGIC cell removes it.

# COMMAND ----------

with db_connect.get_pg_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, display_name) VALUES (%s, %s) RETURNING user_id",
            ("smoke-test@example.com", "Smoke Test User"),
        )
        TEST_USER_ID = cur.fetchone()[0]
print(f"Created test user_id={TEST_USER_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: subscribe_to_area
# MAGIC
# MAGIC Twice with the same arguments: the second reports `already_subscribed` instead of failing.

# COMMAND ----------

TEST_AREA = 25  # Austin; any area works

r1 = agent_tools.subscribe_to_area(user_id=TEST_USER_ID, community_area=TEST_AREA)
print("First subscribe:", r1)
assert r1["ok"] is True and r1["already_subscribed"] is False, "expected a fresh subscription"

r2 = agent_tools.subscribe_to_area(user_id=TEST_USER_ID, community_area=TEST_AREA)
print("Second subscribe (should be idempotent):", r2)
assert r2["ok"] is True and r2["already_subscribed"] is True, "expected already_subscribed=True on repeat"

print("PASS: subscribe_to_area")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: log_resident_report

# COMMAND ----------

r3 = agent_tools.log_resident_report(
    user_id=TEST_USER_ID, community_area=TEST_AREA, description="Smoke test report -- safe to delete."
)
print("log_resident_report:", r3)
assert r3["ok"] is True and isinstance(r3["report_id"], int), "expected a report_id back"
print("PASS: log_resident_report")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: get_area_trend
# MAGIC
# MAGIC Reads the Synced Tables. Empty `metrics` most likely means they haven't synced since `04`.

# COMMAND ----------

r4 = agent_tools.get_area_trend(TEST_AREA, months=6)
print("get_area_trend:", r4)
assert r4["ok"] is True
if not r4["metrics"]:
    print("WARNING: no metrics returned -- check whether the Synced Table has completed a sync yet.")
else:
    print(f"PASS: get_area_trend returned {len(r4['metrics'])} metrics for area {TEST_AREA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: get_recent_activity
# MAGIC
# MAGIC Reads the silver tables (through Spark here, the Statement Execution API in the app). A quiet
# MAGIC area can return no records, so this only checks the call succeeds.

# COMMAND ----------

r5 = agent_tools.get_recent_activity(TEST_AREA, days=30, limit=10)
print("get_recent_activity:", r5)
assert r5["ok"] is True
print(f"PASS: get_recent_activity returned {len(r5['records'])} records for area {TEST_AREA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: search_area_background
# MAGIC
# MAGIC Queries the Wikipedia vector search index (`02d`, `03c`). Once filtered to an area, once
# MAGIC across all of them: "Wicker Park" should find West Town (area 24).

# COMMAND ----------

r6 = agent_tools.search_area_background("history", community_area=TEST_AREA, num_results=3)
print("search_area_background (filtered):", r6)
assert r6["ok"] is True and r6["passages"]
assert all(p["community_area"] == TEST_AREA for p in r6["passages"])
r7 = agent_tools.search_area_background("Wicker Park", num_results=3)
print("search_area_background (all areas):", [(p["area_name"], p["section"]) for p in r7["passages"]])
assert r7["ok"] is True and r7["passages"][0]["community_area"] == 24
print("PASS: search_area_background")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup
# MAGIC
# MAGIC Removes the test user and every row that references it, children first for the foreign
# MAGIC keys (`@_logged` wrote a `tool_invocations` row for each write-tool call above). Run it even
# MAGIC if an earlier cell failed.

# COMMAND ----------

with db_connect.get_pg_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tool_invocations WHERE user_id = %s", (TEST_USER_ID,))
        cur.execute("DELETE FROM resident_reports WHERE user_id = %s", (TEST_USER_ID,))
        cur.execute("DELETE FROM area_subscriptions WHERE user_id = %s", (TEST_USER_ID,))
        cur.execute("DELETE FROM users WHERE user_id = %s", (TEST_USER_ID,))
print(f"Cleaned up test user_id={TEST_USER_ID} and its rows.")
