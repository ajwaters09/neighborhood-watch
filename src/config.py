"""Deployment settings: where this instance of the project lives on Databricks.

Every workspace-specific name is here. Pointing the project at another workspace means changing
these defaults or setting the environment variables, not editing code. The notebooks use them as
their widget defaults, and the app and agent read them at import.
"""

from __future__ import annotations

import os

# Unity Catalog: every Delta table, view, volume and the vector search index live in this schema.
CATALOG = os.environ.get("NW_CATALOG", "bootcamp_students")
SCHEMA = os.environ.get("NW_SCHEMA", "ajwaters_chicago")


def uc(name: str) -> str:
    """Fully qualified Unity Catalog name for an object in the project schema."""
    return f"{CATALOG}.{SCHEMA}.{name}"


# The SQL warehouse for Unity Catalog reads outside Spark (the deployed app). The app sets it from
# its `sql-warehouse` resource (app.yaml).
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "b15d3d6f837ba428")

# Lakebase (managed Postgres): the OLTP tables live in `public`. The Synced Tables land in a
# schema named after the Unity Catalog schema, which is what the Synced Table wizard picks.
LAKEBASE_PROJECT = os.environ.get("NW_LAKEBASE_PROJECT", "ajwaters-chicago-lb")
LAKEBASE_BRANCH = "production"
LAKEBASE_ENDPOINT = "primary"
LAKEBASE_DATABASE = "databricks_postgres"
LAKEBASE_SYNC_SCHEMA = os.environ.get("NW_LAKEBASE_SYNC_SCHEMA", SCHEMA)
TREND_METRICS_LB = f"{LAKEBASE_SYNC_SCHEMA}.area_trend_metrics_lb"
TREND_ROLLING_LB = f"{LAKEBASE_SYNC_SCHEMA}.area_trend_rolling_lb"

# The Databricks secret scope that holds the source API keys (Socrata, setlist.fm, Ticketmaster).
SECRET_SCOPE = os.environ.get("NW_SECRET_SCOPE", "chicago-capstone")

# The Foundation Model endpoint behind the chat agent, the nightly narratives (09) and the
# category fallback (03b). The app sets it from its `chat-model` resource.
SERVING_ENDPOINT = os.environ.get("DATABRICKS_SERVING_ENDPOINT", "databricks-claude-sonnet-4-5")
