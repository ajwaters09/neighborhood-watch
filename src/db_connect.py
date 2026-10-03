"""Connections to the project's two data stores.

- Lakebase Postgres: psycopg 3 with a short-lived OAuth credential from the Databricks SDK.
- Unity Catalog: `spark.sql` where a Spark session exists (notebooks), the SQL Statement
  Execution API where it doesn't (the deployed app).

Authentication is whatever identity runs the code: a notebook's user, or the app's service
principal. `WorkspaceClient()` configures itself from the environment, so no tokens or secrets are
stored. A fresh connection per request is fine at this scale; there's no pooling.

Needs `databricks-sdk>=0.118.0` (older versions lack `w.postgres`) and `psycopg[binary]>=3.1.0`.
Pin both wherever this is installed.
"""

from databricks.sdk import WorkspaceClient

from src import config

LB_ENDPOINT_NAME = (f"projects/{config.LAKEBASE_PROJECT}/branches/{config.LAKEBASE_BRANCH}"
                    f"/endpoints/{config.LAKEBASE_ENDPOINT}")
LB_DB = config.LAKEBASE_DATABASE
UC_CATALOG = config.CATALOG
UC_SCHEMA = config.SCHEMA
UC_CRIMES = config.uc("silver_crimes")
UC_311 = config.uc("silver_311")
WAREHOUSE_ID = config.WAREHOUSE_ID


# ── Lakebase ───────────────────────────────────────────────────────

def get_pg_connection():
    """A fresh psycopg connection to Lakebase. The OAuth credential lasts an hour."""
    import psycopg

    w = WorkspaceClient()
    ep = w.postgres.get_endpoint(name=LB_ENDPOINT_NAME)
    host = ep.status.hosts.host
    user = w.current_user.me().user_name
    token = w.postgres.generate_database_credential(
        endpoint=LB_ENDPOINT_NAME
    ).token

    return psycopg.connect(
        host=host,
        dbname=LB_DB,
        user=user,
        password=token,
        sslmode="require",
    )


# ── Unity Catalog (notebook context – Spark available) ─────────────

def query_uc_spark(sql: str):
    """Run a UC query through the active Spark session; rows as a list of dicts."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession – use query_uc_api() in an app.")
    return [row.asDict() for row in spark.sql(sql).collect()]


# ── Unity Catalog (app context – no Spark, use SQL Statement API) ──

def query_uc_api(sql: str):
    """Run a UC query through the Statement Execution API; rows as a list of dicts.

    Every value comes back as a string, so callers cast.
    """
    from databricks.sdk.service.sql import StatementState
    import time

    w = WorkspaceClient()
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="50s",
    )

    # If the warehouse is cold-starting, poll until done (up to 5 min).
    deadline = time.time() + 300
    while result.status.state in (
        StatementState.PENDING, StatementState.RUNNING
    ) and time.time() < deadline:
        time.sleep(2)
        result = w.statement_execution.get_statement(result.statement_id)

    if result.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Statement failed: {result.status.error}")

    cols = [c.name for c in result.manifest.schema.columns]
    return [dict(zip(cols, row)) for row in result.result.data_array]
