# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC
# MAGIC Creates the project schema and its `landing` volume, and checks the secret scope. Safe to
# MAGIC re-run: everything is `IF NOT EXISTS`. It never creates a catalog, so it only needs
# MAGIC `USE CATALOG` and `CREATE SCHEMA` on an existing one.
# MAGIC
# MAGIC The defaults for every notebook's `catalog` and `schema` widgets come from `src/config.py`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC The landing volume holds the one-time bulk crime export and the seed files for `01c`.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.landing")
LANDING_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/landing"
print(f"Landing volume path: {LANDING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credentials
# MAGIC
# MAGIC Source API keys live in a Databricks secret scope (widgets aren't masked and land in job
# MAGIC history). Create it once from the CLI, with your scope name from `src/config.py`:
# MAGIC
# MAGIC ```bash
# MAGIC databricks secrets create-scope <scope>
# MAGIC databricks secrets put-secret <scope> socrata_key_id --string-value "<key id>"
# MAGIC databricks secrets put-secret <scope> socrata_key_secret --string-value "<key secret>"
# MAGIC databricks secrets put-secret <scope> setlistfm_api_key --string-value "<key>"
# MAGIC databricks secrets put-secret <scope> ticketmaster_api_key --string-value "<key>"
# MAGIC ```
# MAGIC
# MAGIC Lakebase, Unity Catalog, vector search and the model endpoints need no secrets: the
# MAGIC Databricks SDK authenticates as whatever identity runs the code (this notebook's user, or
# MAGIC the deployed app's service principal).

# COMMAND ----------

try:
    dbutils.secrets.get(scope=config.SECRET_SCOPE, key="socrata_key_id")
    print(f"Secret scope '{config.SECRET_SCOPE}' is reachable and populated.")
except Exception as e:
    print(f"Secret scope '{config.SECRET_SCOPE}' not set up yet ({e}). "
          f"Run the `databricks secrets` commands above before 01/02.")
