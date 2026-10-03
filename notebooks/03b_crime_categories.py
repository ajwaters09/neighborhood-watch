# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03b - Crime categories with `ai_classify()`
# MAGIC
# MAGIC Chicago's `primary_type` has ~30 overlapping values (THEFT, BURGLARY, MOTOR VEHICLE THEFT,
# MAGIC CRIMINAL DAMAGE; OTHER OFFENSE spans harassment and protection-order violations), which made
# MAGIC for long menus and thin, noisy series. This maps every distinct `(primary_type,
# MAGIC description)` pair to one of seven groups:
# MAGIC
# MAGIC | category | covers, roughly |
# MAGIC |---|---|
# MAGIC | `violent` | assault, battery, robbery, homicide, sexual offenses, kidnapping, domestic violence |
# MAGIC | `property` | theft, burglary, motor vehicle theft, criminal damage, arson, trespass |
# MAGIC | `drugs` | narcotics, other narcotic violations |
# MAGIC | `weapons` | weapons violations, concealed carry violations |
# MAGIC | `public_order` | public peace, liquor, gambling, prostitution, interference with an officer |
# MAGIC | `fraud_financial` | deceptive practice, forgery, embezzlement |
# MAGIC | `other` | anything that doesn't fit the above |
# MAGIC
# MAGIC **Pairs, not rows.** ~8M crimes hold only a few hundred distinct pairs, so this is a few
# MAGIC hundred AI calls, once. Pairs rather than `primary_type` alone, because `description` is what
# MAGIC separates OTHER OFFENSE / HARASSMENT BY TELEPHONE (public order) from OTHER OFFENSE /
# MAGIC VIOLATE ORDER OF PROTECTION (violent).
# MAGIC
# MAGIC **A persisted, append-only map.** `crime_category_map` is written with
# MAGIC `MERGE ... WHEN NOT MATCHED`, so each pair is classified exactly once: later runs only see
# MAGIC new pairs, and history never shifts because a model answered differently the second time.
# MAGIC `category_override` corrects a row by hand; `category` is
# MAGIC `coalesce(category_override, ai_category, 'other')`.
# MAGIC
# MAGIC `ai_classify` labels must each be under 50 characters, so it gets the bare slugs. If AI
# MAGIC Functions aren't available, set `method` to `ai_query` (a model call that gets the full
# MAGIC descriptions) or `rules` (a `CASE` on `primary_type`, no AI).
# MAGIC
# MAGIC `04` turns these into the `crime_<category>` metrics. Runs nightly after `03`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import config
from src.constants import CRIME_CATEGORY_RULES

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.dropdown("method", "ai_classify", ["ai_classify", "ai_query", "rules"])
dbutils.widgets.text("ai_query_endpoint", config.SERVING_ENDPOINT)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
METHOD = dbutils.widgets.get("method")
AI_QUERY_ENDPOINT = dbutils.widgets.get("ai_query_endpoint")

CRIMES_TABLE = f"{CATALOG}.{SCHEMA}.silver_crimes"
MAP_TABLE = f"{CATALOG}.{SCHEMA}.crime_category_map"

# Category slug -> what it covers. The slugs become metric names (crime_<slug>); keep them in
# sync with webapp/_fake.py's CRIME list. Only the ai_query fallback sees the descriptions.
CATEGORY_DESCRIPTIONS = {
    "violent": "violent crime against a person (assault, battery, robbery, homicide, sexual assault, kidnapping, domestic violence)",
    "property": "property crime (theft, burglary, motor vehicle theft, vandalism or criminal damage, arson, trespass)",
    "drugs": "drug offense (possession, sale, or manufacture of narcotics or other drugs)",
    "weapons": "weapons offense (unlawful possession, use, or sale of a firearm or other weapon)",
    "public_order": "public order offense (disorderly conduct, public peace violation, liquor, gambling, prostitution, interfering with police)",
    "fraud_financial": "fraud or financial crime (deceptive practice, identity theft, forgery, embezzlement)",
    "other": "other offense that fits none of the above",
}
CATEGORIES = sorted(CATEGORY_DESCRIPTIONS)
assert all(len(c) < 50 for c in CATEGORIES), "ai_classify labels must be under 50 characters"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

labels_sql = "array(" + ", ".join(f"'{c}'" for c in CATEGORIES) + ")"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test: is `ai_classify` available here?

# COMMAND ----------

if METHOD == "ai_classify":
    display(spark.sql(f"""
        SELECT ai_classify('Chicago police crime: BURGLARY - FORCIBLE ENTRY', {labels_sql}) AS label
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The persisted map

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {MAP_TABLE} (
        primary_type       STRING NOT NULL,
        description        STRING,
        ai_category        STRING,
        category_override  STRING,
        method             STRING,
        classified_at      TIMESTAMP
    )
""")

pairs = (
    spark.table(CRIMES_TABLE)
    .groupBy("primary_type", "description")
    .agg(F.count(F.lit(1)).alias("row_count"))
)
existing = spark.table(MAP_TABLE).select("primary_type", "description")
# eqNullSafe: description can be null, and null = null is not true in a plain join.
new_pairs = pairs.join(
    existing,
    (pairs.primary_type == existing.primary_type) & pairs.description.eqNullSafe(existing.description),
    "left_anti",
)
new_count = new_pairs.count()
print(f"{pairs.count()} distinct (primary_type, description) pairs; {new_count} not yet classified")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classify only the new pairs

# COMMAND ----------

# The `rules` method: a CASE on primary_type alone (src/constants.CRIME_CATEGORY_RULES).
RULES_SQL = "CASE " + " ".join(
    f"WHEN primary_type IN ({', '.join(repr(t) for t in types)}) THEN '{category}'"
    for category, types in CRIME_CATEGORY_RULES.items()) + " ELSE 'other' END"

if new_count > 0:
    new_pairs.createOrReplaceTempView("new_pairs")
    content = "concat('Chicago police crime: ', primary_type, ' - ', coalesce(description, ''))"
    if METHOD == "ai_classify":
        classified = spark.sql(f"""
            SELECT primary_type, description, ai_classify({content}, {labels_sql}) AS ai_category
            FROM new_pairs
        """)
    elif METHOD == "ai_query":
        categories = "; ".join(f"{c} = {d}" for c, d in CATEGORY_DESCRIPTIONS.items()).replace("'", "''")
        prompt = (
            f"'Classify this Chicago police crime record into exactly one category. Categories: {categories}. "
            f"Answer with only the category slug. Record: ' || {content}"
        )
        classified = spark.sql(f"""
            SELECT primary_type, description,
                   lower(trim(ai_query('{AI_QUERY_ENDPOINT}', {prompt}))) AS ai_category
            FROM new_pairs
        """).withColumn(
            "ai_category",
            F.when(F.col("ai_category").isin(CATEGORIES), F.col("ai_category")).otherwise(F.lit(None)),
        )
    else:
        classified = spark.sql(f"SELECT primary_type, description, {RULES_SQL} AS ai_category FROM new_pairs")

    classified = classified.withColumn("method", F.lit(METHOD)).withColumn("classified_at", F.current_timestamp())
    classified.createOrReplaceTempView("classified")
    spark.sql(f"""
        MERGE INTO {MAP_TABLE} t
        USING classified s
          ON t.primary_type = s.primary_type AND t.description <=> s.description
        WHEN NOT MATCHED THEN INSERT (primary_type, description, ai_category, category_override, method, classified_at)
          VALUES (s.primary_type, s.description, s.ai_category, NULL, s.method, s.classified_at)
    """)
    print(f"Classified and merged {new_count} new pairs via {METHOD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review: eyeball this, then override anything wrong
# MAGIC
# MAGIC Rows are weighted by how many crimes they cover, so a misclassified
# MAGIC high-volume pair stands out. To fix one:
# MAGIC ```sql
# MAGIC UPDATE <catalog>.<schema>.crime_category_map
# MAGIC SET category_override = 'violent'
# MAGIC WHERE primary_type = 'OTHER OFFENSE' AND description = 'VIOLATE ORDER OF PROTECTION';
# MAGIC ```
# MAGIC then re-run `04`. Rows where the AI call returned nothing
# MAGIC (`ai_category IS NULL`) fall into `other` until overridden.

# COMMAND ----------

m = spark.table(MAP_TABLE).withColumn("category", F.coalesce("category_override", "ai_category", F.lit("other")))
mapped = m.join(
    pairs.withColumnRenamed("primary_type", "p_type").withColumnRenamed("description", "p_desc"),
    (F.col("primary_type") == F.col("p_type")) & F.col("description").eqNullSafe(F.col("p_desc")),
    "left",
).drop("p_type", "p_desc").fillna({"row_count": 0})
display(mapped.groupBy("category").agg(F.sum("row_count").alias("crimes"), F.count(F.lit(1)).alias("pairs")).orderBy(F.desc("crimes")))
display(mapped.orderBy(F.desc("row_count")).select("primary_type", "description", "category", "ai_category", "category_override", "row_count").limit(200))
display(mapped.filter("ai_category IS NULL").select("primary_type", "description", "row_count"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Migrating alert rules from per-type metrics
# MAGIC
# MAGIC Crime metrics used to be one per `primary_type` (`crime_theft`). For an existing Lakebase
# MAGIC database with rules on those, this prints `UPDATE` statements that repoint each old metric at
# MAGIC the category most of its crimes fall into. Run them in the Lakebase SQL editor. A fresh
# MAGIC deployment can skip this.

# COMMAND ----------

slug = F.concat(F.lit("crime_"), F.regexp_replace(F.lower(F.trim("primary_type")), r"[^a-z0-9]+", "_"))
majority = (
    mapped.withColumn("old_metric", slug)
    .groupBy("old_metric", "category").agg(F.sum("row_count").alias("n"))
    .withColumn("rk", F.row_number().over(Window.partitionBy("old_metric").orderBy(F.desc("n"))))
    .filter("rk = 1")
    .orderBy("old_metric")
    .collect()
)
print("-- Paste into the Lakebase SQL editor:")
for row in majority:
    print(f"UPDATE alert_rules SET metric = 'crime_{row['category']}' WHERE metric = '{row['old_metric']}';")
