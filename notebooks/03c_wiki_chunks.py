# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03c - Wikipedia passages and their vector search index
# MAGIC
# MAGIC Cuts `bronze_wiki_pages` (from `02d_bronze_wikipedia`) into passages in `silver_wiki_chunks`,
# MAGIC and keeps the vector search index `area_wiki_chunks_index` in step with it. The agent's
# MAGIC `search_area_background` tool queries the index.
# MAGIC
# MAGIC **Chunking** (`src/area_wiki.chunk_article`, tested offline in `tests/test_area_wiki.py`):
# MAGIC - Split at section headings. Reference-type sections (References, See also, External links,
# MAGIC   Notes, ...) are dropped.
# MAGIC - Paragraphs pack into chunks of about 350 words, never across sections. A new chunk in the
# MAGIC   same section repeats the previous paragraph if it's short.
# MAGIC - Each chunk opens with "<area> (Chicago community area N) - <section>", so short passages
# MAGIC   still say where they're from.
# MAGIC - All 77 articles come to about 940 chunks, 4-32 an area, median ~120 words.
# MAGIC
# MAGIC **Only changed articles are re-chunked.** An area is rebuilt when its bronze `revid` differs
# MAGIC from its chunks'. The rewrite is one `replaceWhere` commit over those areas, and Change Data
# MAGIC Feed carries just those rows to the index, so a sync only re-embeds what changed. Set `force`
# MAGIC after changing the chunker.
# MAGIC
# MAGIC **The index** is a Delta Sync index with managed embeddings (`databricks-gte-large-en`), on
# MAGIC an existing endpoint (the `vs_endpoint` widget), synced on trigger. This notebook creates it
# MAGIC on the first run and triggers a sync on later runs that changed something. Creating it in the
# MAGIC UI instead works too, with the same name and settings: primary key `chunk_id`, embedding
# MAGIC source column `content`, sync mode Triggered, and the columns in `SYNC_COLUMNS` below.
# MAGIC
# MAGIC Runs nightly after `02d`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import area_wiki as aw
from src import config
from src.community_areas import AREA_NAMES
from src.spark_bronze_utils import replace_rows

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.text("vs_endpoint", "")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")
dbutils.widgets.dropdown("force", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint").strip()
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint").strip()
FORCE = dbutils.widgets.get("force") == "true"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
PAGES_TABLE = T("bronze_wiki_pages")
CHUNKS_TABLE = T("silver_wiki_chunks")
INDEX_NAME = T("area_wiki_chunks_index")      # must match src/area_data.UC_WIKI_INDEX
CHUNK_SCHEMA = [("chunk_id", "STRING"), ("community_area", "INT"), ("area_name", "STRING"), ("section", "STRING"),
                ("chunk_index", "INT"), ("content", "STRING"), ("word_count", "INT"), ("page_url", "STRING"),
                ("revid", "BIGINT")]
# What the index keeps beside the embedding, i.e. what a query can return. src/area_data.WIKI_COLUMNS
# asks for these.
SYNC_COLUMNS = ["chunk_id", "community_area", "area_name", "section", "content", "page_url"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk the new and changed articles

# COMMAND ----------

pages = spark.table(PAGES_TABLE).select("community_area", "page_url", "revid", "extract").collect()
done = {}
if spark.catalog.tableExists(CHUNKS_TABLE):
    done = {r["community_area"]: r["revid"] for r in spark.sql(
        f"SELECT community_area, max(revid) AS revid FROM {CHUNKS_TABLE} GROUP BY community_area").collect()}

todo = sorted((p for p in pages if FORCE or done.get(p["community_area"]) != p["revid"]), key=lambda p: p["community_area"])
rows = []
for p in todo:
    n = p["community_area"]
    rows += aw.chunk_article(n, AREA_NAMES[n], p["extract"], p["page_url"], p["revid"])

if todo:
    areas_in = ", ".join(str(p["community_area"]) for p in todo)
    written = replace_rows(spark, rows, CHUNKS_TABLE, CHUNK_SCHEMA, f"community_area IN ({areas_in})", source="api")
    print(f"silver_wiki_chunks: {len(todo)} areas re-chunked, {written:,} chunks written" + (" (force)" if FORCE else ""))
else:
    print("silver_wiki_chunks: no article changed since the last run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or sync the index

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn,
                                                 PipelineType, VectorIndexType)

w = WorkspaceClient()
try:
    index = w.vector_search_indexes.get_index(INDEX_NAME)
except NotFound:
    index = None

if index is None:
    if not VS_ENDPOINT:
        raise ValueError("the index doesn't exist yet: set the vs_endpoint widget to the endpoint to create it on")
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_TABLE,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[EmbeddingSourceColumn(name="content", embedding_model_endpoint_name=EMBEDDING_ENDPOINT)],
            columns_to_sync=SYNC_COLUMNS,
        ),
    )
    print(f"created {INDEX_NAME} on {VS_ENDPOINT}; its first sync starts on its own and takes a few minutes")
elif not todo:
    print(f"{INDEX_NAME}: nothing to sync")
elif index.status and index.status.ready:
    w.vector_search_indexes.sync_index(INDEX_NAME)
    print(f"{INDEX_NAME}: sync triggered for {len(todo)} areas")
else:
    print(f"{INDEX_NAME} isn't ready yet ({index.status.message if index.status else 'no status'}); "
          "it picks up the table when its current sync finishes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check
# MAGIC
# MAGIC Re-run this cell once the index is ready. `indexed_row_count` should match the table's chunk
# MAGIC count. The sample queries go through `src/area_data.search_wiki`, the same call the agent's
# MAGIC tool makes: a neighborhood name with no area (Wicker Park should come back as West Town, area
# MAGIC 24), and a topic within one area (Albany Park, area 14).

# COMMAND ----------

from src import area_data as ad

status = w.vector_search_indexes.get_index(INDEX_NAME).status
print(f"table: {spark.table(CHUNKS_TABLE).count():,} chunks; index: ready={status.ready}, "
      f"indexed_row_count={status.indexed_row_count}, {status.message}")

if status.ready:
    for query, area in [("Wicker Park", None), ("history of immigration", 14)]:
        print(f"\n{query!r}" + (f" in area {area}" if area else ""))
        for hit in ad.search_wiki(query, area, 3):
            print(f"  {hit['score']}  {hit['area_name']} - {hit['section']}: {hit['text'][:100]}...")
