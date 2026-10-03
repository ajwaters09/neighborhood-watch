# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02d - Bronze: Wikipedia articles on the 77 community areas
# MAGIC
# MAGIC Background for the chat agent: history, the neighborhoods inside each area, landmarks, parks,
# MAGIC schools, transit. `03c_wiki_chunks` cuts these into passages behind a vector search index, and
# MAGIC the agent's `search_area_background` tool retrieves them.
# MAGIC
# MAGIC **Source.** The first table on
# MAGIC [Community areas of Chicago](https://en.wikipedia.org/wiki/Community_areas_of_Chicago) links an
# MAGIC article for each area number. The articles come through the MediaWiki API as plain text with
# MAGIC their section headings kept (`src/area_wiki.py`), not scraped HTML.
# MAGIC
# MAGIC **Staying current.** Each article's revision id is the watermark. A run asks for the latest
# MAGIC revision of all 77 in two batched requests, then fetches only the articles that changed, and
# MAGIC MERGEs them on `community_area`. A quiet night is three requests. Wikipedia has no API key or
# MAGIC daily budget, so unlike the other sources there's no seed from the volume. Set `force` to
# MAGIC re-fetch everything.
# MAGIC
# MAGIC Runs nightly with no dependencies. Text is CC BY-SA 4.0; `page_url` carries the attribution.

# COMMAND ----------

import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))   # the repo root, if this isn't a Git folder
from src import area_wiki as aw
from src import config
from src.community_areas import AREA_NAMES
from src.spark_bronze_utils import upsert_rows

dbutils.widgets.text("catalog", config.CATALOG)
dbutils.widgets.text("schema", config.SCHEMA)
dbutils.widgets.dropdown("force", "false", ["false", "true"])

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FORCE = dbutils.widgets.get("force") == "true"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
PAGES_TABLE = T("bronze_wiki_pages")
PAGES_SCHEMA = [("community_area", "INT"), ("list_name", "STRING"), ("title", "STRING"), ("page_url", "STRING"),
                ("revid", "BIGINT"), ("rev_timestamp", "TIMESTAMP"), ("extract", "STRING")]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Area number -> article
# MAGIC
# MAGIC The table must list exactly areas 1-77. A name that doesn't match `AREA_NAMES` (the app's
# MAGIC names, from the city's boundary file) is printed, not fatal: the number is the join key, and
# MAGIC Wikipedia's spelling of a name ("(The) Loop") can differ harmlessly.

# COMMAND ----------

areas = aw.fetch_area_titles()
if sorted(areas) != list(range(1, 78)):
    raise ValueError(f"expected areas 1-77 on {aw.LIST_PAGE}, got {sorted(areas)}")

norm = lambda s: re.sub(r"[^a-z]", "", s.lower().replace("(the)", ""))
for n, a in sorted(areas.items()):
    if norm(a["name"]) != norm(AREA_NAMES[n]):
        print(f"  name differs for area {n}: Wikipedia {a['name']!r}, app {AREA_NAMES[n]!r}")
print(f"{len(areas)} areas listed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch what changed

# COMMAND ----------

stored = {}
if spark.catalog.tableExists(PAGES_TABLE):
    stored = {r["community_area"]: r["revid"] for r in spark.table(PAGES_TABLE).select("community_area", "revid").collect()}

latest = aw.fetch_revisions([a["title"] for a in areas.values()])
changed = [n for n, a in sorted(areas.items()) if FORCE or stored.get(n) != latest.get(a["title"])]
print(f"{len(changed)} of 77 articles new or changed" + (" (force)" if FORCE else ""))

rows, failed = [], []
for n in changed:
    try:
        art = aw.fetch_article(areas[n]["title"])
    except Exception as exc:        # keep the stored copy; the next run tries again
        failed.append(n)
        print(f"  area {n} ({areas[n]['title']}): {exc}")
        continue
    rows.append({"community_area": n, "list_name": areas[n]["name"], **art})
    print(f"  area {n}: {art['title']} (rev {art['revid']}, {len(art['extract'].split()):,} words)")

n = upsert_rows(spark, rows, PAGES_TABLE, PAGES_SCHEMA, ["community_area"], source="api", replace_sources=("api",))
print(f"bronze_wiki_pages: {n} articles merged" + (f", {len(failed)} failed: {failed}" if failed else ""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT count(*) AS articles, sum(size(split(extract, '\\\\s+'))) AS words, max(rev_timestamp) AS newest_edit,
           max(_ingested_at) AS last_fetch
    FROM {PAGES_TABLE}
"""))
