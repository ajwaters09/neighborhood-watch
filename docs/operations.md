# Operations

How to stand the project up on a workspace, run it, keep it healthy, and what to watch out for.
For how the pieces fit, see [architecture.md](architecture.md).

## Deployment status

As of 2026-10-02:
- **Live and verified:** every notebook `00`-`09`, the nightly job, both Synced Tables, Lakebase
  CDF into Unity Catalog, the vector search index, the deployed app and its chat agent, and the
  live smoke test (`tests/smoke_test_agent_tools.py`).
- **Pending a live run:**
  - `lakebase/migrations/2026-09_alerts_and_unsubscribe.sql` (trend alerts, quiet periods,
    unsubscribing). Run it before deploying the current app code, then re-run `06` for
    `unsubscription` events, and check that CDF carries the new `alert_rules` columns into
    `lb_alert_rules_history`.
  - The `portfolio-cleanup` changes: notebooks now take their widget defaults from
    `src/config.py`, `db_connect` moved to `src/db_connect.py`, `05` uses
    `src/leading_indicators.py`, and `08` uses `src/outlook/`, writing two new tables
    (`outlook_leaderboard`, `outlook_calibration`) and logging to MLflow. The refactored outlook
    reproduced the previous model's outputs exactly on local data; it now also reads recent
    snowfall (`snow` from `bronze_weather_observed`), which moved the backtest slightly
    (docs/modeling.md). Re-run the nightly job once and the smoke test, and redeploy the app.

## Standing it up

1. **Configure.** Set the names in [`src/config.py`](../src/config.py) (catalog, schema, Lakebase
   project, SQL warehouse, secret scope, model endpoint), or the matching environment variables.
2. **Workspace.** Create the Git folder, then run `00_setup` (schema, `landing` volume) and put
   the API keys in the secret scope it names: `socrata_key_id`, `socrata_key_secret`,
   `setlistfm_api_key`, `ticketmaster_api_key`. Socrata keys are optional but lift throttling;
   the events sources are skipped without theirs.
3. **History.** Run `01` (crimes, bulk CSV) and `01b` (311, resumable). Upload the weather and
   events seed files to `landing/seed/` and run `01c`, or skip it and let `02b`/`02c` backfill
   from the APIs (setlist.fm's daily quota makes that slow).
4. **First pipeline run.** `02`-`02d`, `03`, `03b`, `03c` (set `vs_endpoint` the first time; it
   creates the index), `04`, `05`, `08`, `08b`, `09`.
5. **Lakebase.** Create the project, run [`lakebase/schema.sql`](../lakebase/schema.sql) in its SQL
   editor, create the two Synced Tables from `area_trend_metrics` and `area_trend_rolling`
   (Catalog → table → Create → Synced table, Triggered mode, named `<table>_lb`), and turn on
   Lakebase CDF for the `public` schema into the project's Unity Catalog schema.
6. **Analytics views.** Run `06` and `07` once CDF has taken its first snapshot.
7. **The app.** Create a Databricks App from the repo folder (`app.yaml`), add its resources
   (`session-secret`, `sql-warehouse`, `chat-model`, and a vector search index resource on
   `area_wiki_chunks_index`), then grant its service principal:
   - Lakebase: [`lakebase/grant_app_access.sql`](../lakebase/grant_app_access.sql), with
     `<APP_SP>` replaced by the principal's application ID;
   - Unity Catalog: `USE CATALOG`, `USE SCHEMA` and `SELECT` on the project schema.
8. **Schedule.** A Lakeflow Job with the nightly notebooks and their dependencies (below), on
   serverless compute.

## The nightly job

| Task | Depends on |
|---|---|
| `02_bronze_incremental`, `02b_bronze_weather`, `02c_bronze_events`, `02d_bronze_wikipedia` | none |
| `03_silver_tables` | `02` |
| `03b_crime_categories` | `03` |
| `03c_wiki_chunks` | `02d` |
| `04_gold_area_trend_metrics` | `03b` |
| `05_leading_indicators` | `04` |
| `08_next_week_outlook` | `03`, `02b` |
| `08b_events_lookahead` | `03`, `02c` |
| `09_area_narratives` | `04`, `05`, `08` |

After `04`, re-trigger both Synced Tables (Triggered mode; not part of the job).

## Routine upkeep

- Re-trigger the Synced Tables after each `04`.
- Re-run `tests/smoke_test_agent_tools.py` in the workspace after changing the tools or the
  connection code.
- After changing the Wikipedia chunker (`src/area_wiki.py`), run `03c` with `force` = true. After
  changing the narrative prompt, run `09` with `force` = true.
- Watch `08`'s warning line: the backtest should stay near a quarter of area-weeks called and 75%
  right. Far off means the inputs changed. The MLflow experiment
  (`/Users/<you>/neighborhood-watch-outlook`) has every run's scores.

## Local development

```bash
python -m pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

`tests/test_events_spark.py` also needs pyspark and a JDK, and is skipped without them.

The UI with canned data and no Databricks (alert rules wait for a data pull like the real ones;
visit `/_fake/next-pull` to simulate one):

```bash
NW_FAKE_DATA=1 SESSION_SECRET=local-dev-only uvicorn webapp.main:app --reload
```

The UI against the workspace, after `databricks auth login`:

```bash
uvicorn webapp.main:app --reload
```

The notebooks run only on Databricks. The models, the shaping code and the tutorials run locally.

## Gotchas

**Pipeline**
- **Bronze is all-string for SODA data**, so the bulk CSV and JSON loaders agree. The bulk loader
  rewrites crime dates to ISO, or `max(date)` watermarking breaks.
- **Socrata's nested `location` field** is dropped before `spark.createDataFrame`; per-batch
  schema inference turns it into a MAP column that can't merge.
- **The crime feed's newest day is a stub** (14 records against ~650). `as_of_date` is the earlier
  of each source's last *full* day, and the monthly table is cut at the same day.
- **`ai_classify` labels must each be under 50 characters**; `03b` passes bare slugs.
- **Gold starts at 311's start (late 2018).** Zero-filling uses a real month sequence.
- **Models work in °C and cm** (Open-Meteo's units); everything a person reads is imperial,
  converted in `src/area_data.py` and in `09`'s prompt.

**Lakebase and auth**
- **The app's Postgres role must come from `databricks_create_role()`.** `CREATE ROLE` and the REST
  API make password roles, which the OAuth-only endpoint rejects.
- **Grants need `SELECT` as well as `INSERT`/`UPDATE`**, for `INSERT ... RETURNING` and UPDATE's
  `WHERE`.
- **Synced Tables are created in the UI**, land in a Postgres schema named after the Unity Catalog
  schema, and run in Triggered mode.
- **Pin `databricks-sdk>=0.118.0` and `psycopg[binary]>=3.1.0`.** Older SDKs lack `w.postgres`.
- **A deployed app has no pyspark.** `area_data.query_uc` falls back to the Statement Execution
  API on `ImportError` too. That API returns every value as a string, so all shaping casts.
- **Hosting the app outside Databricks needs a credential an external host can use** (an OAuth
  service principal secret). On a workspace without one, the app has to run as a Databricks App.

**Wikipedia background**
- **The MediaWiki API's section format is `exsectionformat=wiki`** (`== History ==` lines).
- **Redirects:** "Loop, Chicago" → "Chicago Loop". `parse_revisions` follows normalization and
  redirects back to the requested title.
- **Area filters differ by endpoint type**: a dict for Standard endpoints, a SQL string for
  Storage-Optimized ones. `area_data.search_wiki` tries both.
- **The articles' crime and population figures are dated.** The prompt says never to use them for
  current levels or trends.

**Frontend**
- **Chart.js charts inside HTMX swaps render on `htmx:afterSettle`, not `afterSwap`**; the settle
  step resets canvas attributes.
- **maplibre-gl is pinned to 5.x**: the Leaflet bridge needs the global `maplibregl`, and 6.x is
  ES modules only.
- **The basemap is OpenFreeMap's Liberty style** in both themes. CARTO's keyless tiles come back
  watermarked and OSM's servers block app traffic.
- **`#area-detail` has `hx-disinherit="*"`**, or its `hx-include` leaks into nested requests.
- **`/static` URLs carry `?v=<mtime>`**, so a deploy isn't masked by cached JS or CSS.

**Agent**
- **The model guesses metric names.** The prompt lists the real ones, and `resolve_metric` maps
  loose names or returns the valid list.
- **The Insights page calls tools through `__wrapped__`**, so page loads don't count as agent usage.
- **No notifications leave the app.** The prompt says never to promise email, SMS or push.

## Known limitations

- **Crime's incremental pull keys on offense date**, so a record corrected later is never
  re-fetched. A full fix would re-sync on `updated_on` and MERGE.
- **`db_connect.query_uc_api` reads only the first result chunk.** Fine at these table sizes; a
  large result would come back short.
- **`get_recent_activity` builds SQL with f-strings.** Safe because every value is a validated int
  or date first.
- **`area_trend_metrics` keeps month-over-month `pct_change_vs_prior` and `rolling_3mo_avg`.** Only
  the agent sees them, and it's told to prefer the window comparison; dropping them would change
  the Synced Table's schema.
- **Login is email only, with no passwords.** It's a demo: anyone can sign in as any email.
