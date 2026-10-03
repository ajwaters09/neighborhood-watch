"""Writing API pulls into Delta bronze tables. The only src module that needs a Spark session;
pyspark is imported inside each function, so the rest of src stays importable anywhere.

- write_rows_in_chunks: big append-only SODA loads (crimes, 311), all-string, in batches.
- upsert_rows: small keyed pulls (weather, setlists, Wikipedia), MERGEd on their key.
- replace_rows: windows that get re-pulled whole (permits, Ticketmaster snapshots).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Iterable, Iterator


def month_windows(start_iso: str, end: datetime) -> Iterator[tuple[str, str]]:
    """Month-aligned (start, end) ISO bounds from `start_iso` (inclusive) to `end` (exclusive).

    Backfilling a month at a time keeps each pull's $offset small (deep offsets are slow on
    Socrata) and each write small.
    """
    cur = datetime.fromisoformat(start_iso)
    while cur < end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        window_end = min(nxt, end)
        yield cur.strftime("%Y-%m-%dT%H:%M:%S"), window_end.strftime("%Y-%m-%dT%H:%M:%S")
        cur = nxt


def write_rows_in_chunks(
    spark,
    rows: Iterable[dict],
    table: str,
    page_size: int,
    label: str,
    ingest_source: str,
) -> int:
    """Append `rows` to `table` in `page_size` batches as they stream in, rather than collecting
    them first (14M rows of JSON on the driver runs it out of memory). Prints progress after
    every batch. Returns the number of rows written.

    Bronze is all-string, so the bulk CSV loader and these JSON pulls always agree on a schema:
    Socrata serializes booleans unquoted, which Spark would otherwise infer as BooleanType, and
    Delta won't merge boolean into string.
    """
    from pyspark.sql.functions import col, lit

    buffer: list[dict] = []
    total = 0
    start_time = time.time()

    def flush() -> None:
        nonlocal total, buffer
        if not buffer:
            return
        # Some fields (311's `location`) arrive as nested objects, and only on rows that have
        # them. Spark infers each batch's schema afresh, so a nested value becomes a MAP column
        # that can't merge into the table's string one. Nothing downstream reads them (latitude
        # and longitude are separate fields), so they're dropped before Spark sees them.
        dropped: set[str] = set()
        for row in buffer:
            for key in [k for k, v in row.items() if isinstance(v, (dict, list))]:
                del row[key]
                dropped.add(key)
        if dropped:
            print(f"  [{label}] dropped non-scalar field(s) {sorted(dropped)} from this batch (unused downstream)")

        raw_df = spark.createDataFrame(buffer)
        df = (
            raw_df.select([col(c).cast("string").alias(c) for c in raw_df.columns])
            .withColumn("_ingest_source", lit(ingest_source))
        )
        (df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(table))
        total += len(buffer)
        elapsed = time.time() - start_time
        print(f"  [{label}] +{len(buffer):,} rows written (running total: {total:,}, {elapsed:,.0f}s elapsed)")
        buffer = []

    for row in rows:
        buffer.append(row)
        if len(buffer) >= page_size:
            flush()
    flush()
    return total


def upsert_rows(
    spark,
    rows: list[dict],
    table: str,
    schema: list[tuple[str, str]],
    keys: list[str],
    source: str,
    replace_sources: tuple[str, ...],
) -> int:
    """MERGE `rows` into a typed bronze table keyed on `keys`, creating the table on first use.

    For small keyed pulls. Unlike the SODA loads these tables are typed: their sources have
    stable types and no bulk CSV to agree with.

    Each row carries `_source` (e.g. "seed" from the one-time volume upload, "api" from a live
    pull) and `_ingested_at`. A matched existing row is only replaced if its `_source` is in
    `replace_sources`. The seed passes ("seed",), so re-running it never overwrites a fresher
    API row. Live pulls pass ("seed", "api").
    """
    from pyspark.sql.functions import current_timestamp, lit

    ddl = ", ".join(f"{name} {typ}" for name, typ in schema)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({ddl}, _source STRING, _ingested_at TIMESTAMP) "
              f"USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    if not rows:
        return 0
    names = [name for name, _ in schema]
    seen = set()
    for r in rows:
        k = tuple(r[c] for c in keys)
        if k in seen:
            raise ValueError(f"duplicate key {dict(zip(keys, k))} in rows for {table}; MERGE needs unique source keys")
        seen.add(k)
    df = (spark.createDataFrame([tuple(r[c] for c in names) for r in rows], schema=ddl)
          .withColumn("_source", lit(source))
          .withColumn("_ingested_at", current_timestamp()))
    view = f"_upsert_{table.replace('.', '_')}"
    df.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    allowed = ", ".join(f"'{x}'" for x in replace_sources)
    spark.sql(f"""
        MERGE INTO {table} t USING {view} s ON {on}
        WHEN MATCHED AND t._source IN ({allowed}) THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    return len(rows)


def replace_rows(
    spark,
    rows: list[dict],
    table: str,
    schema: list[tuple[str, str]],
    replace_where: str | None,
    source: str,
) -> int:
    """Atomically replace everything in `table` matching `replace_where` with `rows` (the whole
    table when it's None), creating the table on first use. Delta's replaceWhere does it in one
    commit, so a reader never sees the window half-rewritten.

    For sources with no stable key, or whose past rows change: permits (statuses move after
    filing) re-pull a rolling window, and Ticketmaster snapshots replace their own day. Every row
    written must match `replace_where`. An empty pull writes nothing rather than wiping the
    window, since an empty result is far more often an upstream hiccup than a real
    "everything was withdrawn".
    """
    from pyspark.sql.functions import current_timestamp, lit

    ddl = ", ".join(f"{name} {typ}" for name, typ in schema)
    spark.sql(f"CREATE TABLE IF NOT EXISTS {table} ({ddl}, _source STRING, _ingested_at TIMESTAMP) "
              f"USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    if not rows:
        print(f"  {table}: nothing pulled, left as is")
        return 0
    names = [name for name, _ in schema]
    df = (spark.createDataFrame([tuple(r.get(c) for c in names) for r in rows], schema=ddl)
          .withColumn("_source", lit(source))
          .withColumn("_ingested_at", current_timestamp()))
    writer = df.write.format("delta").mode("overwrite")
    if replace_where:
        writer = writer.option("replaceWhere", replace_where)
    writer.saveAsTable(table)
    return len(rows)
