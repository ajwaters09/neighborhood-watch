"""Clean the raw crime + 311 exports in offline_data/ into H3-keyed parquet.

    offline_ml/.venv/bin/python offline_ml/preprocess.py

Inputs (offline_data/):
    crimes_bulk_export.csv   full Chicago crimes dataset, portal CSV export
    part-*.parquet           311 service requests, all-string columns

Outputs (offline_data/clean/):
    crimes.parquet           one row per crime record, sorted by occurred_at
    sr311.parquet            one row per place-based 311 request, sorted by created_at
    sr_type_catalog.csv      per-request-type volume, date span, and quirks
    cleaning_log.json        row counts at every drop/fix step

Every rule below came from profiling the raw files; see offline_ml/README.md
for the numbers behind each one.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import polars as pl
from h3.api import basic_int as h3

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "offline_data"
OUT = RAW / "clean"

H3_BASE_RES = 9            # ~0.1 km^2; crime coords are block-snapped (~16 m spread), so finer adds nothing
H3_PARENT_RES = (8, 7)     # ~0.7 km^2 and ~5 km^2, precomputed for rollups

# Anything outside this box is a bad geocode (the crime file has 149 rows in Missouri).
LAT_RANGE = (41.60, 42.10)
LON_RANGE = (-88.00, -87.50)

# A block needs this many geocoded crimes before its median point is trusted to fill in
# records with missing coordinates.
MIN_BLOCK_POINTS = 3

# 311 before March 2019 is a partial migration: Jan 2019 has ~30% and Feb ~65% of the
# volume the same months have in later years.
SR_START = "2019-03-01"

# Types with no real location: info calls sit on the 311 call center's address and every
# aircraft-noise complaint sits on one O'Hare address. Together they're 53% of raw rows.
SR_NON_PLACE_TYPES = ["311 INFORMATION ONLY CALL", "Aircraft Noise Complaint"]
# Real request types occasionally get parked on those same two addresses when the
# location is unknown.
SR_PLACEHOLDER_ADDRESSES = ["2111 W LEXINGTON", "10510 W ZEMKE"]

SR_RESIDENT_ORIGINS = {
    "Phone Call", "Internet", "Mobile Device", "E-Mail", "Mail", "Fax", "Walk-in", "Phone",
    "Web", "Social Media", "SPOTCSR", "Open311", "Open311 Interface", "Spot311 Interface", "SPOT311",
}
SR_ELECTED_ORIGINS = {
    "Alderman's Office", "ALDERMAN", "State Representatives", "Mayor's Office",
    "Budget Town Hall Meeting", "FY25 Budget Engagement",
}

# UCR Part I codes. 01B (involuntary manslaughter) is Part I but not in the violent index.
UCR_VIOLENT = ["01A", "02", "03", "04A", "04B"]
UCR_PROPERTY = ["05", "06", "07", "09"]

PRIMARY_TYPE_ALIASES = {"CRIM SEXUAL ASSAULT": "CRIMINAL SEXUAL ASSAULT"}  # renamed in 2019

# Spelling variants of the same place, applied after upper-casing and tightening "a / b" to "a/b".
LOCATION_ALIASES = {
    "PARKING LOT/GARAGE(NON.RESID.)": "PARKING LOT/GARAGE (NON RESIDENTIAL)",
    "PARKING LOT": "PARKING LOT/GARAGE (NON RESIDENTIAL)",
    "RESIDENCE-GARAGE": "RESIDENCE - GARAGE",
    "RESIDENCE PORCH/HALLWAY": "RESIDENCE - PORCH/HALLWAY",
    "RESIDENTIAL YARD (FRONT/BACK)": "RESIDENCE - YARD (FRONT/BACK)",
    "SCHOOL, PUBLIC, BUILDING": "SCHOOL - PUBLIC BUILDING",
    "SCHOOL, PUBLIC, GROUNDS": "SCHOOL - PUBLIC GROUNDS",
    "SCHOOL, PRIVATE, BUILDING": "SCHOOL - PRIVATE BUILDING",
    "SCHOOL, PRIVATE, GROUNDS": "SCHOOL - PRIVATE GROUNDS",
    "PUBLIC GRAMMAR SCHOOL": "SCHOOL - PUBLIC BUILDING",
    "PUBLIC HIGH SCHOOL": "SCHOOL - PUBLIC BUILDING",
    "SCHOOL YARD": "SCHOOL - PUBLIC GROUNDS",
    "COLLEGE/UNIVERSITY GROUNDS": "COLLEGE/UNIVERSITY - GROUNDS",
    "COLLEGE/UNIVERSITY RESIDENCE HALL": "COLLEGE/UNIVERSITY - RESIDENCE HALL",
    "POLICE FACILITY/VEH PARKING LOT": "POLICE FACILITY/VEHICLE PARKING LOT",
    "POLICE FACILITY": "POLICE FACILITY/VEHICLE PARKING LOT",
    "OTHER RAILROAD PROP/TRAIN DEPOT": "OTHER RAILROAD PROPERTY/TRAIN DEPOT",
    "RAILROAD PROPERTY": "OTHER RAILROAD PROPERTY/TRAIN DEPOT",
    "NURSING HOME/RETIREMENT HOME": "NURSING/RETIREMENT HOME",
    "NURSING HOME": "NURSING/RETIREMENT HOME",
    "TAXI CAB": "TAXICAB",
    "POOLROOM": "POOL ROOM",
    "VEHICLE-COMMERCIAL": "VEHICLE - COMMERCIAL",
    "VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS": "VEHICLE - COMMERCIAL: ENTERTAINMENT/PARTY BUS",
    "VEHICLE-COMMERCIAL - TROLLEY BUS": "VEHICLE - COMMERCIAL: TROLLEY BUS",
    "VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)": "VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)",
    "AUTO": "VEHICLE NON-COMMERCIAL",
    "OTHER (SPECIFY)": "OTHER",
    "HOTEL": "HOTEL/MOTEL",
    "MOTEL": "HOTEL/MOTEL",
    "GAS STATION DRIVE/PROP.": "GAS STATION",
    "CHA PARKING LOT": "CHA PARKING LOT/GROUNDS",
    "CHA GROUNDS": "CHA PARKING LOT/GROUNDS",
    "CHA HALLWAY": "CHA HALLWAY/STAIRWELL/ELEVATOR",
    "CHA STAIRWELL": "CHA HALLWAY/STAIRWELL/ELEVATOR",
    "CHA ELEVATOR": "CHA HALLWAY/STAIRWELL/ELEVATOR",
    "CHA LOBBY": "CHA HALLWAY/STAIRWELL/ELEVATOR",
    "CHA BREEZEWAY": "CHA HALLWAY/STAIRWELL/ELEVATOR",
    "CHURCH": "CHURCH/SYNAGOGUE/PLACE OF WORSHIP",
    "CHURCH PROPERTY": "CHURCH/SYNAGOGUE/PLACE OF WORSHIP",
    "HOSPITAL": "HOSPITAL BUILDING/GROUNDS",
    "GOVERNMENT BUILDING": "GOVERNMENT BUILDING/PROPERTY",
    "FACTORY": "FACTORY/MANUFACTURING BUILDING",
    "VACANT LOT": "VACANT LOT/LAND",
    "DRIVEWAY": "DRIVEWAY - RESIDENTIAL",
    "TAVERN": "TAVERN/LIQUOR STORE",
    "LIQUOR STORE": "TAVERN/LIQUOR STORE",
    "CLEANERS/LAUNDROMAT": "CLEANING STORE",
    "BARBER SHOP/BEAUTY SALON": "BARBERSHOP",
    'CTA "L" PLATFORM': "CTA PLATFORM",
    'CTA "L" TRAIN': "CTA TRAIN",
    "CTA SUBWAY STATION": "CTA STATION",
    "CTA PROPERTY": "CTA GARAGE/OTHER PROPERTY",
    "CTA PARKING LOT/GARAGE/OTHER PROPERTY": "CTA GARAGE/OTHER PROPERTY",
    "OFFICE": "COMMERCIAL/BUSINESS OFFICE",
    "RETAIL STORE": "SMALL RETAIL STORE",
    "RIVER": "LAKEFRONT/WATERFRONT/RIVERBANK",
    "RIVER BANK": "LAKEFRONT/WATERFRONT/RIVERBANK",
    "LAKE": "LAKEFRONT/WATERFRONT/RIVERBANK",
    "LAGOON": "LAKEFRONT/WATERFRONT/RIVERBANK",
    "BEACH": "LAKEFRONT/WATERFRONT/RIVERBANK",
}

# First match wins, so the order matters: AIRPORT PARKING LOT is airport, ANIMAL HOSPITAL is
# commercial, POLICE FACILITY/VEHICLE PARKING LOT is institutional, CHA PARKING LOT is residential.
LOCATION_GROUP_RULES = [
    ("airport", r"^AIRPORT|^AIRCRAFT"),
    ("transit", r"^CTA|RAILROAD|^OTHER COMMERCIAL TRANSPORTATION"),
    ("vehicle", r"^VEHICLE|^TAXICAB|TRUCK$|^TRAILER|^LIVERY AUTO|^BOAT"),
    ("residential", r"^RESIDENCE|^APARTMENT|^HOUSE$|^CHA |^COACH HOUSE|^ROOMING HOUSE|^DRIVEWAY|^PORCH|^YARD$"
                    r"|^HALLWAY|^STAIRWELL|^VESTIBULE|^BASEMENT|^LAUNDRY ROOM|^ELEVATOR|^GARAGE$|^GANGWAY"),
    ("street", r"^STREET|^SIDEWALK|^ALLEY|^HIGHWAY|^EXPRESSWAY|^BRIDGE"),
    ("school", r"SCHOOL|COLLEGE|^DAY CARE"),
    ("commercial", r"STORE|RETAIL|RESTAURANT|TAVERN|^GAS STATION|^BANK$|CREDIT UNION|SAVINGS|^ATM|CURRENCY EXCHANGE"
                   r"|HOTEL|OFFICE|BARBER|CAR WASH|PAWN|DEALERSHIP|ATHLETIC CLUB|BOWLING|MOVIE|POOL ROOM|CASINO"
                   r"|NEWSSTAND|WAREHOUSE|FACTORY|LOADING DOCK|TRUCKING TERMINAL|ANIMAL HOSPITAL|KENNEL|BANQUET"
                   r"|^CLUB|SPORTS ARENA|COIN OPERATED|FUNERAL|GARAGE/AUTO REPAIR"),
    ("institutional", r"GOVERNMENT|FEDERAL|POLICE|FIRE STATION|JAIL|HOSPITAL|NURSING|CHURCH|LIBRARY|CEMETARY|YMCA"),
    ("parking", r"^PARKING LOT"),
    ("vacant", r"^VACANT LOT|^ABANDONED BUILDING|^JUNK YARD"),
    ("park_open_space", r"^PARK PROPERTY|^FOREST PRESERVE|^LAKEFRONT|^WOODED|^PRAIRIE|^FARM|^HORSE STABLE"),
]

log: dict[str, dict[str, int]] = {"crimes": {}, "sr311": {}}


def note(dataset: str, step: str, value: int) -> None:
    log[dataset][step] = value
    print(f"  {dataset:>6} | {step:<44} {value:>12,}")


def in_city_box(lat: str = "latitude", lon: str = "longitude") -> pl.Expr:
    return pl.col(lat).is_between(*LAT_RANGE) & pl.col(lon).is_between(*LON_RANGE)


def add_h3(df: pl.DataFrame) -> pl.DataFrame:
    """Attach h3_r9 plus its parents, computing each distinct point and cell only once."""
    pts = df.select("latitude", "longitude").drop_nulls().unique()
    base = f"h3_r{H3_BASE_RES}"
    pts = pts.with_columns(
        pl.Series(base, [h3.latlng_to_cell(a, b, H3_BASE_RES)
                         for a, b in zip(pts["latitude"].to_list(), pts["longitude"].to_list())], dtype=pl.Int64)
    )
    cells = pts.select(base).unique()
    cells = cells.with_columns(
        pl.Series(f"h3_r{r}", [h3.cell_to_parent(c, r) for c in cells[base].to_list()], dtype=pl.Int64)
        for r in H3_PARENT_RES
    )
    return df.join(pts, on=["latitude", "longitude"], how="left").join(cells, on=base, how="left")


def slugify(col: str) -> pl.Expr:
    return (
        pl.col(col).str.to_lowercase()
        .str.replace_all(r"\(no longer being accepted\)", "")
        .str.replace_all(r"[^a-z0-9]+", "_")
        .str.strip_chars("_")
    )


# ---------------------------------------------------------------------------------------------
# Crimes
# ---------------------------------------------------------------------------------------------

def clean_crimes() -> pl.DataFrame:
    print("\nCrimes")
    raw = pl.read_csv(RAW / "crimes_bulk_export.csv", infer_schema=False)
    note("crimes", "raw rows", raw.height)

    ts_fmt = "%m/%d/%Y %I:%M:%S %p"
    df = raw.select(
        pl.col("ID").cast(pl.Int64).alias("id"),
        pl.col("Case Number").alias("case_number"),
        pl.col("Date").str.strptime(pl.Datetime("us"), ts_fmt).alias("occurred_at"),
        pl.col("Updated On").str.strptime(pl.Datetime("us"), ts_fmt).alias("updated_at"),
        pl.col("Block").str.strip_chars().alias("block"),
        pl.col("IUCR").alias("iucr"),
        pl.col("Primary Type").replace(PRIMARY_TYPE_ALIASES).alias("primary_type"),
        pl.col("Description").alias("description"),
        pl.col("Location Description")
        .str.to_uppercase().str.strip_chars()
        .str.replace_all(r"\s*/\s*", "/").str.replace_all(r"\s+", " ")
        .replace(LOCATION_ALIASES).alias("location_description"),
        pl.col("FBI Code").alias("fbi_code"),
        (pl.col("Arrest") == "true").alias("arrest"),
        (pl.col("Domestic") == "true").alias("domestic"),
        pl.col("Beat").cast(pl.Int16).alias("beat"),
        pl.col("District").cast(pl.Int16).alias("district"),
        pl.col("Ward").cast(pl.Int16).alias("ward"),
        pl.col("Community Area").cast(pl.Int16).alias("community_area"),
        pl.col("Latitude").cast(pl.Float64).alias("latitude"),
        pl.col("Longitude").cast(pl.Float64).alias("longitude"),
    )
    del raw

    n = df.height
    df = df.sort("updated_at", descending=True).unique("id", keep="first")
    note("crimes", "dropped: duplicate id", n - df.height)

    # Community area 0 means "unassigned"; nulls are mostly 2001-02, before it was recorded.
    df = df.with_columns(pl.when(pl.col("community_area") > 0).then(pl.col("community_area")).alias("community_area"))

    # --- Location -----------------------------------------------------------------------
    bad_geo = df.filter(pl.col("latitude").is_not_null() & ~in_city_box()).height
    note("crimes", "coords outside city box -> nulled", bad_geo)
    df = df.with_columns(
        pl.when(in_city_box()).then(pl.col(c)).alias(c) for c in ("latitude", "longitude")
    )
    note("crimes", "missing coords before block fill", df.filter(pl.col("latitude").is_null()).height)

    block_pts = (
        df.filter(pl.col("latitude").is_not_null())
        .group_by("block")
        .agg(pl.len().alias("_n"), pl.col("latitude").median().alias("_blat"), pl.col("longitude").median().alias("_blon"))
        .filter(pl.col("_n") >= MIN_BLOCK_POINTS)
    )
    df = df.join(block_pts, on="block", how="left").with_columns(
        pl.when(pl.col("latitude").is_not_null()).then(pl.lit("reported"))
        .when(pl.col("_blat").is_not_null()).then(pl.lit("block_median"))
        .alias("geo_source"),
        pl.coalesce("latitude", "_blat").alias("latitude"),
        pl.coalesce("longitude", "_blon").alias("longitude"),
    ).drop("_n", "_blat", "_blon")
    note("crimes", "coords filled from block median", df.filter(pl.col("geo_source") == "block_median").height)

    n = df.height
    df = df.filter(pl.col("latitude").is_not_null())
    note("crimes", "dropped: no usable location", n - df.height)

    # --- Time ---------------------------------------------------------------------------
    # 00:00 and 00:01 are the conventions for "time unknown"; the 1st of the month at those
    # times is the convention for "date unknown within the month" (~2x the normal day's count).
    midnightish = (pl.col("occurred_at").dt.hour() == 0) & (pl.col("occurred_at").dt.minute() <= 1)

    # The case number's two-letter prefix encodes the year the case was opened (G* = 2001,
    # HH = 2002, ..., JK = 2026). Learn the mapping from the data: each busy prefix's modal
    # occurrence year. Occurrences before Jan 1 of that year were reported at least that late.
    prefix = pl.col("case_number").str.slice(0, 2)
    prefix_year = (
        df.group_by(prefix.alias("_pfx"))
        .agg(pl.len().alias("_n"), pl.col("occurred_at").dt.year().mode().first().alias("case_year"))
        .filter(pl.col("_n") >= 1000)
        .select("_pfx", pl.col("case_year").cast(pl.Int16))
    )
    df = (
        df.with_columns(prefix.alias("_pfx"))
        .join(prefix_year, on="_pfx", how="left")
        .drop("_pfx")
        .with_columns(
            midnightish.alias("time_unknown"),
            (midnightish & (pl.col("occurred_at").dt.day() == 1)).alias("date_is_month_placeholder"),
            pl.max_horizontal(
                pl.lit(0),
                (pl.date(pl.col("case_year"), 1, 1) - pl.col("occurred_at").dt.date()).dt.total_days(),
            ).cast(pl.Int32).alias("report_lag_min_days"),
        )
    )
    note("crimes", "time_unknown (00:00/00:01)", df["time_unknown"].sum())
    note("crimes", "date_is_month_placeholder", df["date_is_month_placeholder"].sum())
    note("crimes", "case opened in a later year", df.filter(pl.col("report_lag_min_days") > 0).height)
    note("crimes", "case_year unknown (rare prefix)", df["case_year"].null_count())

    # --- Categories ---------------------------------------------------------------------
    group = pl.lit("other")
    for name, pattern in reversed(LOCATION_GROUP_RULES):
        group = pl.when(pl.col("location_description").str.contains(pattern)).then(pl.lit(name)).otherwise(group)
    df = df.with_columns(
        pl.when(pl.col("location_description").is_null()).then(pl.lit("unknown")).otherwise(group).alias("location_group"),
        pl.when(pl.col("fbi_code").is_in(UCR_VIOLENT)).then(pl.lit("violent"))
        .when(pl.col("fbi_code").is_in(UCR_PROPERTY)).then(pl.lit("property"))
        .otherwise(pl.lit("other")).alias("ucr_class"),
        # Homicides have their own ID range, one record per victim rather than per incident.
        (pl.col("id") < 100_000).alias("is_homicide_victim_record"),
    )

    df = add_h3(df)
    cat = pl.Categorical
    df = df.select(
        "id", "case_number", "occurred_at", "time_unknown", "date_is_month_placeholder",
        "case_year", "report_lag_min_days", "updated_at",
        pl.col("primary_type").cast(cat), "description", "iucr", pl.col("fbi_code").cast(cat),
        pl.col("ucr_class").cast(cat), "is_homicide_victim_record", "arrest", "domestic",
        "location_description", pl.col("location_group").cast(cat),
        "block", "latitude", "longitude", pl.col("geo_source").cast(cat),
        "h3_r9", "h3_r8", "h3_r7", "community_area", "ward", "district", "beat",
    ).sort("occurred_at")
    note("crimes", "final rows", df.height)
    return df


# ---------------------------------------------------------------------------------------------
# 311
# ---------------------------------------------------------------------------------------------

def clean_311() -> tuple[pl.DataFrame, pl.DataFrame]:
    print("\n311")
    cols = [
        "sr_number", "sr_type", "sr_short_code", "owner_department", "origin", "status", "duplicate",
        "parent_sr_number", "created_date", "closed_date", "last_modified_date", "street_address",
        "zip_code", "ward", "community_area", "police_district", "police_beat", "latitude", "longitude",
    ]
    raw = pl.read_parquet(RAW / "part-*.parquet", columns=cols)
    note("sr311", "raw rows", raw.height)

    iso = "%Y-%m-%dT%H:%M:%S%.f"
    df = raw.select(
        "sr_number", "sr_type", "sr_short_code", "owner_department", "origin",
        pl.col("status").replace({"Closed": "Completed"}),
        pl.col("duplicate").fill_null(False).alias("is_duplicate"),
        "parent_sr_number",
        pl.col("created_date").str.strptime(pl.Datetime("us"), iso).alias("created_at"),
        pl.col("closed_date").str.strptime(pl.Datetime("us"), iso).alias("closed_at"),
        pl.col("last_modified_date").str.strptime(pl.Datetime("us"), iso).alias("last_modified_at"),
        pl.col("street_address").str.to_uppercase().str.strip_chars().str.replace_all(r"\s+", " ").alias("street_address"),
        pl.col("zip_code").str.slice(0, 5),
        pl.col("ward").cast(pl.Int16, strict=False),
        pl.col("community_area").cast(pl.Int16, strict=False),
        pl.col("police_district").cast(pl.Int16, strict=False),
        pl.col("police_beat").cast(pl.Int16, strict=False),
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False),
    )
    del raw

    n = df.height
    df = df.sort("last_modified_at", descending=True).unique("sr_number", keep="first")
    note("sr311", "dropped: duplicate sr_number", n - df.height)

    n = df.height
    df = df.filter(~pl.col("sr_type").is_in(SR_NON_PLACE_TYPES))
    note("sr311", "dropped: info-only calls + aircraft noise", n - df.height)

    n = df.height
    df = df.filter(pl.col("created_at") >= pl.lit(SR_START).str.to_datetime())
    note("sr311", f"dropped: created before {SR_START}", n - df.height)

    n = df.height
    placeholder = pl.any_horizontal(pl.col("street_address").str.starts_with(a) for a in SR_PLACEHOLDER_ADDRESSES)
    df = df.filter(~placeholder.fill_null(False))
    note("sr311", "dropped: placeholder address", n - df.height)

    n = df.height
    df = df.filter(pl.col("latitude").is_not_null() & in_city_box())
    note("sr311", "dropped: no usable location", n - df.height)

    bad_close = df.filter(pl.col("closed_at") < pl.col("created_at")).height
    note("sr311", "closed before created -> closed_at nulled", bad_close)
    df = df.with_columns(
        pl.when(pl.col("closed_at") >= pl.col("created_at")).then(pl.col("closed_at")).alias("closed_at"),
        pl.when(pl.col("community_area") > 0).then(pl.col("community_area")).alias("community_area"),
    ).with_columns(
        ((pl.col("closed_at") - pl.col("created_at")).dt.total_seconds() / 86_400).cast(pl.Float32).alias("days_to_close"),
        pl.when(pl.col("origin").is_in(SR_RESIDENT_ORIGINS) | pl.col("origin").str.starts_with("spot-open311"))
        .then(pl.lit("resident"))
        .when(pl.col("origin").is_in(SR_ELECTED_ORIGINS)).then(pl.lit("elected_official"))
        .when(pl.col("origin").is_null()).then(pl.lit("unknown"))
        .otherwise(pl.lit("city_internal")).alias("origin_group"),
        slugify("sr_type").alias("sr_type_slug"),
    )

    df = add_h3(df)
    cat = pl.Categorical
    df = df.select(
        "sr_number", "created_at", "closed_at", "last_modified_at", "days_to_close",
        pl.col("sr_type").cast(cat), pl.col("sr_type_slug").cast(cat), pl.col("sr_short_code").cast(cat),
        pl.col("owner_department").cast(cat), pl.col("status").cast(cat),
        pl.col("origin").cast(cat), pl.col("origin_group").cast(cat),
        "is_duplicate", "parent_sr_number",
        "street_address", "latitude", "longitude", "h3_r9", "h3_r8", "h3_r7",
        "zip_code", "community_area", "ward", "police_district", "police_beat",
    ).sort("created_at")
    note("sr311", "final rows", df.height)

    # A year, not a few months: the snow types go quiet every summer.
    discontinued_before = df["created_at"].max().date() - timedelta(days=365)
    catalog = (
        df.group_by("sr_type", "sr_type_slug")
        .agg(
            pl.len().alias("n"),
            pl.col("created_at").min().dt.date().alias("first_seen"),
            pl.col("created_at").max().dt.date().alias("last_seen"),
            pl.col("created_at").dt.truncate("1mo").n_unique().alias("months_with_any"),
            pl.col("is_duplicate").mean().round(3).alias("share_duplicate"),
            (pl.col("status") == "Canceled").mean().round(3).alias("share_canceled"),
            (pl.col("origin_group") == "resident").mean().round(3).alias("share_resident_origin"),
            pl.col("days_to_close").median().round(2).alias("median_days_to_close"),
            pl.col("owner_department").mode().first().cast(pl.String).alias("owner_department"),
        )
        .with_columns(
            (pl.col("last_seen") < discontinued_before).alias("looks_discontinued"),
            pl.col("sr_type").cast(pl.String), pl.col("sr_type_slug").cast(pl.String),
        )
        .sort("n", descending=True)
    )
    return df, catalog


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    crimes = clean_crimes()
    crimes.write_parquet(OUT / "crimes.parquet", compression="zstd", row_group_size=500_000)
    del crimes

    sr, catalog = clean_311()
    sr.write_parquet(OUT / "sr311.parquet", compression="zstd", row_group_size=500_000)
    catalog.write_csv(OUT / "sr_type_catalog.csv")

    (OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nWrote {OUT.relative_to(ROOT)}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
