"""
Breathe Easy - data preparation
DSC 205 Final Project

Run this ONCE, before running the app:

    python prep_data.py

It reads the raw EPA files in data/raw/, joins on population, income and map
shapes, works out the columns the dashboard needs, and saves everything as
Parquet. The app itself never does any of this work - it just opens the
finished files. That is what keeps the dashboard fast.

Before you run it, download the EPA files:

    https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI
    Get "daily_aqi_by_county_YYYY.zip" for each year 2015-2025
    Put the zip files (no need to unzip) in data/raw/

Everything else downloads itself.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

YEARS = list(range(2015, 2026))          # 2015 through 2025

DATA = Path("data")
RAW = DATA / "raw"

# We keep all 50 states plus Washington DC.
# Puerto Rico (72) and the other territories are dropped because the map
# projection Plotly uses for the USA cannot place them anywhere sensible.
DROP_STATE_CODES = {"60", "66", "69", "72", "78"}

# County shapes, already simplified so the map draws quickly.
# This file also carries each county's land area, which saves a download.
GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "geojson-counties-fips.json"
)

# Population and median household income, 5-year ACS.
ACS_YEAR = 2023

# The six official EPA health categories. Everything in the app colours by
# these, and the boundaries never change no matter what is filtered.
AQI_BREAKS = [-1, 50, 100, 150, 200, 300, 10_000]
AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

# Census divisions, keyed by state FIPS code.
DIVISION = {
    "09": "New England", "23": "New England", "25": "New England",
    "33": "New England", "44": "New England", "50": "New England",
    "34": "Middle Atlantic", "36": "Middle Atlantic", "42": "Middle Atlantic",
    "17": "East North Central", "18": "East North Central",
    "26": "East North Central", "39": "East North Central",
    "55": "East North Central",
    "19": "West North Central", "20": "West North Central",
    "27": "West North Central", "29": "West North Central",
    "31": "West North Central", "38": "West North Central",
    "46": "West North Central",
    "10": "South Atlantic", "11": "South Atlantic", "12": "South Atlantic",
    "13": "South Atlantic", "24": "South Atlantic", "37": "South Atlantic",
    "45": "South Atlantic", "51": "South Atlantic", "54": "South Atlantic",
    "01": "East South Central", "21": "East South Central",
    "28": "East South Central", "47": "East South Central",
    "05": "West South Central", "22": "West South Central",
    "40": "West South Central", "48": "West South Central",
    "04": "Mountain", "08": "Mountain", "16": "Mountain", "30": "Mountain",
    "32": "Mountain", "35": "Mountain", "49": "Mountain", "56": "Mountain",
    "02": "Pacific", "06": "Pacific", "15": "Pacific", "41": "Pacific",
    "53": "Pacific",
}

REGION = {
    "New England": "Northeast", "Middle Atlantic": "Northeast",
    "East North Central": "Midwest", "West North Central": "Midwest",
    "South Atlantic": "South", "East South Central": "South",
    "West South Central": "South",
    "Mountain": "West", "Pacific": "West",
}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 1 - read the raw EPA files and tidy them up
# ---------------------------------------------------------------------------

def read_epa_year(year):
    """Read one year of EPA daily AQI readings. Returns None if not found."""
    zip_path = RAW / f"daily_aqi_by_county_{year}.zip"
    csv_path = RAW / f"daily_aqi_by_county_{year}.csv"

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                df = pd.read_csv(f, dtype=str)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)
    else:
        log(f"  ! {year}: no file found, skipping")
        return None

    # Column names drift between years and one of them is lower-cased in the
    # real files ("county Name"). Normalise before touching anything.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "state_name": "state",
        "county_name": "county",
        "defining_parameter": "pollutant",
        "number_of_sites_reporting": "num_sites",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    needed = ["state", "county", "state_code", "county_code", "date", "aqi"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{year}: expected columns missing: {missing}")

    # Build the 5-digit county ID. This is the key every dataset joins on, and
    # zero-padding it is the single most common place this kind of join breaks:
    # Alabama is state 1, and "1001" will never match "01001".
    df["fips"] = (
        df["state_code"].str.strip().str.zfill(2)
        + df["county_code"].str.strip().str.zfill(3)
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df["num_sites"] = pd.to_numeric(df.get("num_sites"), errors="coerce")

    # Drop rows we cannot use: no date or no reading.
    before = len(df)
    df = df.dropna(subset=["date", "aqi"])
    dropped = before - len(df)

    # Drop exact duplicate county-days. A handful appear in the raw files.
    before = len(df)
    df = df.drop_duplicates(subset=["fips", "date"], keep="first")
    dupes = before - len(df)

    # Territories out, 50 states + DC in.
    df = df[~df["fips"].str[:2].isin(DROP_STATE_CODES)]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    keep = ["fips", "date", "year", "month", "state", "county",
            "aqi", "pollutant", "num_sites"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    log(f"  {year}: {len(df):>7,} rows  "
        f"({df['fips'].nunique():,} counties, "
        f"{dropped:,} unusable, {dupes:,} duplicates removed)")
    return df


# ---------------------------------------------------------------------------
# Step 2 - the things we join on: map shapes, population, income
# ---------------------------------------------------------------------------

def get_geojson():
    """County outlines. Cached on disk so we only download once."""
    out = DATA / "counties.json"
    if out.exists():
        log("  county shapes: already downloaded")
        return json.loads(out.read_text())

    log("  county shapes: downloading...")
    geo = requests.get(GEOJSON_URL, timeout=120).json()
    geo["features"] = [
        f for f in geo["features"]
        if f["properties"]["STATE"] not in DROP_STATE_CODES
    ]
    out.write_text(json.dumps(geo))
    log(f"  county shapes: {len(geo['features']):,} counties saved")
    return geo


def fetch_acs():
    """Population and median income from the Census API.

    The Census API is fussy: a given ACS year may not be published yet, and it
    sometimes rejects the default requests user-agent. So we try a few
    variations and give up gracefully rather than crashing - population and
    income are only needed for the "who is exposed" question, and the map,
    seasonality and trend charts work fine without them.

    Returns a DataFrame, or None if every attempt failed.
    """
    fields = "NAME,B01003_001E,B19013_001E"

    # The Census API requires a free key. Get one in about a minute at
    # https://api.census.gov/data/key_signup.html (remember to click the
    # activation link in the email), then set it before running:
    #     export CENSUS_API_KEY=your_key_here      (Mac/Linux terminal)
    #     %env CENSUS_API_KEY=your_key_here        (Colab / Jupyter cell)
    # It is read from the environment on purpose, so the key never gets
    # committed to a public repo.
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    suffix = f"&key={key}" if key else ""

    if not key:
        log("    no CENSUS_API_KEY set - the API will refuse the request")

    attempts = [
        (f"ACS {ACS_YEAR}",
         f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
         f"?get={fields}&for=county:*&in=state:*{suffix}"),
        (f"ACS {ACS_YEAR}, no state clause",
         f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
         f"?get={fields}&for=county:*{suffix}"),
        (f"ACS {ACS_YEAR - 1}",
         f"https://api.census.gov/data/{ACS_YEAR - 1}/acs/acs5"
         f"?get={fields}&for=county:*{suffix}"),
        (f"ACS {ACS_YEAR - 2}",
         f"https://api.census.gov/data/{ACS_YEAR - 2}/acs/acs5"
         f"?get={fields}&for=county:*{suffix}"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 (DSC205 student project)"}

    for label, url in attempts:
        try:
            r = requests.get(url, timeout=120, headers=headers)
        except Exception as e:
            log(f"    {label}: request failed ({type(e).__name__})")
            continue

        if r.status_code != 200:
            log(f"    {label}: HTTP {r.status_code}")
            continue

        # The Census API returns its error pages as HTML with status 200, so
        # the status code alone does not tell us whether this worked.
        if r.text.lstrip()[:1] == "<":
            if "missing key" in r.text.lower():
                log(f"    {label}: rejected - CENSUS_API_KEY is missing or "
                    f"not yet activated")
            elif "invalid key" in r.text.lower():
                log(f"    {label}: rejected - CENSUS_API_KEY is not valid")
            else:
                log(f"    {label}: got an HTML error page, not data")
            continue

        try:
            rows = r.json()
        except Exception:
            # Not JSON - show what actually came back so it can be diagnosed.
            log(f"    {label}: response was not JSON -> {r.text[:120]!r}")
            continue

        acs = pd.DataFrame(rows[1:], columns=rows[0])
        acs["fips"] = acs["state"].str.zfill(2) + acs["county"].str.zfill(3)
        acs["population"] = pd.to_numeric(acs["B01003_001E"], errors="coerce")
        acs["median_income"] = pd.to_numeric(acs["B19013_001E"], errors="coerce")

        # The Census API uses large negative numbers as "no estimate available".
        acs.loc[acs["population"] < 0, "population"] = pd.NA
        acs.loc[acs["median_income"] < 0, "median_income"] = pd.NA

        acs = acs[["fips", "population", "median_income"]]
        acs = acs[~acs["fips"].str[:2].isin(DROP_STATE_CODES)]
        log(f"    {label}: OK, {len(acs):,} counties")
        return acs

    return None


def get_county_meta(geo):
    """County name, land area, division, and (if available) population/income."""
    # Land area and county names ride along in the shape file - no download.
    meta = pd.DataFrame([
        {"fips": f["id"],
         "county_name": f["properties"]["NAME"],
         "land_area_sqmi": f["properties"].get("CENSUSAREA")}
        for f in geo["features"]
    ])

    log("  population + income: downloading from Census API...")
    acs = fetch_acs()

    if acs is None:
        log("  ! Census API unavailable. Continuing WITHOUT population and")
        log("    income. The map, seasonality and trend charts will all work;")
        log("    the 'who is exposed' equity question will not.")
        meta["population"] = pd.NA
        meta["median_income"] = pd.NA
    else:
        meta = meta.merge(acs, on="fips", how="left")

    meta["division"] = meta["fips"].str[:2].map(DIVISION)
    meta["region"] = meta["division"].map(REGION)
    meta["density"] = (
        pd.to_numeric(meta["population"], errors="coerce")
        / pd.to_numeric(meta["land_area_sqmi"], errors="coerce")
    ).round(1)

    log(f"  county details: {len(meta):,} counties "
        f"({pd.to_numeric(meta['population'], errors='coerce').isna().sum():,} "
        f"missing population)")
    return meta


# ---------------------------------------------------------------------------
# Step 3 - add the columns the dashboard actually needs
# ---------------------------------------------------------------------------

def add_derived(df, meta):
    """Sort each day into a health category and attach county details."""
    df["category"] = pd.cut(
        df["aqi"], bins=AQI_BREAKS, labels=AQI_LABELS, right=True
    ).astype(str)

    # "Unhealthy" for our purposes means above 100 - the point where the EPA
    # starts warning sensitive groups. This is the number the app counts.
    df["is_unhealthy"] = df["aqi"] > 100

    lookup = meta.set_index("fips")
    df["division"] = df["fips"].map(lookup["division"])
    df["region"] = df["fips"].map(lookup["region"])

    unmatched = df["division"].isna().sum()
    if unmatched:
        log(f"  ! {unmatched:,} rows have a county ID with no matching state")
    return df


# ---------------------------------------------------------------------------
# Step 4 - pre-compute the summaries, then save
# ---------------------------------------------------------------------------

def build_annual(df, meta):
    """One row per county per year. This is what the map and rankings use."""
    g = df.groupby(["fips", "year"], observed=True)

    annual = g.agg(
        state=("state", "first"),
        county=("county", "first"),
        days_reported=("aqi", "size"),
        median_aqi=("aqi", "median"),
        mean_aqi=("aqi", "mean"),
        max_aqi=("aqi", "max"),
        unhealthy_days=("is_unhealthy", "sum"),
    ).reset_index()

    # How complete is each county-year? We report this rather than filling
    # the gaps in - inventing a clean day during a wildfire is a real harm.
    days_in_year = (
        pd.to_datetime(annual["year"].astype(str) + "-12-31").dt.dayofyear
    )
    annual["days_missing"] = days_in_year - annual["days_reported"]
    annual["pct_reported"] = (annual["days_reported"] / days_in_year * 100).round(1)
    annual["pct_unhealthy"] = (
        annual["unhealthy_days"] / annual["days_reported"] * 100
    ).round(1)

    # Which pollutant was to blame most often that year?
    if "pollutant" in df.columns:
        top = (
            df.groupby(["fips", "year", "pollutant"], observed=True)
              .size().rename("n").reset_index()
              .sort_values("n", ascending=False)
              .drop_duplicates(subset=["fips", "year"])
              .rename(columns={"pollutant": "main_pollutant"})
        )
        annual = annual.merge(
            top[["fips", "year", "main_pollutant"]], on=["fips", "year"], how="left"
        )

    annual = annual.merge(
        meta[["fips", "county_name", "division", "region",
              "population", "median_income", "land_area_sqmi", "density"]],
        on="fips", how="left",
    )

    for c in ["median_aqi", "mean_aqi"]:
        annual[c] = annual[c].round(1)

    return annual


def build_region_month(df):
    """Month by division. This is what the seasonality heatmap uses."""
    rm = (
        df.groupby(["division", "year", "month"], observed=True)
          .agg(median_aqi=("aqi", "median"),
               mean_aqi=("aqi", "mean"),
               unhealthy_days=("is_unhealthy", "sum"),
               readings=("aqi", "size"))
          .reset_index()
    )
    rm["median_aqi"] = rm["median_aqi"].round(1)
    rm["mean_aqi"] = rm["mean_aqi"].round(1)
    return rm


def build_pollutant_mix(df):
    """Share of unhealthy days by pollutant, per year. Powers 'what changed'."""
    if "pollutant" not in df.columns:
        return pd.DataFrame()
    bad = df[df["is_unhealthy"]]
    mix = (
        bad.groupby(["year", "pollutant"], observed=True)
           .size().rename("unhealthy_days").reset_index()
    )
    total = mix.groupby("year")["unhealthy_days"].transform("sum")
    mix["share_pct"] = (mix["unhealthy_days"] / total * 100).round(1)
    return mix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)

    if not any(RAW.glob("daily_aqi_by_county_*")):
        raise SystemExit(
            f"No EPA files found in {RAW}/\n"
            "Download daily_aqi_by_county_YYYY.zip for 2015-2025 from\n"
            "https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI\n"
            "and put them there, then run this again."
        )

    log("\n[1/4] Reading the raw EPA files")
    frames = [f for f in (read_epa_year(y) for y in YEARS) if f is not None]
    if not frames:
        raise SystemExit("No usable EPA files were read.")
    df = pd.concat(frames, ignore_index=True)
    log(f"  total: {len(df):,} county-days across {df['year'].nunique()} years")

    log("\n[2/4] Joining on shapes, population and income")
    geo = get_geojson()
    meta = get_county_meta(geo)

    log("\n[3/4] Adding the columns the dashboard needs")
    df = add_derived(df, meta)
    annual = build_annual(df, meta)
    region_month = build_region_month(df)
    mix = build_pollutant_mix(df)
    log(f"  annual summary:   {len(annual):>7,} rows")
    log(f"  month by region:  {len(region_month):>7,} rows")

    log("\n[4/4] Saving")
    meta.to_parquet(DATA / "county_meta.parquet", index=False)
    annual.to_parquet(DATA / "county_annual.parquet", index=False)
    region_month.to_parquet(DATA / "region_month.parquet", index=False)
    if not mix.empty:
        mix.to_parquet(DATA / "pollutant_mix.parquet", index=False)

    daily_cols = ["fips", "date", "aqi", "category", "pollutant",
                  "num_sites", "division", "region"]
    for year, part in df.groupby("year", observed=True):
        part[[c for c in daily_cols if c in part.columns]].to_parquet(
            DATA / f"daily_{year}.parquet", index=False
        )

    log("\nDone. Files written to data/:")
    total_mb = 0
    for p in sorted(DATA.glob("*.parquet")) + [DATA / "counties.json"]:
        mb = p.stat().st_size / 1_000_000
        total_mb += mb
        log(f"  {p.name:<28} {mb:>7.1f} MB")
    log(f"  {'TOTAL':<28} {total_mb:>7.1f} MB")

    # Numbers worth copying into your slides.
    log("\n--- for your slides ---")
    log(f"  daily rows, 2015-2025:      {len(df):,}")
    log(f"  counties with a monitor:    {df['fips'].nunique():,}")
    log(f"  counties in the shape file: {len(meta):,}")
    log(f"  counties with NO monitor:   {len(meta) - df['fips'].nunique():,}")
    log(f"  annual summary rows:        {len(annual):,}")


if __name__ == "__main__":
    sys.exit(main())
