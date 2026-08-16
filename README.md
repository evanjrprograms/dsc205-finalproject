# Breathe Easy

County-level U.S. air quality dashboard. DSC 205 final project.

**Live app:** breathe-easy.streamlit.app

## Run order

```bash
pip install -r requirements.txt

# 1. Get the raw EPA files (see below), put them in data/raw/
# 2. Build the tables the app reads
python prep_data.py

# 3. Run the app
streamlit run app.py
```

### Getting the raw data

Download `daily_aqi_by_county_YYYY.zip` for 2015 through 2025 from
<https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI> and drop the zip
files into `data/raw/`. No need to unzip them — `prep_data.py` reads the zips
directly.

County shapes and Census population/income download themselves on first run.

## What prep_data.py produces

| File | Rows | What it's for |
|---|---|---|
| `county_annual.parquet` | ~35k | One row per county per year. Drives the map, the rankings and the trend lines. |
| `daily_YYYY.parquet` | ~400k each | Daily readings, split by year so the app only loads the year on screen. |
| `region_month.parquet` | ~1.2k | Month by census division. Drives the seasonality heatmap. |
| `pollutant_mix.parquet` | ~70 | Share of unhealthy days by pollutant per year. Drives "what changed". |
| `county_meta.parquet` | 3,143 | Name, division, population, income, land area, density. |
| `counties.json` | 3,143 | Simplified county outlines for the choropleth. |

The script prints a "for your slides" block at the end with the real row counts
— **use those numbers, not the estimates currently in the deck.**

## Decisions worth defending

**Scope is all 50 states plus DC.** Puerto Rico and the other territories are
dropped (`DROP_STATE_CODES`) because the Albers USA projection Plotly uses has
nowhere to put them. Alaska and Hawaii are kept — Albers USA composites them as
insets.

**Missing days are flagged, never filled.** Each county-year carries
`days_reported`, `days_missing` and `pct_reported`. Forward-filling a gap would
invent a clean day during a wildfire, which is a real harm.

**Counties with no monitor are absent, not zero.** Roughly 2,000 of the 3,143
counties have no EPA monitor. They simply have no rows, so the map draws them
grey. Showing an unmeasured county as green would be the most misleading thing
this dashboard could do.

**FIPS is built by zero-padding, not concatenation.** Alabama is state code `1`
in the raw file; `"1" + "001"` gives `1001`, which matches nothing. The script
pads to 2 + 3 characters. This is the single most common silent join failure in
this dataset.

**"Unhealthy" means AQI above 100** — the point where the EPA begins warning
sensitive groups. The six category labels and their boundaries come straight
from the EPA and never change with the filters.

## Data sources

- EPA AQS, Daily AQI by County, 2015–2025 — <https://aqs.epa.gov/aqsweb/airdata/>
- U.S. Census ACS 5-year (2023), tables B01003 and B19013 — <https://api.census.gov>
- County outlines: Plotly's pre-simplified GeoJSON, derived from Census
  TIGER/Line; land area comes from its `CENSUSAREA` property.
