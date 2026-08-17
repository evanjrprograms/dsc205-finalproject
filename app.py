"""
Breathe Easy - county-level U.S. air quality dashboard
DSC 205 Final Project

Run with:  streamlit run app.py

Everything this app reads was built once by prep_data.py. The app itself never
cleans, joins or aggregates - it opens finished tables and draws them. That is
what keeps it responsive.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path("data")

st.set_page_config(
    page_title="Breathe Easy",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# The colour scale
#
# These are the six official EPA health categories. The scale below turns them
# into hard steps rather than a smooth gradient, and every chart in the app is
# locked to the range 0-300 no matter what the user has filtered to.
#
# This is the most important design decision in the app. If the scale rescaled
# itself to whatever is on screen, then filtering down to a clean week in
# Vermont would paint the worst county in it bright red - even though the air
# was fine. Fixing the scale keeps colours comparable across every filter state.
# ---------------------------------------------------------------------------

EPA_COLORS = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
EPA_LABELS = [
    "Good (0-50)",
    "Moderate (51-100)",
    "Unhealthy for sensitive groups (101-150)",
    "Unhealthy (151-200)",
    "Very unhealthy (201-300)",
    "Hazardous (301+)",
]

AQI_MAX = 300  # top of the fixed scale


def epa_step_scale():
    """A stepped colour scale, so 99 and 51 look the same and 101 does not."""
    stops = [0, 50, 100, 150, 200, 300]
    scale = []
    for i, colour in enumerate(EPA_COLORS[:5]):
        lo = stops[i] / AQI_MAX
        hi = stops[i + 1] / AQI_MAX
        scale.append([lo, colour])
        scale.append([hi, colour])
    return scale


# Counties with no monitor are drawn in this grey. They are NOT drawn green.
# A neutral mid-grey reads as "nothing here" rather than as a low value on
# whichever colour scale is active.
NO_DATA = "#C3CDD3"
INK = "#0F3A4D"
TEAL = "#1C7293"
AMBER = "#E8833A"

DAYS_MAX = 120  # fixed top of the "unhealthy days" scale, also never rescales


# ---------------------------------------------------------------------------
# Loading
#
# @st.cache_data means "run this once, then remember the answer". Streamlit
# re-runs this whole file top to bottom every time you touch a widget, so
# without caching every slider drag would re-read these files from disk.
# ---------------------------------------------------------------------------

@st.cache_data
def load_annual():
    """One row per county per year. ~11,000 rows."""
    return pd.read_parquet(DATA / "county_annual.parquet")


@st.cache_data
def load_meta():
    return pd.read_parquet(DATA / "county_meta.parquet")


@st.cache_data
def load_region_month():
    return pd.read_parquet(DATA / "region_month.parquet")


@st.cache_data
def load_mix():
    path = DATA / "pollutant_mix.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_geo():
    """County outlines. 3.2 MB, so definitely load this only once."""
    with open(DATA / "counties.json") as f:
        return json.load(f)


@st.cache_data
def load_daily(year):
    """Daily readings for ONE year - about 320,000 rows.

    Loading all eleven years at once would be 3.5 million rows, which is how
    you exhaust the memory Streamlit's free hosting gives each app.
    """
    path = DATA / f"daily_{year}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def quiet_layout(fig, height=460, has_legend=True):
    """Shared chart styling, so every panel looks like it belongs to the app.

    The legend sits BELOW the plot. Putting it above puts it in the same band
    as the title, and the two overlap as soon as either one gets long.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=64, b=86 if has_legend else 24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        title=dict(font=dict(size=17), x=0, xanchor="left",
                   y=0.97, yanchor="top"),
        showlegend=has_legend,
    )
    if has_legend:
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="top", y=-0.16,
            xanchor="left", x=0,
            title_text="",
        ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_map(df, geo, metric):
    """The choropleth. Counties absent from df simply are not drawn, which is
    how a county with no monitor ends up grey rather than green."""
    if metric == "unhealthy_days":
        colour_args = dict(
            color_continuous_scale=[
                [0.0, "#DFF3E4"], [0.25, "#FFD97D"],
                [0.5, "#FF9F4A"], [0.75, "#E8552F"], [1.0, "#7E0023"],
            ],
            range_color=(0, DAYS_MAX),
            labels={"unhealthy_days": "Days above 100"},
        )
        title = "Days with AQI above 100"
    else:
        colour_args = dict(
            color_continuous_scale=epa_step_scale(),
            range_color=(0, AQI_MAX),
            labels={"median_aqi": "Median AQI"},
        )
        title = "Median AQI"

    fig = px.choropleth(
        df,
        geojson=geo,
        locations="fips",
        color=metric,
        scope="usa",
        hover_name="hover_name",
        hover_data={
            "fips": False,
            metric: True,
            "days_reported": True,
            "main_pollutant": True,
        },
        **colour_args,
    )
    fig.update_traces(marker_line_width=0.15, marker_line_color="rgba(255,255,255,0.4)")
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        lakecolor="rgba(0,0,0,0)",
        landcolor=NO_DATA,          # counties with no monitor show through as this
        showland=True,
        showlakes=False,
        subunitcolor="rgba(255,255,255,0.25)",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=17), x=0, xanchor="left",
                   y=0.97, yanchor="top"),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_ranked_bars(df, metric, n=15):
    top = df.nlargest(n, metric).sort_values(metric)
    label = "Days above 100" if metric == "unhealthy_days" else "Median AQI"
    fig = px.bar(
        top, x=metric, y="hover_name", orientation="h",
        text=metric, labels={metric: label, "hover_name": ""},
    )
    fig.update_traces(marker_color=AMBER, textposition="outside", cliponaxis=False)
    fig.update_layout(title=f"Worst {n} counties")
    return quiet_layout(fig, height=520, has_legend=False)


def build_county_trend(annual, fips, metric):
    """The selected county against the national median, on the same axes."""
    county = annual[annual["fips"] == fips].sort_values("year")
    national = annual.groupby("year", as_index=False)[metric].median()
    label = "Days above 100" if metric == "unhealthy_days" else "Median AQI"
    name = county["hover_name"].iloc[0] if len(county) else "Selected county"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national["year"], y=national[metric], name="National median",
        mode="lines", line=dict(color="#9AAEB8", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=county["year"], y=county[metric], name=name,
        mode="lines+markers", line=dict(color=AMBER, width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(title=f"{name} vs. the national median",
                      yaxis_title=label, xaxis_title="")
    return quiet_layout(fig, height=380)


def build_seasonality(rm, year):
    """Month by census division. Answers: does peak season differ by region?"""
    sub = rm[rm["year"] == year]
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="division", columns="month",
                            values="median_aqi", aggfunc="mean")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = months

    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]

    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale=epa_step_scale(),
        zmin=0, zmax=AQI_MAX, labels=dict(color="Median AQI"),
        text_auto=".0f",
    )
    fig.update_layout(title=f"Median AQI by month and region, {year}",
                      xaxis_title="", yaxis_title="")
    return quiet_layout(fig, height=440, has_legend=False)


def build_slope(annual, first, last, metric, n=12):
    """Which counties improved most, and which got worse."""
    a = annual[annual["year"] == first][["fips", "hover_name", metric]]
    b = annual[annual["year"] == last][["fips", metric]]
    m = a.merge(b, on="fips", suffixes=("_first", "_last")).dropna()
    if m.empty:
        return None, None
    m["change"] = m[f"{metric}_last"] - m[f"{metric}_first"]

    worst = m.nlargest(n // 2, "change")
    best = m.nsmallest(n // 2, "change")
    show = pd.concat([best, worst])

    fig = go.Figure()
    for _, r in show.iterrows():
        improving = r["change"] < 0
        fig.add_trace(go.Scatter(
            x=[str(first), str(last)],
            y=[r[f"{metric}_first"], r[f"{metric}_last"]],
            mode="lines+markers",
            line=dict(color=TEAL if improving else AMBER, width=2),
            marker=dict(size=7),
            name=r["hover_name"], hovertemplate=f"{r['hover_name']}<br>%{{y:.0f}}<extra></extra>",
            showlegend=False,
        ))
    label = "Days above 100" if metric == "unhealthy_days" else "Median AQI"
    fig.update_layout(
        title=f"Biggest movers, {first} to {last}  "
              f"(teal = improved, orange = got worse)",
        yaxis_title=label,
    )
    return quiet_layout(fig, height=480, has_legend=False), show


def build_mix(mix):
    """Is the main pollutant changing? Ozone vs particle pollution over time."""
    if mix.empty:
        return None
    fig = px.area(
        mix.sort_values(["year", "pollutant"]),
        x="year", y="share_pct", color="pollutant",
        labels={"share_pct": "Share of unhealthy days (%)",
                "year": "", "pollutant": "Pollutant"},
    )
    fig.update_layout(title="What is causing the unhealthy days?")
    return quiet_layout(fig, height=420)


def build_equity(df, metric):
    """Are counties with worse air poorer? Association, not cause.

    The fit line is worked out here with numpy rather than handed to Plotly's
    built-in trendline, which needs statsmodels. One less dependency, and one
    less thing that can break on a version mismatch during a live demo.
    """
    sub = df.dropna(subset=["median_income", metric, "population"])
    if sub.empty or len(sub) < 10:
        return None, None

    fig = px.scatter(
        sub, x="median_income", y=metric,
        size="population", size_max=38, color="region",
        hover_name="hover_name", opacity=0.72,
        labels={
            "median_income": "Median household income ($)",
            metric: "Days above 100" if metric == "unhealthy_days" else "Median AQI",
            "region": "Region",
        },
    )

    # Least-squares line through the points, plus how strong the relationship is.
    x = sub["median_income"].astype(float).to_numpy()
    y = sub[metric].astype(float).to_numpy()
    stats = None
    if len(x) >= 3 and x.std() > 0 and y.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            name="Overall trend",
            line=dict(color="#5F7480", width=2.5, dash="dash"),
            hoverinfo="skip",
        ))
        # Change in the metric per $10,000 of income.
        stats = {"slope_per_10k": slope * 10_000, "r": r, "n": len(x)}

    fig.update_layout(title="Income against air quality  ·  bubble size = population")
    return quiet_layout(fig, height=520), stats


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    missing = [f for f in ["county_annual.parquet", "counties.json"]
               if not (DATA / f).exists()]
    if missing:
        st.error(
            f"Missing data files: {', '.join(missing)}\n\n"
            "Run `python prep_data.py` first, then make sure the `data/` folder "
            "is committed to the repo."
        )
        st.stop()

    annual = load_annual().copy()
    meta = load_meta()
    geo = load_geo()
    rm = load_region_month()
    mix = load_mix()

    # A readable label for every county, used in hovers and the dropdown.
    annual["hover_name"] = annual["county"].fillna(annual["county_name"]) \
        + ", " + annual["state"].fillna("")
    annual["hover_name"] = annual["hover_name"].str.strip(", ")

    years = sorted(annual["year"].dropna().unique().astype(int))
    first_year, last_year = years[0], years[-1]

    # -- header ------------------------------------------------------------
    st.title("Breathe Easy")
    st.caption(
        "County-level U.S. air quality, 2015–2025  ·  "
        "EPA Air Quality System + Census ACS  ·  DSC 205"
    )

    # -- sidebar: every global control lives here --------------------------
    with st.sidebar:
        st.header("Filters")

        year = st.slider("Year", first_year, last_year, last_year, step=1)

        metric_label = st.radio(
            "Colour the map by",
            ["Days above AQI 100", "Median AQI"],
            help="Days above 100 is where the EPA starts warning sensitive groups.",
        )
        metric = "unhealthy_days" if metric_label.startswith("Days") else "median_aqi"

        regions = sorted(annual["region"].dropna().unique())
        picked_regions = st.multiselect("Regions", regions, default=regions)

        min_coverage = st.slider(
            "Minimum days reported (%)", 0, 100, 50, step=5,
            help="Counties that reported on fewer days than this are hidden. "
                 "A county with 20 readings all year is not a reliable annual figure.",
        )

        st.divider()
        st.caption(
            f"{len(meta):,} counties in the map. Only {annual['fips'].nunique():,} "
            "have an EPA monitor — the rest are drawn grey, never green."
        )

    # -- apply the filters --------------------------------------------------
    view = annual[
        (annual["year"] == year)
        & (annual["region"].isin(picked_regions))
        & (annual["pct_reported"] >= min_coverage)
    ].copy()

    if view.empty:
        st.warning("No counties match these filters. Try widening the region "
                   "selection or lowering the minimum days reported.")
        st.stop()

    # -- county selection, remembered across re-runs ------------------------
    # Streamlit re-runs this file top to bottom on every interaction, so an
    # ordinary variable would be wiped each time. session_state is the one box
    # whose contents survive, and every chart below reads the county from it.
    options = view.sort_values("hover_name")[["fips", "hover_name"]]
    names = dict(zip(options["fips"], options["hover_name"]))

    if "county" not in st.session_state or st.session_state.county not in names:
        st.session_state.county = options["fips"].iloc[0]

    with st.sidebar:
        st.session_state.county = st.selectbox(
            "County", options=list(options["fips"]),
            format_func=lambda f: names.get(f, f),
            index=list(options["fips"]).index(st.session_state.county),
        )

    selected = st.session_state.county

    # -- KPI strip ----------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Counties reporting", f"{len(view):,}")
    c2.metric("Median AQI", f"{view['median_aqi'].median():.0f}")
    c3.metric("Counties with 30+ bad days",
              f"{int((view['unhealthy_days'] >= 30).sum()):,}")
    exposed = view.loc[view["unhealthy_days"] >= 30, "population"].sum()
    c4.metric("People in those counties",
              f"{exposed/1e6:.1f}M" if pd.notna(exposed) and exposed else "—")

    st.divider()

    # -- the four questions, one per tab ------------------------------------
    t1, t2, t3, t4 = st.tabs([
        "Where is it worst?",
        "When is it worst?",
        "Is it getting better?",
        "Who is exposed?",
    ])

    with t1:
        st.markdown(
            "**The question:** how does air quality differ across the country, "
            "and has the gap between West and East widened since 2015?"
        )
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(build_map(view, geo, metric), width="stretch")
            st.caption(
                "Grey counties have no EPA monitor, so there is nothing to "
                "report. That is not the same as clean air, and the map never "
                "colours them as though it were."
            )
        with right:
            st.plotly_chart(build_ranked_bars(view, metric), width="stretch")

        st.plotly_chart(
            build_county_trend(annual[annual["region"].isin(picked_regions)],
                               selected, metric),
            width="stretch",
        )
        row = view[view["fips"] == selected]
        if not row.empty:
            r = row.iloc[0]
            st.caption(
                f"{r['hover_name']} reported on {int(r['days_reported'])} days in "
                f"{year} ({r['pct_reported']:.0f}% of the year). Most common "
                f"cause of its worst days: {r['main_pollutant']}."
            )

    with t2:
        st.markdown(
            "**The question:** does peak pollution season differ by region — "
            "summer ozone in the Southwest against winter particle pollution "
            "in mountain valleys?"
        )
        fig = build_seasonality(rm, year)
        if fig is None:
            st.info("No monthly data for this year.")
        else:
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Regions are sorted by annual average, cleanest at the top. "
                "The colour scale is the same fixed EPA one used on the map, so "
                "a yellow cell here means exactly what a yellow county means there."
            )

    with t3:
        st.markdown(
            f"**The question:** which counties improved between {first_year} and "
            f"{last_year}, which got worse, and is the main pollutant changing?"
        )
        fig, movers = build_slope(
            annual[annual["region"].isin(picked_regions)],
            first_year, last_year, metric,
        )
        if fig is None:
            st.info("Not enough overlapping counties to compare those years.")
        else:
            st.plotly_chart(fig, width="stretch")
        mixfig = build_mix(mix)
        if mixfig is not None:
            st.plotly_chart(mixfig, width="stretch")
            st.caption(
                "Each band is that pollutant's share of all unhealthy days "
                "nationally. A rising particle band with a falling ozone band "
                "means the problem is changing character, not just size."
            )

    with t4:
        st.markdown(
            "**The question:** are the counties with the worst air also the "
            "poorer ones?"
        )
        fig, stats = build_equity(view, metric)
        if fig is None:
            st.info(
                "Income data is not available in this build, so this panel "
                "cannot be drawn. Re-run prep_data.py with a working "
                "CENSUS_API_KEY to enable it."
            )
        else:
            st.plotly_chart(fig, width="stretch")
            if stats:
                unit = ("days above 100" if metric == "unhealthy_days"
                        else "points of median AQI")
                direction = "fewer" if stats["slope_per_10k"] < 0 else "more"
                st.markdown(
                    f"Across **{stats['n']:,} counties** in {year}, each extra "
                    f"$10,000 of median household income goes with "
                    f"**{abs(stats['slope_per_10k']):.1f} {direction} {unit}** "
                    f"(correlation r = {stats['r']:.2f})."
                )
            st.warning(
                "This is an association, not a cause. Poorer counties are also "
                "more likely to sit near ports, highways and industry, and this "
                "chart cannot separate those from income itself.",
                icon="⚠️",
            )

    # -- footer -------------------------------------------------------------
    st.divider()
    st.caption(
        f"Data: EPA Air Quality System daily AQI by county, {first_year}–{last_year} "
        f"({len(annual):,} county-years)  ·  U.S. Census ACS 5-year (2023)  ·  "
        "County outlines from Census TIGER/Line.  "
        "Built with Streamlit and Plotly for DSC 205."
    )


if __name__ == "__main__":
    main()
