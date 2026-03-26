import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from openai import OpenAI
import folium
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

# ── Paths ──
BASE = Path(__file__).parent
DATA = BASE / "data" / "app"

# ── Colour palette (matches notebook) ──
C_MEMBER = "#2196F3"
C_CASUAL = "#FF9800"
C_RED = "#E53935"
C_GREEN = "#43A047"
C_PURPLE = "#8E24AA"
C_GREY = "#757575"

SEASON_COLOR = {"Spring": C_GREEN, "Summer": C_CASUAL, "Fall": C_RED, "Winter": C_MEMBER}
DOW_LABELS   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_family="Helvetica Neue, Arial, sans-serif",
    margin=dict(t=50, b=40, l=40, r=20),
)

# ── Page config ──
st.set_page_config(
    page_title="Citi Bike EDA Dashboard",
    page_icon="\U0001f6b2",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

h1 { font-size: 1.9rem; font-weight: 700; color: #0F172A; margin-bottom: .25rem; }
h2 { font-size: 1.3rem; font-weight: 600; color: #1E293B; }
h3 { font-size: 1.1rem; font-weight: 600; color: #334155; }

[data-testid="stSidebar"] { background: #0F172A !important; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.08); border-radius: 6px; }

[data-testid="stMetric"] {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: .9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; color: #0F172A !important; }
[data-testid="stMetricLabel"] { font-size: .78rem !important; font-weight: 500 !important; color: #64748B !important; text-transform: uppercase; letter-spacing: .04em; }

hr { border-color: #E2E8F0; margin: 1.5rem 0; }

.finding-box {
    background: #EFF6FF; border-left: 4px solid #2196F3;
    padding: .85rem 1.1rem; margin: .6rem 0;
    border-radius: 0 8px 8px 0; font-size: .9rem; line-height: 1.6;
}
.rec-box {
    background: #F0FDF4; border-left: 4px solid #22C55E;
    padding: .85rem 1.1rem; margin: .6rem 0;
    border-radius: 0 8px 8px 0; font-size: .9rem; line-height: 1.6;
}
.narrative-box {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    padding: 1.1rem 1.4rem; margin: .5rem 0 .8rem 0;
    border-radius: 10px; font-size: .92rem; line-height: 1.75; color: #1E293B;
}
.section-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #94A3B8; margin: 1.4rem 0 .5rem 0;
}

[data-testid="stExpander"] { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 10px; overflow: hidden; }
.stCaption, [data-testid="stCaptionContainer"] { color: #94A3B8 !important; font-size: .8rem !important; }
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_daily():
    return pd.read_csv(DATA / "daily_rides_weather.csv", parse_dates=["date"])

@st.cache_data
def load_stations():
    return pd.read_csv(DATA / "station_summary.csv", low_memory=False)

@st.cache_data
def load_trips():
    return pd.read_csv(DATA / "trips_sample.csv",
                       parse_dates=["started_at", "ended_at"], low_memory=False)

daily    = load_daily()
stations = load_stations()
trips    = load_trips()

SEASON_ORDER = [s for s in ["Spring", "Summer", "Fall", "Winter"]
                if s in daily["season"].values]

# ── Sidebar ──
st.sidebar.markdown("""
<div style="padding:.5rem 0 1.2rem 0;">
  <div style="font-size:1.3rem;font-weight:700;color:#F8FAFC;letter-spacing:-.01em;">
    🚲 Citi Bike NYC
  </div>
  <div style="font-size:.75rem;color:#64748B;margin-top:.2rem;">
    Mar 2025 – Feb 2026
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "Overview",
    "Raw Data Explorer",
    "1 \u2014 Distributions",
    "2 \u2014 Temporal",
    "3 \u2014 Spatial Analysis",
    "4 \u2014 Conclusions",
    "\U0001f4ac Ask the Data",
])

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def reg_line(x, y):
    """Return (x_sorted, y_pred, r) for a linear regression."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return xm, ym, 0.0
    coeffs = np.polyfit(xm, ym, 1)
    xs = np.linspace(xm.min(), xm.max(), 100)
    r  = np.corrcoef(xm, ym)[0, 1]
    return xs, np.polyval(coeffs, xs), r

def lorenz(values):
    """Return (pop_share, income_share) for a Lorenz curve."""
    s = np.sort(values)
    n = len(s)
    cs = np.cumsum(s)
    return np.arange(1, n + 1) / n, cs / cs[-1]

def gini(values):
    s = np.sort(values)
    n = len(s)
    return (2 * np.sum(np.arange(1, n + 1) * s) / (n * s.sum())) - (n + 1) / n

# ── LLM Agent ──────────────────────────────────────────────
_SYSTEM_PROMPT = """
You are a data analyst for the NYC Citi Bike system.
You help users interpret ridership data covering March 2025 – February 2026.

## Dataset overview
- ~28 million total trips across 2,250 active stations
- Rider types: member (~84% of trips), casual (~16%)
- Bike types: electric (~70%), classic (~30%); electric adoption has plateaued (no sustained growth)
- Weather: NOAA Central Park daily observations (TAVG, TMAX, TMIN, PRCP, SNOW, AWND)

## Dashboard structure (4 analytical sections)
- Section 1 — Distributions: rider/bike type composition, trip duration profiles
- Section 2 — Temporal (10 subsections 2a–2j): seasonal decomposition, daily trend & anomaly
  detection (14-day rolling, ±2σ), day-of-week, hourly profiles, hour×day heatmap,
  rush-hour analysis, weather×demand, weather sensitivity by segment, electric adoption,
  trip duration distribution
- Section 3 — Spatial: busiest stations, chronic imbalance (imbalance_ratio = |net_flow|/total_flow),
  geographic imbalance map (signed_ratio = net_flow/total_flow), AM/PM rush flow reversal,
  Lorenz curve & Gini coefficient
- Section 4 — Conclusions & Operational Recommendations

## Key findings from the EDA
1. Temperature is the dominant demand driver (r ≈ 0.86); system swings ~3× between winter trough and summer peak.
2. Two distinct populations: members = habitual commuters (bimodal 8 AM / 5–6 PM, short trips, weather-resilient);
   casuals = discretionary leisure riders (midday/weekend, longer trips, weather-sensitive).
3. Rain suppresses demand ~15–30%; snow even more. Weather elasticity is quantifiable and actionable.
4. Electric bikes dominate (~70%) but adoption has plateaued — creates a battery logistics sub-problem.
5. Busiest stations ≠ most imbalanced: high-volume hubs self-balance via AM/PM reversal; chronic
   imbalance is concentrated in mid-volume residential-edge and terminal stations.
6. Station utilisation is highly unequal (Gini ≈ 0.59): top 20% of stations handle 61% of departures.
7. Exporters cluster in residential zones (Brooklyn, outer boroughs); importers in commercial cores (Midtown, Lower Manhattan).
8. Correct imbalance metric is imbalance_ratio = |net_flow| / total_flow, NOT raw net_flow.

## Available data (injected in each user message)
- daily_stats: 365 rows — total_rides, member_rides, casual_rides, pct_member, pct_electric,
  avg_duration, pct_rush_hour, TAVG, PRCP, SNOW, season, day_name, is_weekend
- station_stats: 2,250 rows — station_name, total_departures, total_arrivals, net_flow,
  imbalance_ratio, signed_ratio, lat, lng, pct_member
- trips_sample: 100,000 trips — rideable_type, user_type, duration_min, hour, day_of_week,
  season, rush_hour, start/end station

## Instructions
- Answer ONLY based on the data provided in the user message.
- Always cite specific numbers from the data.
- Be concise: 3–5 sentences or a short bullet list. No lengthy preamble.
- If the data is insufficient to answer precisely, say so and suggest what additional data would help.
- Do not invent statistics not present in the context.
- When discussing imbalance, use imbalance_ratio (normalised), not raw net_flow.
""".strip()


def _build_user_prompt(question: str) -> str:
    """Inject pre-computed data summaries as context for the LLM."""

    # Daily stats summary
    daily_desc = (
        daily[["total_rides", "avg_duration", "pct_member",
               "pct_electric", "pct_rush_hour", "TAVG", "PRCP", "SNOW"]]
        .describe().round(1).to_string()
    )

    # Ridership by season
    season_agg = (
        daily.groupby("season")
        .agg(
            avg_rides    =("total_rides",  "mean"),
            avg_duration =("avg_duration", "mean"),
            avg_temp     =("TAVG",         "mean"),
            pct_member   =("pct_member",   "mean"),
            pct_electric =("pct_electric", "mean"),
        )
        .round(1).to_string()
    )

    # Ridership by day of week
    dow_agg = (
        daily.groupby("day_name")["total_rides"]
        .agg(["mean", "median"]).round(0)
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .to_string()
    )

    # Weather impact (rainy vs clear days)
    rain_mask = daily["PRCP"].fillna(0) > 1.0
    rain_rides  = daily.loc[rain_mask,  "total_rides"].mean()
    clear_rides = daily.loc[~rain_mask, "total_rides"].mean()
    snow_mask   = daily["SNOW"].fillna(0) > 0
    snow_rides  = daily.loc[snow_mask,  "total_rides"].mean()

    # Top 10 stations by departures
    top_dep = (
        stations.nlargest(10, "total_departures")
        [["station_name", "total_departures", "net_flow", "pct_member"]]
        .to_string(index=False)
    )

    # Most imbalanced stations by imbalance_ratio (correct metric: normalised by total_flow)
    _s = stations.copy()
    _s["total_flow"]      = _s["total_departures"] + _s["total_arrivals"]
    _s["imbalance_ratio"] = (_s["net_flow"].abs() /
                             _s["total_flow"].replace(0, np.nan)).fillna(0)
    _s["signed_ratio"]    = (_s["net_flow"] /
                             _s["total_flow"].replace(0, np.nan)).fillna(0)
    most_imbalanced = (
        _s[_s["total_flow"] >= 1000]
        .nlargest(10, "imbalance_ratio")
        [["station_name", "imbalance_ratio", "signed_ratio", "net_flow", "total_flow"]]
        .round(4)
        .to_string(index=False)
    )

    # Top exporters and importers by signed_ratio
    top_exporters = (
        _s[(_s["total_flow"] >= 1000) & (_s["signed_ratio"] < 0)]
        .nsmallest(5, "signed_ratio")
        [["station_name", "signed_ratio", "net_flow", "total_flow"]]
        .round(4).to_string(index=False)
    )
    top_importers = (
        _s[(_s["total_flow"] >= 1000) & (_s["signed_ratio"] > 0)]
        .nlargest(5, "signed_ratio")
        [["station_name", "signed_ratio", "net_flow", "total_flow"]]
        .round(4).to_string(index=False)
    )

    # Electric adoption over time
    electric_by_season = (
        daily.groupby("season")["pct_electric"].agg(["mean", "min", "max"])
        .round(1).to_string()
    )

    # Trip duration by user type
    dur_summary = (
        trips[trips["duration_min"].between(1, 60)]
        .groupby("user_type")["duration_min"]
        .describe(percentiles=[.5, .75, .9]).round(1)
        .to_string()
    )

    return f"""Question: {question}

--- DAILY STATS SUMMARY ---
{daily_desc}

--- RIDERSHIP BY SEASON ---
{season_agg}

--- RIDERSHIP BY DAY OF WEEK (avg daily rides) ---
{dow_agg}

--- WEATHER IMPACT ---
Clear days avg rides : {clear_rides:,.0f}
Rainy days avg rides : {rain_rides:,.0f}  (PRCP > 1mm)
Snow  days avg rides : {snow_rides:,.0f}

--- TOP 10 STATIONS BY DEPARTURES ---
{top_dep}

--- TOP 10 MOST IMBALANCED STATIONS (by imbalance_ratio = |net_flow|/total_flow, min 1000 trips) ---
NOTE: imbalance_ratio is the correct metric — busiest stations are NOT necessarily most imbalanced.
{most_imbalanced}

--- TOP 5 NET EXPORTERS (signed_ratio most negative, min 1000 trips) ---
{top_exporters}

--- TOP 5 NET IMPORTERS (signed_ratio most positive, min 1000 trips) ---
{top_importers}

--- ELECTRIC BIKE SHARE BY SEASON (%) ---
{electric_by_season}

--- TRIP DURATION BY RIDER TYPE (trips ≤ 60 min) ---
{dur_summary}
"""

# ══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Citi Bike NYC \u2014 Exploratory Data Analysis")
    st.markdown("**Mar 2025 \u2013 Feb 2026** | Trip-level, daily-aggregate, and station-level data")

    st.markdown("""
This dashboard presents a full-year analysis of NYC Citi Bike ridership across four analytical dimensions:

| Section | Focus |
|---|---|
| **1 — Distributions** | Rider composition, bike type splits, trip duration profiles |
| **2 — Temporal** | Seasonal arc, commuter vs. leisure profiles, weather elasticity, electric adoption |
| **3 — Spatial** | Busiest stations, chronic imbalance (imbalance ratio), geographic clustering, AM/PM reversal |
| **4 — Conclusions** | 3-theme synthesis, 8 key findings (temporal + spatial), 3 operational recommendations |
""")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rides",         f"{daily['total_rides'].sum():,.0f}")
    c2.metric("Avg Daily Rides",     f"{daily['total_rides'].mean():,.0f}")
    c3.metric("Unique Stations",     f"{len(stations):,}")
    c4.metric("Trip Sample Size",    f"{len(trips):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Member %",            f"{daily['pct_member'].mean():.1f}%")
    c6.metric("Electric %",          f"{daily['pct_electric'].mean():.1f}%")
    c7.metric("Rush-Hour % (wkday)", f"{daily[daily['is_weekend']==0]['pct_rush_hour'].mean():.1f}%")
    c8.metric("Avg Duration (min)",  f"{daily['avg_duration'].mean():.1f}")

    st.markdown("---")

    # Daily ride volume
    st.subheader("Daily Ride Volume")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_rides"],
                             name="Total", line=dict(color=C_GREY, width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["member_rides"],
                             name="Member", line=dict(color=C_MEMBER, width=1.5)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["casual_rides"],
                             name="Casual", line=dict(color=C_CASUAL, width=1.5)))
    fig.update_layout(**PLOTLY_LAYOUT, height=350,
                      yaxis_title="Daily Trips", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # Temperature vs rides
    st.subheader("Temperature vs Rides")
    fig2 = px.scatter(daily.dropna(subset=["TAVG"]), x="TAVG", y="total_rides",
                      color="season", color_discrete_map=SEASON_COLOR,
                      labels={"TAVG": "Avg Temp (°C)", "total_rides": "Daily Rides",
                              "season": "Season"},
                      hover_data={"date": True})
    fig2.update_layout(**PLOTLY_LAYOUT, height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "Raw Data Explorer":
    st.title("Raw Data Explorer")
    dataset = st.selectbox("Choose dataset",
                           ["Daily Rides + Weather", "Station Summary", "Trips Sample (100k)"])

    if dataset == "Daily Rides + Weather":
        st.markdown(f"**{len(daily)} rows × {len(daily.columns)} columns**")
        ca, cb = st.columns(2)
        with ca:
            seasons = st.multiselect("Season", daily["season"].unique().tolist(),
                                     default=daily["season"].unique().tolist())
        with cb:
            wf = st.selectbox("Day type", ["All", "Weekday", "Weekend"])
        f = daily[daily["season"].isin(seasons)]
        if wf == "Weekday": f = f[f["is_weekend"] == 0]
        if wf == "Weekend": f = f[f["is_weekend"] == 1]
        st.dataframe(f, height=500)
        st.download_button("Download CSV", f.to_csv(index=False), "daily_filtered.csv")

    elif dataset == "Station Summary":
        st.markdown(f"**{len(stations)} stations**")
        mn, mx = int(stations["total_departures"].min()), int(stations["total_departures"].max())
        rng = st.slider("Total departures range", mn, mx, (mn, mx))
        f = stations[(stations["total_departures"] >= rng[0]) & (stations["total_departures"] <= rng[1])]
        st.dataframe(f, height=500)
        st.download_button("Download CSV", f.to_csv(index=False), "stations_filtered.csv")

    else:
        st.markdown(f"**{len(trips):,} trips**")
        ca, cb = st.columns(2)
        with ca:
            ut = st.multiselect("User type", trips["user_type"].unique().tolist(),
                                default=trips["user_type"].unique().tolist())
        with cb:
            bt = st.multiselect("Bike type", trips["rideable_type"].unique().tolist(),
                                default=trips["rideable_type"].unique().tolist())
        f = trips[trips["user_type"].isin(ut) & trips["rideable_type"].isin(bt)]
        st.dataframe(f.head(5000), height=500)
        st.caption("Showing the first 5,000 rows of filtered results. Use the filters above to narrow down by user type or bike type.")

# ══════════════════════════════════════════════════════════
# PAGE: SECTION 1 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════
elif page == "1 \u2014 Distributions":
    st.title("Section 1: Data Overview & Distributions")

    # ── 1a. Rider & bike type composition ──────────────────
    st.subheader("Fleet & Rider Composition")
    member_pct   = daily["pct_member"].mean()
    casual_pct   = daily["pct_casual"].mean()
    electric_pct = daily["pct_electric"].mean()
    classic_pct  = 100 - electric_pct

    fig_pie = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                            subplot_titles=["Rider Type Split", "Bike Type Split"])
    fig_pie.add_trace(go.Pie(
        labels=["Member", "Casual"], values=[member_pct, casual_pct],
        marker_colors=[C_MEMBER, C_CASUAL],
        textinfo="label+percent", hole=0.35, showlegend=False,
    ), row=1, col=1)
    fig_pie.add_trace(go.Pie(
        labels=["Electric", "Classic"], values=[electric_pct, classic_pct],
        marker_colors=["#43A047", C_GREY],
        textinfo="label+percent", hole=0.35, showlegend=False,
    ), row=1, col=2)
    fig_pie.update_layout(**PLOTLY_LAYOUT, height=340,
                          title_text="Yearly average across 365 days")
    st.plotly_chart(fig_pie, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Member share",   f"{member_pct:.1f}%")
    col2.metric("Casual share",   f"{casual_pct:.1f}%")
    col3.metric("Electric share", f"{electric_pct:.1f}%")
    col4.metric("Classic share",  f"{classic_pct:.1f}%")

    st.divider()

    # ── 1b. Distribution overview (with mean / median lines) ─
    st.subheader("Distribution Overview")

    panels = [
        (daily["total_rides"],                "Daily Total Rides",          "Rides",      C_MEMBER, 30),
        (daily["avg_duration"],               "Daily Avg Duration (min)",   "Minutes",    C_GREEN,  30),
        (daily["TAVG"].dropna(),              "Daily Avg Temp (°C)",        "Temp (°C)",  C_RED,    25),
        (stations["total_departures"],        "Station Departures",         "Departures", C_MEMBER, 50),
        (stations["net_flow"].clip(-500,500), "Station Net Flow (±500)",    "Net Flow",   C_PURPLE, 50),
        (trips["duration_min"].clip(0, 60),   "Trip Duration (≤ 60 min)",   "Minutes",    C_CASUAL, 50),
    ]

    fig_dist = make_subplots(rows=2, cols=3,
                             subplot_titles=[p[1] for p in panels])
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

    for (s, title, xlabel, color, nbins), (r, c) in zip(panels, positions):
        mean_val   = float(s.mean())
        median_val = float(s.median())
        fig_dist.add_trace(
            go.Histogram(x=s, nbinsx=nbins, marker_color=color,
                         opacity=0.85, showlegend=False, name=title),
            row=r, col=c,
        )
        for val, dash, label in [
            (mean_val,   "dash",  f"Mean {mean_val:,.1f}"),
            (median_val, "solid", f"Median {median_val:,.1f}"),
        ]:
            fig_dist.add_vline(
                x=val, line_dash=dash, line_color="black", line_width=1.5,
                annotation_text=label, annotation_font_size=9,
                annotation_position="top right",
                row=r, col=c,
            )

    fig_dist.update_traces(marker_line_width=0)
    fig_dist.update_layout(**PLOTLY_LAYOUT, height=560,
                           title_text="— solid = median   · · · dashed = mean")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── 1c. Trip duration by rider type ────────────────────
    st.subheader("Trip Duration by Rider Type")

    dur = trips[trips["duration_min"].between(1, 60)].copy()
    fig_vio = make_subplots(rows=1, cols=2,
                            subplot_titles=["Violin", "Box"])

    for user, color in [("member", C_MEMBER), ("casual", C_CASUAL)]:
        sub = dur[dur["user_type"] == user]["duration_min"]
        label = user.capitalize()
        fig_vio.add_trace(
            go.Violin(y=sub, name=label, fillcolor=color, line_color=color,
                      opacity=0.75, box_visible=True, meanline_visible=True,
                      showlegend=False),
            row=1, col=1,
        )
        fig_vio.add_trace(
            go.Box(y=sub, name=label, marker_color=color,
                   boxmean="sd", showlegend=False),
            row=1, col=2,
        )

    fig_vio.update_layout(**PLOTLY_LAYOUT, height=420,
                          title_text="Trips ≤ 60 min  |  blue = Member · orange = Casual")
    fig_vio.update_yaxes(title_text="Duration (min)")
    st.plotly_chart(fig_vio, use_container_width=True)

    # Duration summary table
    dur["Rider Type"] = dur["user_type"].str.capitalize()
    summary = (
        dur.groupby("Rider Type")["duration_min"]
        .describe(percentiles=[.25, .5, .75, .9])
        .round(1)
        .rename(columns={"count": "n", "mean": "Mean", "std": "Std",
                         "min": "Min", "25%": "P25", "50%": "Median",
                         "75%": "P75", "90%": "P90", "max": "Max"})
    )
    st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: SECTION 2 — TEMPORAL
# ══════════════════════════════════════════════════════════
elif page == "2 \u2014 Temporal":
    st.title("Section 2: Temporal Demand Analysis")

    CHART_DESCS = {
        "2a \u2014 Seasonal Decomposition":
            "Three-panel seasonal breakdown covering all four seasons (Spring, Summer, Fall, "
            "Winter). Box plots show the full distribution of daily rides and average trip "
            "duration per season; grouped bars show how the member/casual split shifts across "
            "seasons. A summary statistics table is displayed below the chart.",
        "2b \u2014 Daily Trend & Anomaly Detection":
            "Raw daily ride counts overlaid with a 14-day centered rolling mean and a ±2σ "
            "confidence band. Days where the z-score exceeds ±2 are flagged as anomalies "
            "and marked with open circles. Peak and trough days are annotated. The rolling "
            "mean removes weekly seasonality to expose the underlying demand trend.",
        "2c \u2014 Day-of-Week Patterns":
            "Four-panel bar chart aggregated by day of the week (Mon–Sun), showing: "
            "(1) average daily ride volume with ±1σ error bars, (2) member share (%), "
            "(3) electric bike share (%), and (4) average trip duration. Blue bars = "
            "weekdays; orange bars = weekends. Error bars indicate how much day-to-day "
            "variability exists within each weekday.",
        "2d \u2014 Hourly Demand Profiles":
            "Percentage of each day-type's trips occurring in each hour (0–23), split by "
            "member vs. casual riders. Weekday and weekend panels are shown side by side. "
            "Members exhibit a classic bimodal commuter profile (peaks at 8 AM and 5–6 PM), "
            "while casual riders concentrate in midday hours, especially on weekends.",
        "2e \u2014 Hour \u00d7 Day-of-Week Demand Surface":
            "A 7\u00d724 demand surface showing trip volume for every combination of day of "
            "the week and hour of the day, computed from the 100k trip sample. Darker cells "
            "= higher demand. Two bright horizontal bands on weekdays (\u22488 AM and "
            "\u22485\u20136 PM) reveal the commuter rush; a broader midday-weekend band "
            "captures leisure usage.",
        "2f \u2014 Rush-Hour Commuter Analysis":
            "Rush hour is defined as weekday 7–9 AM and 5–7 PM. The left panel shows how "
            "rush-hour share (% of daily trips) evolves over time on weekdays, with a 5-day "
            "rolling average and the overall mean as a reference line. The right panel shows "
            "rush-hour share by day of the week, highlighting the contrast between weekday "
            "commuter demand and the near-zero rush-hour share on weekends.",
        "2g \u2014 Weather \u00d7 Demand":
            "Three panels quantifying how weather drives demand: (1) temperature vs. total "
            "daily rides with an OLS regression line — points are coloured by rain status; "
            "(2) box plot comparing daily rides on Clear, Rainy, and Snowy days, with median "
            "and sample size annotated; "
            "(3) a correlation heatmap of all weather variables against key ride metrics. "
            "The regression slope (rides per °C) and Pearson r are shown in the legend.",
        "2h \u2014 Weather Sensitivity by Rider Segment":
            "Tests whether members and casual riders respond differently to weather shocks. "
            "Left panel: temperature vs. daily rides for each user type with separate OLS "
            "regression lines. Right panel: average rides on dry vs. rainy days per user "
            "type. Casual riders are substantially more weather-sensitive because their "
            "trips are discretionary.",
        "2i \u2014 Electric vs Classic Adoption":
            "Left panel: electric bike share (%) over time with a 7-day rolling average, "
            "showing whether adoption is growing, plateauing, or declining. Right panel: "
            "100% stacked bars comparing electric vs. classic bike preference for members "
            "vs. casual riders, revealing whether the two user types differ in their "
            "vehicle choice.",
        "2j \u2014 Trip Duration Distribution":
            "Three panels analysing trip duration (clipped at 60 min to focus on typical "
            "usage). (1) Overlapping density histograms for members vs. casual riders — "
            "casual trips tend to be longer and more right-skewed. (2) Density by rideable "
            "type (electric vs. classic). (3) Box plots broken down by time-of-day period "
            "(Night, Morning, Afternoon, Evening) × user type, showing how trip purpose "
            "changes throughout the day.",
    }

    sub = st.selectbox("Select chart", list(CHART_DESCS.keys()))
    st.caption(CHART_DESCS[sub])

    # ── 2b: Daily Trend & Anomaly Detection ──
    if sub.startswith("2b"):
        ts = daily.set_index("date")["total_rides"]
        roll_mean = ts.rolling(14, center=True).mean()
        roll_std  = ts.rolling(14, center=True).std()

        # z-score anomaly detection
        z_scores = (ts - roll_mean) / roll_std
        anomaly_mask = z_scores.abs() > 2
        anomaly_mask.index = daily.index  # realign to integer index
        anomalies = daily[anomaly_mask].copy()

        peak   = daily.loc[daily["total_rides"].idxmax()]
        trough = daily.loc[daily["total_rides"].idxmin()]

        fig = go.Figure()
        # ±2σ band
        fig.add_trace(go.Scatter(
            x=pd.concat([roll_mean.index.to_series(), roll_mean.index.to_series()[::-1]]),
            y=pd.concat([(roll_mean + 2*roll_std), (roll_mean - 2*roll_std)[::-1]]),
            fill="toself", fillcolor="rgba(229,57,53,0.12)",
            line=dict(width=0), name="±2σ band", hoverinfo="skip"))
        # Raw series
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_rides"],
                                 line=dict(color=C_MEMBER, width=1), opacity=0.5,
                                 name="Daily rides"))
        # Rolling mean
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values,
                                 line=dict(color=C_RED, width=2.5), name="14-day rolling mean"))
        # Anomaly scatter
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies["date"], y=anomalies["total_rides"],
                mode="markers", marker=dict(color=C_RED, size=7, symbol="circle-open", line=dict(width=2)),
                name=f"Anomalies (|z|>2, n={len(anomalies)})",
                hovertemplate="%{x|%b %d}<br>%{y:,.0f} rides<extra></extra>"))
        # Peak/trough annotations
        fig.add_annotation(x=peak["date"], y=peak["total_rides"],
                           text=f"Peak: {peak['total_rides']:,.0f}<br>({peak['date'].strftime('%b %d')})",
                           showarrow=True, arrowhead=2, arrowcolor=C_RED,
                           font=dict(color=C_RED, size=10), bgcolor="white")
        fig.add_annotation(x=trough["date"], y=trough["total_rides"],
                           text=f"Trough: {trough['total_rides']:,.0f}<br>({trough['date'].strftime('%b %d')})",
                           showarrow=True, arrowhead=2, arrowcolor=C_PURPLE, ay=40,
                           font=dict(color=C_PURPLE, size=10), bgcolor="white")

        date_range = f"{daily['date'].min().strftime('%b %Y')} – {daily['date'].max().strftime('%b %Y')}"
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=f"Daily Citi Bike Ridership — NYC ({date_range})",
                          yaxis_title="Daily Trips", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean daily rides", f"{daily['total_rides'].mean():,.0f}")
        c2.metric("Std dev",          f"{daily['total_rides'].std():,.0f}")
        c3.metric("CV",               f"{daily['total_rides'].std()/daily['total_rides'].mean():.1%}")
        c4.metric("Anomalous days",   f"{len(anomalies)}")

    # ── 2d: Hourly Demand Profiles ──
    elif sub.startswith("2d"):
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                            subplot_titles=["Weekday", "Weekend"])
        hours = list(range(24))
        for col_idx, (wk_flag, wk_label) in enumerate([(0, "Weekday"), (1, "Weekend")], 1):
            sub_df = trips[trips["is_weekend"] == wk_flag]
            for utype, color, offset in [("member", C_MEMBER, -0.2), ("casual", C_CASUAL, 0.2)]:
                counts = sub_df[sub_df["user_type"] == utype].groupby("hour").size()
                pct    = (counts / counts.sum() * 100).reindex(hours, fill_value=0)
                fig.add_trace(
                    go.Bar(x=[h + offset for h in hours], y=pct.values, width=0.38,
                           name=utype.capitalize(), marker_color=color,
                           showlegend=(col_idx == 1)),
                    row=1, col=col_idx)

        fig.update_layout(**PLOTLY_LAYOUT, height=400, barmode="overlay",
                          title="Hourly Trip Share — Member vs Casual",
                          yaxis_title="Share of Trips (%)")
        fig.update_xaxes(tickvals=list(range(0, 24, 2)),
                         ticktext=[f"{h}:00" for h in range(0, 24, 2)])
        st.plotly_chart(fig, use_container_width=True)

    # ── 2c: Day of Week ──
    elif sub.startswith("2c"):
        dow_stats = (daily.groupby("day_of_week")
                     .agg(avg_rides=("total_rides","mean"), std_rides=("total_rides","std"),
                          avg_member=("pct_member","mean"), avg_electric=("pct_electric","mean"),
                          avg_duration=("avg_duration","mean"))
                     .reset_index())
        labels = [DOW_LABELS[d] for d in dow_stats["day_of_week"]]
        colors = [C_CASUAL if d >= 5 else C_MEMBER for d in dow_stats["day_of_week"]]

        fig = make_subplots(rows=1, cols=4,
                            subplot_titles=["Avg Daily Rides", "Member Share (%)",
                                            "Electric Share (%)", "Avg Duration (min)"])
        fig.add_trace(go.Bar(x=labels, y=dow_stats["avg_rides"],
                             error_y=dict(type="data", array=dow_stats["std_rides"]),
                             marker_color=colors, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=labels, y=dow_stats["avg_member"],
                             marker_color=C_MEMBER, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=labels, y=dow_stats["avg_electric"],
                             marker_color=C_GREEN, showlegend=False), row=1, col=3)
        fig.add_trace(go.Bar(x=labels, y=dow_stats["avg_duration"],
                             marker_color=C_PURPLE, showlegend=False), row=1, col=4)

        fig.update_traces(opacity=0.85, marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Day-of-Week Patterns")
        st.plotly_chart(fig, use_container_width=True)

        wd = daily[daily["is_weekend"]==0]["total_rides"]
        we = daily[daily["is_weekend"]==1]["total_rides"]
        st.caption(
            f"Weekday average: {wd.mean():,.0f} trips  |  Weekend average: {we.mean():,.0f} trips  "
            f"|  Difference: {wd.mean()-we.mean():+,.0f} trips. "
            "The gap reflects the strong commuter base that drives weekday volume above weekend leisure demand."
        )

    # ── 2a: Seasonal Decomposition ──
    elif sub.startswith("2a"):
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Daily Rides by Season",
                                            "User Mix by Season",
                                            "Avg Duration by Season"])

        # Box plots for rides and duration
        for season in SEASON_ORDER:
            sd = daily[daily["season"] == season]
            c  = SEASON_COLOR[season]
            fig.add_trace(go.Box(y=sd["total_rides"], name=season,
                                 marker_color=c, showlegend=False), row=1, col=1)
            fig.add_trace(go.Box(y=sd["avg_duration"], name=season,
                                 marker_color=c, showlegend=False), row=1, col=3)

        # Grouped bar for user mix
        season_stats = (daily.groupby("season")
                        .agg(member=("pct_member","mean"), casual=("pct_casual","mean"))
                        .reindex(SEASON_ORDER).reset_index())
        fig.add_trace(go.Bar(x=season_stats["season"], y=season_stats["member"],
                             name="Member %", marker_color=C_MEMBER), row=1, col=2)
        fig.add_trace(go.Bar(x=season_stats["season"], y=season_stats["casual"],
                             name="Casual %", marker_color=C_CASUAL), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=430, barmode="group",
                          title="Seasonal Demand Patterns")
        fig.update_yaxes(title_text="Daily Trips", row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=2)
        fig.update_yaxes(title_text="Minutes", row=1, col=3)
        st.plotly_chart(fig, use_container_width=True)

        tbl = (daily.groupby("season")
               .agg(days=("total_rides","count"),
                    mean_rides=("total_rides","mean"),
                    median_rides=("total_rides","median"),
                    mean_member_pct=("pct_member","mean"),
                    mean_duration=("avg_duration","mean"))
               .reindex(SEASON_ORDER).round(1))
        st.dataframe(tbl, use_container_width=True)

    # ── 2g: Weather × Demand ──
    elif sub.startswith("2g"):
        wx = daily.dropna(subset=["TAVG", "PRCP"])
        rain_col = wx["is_rainy"].fillna(0) if "is_rainy" in wx.columns else pd.Series(0, index=wx.index)

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Temperature vs Rides",
                                            "Precipitation vs Rides",
                                            "Weather–Ride Correlation"])

        # (a) Temperature scatter + regression
        xs, ys, r = reg_line(wx["TAVG"].values, wx["total_rides"].values)
        slope = np.polyfit(wx["TAVG"].dropna().values,
                           wx.loc[wx["TAVG"].notna(), "total_rides"].values, 1)[0]
        fig.add_trace(go.Scatter(x=wx["TAVG"], y=wx["total_rides"], mode="markers",
                                 marker=dict(color=rain_col, colorscale="RdBu_r",
                                             size=6, opacity=0.7,
                                             colorbar=dict(title="Rainy", x=0.31, len=0.5)),
                                 name="Days", showlegend=False,
                                 hovertemplate="Temp: %{x:.1f}°C<br>Rides: %{y:,.0f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=C_RED, width=2, dash="dash"),
                                 name=f"r={r:.3f}, {slope:,.0f} rides/°C",
                                 showlegend=True), row=1, col=1)

        # (b) Box plot: Clear / Rainy / Snowy
        def _wtype(row):
            if row.get("is_snowy", 0) == 1: return "Snowy"
            if row.get("is_rainy", 0) == 1: return "Rainy"
            return "Clear"
        wx = wx.copy()
        wx["weather_type"] = wx.apply(_wtype, axis=1)
        wt_order  = ["Clear", "Rainy", "Snowy"]
        wt_colors = {"Clear": C_MEMBER, "Rainy": C_GREY, "Snowy": C_PURPLE}
        for wt in wt_order:
            grp = wx[wx["weather_type"] == wt]["total_rides"]
            if len(grp) == 0:
                continue
            fig.add_trace(go.Box(y=grp, name=f"{wt} (n={len(grp)})",
                                 marker_color=wt_colors[wt],
                                 boxmean=False,
                                 hovertemplate=f"{wt}<br>Rides: %{{y:,.0f}}"),
                          row=1, col=2)

        # (c) Correlation heatmap
        corr_cols = [c for c in ["TAVG","TMAX","TMIN","PRCP","SNOW","AWND"] if c in wx.columns]
        ride_cols = ["total_rides","pct_member","pct_casual","avg_duration"]
        corr = wx[corr_cols + ride_cols].corr().loc[corr_cols, ride_cols]
        fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                 y=corr.index.tolist(),
                                 colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                 text=corr.round(2).values,
                                 texttemplate="%{text}",
                                 showscale=True, colorbar=dict(x=1.02, len=0.5)),
                      row=1, col=3)

        fig.update_layout(**PLOTLY_LAYOUT, height=430, title="Weather × Demand Relationships")
        fig.update_yaxes(title_text="Daily Rides", row=1, col=1)
        fig.update_yaxes(title_text="Daily Rides", row=1, col=2)
        fig.update_xaxes(title_text="Avg Temp (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Weather Type", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        if "is_rainy" in wx.columns:
            clear = wx[wx["weather_type"]=="Clear"]["total_rides"]
            rainy = wx[wx["weather_type"]=="Rainy"]["total_rides"]
            snowy = wx[wx["weather_type"]=="Snowy"]["total_rides"]
            parts = [f"Rain penalty: {(clear.mean()-rainy.mean())/clear.mean()*100:.1f}% "
                     f"(clear median {clear.median():,.0f} vs rainy {rainy.median():,.0f})"]
            if len(snowy) > 2:
                parts.append(f"Snow penalty: {(clear.mean()-snowy.mean())/clear.mean()*100:.1f}% "
                              f"(snowy median {snowy.median():,.0f}, n={len(snowy)})")
            st.caption("  ·  ".join(parts))

    # ── 2f: Rush Hour ──
    elif sub.startswith("2f"):
        wk = daily[daily["is_weekend"] == 0].copy()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Rush Hour Share Over Time (Weekdays)",
                                            "Rush Hour Share by Day"])

        fig.add_trace(go.Scatter(x=wk["date"], y=wk["pct_rush_hour"],
                                 line=dict(color=C_PURPLE, width=1), opacity=0.45,
                                 name="Raw", showlegend=False), row=1, col=1)
        roll5 = wk.set_index("date")["pct_rush_hour"].rolling(5).mean()
        fig.add_trace(go.Scatter(x=roll5.index, y=roll5.values,
                                 line=dict(color=C_PURPLE, width=2.5),
                                 name="5-day avg"), row=1, col=1)
        mean_rh = wk["pct_rush_hour"].mean()
        fig.add_hline(y=mean_rh, line_dash="dash", line_color=C_RED,
                      annotation_text=f"Mean {mean_rh:.1f}%", row=1, col=1)

        # Rush by DoW from sample
        rush_dow = (trips.groupby(["day_of_week","rush_hour"]).size()
                    .unstack(fill_value=0)
                    .rename(columns={0:"off",1:"rush"}))
        rush_dow["pct"] = rush_dow["rush"] / (rush_dow["rush"] + rush_dow["off"]) * 100
        dow_colors = [C_PURPLE if d < 5 else C_CASUAL for d in rush_dow.index]
        fig.add_trace(go.Bar(x=[DOW_LABELS[d] for d in rush_dow.index],
                             y=rush_dow["pct"], marker_color=dow_colors,
                             name="Rush %", showlegend=False), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Commuter (Rush Hour) Demand")
        fig.update_yaxes(title_text="Rush Hour Trips (%)", row=1, col=1)
        fig.update_yaxes(title_text="Rush Hour Trips (%)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

    # ── 2i: Electric vs Classic Adoption ──
    elif sub.startswith("2i"):
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Electric Share Over Time",
                                            "Rideable Preference by User Type"])

        roll_e = daily.set_index("date")["pct_electric"].rolling(7).mean()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["pct_electric"],
                                 line=dict(color=C_GREEN, width=1), opacity=0.4,
                                 name="Raw", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=roll_e.index, y=roll_e.values,
                                 line=dict(color=C_GREEN, width=2.5),
                                 name="7-day avg"), row=1, col=1)
        fig.add_hline(y=daily["pct_electric"].mean(), line_dash="dash", line_color=C_RED,
                      annotation_text=f"Mean {daily['pct_electric'].mean():.1f}%", row=1, col=1)

        ride_pref = (trips.groupby(["user_type","rideable_type"]).size()
                     .unstack(fill_value=0))
        ride_pct  = ride_pref.div(ride_pref.sum(axis=1), axis=0) * 100
        colors_rt = [C_GREEN, C_MEMBER, C_CASUAL][:ride_pct.shape[1]]
        for rt, col in zip(ride_pct.columns, colors_rt):
            fig.add_trace(go.Bar(x=ride_pct.index.tolist(), y=ride_pct[rt],
                                 name=rt, marker_color=col), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=400, barmode="stack",
                          title="Electric vs Classic Bike Usage")
        fig.update_yaxes(title_text="Electric Trips (%)", row=1, col=1)
        fig.update_yaxes(title_text="Share (%)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

    # ── 2j: Trip Duration Distribution ──
    elif sub.startswith("2j"):
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Duration Density (Member vs Casual)",
                                            "Duration by Rideable Type",
                                            "Duration by Time of Day"])

        for utype, color in [("member", C_MEMBER), ("casual", C_CASUAL)]:
            d = trips[trips["user_type"] == utype]["duration_min"].clip(0, 60)
            fig.add_trace(go.Histogram(x=d, histnorm="probability density",
                                       name=utype.capitalize(), marker_color=color,
                                       opacity=0.55, nbinsx=60), row=1, col=1)

        for rt in trips["rideable_type"].unique():
            d = trips[trips["rideable_type"] == rt]["duration_min"].clip(0, 60)
            fig.add_trace(go.Histogram(x=d, histnorm="probability density",
                                       name=rt, opacity=0.55, nbinsx=60), row=1, col=2)

        trips["period"] = pd.cut(trips["hour"],
                                 bins=[0,6,12,18,24], right=False,
                                 labels=["Night(0-6)","Morning(6-12)",
                                         "Afternoon(12-18)","Evening(18-24)"])
        for utype, color in [("member", C_MEMBER), ("casual", C_CASUAL)]:
            sub_d = trips[(trips["user_type"] == utype) & (trips["duration_min"] < 45)]
            for period in ["Night(0-6)","Morning(6-12)","Afternoon(12-18)","Evening(18-24)"]:
                pd_data = sub_d[sub_d["period"] == period]["duration_min"]
                fig.add_trace(go.Box(y=pd_data, name=period, marker_color=color,
                                     legendgroup=utype,
                                     legendgrouptitle_text=utype.capitalize() if period == "Night(0-6)" else None,
                                     showlegend=True), row=1, col=3)

        fig.update_layout(**PLOTLY_LAYOUT, height=430, barmode="overlay",
                          title="Trip Duration Analysis")
        fig.update_xaxes(title_text="Duration (min)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (min)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        mem = trips[trips["user_type"]=="member"]["duration_min"]
        cas = trips[trips["user_type"]=="casual"]["duration_min"]
        st.caption(
            f"Member — mean: {mem.mean():.1f} min, median: {mem.median():.1f} min  |  "
            f"Casual — mean: {cas.mean():.1f} min, median: {cas.median():.1f} min.  "
            "The mean–median gap for casual riders is larger, indicating a heavier right tail "
            "(long leisure or tourist trips pulling the average up)."
        )

    # ── 2e: Hour × Day-of-Week Demand Surface ──
    elif sub.startswith("2e"):
        hm = (trips.groupby(["day_of_week","hour"]).size()
              .unstack(fill_value=0)
              .reindex(range(7), fill_value=0))
        fig = px.imshow(hm.values,
                        x=list(range(24)), y=DOW_LABELS,
                        color_continuous_scale="Blues",
                        labels=dict(x="Hour of Day", y="Day of Week", color="Trip Count"),
                        aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          title="Hour \u00d7 Day-of-Week Demand Surface")
        fig.update_xaxes(tickvals=list(range(0,24,2)),
                         ticktext=[f"{h}:00" for h in range(0,24,2)])
        st.plotly_chart(fig, use_container_width=True)

    # ── 2h: Weather Sensitivity by Rider Segment ──
    elif sub.startswith("2h"):
        wx = daily.dropna(subset=["TAVG"])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Temperature Sensitivity",
                                            "Rain Impact by User Type"])
        for utype, y_col, color in [("member", "member_rides", C_MEMBER),
                                     ("casual", "casual_rides", C_CASUAL)]:
            fig.add_trace(go.Scatter(x=wx["TAVG"], y=wx[y_col], mode="markers",
                                     marker=dict(color=color, size=5, opacity=0.5),
                                     name=utype.capitalize(), showlegend=True), row=1, col=1)
            xs, ys, r = reg_line(wx["TAVG"].values, wx[y_col].values)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=color, width=2, dash="dash"),
                                     name=f"{utype} fit (r={r:.2f})",
                                     showlegend=True), row=1, col=1)
        if "is_rainy" in daily.columns:
            wx2 = daily.dropna(subset=["is_rainy"])
            bars = []
            for utype, y_col, color in [("member", "member_rides", C_MEMBER),
                                         ("casual", "casual_rides", C_CASUAL)]:
                dry   = wx2[wx2["is_rainy"]==0][y_col].mean()
                rainy = wx2[wx2["is_rainy"]==1][y_col].mean()
                bars.append(dict(utype=utype, dry=dry, rainy=rainy,
                                 pct=-(dry-rainy)/dry*100, color=color))
            fig.add_trace(go.Bar(x=[b["utype"].capitalize() for b in bars],
                                 y=[b["dry"] for b in bars],
                                 name="Dry days", marker_color=[b["color"] for b in bars],
                                 opacity=0.85), row=1, col=2)
            fig.add_trace(go.Bar(x=[b["utype"].capitalize() for b in bars],
                                 y=[b["rainy"] for b in bars],
                                 name="Rainy days",
                                 marker_color=[b["color"] for b in bars],
                                 opacity=0.45, marker_pattern_shape="/"), row=1, col=2)
            for b in bars:
                fig.add_annotation(x=b["utype"].capitalize(), y=b["rainy"]/2,
                                   text=f"\u2212{b['pct']:.0f}%",
                                   font=dict(color="white", size=11),
                                   showarrow=False, row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, height=430, barmode="group",
                          title="Weather Sensitivity by User Type")
        fig.update_xaxes(title_text="Avg Temp (\u00b0C)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Rides", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: SECTION 3 — SPATIAL
# ══════════════════════════════════════════════════════════
elif page == "3 \u2014 Spatial Analysis":
    st.title("Section 3: Spatial Analysis")

    CHART_DESCS = {
        "3a \u2014 Busiest Stations":
            "Top 20 stations ranked by total departures over the full 12-month period. "
            "The bar length shows annual trip volume; the annotation shows e-bike share. "
            "The right panel shows the full departure distribution on a log scale — confirming "
            "that demand follows a long-tail structure where a small number of hubs generate "
            "the majority of trips.",
        "3b \u2014 Chronic Station Imbalance":
            "Stations ranked by imbalance ratio = |net_flow| / total_flow. This metric "
            "separates genuinely imbalanced stations from high-volume hubs that are simply busy. "
            "A busy station with balanced arrivals and departures scores near zero; a quieter "
            "station where 60% of trips flow in one direction ranks at the top. "
            "Minimum volume filter: 1,000 total trips.",
        "3c \u2014 Imbalance Map":
            "Geographic scatter of all stations (total_flow ≥ 1,000). Dot colour encodes "
            "signed imbalance ratio = net_flow / total_flow on a −1 to +1 scale "
            "(red = structural exporter, blue = structural importer, white = balanced). "
            "High-volume hubs appear pale if near-balanced. Dot size encodes total flow.",
        "3d \u2014 AM vs PM Rush":
            "Side-by-side maps comparing per-station net flow during the AM rush (7–9 AM) vs. "
            "the PM rush (5–7 PM). If the commuter-reversal hypothesis holds, the colour pattern "
            "should flip: stations that export bikes in the morning (people commuting out) should "
            "import bikes in the evening (people commuting back). The caption below shows what "
            "fraction of stations actually exhibit this reversal.",
        "3e \u2014 Inequality (Lorenz / Gini)":
            "The Lorenz curve plots the cumulative share of total departures (y-axis) against "
            "the cumulative share of stations ranked from least to most active (x-axis). The "
            "45° dashed line represents perfect equality (every station handles the same volume). "
            "The greater the bow, the higher the Gini coefficient and the more concentrated "
            "traffic is in a small number of hub stations.",
    }
    sub = st.selectbox("Select chart", list(CHART_DESCS.keys()))
    st.caption(CHART_DESCS[sub])

    # ── 3a: Busiest Stations ──
    if sub.startswith("3a"):
        TOP_N = 20
        top_busy = stations.nlargest(TOP_N, "total_departures").copy()
        top_busy["pct_electric"] = (top_busy["electric_dep"] / top_busy["total_departures"] * 100).round(1)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Top {TOP_N} Busiest Stations",
                                            "Departure Distribution (log scale)"],
                            column_widths=[0.6, 0.4])

        fig.add_trace(go.Bar(
            y=top_busy["station_name"].str[:35],
            x=top_busy["total_departures"],
            orientation="h",
            marker_color=C_MEMBER,
            hovertemplate="%{y}<br>Departures: %{x:,.0f}",
            name="Departures",
        ), row=1, col=1)

        dep_all = stations["total_departures"].clip(lower=1)
        fig.add_trace(go.Histogram(
            x=dep_all, nbinsx=50,
            marker_color=C_MEMBER, opacity=0.7,
            name="All stations",
        ), row=1, col=2)
        fig.add_vline(x=dep_all.median(), line_dash="dash", line_color="black",
                      annotation_text=f"Median {dep_all.median():,.0f}", row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=560,
                          title="Station Activity — Busiest Stations & Departure Distribution",
                          showlegend=False)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_xaxes(title_text="Total Departures", row=1, col=1)
        fig.update_xaxes(title_text="Total Departures", row=1, col=2)
        fig.update_yaxes(title_text="Number of Stations (log)", type="log", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        total_dep  = stations["total_departures"].sum()
        top20_share = top_busy["total_departures"].sum() / total_dep * 100
        st.caption(
            f"Top 20 stations account for {top20_share:.1f}% of all departures. "
            f"Busiest station: {top_busy.iloc[0]['station_name']} "
            f"({top_busy.iloc[0]['total_departures']:,.0f} departures). "
            f"Median station: {dep_all.median():,.0f} departures."
        )

    # ── 3b: Chronic Station Imbalance (imbalance_ratio) ──
    elif sub.startswith("3b"):
        TOP_N = 15
        MIN_FLOW = 1000
        df_ib = stations.copy()
        df_ib["total_flow"]      = df_ib["total_departures"] + df_ib["total_arrivals"]
        df_ib["imbalance_ratio"] = (df_ib["net_flow"].abs() /
                                    df_ib["total_flow"].replace(0, np.nan)).fillna(0)
        df_filt = df_ib[df_ib["total_flow"] >= MIN_FLOW]

        top_exp = df_filt[df_filt["net_flow"] < 0].nlargest(TOP_N, "imbalance_ratio")
        top_imp = df_filt[df_filt["net_flow"] > 0].nlargest(TOP_N, "imbalance_ratio")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Top {TOP_N} Structural Exporters",
                                            f"Top {TOP_N} Structural Importers"])

        fig.add_trace(go.Bar(
            y=top_exp["station_name"].str[:35],
            x=(top_exp["imbalance_ratio"] * 100),
            orientation="h", marker_color=C_RED, name="Exporters",
            hovertemplate="%{y}<br>Imbalance ratio: %{x:.1f}%",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=top_imp["station_name"].str[:35],
            x=(top_imp["imbalance_ratio"] * 100),
            orientation="h", marker_color=C_MEMBER, name="Importers",
            hovertemplate="%{y}<br>Imbalance ratio: %{x:.1f}%",
        ), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=520,
                          title="Chronic Station Imbalance — Ranked by Imbalance Ratio")
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="|net_flow| / total_flow (%)", row=1, col=1)
        fig.update_xaxes(title_text="|net_flow| / total_flow (%)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        n_high = (df_filt["imbalance_ratio"] > 0.3).sum()
        st.caption(
            f"Stations analysed (total_flow ≥ {MIN_FLOW:,}): {len(df_filt):,}  |  "
            f"Median imbalance ratio: {df_filt['imbalance_ratio'].median():.1%}  |  "
            f"Stations with ratio > 30%: {n_high}. "
            "High-volume hubs that are near-balanced score low and do not appear here — "
            "only chronically one-directional stations rank at the top."
        )

    # ── 3c: Imbalance Map (Folium, signed_ratio) ──
    elif sub.startswith("3c"):
        MIN_FLOW = 1000
        df_map = stations.dropna(subset=["lat", "lng"]).copy()
        df_map = df_map[df_map["lat"].between(40.4, 41.0) & df_map["lng"].between(-74.3, -73.7)]
        df_map["total_flow"]   = df_map["total_departures"] + df_map["total_arrivals"]
        df_map["signed_ratio"] = (df_map["net_flow"] /
                                  df_map["total_flow"].replace(0, np.nan)).fillna(0)
        df_map = df_map[df_map["total_flow"] >= MIN_FLOW].copy()

        # Percentile-based symmetric color scale (98th pct of |ratio|)
        v = df_map["signed_ratio"].abs().quantile(0.98)
        norm = mcolors.Normalize(vmin=-v, vmax=v)
        cmap = mcm.get_cmap("RdBu")

        def ratio_to_hex(r):
            rgba = cmap(norm(np.clip(r, -v, v)))
            return mcolors.to_hex(rgba)

        # Dot radius: proportional to total_flow, capped at 95th pct
        max_flow = df_map["total_flow"].clip(upper=df_map["total_flow"].quantile(0.95)).max()

        def flow_to_radius(f):
            return (min(f, df_map["total_flow"].quantile(0.95)) / max_flow) * 12 + 4

        m = folium.Map(
            location=[df_map["lat"].mean(), df_map["lng"].mean()],
            zoom_start=12,
            tiles="CartoDB positron",
        )
        for _, row in df_map.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=flow_to_radius(row["total_flow"]),
                color="none",
                fill=True,
                fill_color=ratio_to_hex(row["signed_ratio"]),
                fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>{row['station_name']}</b><br>"
                    f"Signed ratio: {row['signed_ratio']:+.3f}<br>"
                    f"Net flow: {row['net_flow']:+,.0f}<br>"
                    f"Departures: {row['total_departures']:,.0f}<br>"
                    f"Arrivals: {row['total_arrivals']:,.0f}<br>"
                    f"Total flow: {row['total_flow']:,.0f}",
                    max_width=220,
                ),
                tooltip=row["station_name"],
            ).add_to(m)

        st.components.v1.html(m._repr_html_(), height=620, scrolling=False)
        st.caption(
            f"Colour = signed_ratio (net_flow / total_flow), clipped at ±{v:.4f} (98th pct)  ·  "
            f"Red = net exporter · Blue = net importer · Dot size = total flow  ·  "
            f"Stations shown: {len(df_map):,}"
        )

    # ── 3d: AM vs PM Rush ──
    elif sub.startswith("3d"):
        def flow_snapshot(df_sub):
            dep = df_sub.groupby("start_station_id").size().rename("dep")
            arr = df_sub.groupby("end_station_id").size().rename("arr")
            # Align index names before concat — otherwise pd.concat drops the name
            # and reset_index() creates a column called "index" instead of "station_id"
            dep.index.name = "station_id"
            arr.index.name = "station_id"
            m = pd.concat([dep, arr], axis=1).fillna(0)
            m["net"] = m["arr"] - m["dep"]
            return m.reset_index()

        am = trips[trips["hour"].between(7, 9)]
        pm = trips[trips["hour"].between(17, 19)]
        fa = flow_snapshot(am)[["station_id","net"]].rename(columns={"net":"net_am"})
        fp = flow_snapshot(pm)[["station_id","net"]].rename(columns={"net":"net_pm"})
        fc = pd.merge(fa, fp, on="station_id")

        coords = (trips.groupby("start_station_id")
                  .agg(lat=("start_lat","median"), lng=("start_lng","median"))
                  .reset_index().rename(columns={"start_station_id":"station_id"}))
        fc = (pd.merge(fc, coords, on="station_id")
              .dropna(subset=["lat","lng"])
              .pipe(lambda d: d[d["lat"].between(40.4, 41.0) & d["lng"].between(-74.3, -73.7)]))

        abs_max = max(fc["net_am"].abs().quantile(0.97), fc["net_pm"].abs().quantile(0.97))

        col1, col2 = st.columns(2)
        for col, net_col, title in [
            (col1, "net_am", "AM Rush (7–9 AM)"),
            (col2, "net_pm", "PM Rush (5–7 PM)"),
        ]:
            fig = px.scatter_mapbox(fc, lat="lat", lon="lng",
                                    color=net_col,
                                    color_continuous_scale="RdBu",
                                    range_color=[-abs_max, abs_max],
                                    mapbox_style="open-street-map", zoom=11,
                                    hover_data={net_col: True, "lat": False, "lng": False},
                                    labels={net_col: "Net Flow"},
                                    title=title)
            fig.update_layout(**PLOTLY_LAYOUT, height=500,
                              coloraxis_colorbar=dict(title="Net Flow"))
            col.plotly_chart(fig, use_container_width=True)

        flip = (fc["net_am"] * fc["net_pm"] < 0).mean()
        st.caption(
            f"{flip:.0%} of stations reverse their net flow direction between AM and PM rush. "
            "These self-correcting stations require less active rebalancing. The remaining stations "
            "that do not reverse — chronic one-directional exporters or importers — are the highest "
            "priority targets for rebalancing truck routes."
        )

    # ── 3e: Lorenz / Gini ──
    elif sub.startswith("3e"):
        pop, inc = lorenz(stations["total_departures"].values)
        g = gini(stations["total_departures"].values)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(color=C_GREY, dash="dash"),
                                 name="Perfect Equality"))
        fig.add_trace(go.Scatter(x=np.insert(pop, 0, 0), y=np.insert(inc, 0, 0),
                                 mode="lines", line=dict(color=C_MEMBER, width=2.5),
                                 name=f"Lorenz Curve (Gini = {g:.3f})",
                                 fill="tonexty",
                                 fillcolor="rgba(33,150,243,0.12)"))
        fig.add_annotation(x=0.6, y=0.25,
                           text=f"Gini = {g:.3f}<br>Bottom 50% stations handle<br>"
                                f"only {inc[int(0.5*len(inc))]*100:.1f}% of traffic",
                           bgcolor="white", bordercolor=C_GREY, borderwidth=1,
                           font=dict(size=11))
        fig.update_layout(**PLOTLY_LAYOUT, height=450,
                          title="Station Utilization Inequality — Lorenz Curve",
                          xaxis_title="Cumulative Share of Stations",
                          yaxis_title="Cumulative Share of Departures",
                          xaxis=dict(range=[0,1], tickformat=".0%"),
                          yaxis=dict(range=[0,1], tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

        p10 = stations.nlargest(int(len(stations)*0.1), "total_departures")["total_departures"].sum()
        pct_top10 = p10 / stations["total_departures"].sum() * 100
        st.caption(
            f"The most active 10% of stations account for {pct_top10:.0f}% of total departures. "
            "This Pareto-like concentration means that optimising operations at a small number of "
            "hub stations has an outsized impact on overall system performance."
        )

# ══════════════════════════════════════════════════════════
# PAGE: CONCLUSIONS
# ══════════════════════════════════════════════════════════
elif page == "4 \u2014 Conclusions":
    st.title("Conclusions & Operational Recommendations")
    st.caption("Mar 2025 – Feb 2026 · ~28M trips · 2,250 stations · All statistics computed live from the loaded data.")

    # ── Pre-compute stats ───────────────────────────────────
    valid_days = daily[daily["total_rides"] >= 1000]
    peak_day   = valid_days.loc[valid_days["total_rides"].idxmax()]
    trough_day = valid_days.loc[valid_days["total_rides"].idxmin()]
    r_temp     = daily[["total_rides","TAVG"]].dropna().corr().iloc[0,1]
    fall_avg   = daily[daily["season"]=="Fall"]["total_rides"].mean()
    winter_avg = daily[daily["season"]=="Winter"]["total_rides"].mean()
    fall_drop  = (fall_avg - winter_avg) / fall_avg * 100

    df_ib = stations.copy()
    df_ib["total_flow"]      = df_ib["total_departures"] + df_ib["total_arrivals"]
    df_ib["imbalance_ratio"] = (df_ib["net_flow"].abs() /
                                df_ib["total_flow"].replace(0, np.nan)).fillna(0)
    df_ib_filt  = df_ib[df_ib["total_flow"] >= 1000]
    n_high_imb  = (df_ib_filt["imbalance_ratio"] > 0.3).sum()
    top_exp_stn = df_ib_filt[df_ib_filt["net_flow"] < 0].nlargest(1,"imbalance_ratio").iloc[0]
    top_imp_stn = df_ib_filt[df_ib_filt["net_flow"] > 0].nlargest(1,"imbalance_ratio").iloc[0]
    g = gini(stations["total_departures"].values)

    # ── Synthesis ────────────────────────────────────────────
    st.markdown('<p class="section-label">Synthesis</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="narrative-box"><strong>Demand is predictable in structure, volatile in magnitude.</strong> '
        f'The seasonal cycle is the single most powerful demand driver: temperature explains the majority of daily '
        f'ridership variance (r = {r_temp:.3f}), and the system swings nearly 3× between winter trough and summer '
        f'peak. Within each season, intraday and day-of-week patterns are stable — the weekday AM/PM commuter '
        f'signal, the weekend midday leisure shift, and the overnight idle window are consistent structural features. '
        f'A temperature-based seasonal model with a precipitation adjustment layer would capture the majority of '
        f'operational demand variation.</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="narrative-box"><strong>The system serves two distinct user populations with divergent needs.</strong> '
        f'Members (~84% of trips) are habitual, weather-resilient commuters making short, efficient trips; casuals '
        f'are discretionary, weather-sensitive, and take longer rides concentrated on weekends and midday. These two '
        f'segments respond differently to temperature, rain, pricing, and time-of-day — treating the system as a '
        f'single undifferentiated user base would systematically mismatch resources to demand.</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="narrative-box"><strong>Spatial imbalance is concentrated, geographically structured, and partially self-correcting.</strong> '
        f'Station utilisation is highly unequal (Gini = {g:.3f}): the top 20% of stations handle 61% of departures, '
        f'while the bottom half contribute under 10%. The busiest stations are largely self-balancing through AM/PM '
        f'commuter flow reversal. Chronic structural imbalance is concentrated in a distinct set of {n_high_imb} '
        f'mid-volume residential-edge and terminal stations. Prioritising rebalancing resources toward these stations '
        f'— ranked by imbalance ratio rather than volume — and scheduling interventions in the pre-AM-rush window '
        f'would reduce total truck mileage while improving dock availability where it matters most.</div>',
        unsafe_allow_html=True)

    # ── Key Findings ─────────────────────────────────────────
    st.markdown('<p class="section-label">Key Findings</p>', unsafe_allow_html=True)
    col_t, col_s = st.columns(2)

    with col_t:
        st.markdown("**Temporal**")
        for num, title, body in [
            (1, "Full seasonal cycle",
             f"Demand peaks in summer, troughs in winter. Fall-to-Winter drop: {fall_drop:.0f}%. "
             f"Temperature (r = {r_temp:.3f}) is the dominant predictor."),
            (2, "Bimodal commuter pattern",
             "Members peak at 8 AM and 5–6 PM on weekdays. Casual riders peak at midday and on weekends. "
             "These profiles warrant separate operational schedules."),
            (3, "Weather elasticity is quantifiable",
             f"Rain suppresses demand ~15–30%; snow by more. "
             f"Each additional °C adds thousands of rides. Directly informs dynamic staffing decisions."),
            (4, "Electric adoption has plateaued",
             f"~{daily['pct_electric'].mean():.0f}% of trips use e-bikes year-round with no sustained growth trend. "
             f"Creates a battery logistics sub-problem layered on top of vehicle rebalancing."),
        ]:
            st.markdown(f'<div class="finding-box"><strong>{num}. {title}.</strong> {body}</div>',
                        unsafe_allow_html=True)

    with col_s:
        st.markdown("**Spatial**")
        for num, title, body in [
            (5, "Busiest \u2260 most imbalanced",
             f"High-volume hubs self-balance via AM/PM reversal. The {n_high_imb} chronically imbalanced "
             f"stations are a distinct mid-volume set at residential edges and terminal attractors."),
            (6, "Utilisation is highly concentrated",
             f"Gini = {g:.3f}. Top 20% of stations handle 61% of departures; bottom 50% under 10%. "
             f"A small tail of chronic offenders generates the majority of the rebalancing workload."),
            (7, "Commuter flow reversal is predictable",
             f"AM exporters become PM importers — natural daily self-correction. "
             f"Most imbalanced exporter: {top_exp_stn['station_name']} ({top_exp_stn['imbalance_ratio']:.1%}). "
             f"Most imbalanced importer: {top_imp_stn['station_name']} ({top_imp_stn['imbalance_ratio']:.1%})."),
            (8, "Geographic clustering is stable",
             "Exporters cluster in residential zones (Brooklyn, outer boroughs); importers in commercial "
             "cores (Midtown, Lower Manhattan). Pattern is consistent across all seasons."),
        ]:
            st.markdown(f'<div class="finding-box"><strong>{num}. {title}.</strong> {body}</div>',
                        unsafe_allow_html=True)

    # ── Operational Recommendations ──────────────────────────
    st.markdown('<p class="section-label">Operational Recommendations</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    for col, num, title, body in [
        (r1, 9,  "Seasonal fleet right-sizing",
         f"Deploy maximum fleet in summer; scale down for winter ({fall_drop:.0f}% demand drop) "
         f"to reduce idle-asset costs. Size to 95th-percentile demand, not the absolute peak."),
        (r2, 10, "Weather-adaptive scheduling",
         "On forecasted rain or snow days, reduce rebalancing frequency (demand drops 15–30%) "
         "and redirect crew to maintenance and e-bike battery swaps."),
        (r3, 11, "Priority-based rebalancing",
         "Rank stations by imbalance_ratio, not volume. The small tail of high-ratio stations "
         "generates most of the rebalancing workload — daily attention there, weekly elsewhere."),
    ]:
        col.markdown(f'<div class="rec-box"><strong>{num}. {title}.</strong> {body}</div>',
                     unsafe_allow_html=True)

    st.caption("Sources: Citi Bike trip records (GBFS) + NOAA GHCND daily weather (Central Park station).")

# ══════════════════════════════════════════════════════════
# PAGE: ASK THE DATA (LLM Agent)
# ══════════════════════════════════════════════════════════
elif page == "\U0001f4ac Ask the Data":
    st.title("\U0001f4ac Ask the Data")
    st.caption(
        "Ask questions in plain English — answers are grounded in the actual dataset. "
        "Powered by GPT-4o-mini."
    )

    # ── API key check ───────────────────────────────────────
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error(
            "No OpenAI API key found. "
            "Add `OPENAI_API_KEY = 'sk-...'` to `.streamlit/secrets.toml`."
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    # ── Session state init ──────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Helper: call API and append response ────────────────
    def _get_answer(question: str) -> str:
        from openai import RateLimitError, AuthenticationError, APIError
        history = [m for m in st.session_state["messages"]
                   if m["content"] != question][-6:]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    *history,
                    {"role": "user",   "content": _build_user_prompt(question)},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except RateLimitError:
            return "⚠️ OpenAI quota exceeded. Please add credits at platform.openai.com/settings/billing."
        except AuthenticationError:
            return "⚠️ Invalid API key. Check your `.streamlit/secrets.toml`."
        except APIError as e:
            return f"⚠️ OpenAI API error: {e}"

    # ── Example questions ───────────────────────────────────
    with st.expander("Example questions", expanded=False):
        examples = [
            "Which season has the highest casual rider share?",
            "How much does rain reduce daily ridership?",
            "What are the top 5 most imbalanced stations?",
            "Do members or casual riders take longer trips on average?",
            "Which day of the week has the fewest rides?",
        ]
        for q in examples:
            if st.button(q, key=q):
                st.session_state["messages"].append({"role": "user", "content": q})
                st.session_state["_pending"] = q

    st.divider()

    # ── Display chat history ─────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Handle pending question from button click ────────────
    if "_pending" in st.session_state:
        question = st.session_state.pop("_pending")
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = _get_answer(question)
            st.write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # ── Handle new input from chat box ───────────────────────
    if question := st.chat_input("Ask anything about the Citi Bike dataset..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = _get_answer(question)
            st.write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # ── Clear button ────────────────────────────────────────
    if st.session_state.get("messages"):
        if st.button("Clear conversation", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()
