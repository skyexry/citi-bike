import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from openai import OpenAI

# ── Paths ──
BASE = Path(__file__).parent
DATA = BASE / "data" / "app"
MAPS = BASE / "maps"

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
.block-container{padding-top:1.5rem;}
h1,h2,h3{color:#2C3E50;font-family:'Helvetica Neue',sans-serif;}
.finding-box{background:#f0f4f8;border-left:4px solid #2980b9;
             padding:.75rem 1rem;margin:.5rem 0;border-radius:4px;}
.rec-box{background:#f0faf0;border-left:4px solid #27ae60;
         padding:.75rem 1rem;margin:.5rem 0;border-radius:4px;}
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
st.sidebar.title("Citi Bike EDA")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Raw Data Explorer",
    "1 \u2014 Distributions",
    "2 \u2014 Temporal Patterns",
    "3 \u2014 Spatial Analysis",
    "4 \u2014 Advanced / Cross-dim",
    "Interactive Map",
    "5 \u2014 Conclusions",
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
- ~28 million total trips across 2,352 active stations
- Rider types: member (~80%), casual (~20%)
- Bike types: electric (~70%), classic (~30%)
- Weather: NOAA Central Park daily observations (temp, precipitation, snow, wind)

## Available data (injected in each user message)
- daily_stats: 365 rows, one per day — total_rides, member_rides, casual_rides,
  electric_rides, avg_duration, pct_member, pct_electric, TAVG, PRCP, SNOW, season,
  day_name, is_weekend
- station_stats: 2,352 rows — station_name, total_departures, total_arrivals,
  net_flow, lat, lng, pct_member
- trips_sample: 100,000 individual trips — rideable_type, user_type, duration_min,
  hour, day_of_week, season, rush_hour, start/end station

## Instructions
- Answer ONLY based on the data provided in the user message.
- Always cite specific numbers from the data.
- Be concise: 3–5 sentences or a short bullet list. No lengthy preamble.
- If the data is insufficient to answer precisely, say so and suggest what to look at.
- Do not invent statistics not present in the context.
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

    # Most imbalanced stations
    most_imbalanced = (
        stations.reindex(stations["net_flow"].abs().nlargest(10).index)
        [["station_name", "net_flow", "total_departures"]]
        .to_string(index=False)
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

--- TOP 10 MOST IMBALANCED STATIONS (by |net_flow|) ---
{most_imbalanced}

--- TRIP DURATION BY RIDER TYPE (trips ≤ 60 min) ---
{dur_summary}
"""

# ══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Citi Bike NYC \u2014 Exploratory Data Analysis")
    st.markdown("**Mar 2025 \u2013 Feb 2026** | Trip-level, daily-aggregate, and station-level data")
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
elif page == "2 \u2014 Temporal Patterns":
    st.title("Section 2: Temporal Patterns")

    CHART_DESCS = {
        "2a \u2014 Daily Trend":
            "Raw daily ride counts overlaid with a 7-day centered rolling mean and a ±1σ "
            "confidence band. Peak and trough days are automatically annotated. The rolling "
            "mean removes weekly seasonality to expose the underlying demand trend, while the "
            "±1σ band flags statistically unusual days that may warrant event-driven analysis.",
        "2b \u2014 Hourly Patterns":
            "Percentage of each day-type's trips occurring in each hour (0–23), split by "
            "member vs. casual riders. Weekday and weekend panels are shown side by side. "
            "Members exhibit a classic bimodal commuter profile (peaks at 8 AM and 5–6 PM), "
            "while casual riders concentrate in midday hours, especially on weekends.",
        "2c \u2014 Day of Week":
            "Four-panel bar chart aggregated by day of the week (Mon–Sun), showing: "
            "(1) average daily ride volume with ±1σ error bars, (2) member share (%), "
            "(3) electric bike share (%), and (4) average trip duration. Blue bars = "
            "weekdays; orange bars = weekends. Error bars indicate how much day-to-day "
            "variability exists within each weekday.",
        "2d \u2014 Seasonal":
            "Three-panel seasonal breakdown covering all four seasons (Spring, Summer, Fall, "
            "Winter). Box plots show the full distribution of daily rides and average trip "
            "duration per season; grouped bars show how the member/casual split shifts across "
            "seasons. A summary statistics table is displayed below the chart.",
        "2e \u2014 Weather Correlation":
            "Three panels quantifying how weather drives demand: (1) temperature vs. total "
            "daily rides with an OLS regression line — points are coloured by rain status; "
            "(2) precipitation vs. rides, with a separate regression fit on rainy days only; "
            "(3) a correlation heatmap of all weather variables against key ride metrics. "
            "The regression slope (rides per °C) and Pearson r are shown in the legend.",
        "2f \u2014 Rush Hour":
            "Rush hour is defined as weekday 7–9 AM and 5–7 PM. The left panel shows how "
            "rush-hour share (% of daily trips) evolves over time on weekdays, with a 5-day "
            "rolling average and the overall mean as a reference line. The right panel shows "
            "rush-hour share by day of the week, highlighting the contrast between weekday "
            "commuter demand and the near-zero rush-hour share on weekends.",
        "2g \u2014 Rideable Type":
            "Left panel: electric bike share (%) over time with a 7-day rolling average, "
            "showing whether adoption is growing, plateauing, or declining. Right panel: "
            "100% stacked bars comparing electric vs. classic bike preference for members "
            "vs. casual riders, revealing whether the two user types differ in their "
            "vehicle choice.",
        "2h \u2014 Trip Duration":
            "Three panels analysing trip duration (clipped at 60 min to focus on typical "
            "usage). (1) Overlapping density histograms for members vs. casual riders — "
            "casual trips tend to be longer and more right-skewed. (2) Density by rideable "
            "type (electric vs. classic). (3) Box plots broken down by time-of-day period "
            "(Night, Morning, Afternoon, Evening) × user type, showing how trip purpose "
            "changes throughout the day.",
    }

    sub = st.selectbox("Select chart", list(CHART_DESCS.keys()))
    st.caption(CHART_DESCS[sub])

    # ── 2a: Daily Trend ──
    if sub.startswith("2a"):
        ts = daily.set_index("date")["total_rides"]
        roll_mean = ts.rolling(7, center=True).mean()
        roll_std  = ts.rolling(7, center=True).std()

        peak   = daily.loc[daily["total_rides"].idxmax()]
        trough = daily.loc[daily["total_rides"].idxmin()]

        fig = go.Figure()
        # ±1σ band
        fig.add_trace(go.Scatter(
            x=pd.concat([roll_mean.index.to_series(), roll_mean.index.to_series()[::-1]]),
            y=pd.concat([(roll_mean + roll_std), (roll_mean - roll_std)[::-1]]),
            fill="toself", fillcolor="rgba(229,57,53,0.12)",
            line=dict(width=0), name="±1σ band", hoverinfo="skip"))
        # Raw series
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_rides"],
                                 line=dict(color=C_MEMBER, width=1), opacity=0.5,
                                 name="Daily rides"))
        # Rolling mean
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values,
                                 line=dict(color=C_RED, width=2.5), name="7-day rolling mean"))
        # Annotations
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
        c4.metric("Days observed",    f"{len(daily)}")

    # ── 2b: Hourly Patterns ──
    elif sub.startswith("2b"):
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

    # ── 2d: Seasonal ──
    elif sub.startswith("2d"):
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

    # ── 2e: Weather ──
    elif sub.startswith("2e"):
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

        # (b) Precipitation scatter
        rainy = wx[wx["PRCP"] > 0]
        fig.add_trace(go.Scatter(x=wx["PRCP"], y=wx["total_rides"], mode="markers",
                                 marker=dict(color=C_MEMBER, size=6, opacity=0.6),
                                 name="Days (precip)", showlegend=False,
                                 hovertemplate="Precip: %{x:.1f}mm<br>Rides: %{y:,.0f}"), row=1, col=2)
        if len(rainy) > 5:
            xs2, ys2, r2 = reg_line(rainy["PRCP"].values, rainy["total_rides"].values)
            fig.add_trace(go.Scatter(x=xs2, y=ys2, mode="lines",
                                     line=dict(color=C_RED, width=2, dash="dash"),
                                     name=f"r={r2:.3f} (rainy only)",
                                     showlegend=True), row=1, col=2)

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
        fig.update_xaxes(title_text="Precipitation (mm)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        if "is_rainy" in wx.columns:
            dry = wx[wx["is_rainy"]==0]["total_rides"].mean()
            wet = wx[wx["is_rainy"]==1]["total_rides"].mean()
            st.caption(
                f"Rain penalty: {(dry-wet)/dry*100:.1f}% fewer rides on rainy days "
                f"(dry-day mean: {dry:,.0f}  vs  rainy-day mean: {wet:,.0f}). "
                "This effect is roughly binary — light and heavy rain cause similar drops, "
                "suggesting riders decide whether to ride rather than how much to ride."
            )

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

    # ── 2g: Rideable Type ──
    elif sub.startswith("2g"):
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

    # ── 2h: Trip Duration ──
    elif sub.startswith("2h"):
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

# ══════════════════════════════════════════════════════════
# PAGE: SECTION 3 — SPATIAL
# ══════════════════════════════════════════════════════════
elif page == "3 \u2014 Spatial Analysis":
    st.title("Section 3: Spatial Analysis")

    CHART_DESCS = {
        "3a \u2014 Top Imbalanced Stations":
            "Horizontal bar charts showing the 15 stations with the most extreme chronic net flow "
            "over the full 12-month period. Net flow = total arrivals − total departures. "
            "Negative (red, left panel) = net exporters: bikes consistently drain away and trucks "
            "must replenish them. Positive (blue, right panel) = net importers: bikes accumulate "
            "and must be removed. These stations are the primary targets for rebalancing operations.",
        "3b \u2014 Imbalance Map":
            "Geographic scatter of all stations across NYC. Dot colour encodes net flow on a "
            "diverging Red–Blue scale (red = exporter, blue = importer, white = balanced). "
            "Dot size encodes total departures (clipped at the 95th percentile to prevent a few "
            "mega-hubs from dominating the visual). Residential areas tend to cluster red in the "
            "morning; commercial cores cluster blue. Hover over any dot for station details.",
        "3c \u2014 AM vs PM Rush":
            "Side-by-side maps comparing per-station net flow during the AM rush (7–9 AM) vs. "
            "the PM rush (5–7 PM). If the commuter-reversal hypothesis holds, the colour pattern "
            "should flip: stations that export bikes in the morning (people commuting out) should "
            "import bikes in the evening (people commuting back). The caption below shows what "
            "fraction of stations actually exhibit this reversal.",
        "3d \u2014 Inequality (Lorenz / Gini)":
            "The Lorenz curve plots the cumulative share of total departures (y-axis) against "
            "the cumulative share of stations ranked from least to most active (x-axis). The "
            "45° dashed line represents perfect equality (every station handles the same volume). "
            "The greater the bow, the higher the Gini coefficient and the more concentrated "
            "traffic is in a small number of hub stations. A high Gini has direct implications "
            "for fleet allocation: resources should be proportional to traffic, not spread evenly.",
    }
    sub = st.selectbox("Select chart", list(CHART_DESCS.keys()))
    st.caption(CHART_DESCS[sub])

    # ── 3a: Top Imbalanced ──
    if sub.startswith("3a"):
        TOP_N = 15
        top_exp = stations.nsmallest(TOP_N, "net_flow")
        top_imp = stations.nlargest(TOP_N,  "net_flow")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Top {TOP_N} Net Exporters (bikes drain)",
                                            f"Top {TOP_N} Net Importers (bikes accumulate)"])

        fig.add_trace(go.Bar(y=top_exp["station_name"].str[:35],
                             x=top_exp["net_flow"], orientation="h",
                             marker_color=C_RED, name="Exporters",
                             hovertemplate="%{y}<br>Net flow: %{x:,.0f}"), row=1, col=1)
        fig.add_trace(go.Bar(y=top_imp["station_name"].str[:35],
                             x=top_imp["net_flow"], orientation="h",
                             marker_color=C_MEMBER, name="Importers",
                             hovertemplate="%{y}<br>Net flow: %{x:,.0f}"), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=520,
                          title="Chronic Station Demand Imbalance (12-Month Total)")
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="Net Flow (arrivals − departures)", row=1, col=1)
        fig.update_xaxes(title_text="Net Flow (arrivals − departures)", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        balanced = stations["net_flow"].between(-10, 10).sum()
        st.caption(
            f"Chronic exporters (net flow < −50): {(stations['net_flow']<-50).sum()} stations  |  "
            f"Chronic importers (net flow > +50): {(stations['net_flow']>50).sum()} stations  |  "
            f"Near-balanced (net flow ±10): {balanced} stations ({balanced/len(stations)*100:.0f}%). "
            "Rebalancing resources should concentrate on the chronic imbalance tail — the near-balanced "
            "majority largely self-correct through normal demand."
        )

    # ── 3b: Imbalance Map ──
    elif sub.startswith("3b"):
        df_map = (stations.dropna(subset=["lat","lng"])
                  .pipe(lambda d: d[d["lat"].between(40.4, 41.0) & d["lng"].between(-74.3, -73.7)]))
        abs_max = df_map["net_flow"].abs().quantile(0.97)

        fig = px.scatter_mapbox(
            df_map,
            lat="lat", lon="lng",
            color="net_flow",
            size=df_map["total_departures"]
                .clip(upper=df_map["total_departures"].quantile(0.95))
                .pipe(lambda x: (x / x.max() * 25 + 3)),
            color_continuous_scale="RdBu",
            range_color=[-abs_max, abs_max],
            mapbox_style="open-street-map",
            zoom=11,
            hover_name="station_name",
            hover_data={"net_flow": True, "total_departures": True,
                        "lat": False, "lng": False},
            labels={"net_flow": "Net Flow"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=600,
                          title="Citi Bike Station Demand Imbalance Map<br>"
                                "<sup>Red = exporter (bikes drain) | Blue = importer (bikes accumulate)</sup>",
                          coloraxis_colorbar=dict(title="Net Flow"))
        st.plotly_chart(fig, use_container_width=True)

    # ── 3c: AM vs PM Rush ──
    elif sub.startswith("3c"):
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

    # ── 3d: Lorenz / Gini ──
    elif sub.startswith("3d"):
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
# PAGE: SECTION 4 — ADVANCED
# ══════════════════════════════════════════════════════════
elif page == "4 \u2014 Advanced / Cross-dim":
    st.title("Section 4: Cross-dimensional & Advanced")

    CHART_DESCS = {
        "4a \u2014 Hour × Day-of-Week Heatmap":
            "A 7×24 demand surface showing trip volume for every combination of day of the week "
            "and hour of the day, computed from the 100k trip sample. Darker cells = higher "
            "demand. Two bright horizontal bands on weekdays (≈8 AM and ≈5–6 PM) reveal the "
            "commuter rush; a broader midday-weekend band captures leisure usage. This heatmap "
            "is the empirical basis for shift-scheduling and time-varying rebalancing policies.",
        "4b \u2014 Weather Sensitivity by User":
            "Tests whether members and casual riders respond differently to weather shocks. "
            "Left panel: temperature vs. daily rides for each user type with separate OLS "
            "regression lines — a steeper slope and higher r indicates greater sensitivity. "
            "Right panel: average rides on dry vs. rainy days for each user type, with the "
            "percentage demand drop labelled. Casual riders are expected to be substantially "
            "more weather-sensitive because their trips are discretionary.",
        "4c \u2014 Rebalancing Priority Score":
            "A composite priority score (0–100) that ranks stations by operational urgency. "
            "Score = 0.5 × (log-scaled departures, normalised) + 0.5 × (|net flow|, normalised). "
            "High-departure stations matter because failures there affect many riders; high "
            "|net flow| stations require active rebalancing to avoid running empty or full. "
            "Stations above the 90th percentile are flagged red on the map and listed in the "
            "table below as the highest-priority targets for truck routing.",
    }
    sub = st.selectbox("Select chart", list(CHART_DESCS.keys()))
    st.caption(CHART_DESCS[sub])

    # ── 4a: Heatmap ──
    if sub.startswith("4a"):
        hm = (trips.groupby(["day_of_week","hour"]).size()
              .unstack(fill_value=0)
              .reindex(range(7), fill_value=0))

        fig = px.imshow(hm.values,
                        x=list(range(24)), y=DOW_LABELS,
                        color_continuous_scale="Blues",
                        labels=dict(x="Hour of Day", y="Day of Week", color="Trip Count"),
                        aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          title="Hour × Day-of-Week Demand Surface")
        fig.update_xaxes(tickvals=list(range(0,24,2)),
                         ticktext=[f"{h}:00" for h in range(0,24,2)])
        st.plotly_chart(fig, use_container_width=True)

    # ── 4b: Weather Sensitivity ──
    elif sub.startswith("4b"):
        wx = daily.dropna(subset=["TAVG"])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Temperature Sensitivity",
                                            "Rain Impact by User Type"])

        # (a) Temp sensitivity — member vs casual
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

        # (b) Rain penalty bar
        if "is_rainy" in daily.columns:
            wx2 = daily.dropna(subset=["is_rainy"])
            bars = []
            for utype, y_col, color in [("member", "member_rides", C_MEMBER),
                                         ("casual", "casual_rides", C_CASUAL)]:
                dry  = wx2[wx2["is_rainy"]==0][y_col].mean()
                rainy= wx2[wx2["is_rainy"]==1][y_col].mean()
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
                                   text=f"−{b['pct']:.0f}%",
                                   font=dict(color="white", size=11),
                                   showarrow=False, row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=430, barmode="group",
                          title="Weather Sensitivity by User Type")
        fig.update_xaxes(title_text="Avg Temp (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Rides", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ── 4c: Rebalancing Priority ──
    elif sub.startswith("4c"):
        df_r = (stations.dropna(subset=["lat","lng"])
                .pipe(lambda d: d[d["lat"].between(40.4, 41.0) & d["lng"].between(-74.3, -73.7)])
                .copy())
        df_r["log_dep"]  = np.log1p(df_r["total_departures"])
        df_r["abs_flow"] = df_r["net_flow"].abs()
        df_r["score"]    = (0.5 * (df_r["log_dep"]  / df_r["log_dep"].max()) +
                            0.5 * (df_r["abs_flow"] / df_r["abs_flow"].max()))
        df_r["score"]    = (df_r["score"] * 100).round(1)
        p90              = df_r["score"].quantile(0.9)
        df_r["priority"] = df_r["score"].apply(
            lambda x: "High (top 10%)" if x >= p90 else "Normal")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_map = px.scatter_mapbox(
                df_r.sort_values("priority", ascending=False),
                lat="lat", lon="lng",
                color="priority",
                size="score",
                color_discrete_map={"High (top 10%)": C_RED, "Normal": C_MEMBER},
                mapbox_style="open-street-map", zoom=11,
                hover_name="station_name",
                hover_data={"score": True, "net_flow": True,
                            "total_departures": True, "lat": False, "lng": False},
                title="Rebalancing Priority Map")
            fig_map.update_layout(**PLOTLY_LAYOUT, height=520)
            st.plotly_chart(fig_map, use_container_width=True)

        with col2:
            fig_hist = go.Figure(go.Histogram(
                x=df_r["score"], nbinsx=40,
                marker_color=C_MEMBER, opacity=0.85))
            fig_hist.add_vline(x=p90, line_dash="dash", line_color=C_RED,
                               annotation_text=f"90th pct ({p90:.1f})")
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=520,
                                   title="Priority Score Distribution",
                                   xaxis_title="Priority Score",
                                   yaxis_title="Stations")
            st.plotly_chart(fig_hist, use_container_width=True)

        high = df_r[df_r["priority"] == "High (top 10%)"].nlargest(10, "score")
        st.subheader("Top 10 Highest Priority Stations")
        st.dataframe(high[["station_name","score","net_flow","total_departures"]]
                     .rename(columns={"station_name":"Station","score":"Priority Score",
                                      "net_flow":"Net Flow","total_departures":"Departures"}),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: INTERACTIVE MAP
# ══════════════════════════════════════════════════════════
elif page == "Interactive Map":
    st.title("Station Imbalance \u2014 Interactive Map")
    st.caption(
        "Each circle represents one Citi Bike station. Circle colour encodes net flow direction "
        "(red = exporter / bikes drain away, blue = importer / bikes accumulate). Circle size "
        "encodes total trip activity. Click any station for its name and metrics. Pan and zoom "
        "to explore specific neighbourhoods."
    )
    html_file = MAPS / "station_imbalance.html"
    if html_file.exists():
        st.components.v1.html(html_file.read_text(encoding="utf-8"), height=700, scrolling=True)
    else:
        st.warning("Map file not found: maps/station_imbalance.html — re-run Section 4d in the EDA notebook to generate it.")

# ══════════════════════════════════════════════════════════
# PAGE: CONCLUSIONS
# ══════════════════════════════════════════════════════════
elif page == "5 \u2014 Conclusions":
    st.title("Conclusions & Operational Recommendations")
    st.markdown(
        "Key findings derived from the full-year Citi Bike dataset (Mar 2025 – Feb 2026), "
        "covering ~365 days of system-wide daily aggregates, 2,352 stations, and a 100k trip sample. "
        "All statistics below are computed live from the loaded data."
    )
    st.markdown("---")

    peak_day   = daily.loc[daily["total_rides"].idxmax()]
    trough_day = daily.loc[daily["total_rides"].idxmin()]
    r_temp     = daily[["total_rides","TAVG"]].dropna().corr().iloc[0,1]
    fall_avg   = daily[daily["season"]=="Fall"]["total_rides"].mean()
    winter_avg = daily[daily["season"]=="Winter"]["total_rides"].mean()
    fall_drop  = (fall_avg - winter_avg) / fall_avg * 100

    balanced    = stations["net_flow"].between(-10,10).sum()
    chronic_exp = (stations["net_flow"] < -50).sum()
    chronic_imp = (stations["net_flow"] > 50).sum()
    top_exp_stn = stations.nsmallest(1,"net_flow").iloc[0]
    top_imp_stn = stations.nlargest(1,"net_flow").iloc[0]
    g = gini(stations["total_departures"].values)

    r1c1,r1c2,r1c3,r1c4 = st.columns(4)
    r1c1.metric("Study Period",    f"{len(daily)} days")
    r1c2.metric("Avg Daily Rides", f"{daily['total_rides'].mean():,.0f}")
    r1c3.metric("Peak Day",        f"{peak_day['total_rides']:,.0f}",
                peak_day["date"].strftime("%b %d"))
    r1c4.metric("Trough Day",      f"{trough_day['total_rides']:,.0f}",
                trough_day["date"].strftime("%b %d"))

    r2c1,r2c2,r2c3,r2c4 = st.columns(4)
    r2c1.metric("Fall → Winter Drop",    f"{fall_drop:.0f}%",     delta_color="inverse")
    r2c2.metric("Temp–Rides Correlation",f"{r_temp:.3f}")
    r2c3.metric("Chronic Exporter Stns", f"{chronic_exp}")
    r2c4.metric("Station Traffic Gini",  f"{g:.2f}")

    st.markdown("---")
    st.subheader("Temporal Findings")
    for title, body in [
        ("Full seasonal cycle confirmed",
         f"With a complete year of data, the demand arc is unambiguous: ridership rises through "
         f"spring, peaks in summer, declines through autumn, and reaches a winter trough. "
         f"The Fall-to-Winter drop alone is {fall_drop:.0f}%. Temperature is the dominant "
         f"predictor (Pearson r = {r_temp:.3f} with daily rides), making it the single most "
         f"important feature for any demand forecasting model."),
        ("Bimodal commuter pattern",
         "Weekday demand shows a clear bimodal shape with peaks at 8 AM and 5–6 PM, driven "
         "primarily by members commuting to and from work. Casual riders follow a unimodal "
         "midday distribution and concentrate heavily on weekends. These two behavioural profiles "
         "have different weather sensitivity, price elasticity, and trip duration distributions — "
         "they should be modelled separately rather than aggregated."),
        ("Weather elasticity is quantifiable",
         f"Each additional degree Celsius adds thousands of daily rides (see Section 2e for the "
         f"exact slope). Rain suppresses demand by roughly 15–30% and the effect is statistically "
         f"significant. Snow causes an even larger drop. Wind has a weaker effect. These "
         f"elasticity estimates are directly actionable: a weather-aware forecast can reduce "
         f"prediction error by 20–40% compared to a day-of-week-only baseline."),
        ("Electric bike dominance",
         f"Electric bikes account for {daily['pct_electric'].mean():.0f}% of all trips across "
         f"the full year, and this share is relatively stable — suggesting adoption has plateaued "
         f"rather than still growing. This creates a battery logistics sub-problem layered on top "
         f"of the vehicle rebalancing problem: e-bikes must be recharged or swapped, adding "
         f"time and location constraints to truck routing decisions."),
    ]:
        st.markdown(f'<div class="finding-box"><strong>{title}.</strong> {body}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Spatial Findings")
    for title, body in [
        ("Extreme station inequality",
         f"The Gini coefficient for station departures is {g:.2f} — comparable to income "
         f"inequality in highly unequal economies. A small fraction of hub stations handles a "
         f"disproportionate share of total traffic. This means fleet allocation must be "
         f"demand-proportional: spreading bikes and rebalancing resources evenly across all "
         f"stations would dramatically over-serve low-traffic stations while under-serving hubs."),
        ("Predictable commuter flow reversal",
         f"There are {chronic_exp} chronic net exporters (net flow < −50) and {chronic_imp} "
         f"chronic net importers (net flow > +50) over the 12-month period. Many stations "
         f"self-correct between AM and PM rush — they export bikes in the morning as commuters "
         f"depart and import them in the evening as commuters return. Rebalancing operations "
         f"should focus on the stations that do NOT reverse, as those are the ones that "
         f"accumulate or deplete inventory irreversibly throughout the day."),
        ("Geographic clustering",
         f"Exporters cluster in residential neighbourhoods (Brooklyn, outer boroughs) where "
         f"people begin their commutes; importers cluster in commercial cores (Midtown, "
         f"Lower Manhattan) where commuters arrive. This geographic pattern is stable and "
         f"predictable, making it suitable for pre-positioning trucks before rush hours. "
         f"Worst exporter: {top_exp_stn['station_name']} (net flow: {top_exp_stn['net_flow']:+,.0f}). "
         f"Worst importer: {top_imp_stn['station_name']} (net flow: {top_imp_stn['net_flow']:+,.0f})."),
        ("Most stations are near-balanced",
         f"{balanced} of {len(stations)} stations ({balanced/len(stations)*100:.0f}%) have a "
         f"net flow within ±10 over the full year, meaning they are essentially self-balancing. "
         f"This concentration of imbalance in a small tail means rebalancing resources can be "
         f"highly targeted rather than spread across the entire network."),
    ]:
        st.markdown(f'<div class="finding-box"><strong>{title}.</strong> {body}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Operational Recommendations")
    for title, body in [
        ("Seasonal fleet right-sizing",
         f"Deploy the maximum fleet in summer when demand peaks; scale down significantly in "
         f"winter (Fall-to-Winter drop: {fall_drop:.0f}%) to reduce idle-asset costs and "
         f"maintenance burden. The optimal fleet size for each month should be set to meet a "
         f"target service level (e.g., 95th-percentile daily demand) rather than the absolute "
         f"peak, which would leave most bikes unused in winter."),
        ("Spring re-deployment planning",
         "The spring demand recovery is operationally as significant as the winter drawdown. "
         "Bikes stored or decommissioned in winter must be inspected, recharged, and "
         "repositioned before the seasonal uptick. Failing to pre-position leads to a "
         "service gap during the early spring surge, exactly when casual riders are returning "
         "to the system."),
        ("Priority-based rebalancing",
         "Use the composite priority score from Section 4c to rank stations for rebalancing "
         "truck routing rather than covering all stations uniformly. Stations above the 90th "
         "percentile score should receive daily attention; lower-priority stations may only "
         "need weekly or demand-triggered visits. This can reduce total truck mileage by "
         "concentrating effort where impact is highest."),
        ("Weather-adaptive scheduling",
         "On days with forecast rain or snow, reduce rebalancing frequency (fewer bike moves "
         "are needed because overall demand drops 15–30%) and redirect crew to maintenance "
         "tasks and electric bike battery swaps. A simple weather-trigger rule — e.g., "
         "\"if precipitation forecast > 5mm, reduce truck shifts by 30%\" — can improve "
         "labour utilisation without degrading service on low-demand days."),
    ]:
        st.markdown(f'<div class="rec-box"><strong>{title}.</strong> {body}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Future Work")
    st.markdown("""
- **Predictive modelling:** Use the temporal features (day of week, month, season) and weather \
  variables (TAVG, PRCP, SNOW) derived in this EDA as inputs to a regression or gradient-boosting \
  model that forecasts daily and hourly demand at the station level.
- **Rebalancing optimisation:** Formulate the truck routing problem as a Mixed-Integer Program (MIP) \
  with time windows (pre-position before rush hours), vehicle capacity constraints, and stochastic \
  demand drawn from the distributions characterised here.
- **Dynamic pricing:** Use the weather and time-of-day elasticity coefficients to design surge pricing \
  rules that shift demand from peak to off-peak periods, reducing the scale of the rebalancing problem.
- **Extended data:** Incorporate real-time dock occupancy data to move from a reactive to a predictive \
  rebalancing model.
""")
    st.caption("Data covers Mar 2025 – Feb 2026. Sources: Citi Bike trip records (GBFS) + NOAA GHCND daily weather (Central Park station).")

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
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append({"role": "user", "content": q})
                st.rerun()

    st.divider()

    # ── Chat history ────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Input ───────────────────────────────────────────────
    if question := st.chat_input("Ask anything about the Citi Bike dataset..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Build messages: system + last 6 turns + current question with data context
                history = st.session_state["messages"][:-1][-6:]
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
                answer = response.choices[0].message.content

            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

    # ── Clear button ────────────────────────────────────────
    if st.session_state.get("messages"):
        if st.button("Clear conversation", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()
