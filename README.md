# Citi Bike NYC — Data Engineering & Exploratory Analysis

A full-year (Mar 2025 – Feb 2026) analysis of NYC Citi Bike ridership, covering data acquisition, cleaning, EDA, and an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset Summary](#dataset-summary)
4. [Project Structure](#project-structure)
5. [Setup & Usage](#setup--usage)
6. [EDA Coverage](#eda-coverage)
7. [Dashboard Pages](#dashboard-pages)
8. [Further Analysis Directions](#further-analysis-directions)
9. [Data Sources](#data-sources)

---

## Project Overview

**Goal:** Analyze how Citi Bike ridership varies by time, weather, geography, and rider segment — and lay a clean foundation for downstream ML forecasting and rebalancing optimization.

**What this project demonstrates:**
- End-to-end data pipeline from raw CSVs to analytics-ready tables
- Cross-dimensional EDA across time, space, weather, and user segments
- Interactive dashboard with 18 Plotly charts across 8 pages

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Folium · Plotly · Streamlit · OpenAI API · NOAA CDO API · Jupyter

---

## Architecture

```
Raw CSVs (Citi Bike S3, one folder per month)
        │
        ▼
build_app_data.py       ← clean, feature-engineer, aggregate
        │
        ├── data/app/daily_rides_weather.csv   (365 rows)
        ├── data/app/station_summary.csv       (2,352 stations)
        └── data/app/trips_sample.csv          (100k trips)
                │
                ▼
            app.py (Streamlit)
```

`data/app/` CSVs are committed to the repo — the dashboard runs immediately after cloning, no raw data needed.

---

## Dataset Summary

| File | Rows | Columns | Description |
|---|---|---|---|
| `trips_sample.csv` | 100,000 | 24 | Random sample of all trips, with derived features |
| `daily_rides_weather.csv` | 365 | 29 | Daily ridership aggregates merged with NOAA weather |
| `station_summary.csv` | 2,352 | 12 | Per-station totals, net flow, coordinates |

**Key figures:**
- ~28M total rides across the 12-month window
- 2,352 active stations across NYC boroughs
- Temperature range: −11.9 °C to 32.2 °C
- 12 snow days, 90 rain days
- 70%+ of trips on electric bikes
- Monthly demand: ~1.2M (Feb 2026, winter) → ~5.3M (Sep 2025, summer peak)

---

## Project Structure

```
citi_bike/
├── app.py                          # Streamlit dashboard (9 pages, 18+ charts + LLM agent)
├── build_app_data.py               # Reproduce data/app/*.csv from raw CSVs
├── requirements.txt
├── README.md
│
├── .streamlit/
│   ├── config.toml                 # Theme config (committed)
│   └── secrets.toml                # API keys — gitignored, create manually
│
├── files/
│   ├── 01_data_acquisition.ipynb   # Data pipeline notebook
│   └── 02_eda.ipynb                # Full EDA notebook (Sections 1–5)
│
├── data/
│   └── app/                        # Pre-built CSVs (committed to repo)
│       ├── daily_rides_weather.csv
│       ├── station_summary.csv
│       └── trips_sample.csv
│
├── figures/                        # Static PNG exports from EDA
└── maps/                           # Folium HTML interactive maps
```

> Raw monthly CSVs (`data/YYYYMM-citibike-*/`) are gitignored. `data/app/` is the only data committed. `.streamlit/secrets.toml` is gitignored and must be created manually.

---

## Setup & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/skyexry/citi-bike.git
cd citi-bike
pip install -r requirements.txt
```

### 2. Configure secrets (required for the LLM agent)

The **💬 Ask the Data** page calls the OpenAI API. Create the secrets file:

```bash
mkdir -p .streamlit
```

Then create `.streamlit/secrets.toml` with your key:

```toml
OPENAI_API_KEY = "sk-..."   # get one at platform.openai.com/api-keys
```

> This file is gitignored and will never be committed. The dashboard runs without it — only the Ask the Data page will show an error if the key is missing.

### 3. Run the dashboard *(no raw data needed)*

```bash
streamlit run app.py
```

The pre-built CSVs in `data/app/` are committed to the repo — the dashboard works immediately after cloning.

### 4. Reproduce `data/app/` from raw CSVs *(optional)*

Download the raw monthly folders from the [Citi Bike system data page](https://citibikenyc.com/system-data) into `data/`, then run:

```bash
# Without weather data
python build_app_data.py

# With NOAA weather (recommended)
python build_app_data.py --noaa-token YOUR_TOKEN
# Or: export NOAA_TOKEN=YOUR_TOKEN && python build_app_data.py
```

Get a free NOAA token at [ncdc.noaa.gov/cdo-web/token](https://www.ncdc.noaa.gov/cdo-web/token).

Optional flags:
```
--months 2025-03 2025-04   # process specific months only
--sample-size 50000        # change trips_sample size (default: 100000)
--data-dir path/to/data    # custom raw data location
```

### 5. Explore the EDA notebook

```bash
jupyter lab files/02_eda.ipynb
```

---

## EDA Coverage

### Section 1 — Data Overview & Quality Assessment
- Missing value check and data completeness summary
- Key summary statistics (mean, median, min, max, std per metric)
- Rider & bike type composition: member/casual and electric/classic splits (pie charts)
- Distribution overview: daily rides, avg duration, temperature, station departures, net flow, trip duration — with mean/median reference lines
- Trip duration by rider type: violin + box plot comparison (member vs. casual)

### Section 2 — Temporal Patterns
- Daily ridership time series with 7-day rolling average
- Monthly totals bar chart (full seasonal arc)
- Hourly demand profile by weekday vs. weekend
- Day-of-week average ridership
- Heatmap: hour of day × day of week
- AM vs. PM rush hour comparison
- Holiday vs. regular-day ridership

### Section 3 — Spatial Patterns
- Top-20 departure stations (bar chart)
- Station net flow map (arrivals minus departures, Folium choropleth)
- AM vs. PM rush flow direction snapshot
- Station utilization Lorenz curve & Gini coefficient

### Section 4 — Weather Impact
- Daily rides vs. temperature (scatter + OLS regression)
- Rides on rain days vs. clear days (box + strip)
- Daily rides vs. wind speed
- Rides on snow days vs. non-snow days

### Section 5 — Advanced / Cross-Dimensional
- Member/casual ratio across seasons (stacked bar)
- Classic vs. electric bike share across seasons
- Composite station priority score: top-10 stations flagged for rebalancing

---

## Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Dataset KPIs, seasonal ride arc, key findings |
| **Raw Data Explorer** | Filterable trip-level table with download |
| **1 – Distributions** | Rider type, bike type, duration distributions |
| **2 – Temporal** | 8 time-series and pattern charts (selectbox) |
| **3 – Spatial** | Station map, net flow, AM/PM rush, Lorenz curve |
| **4 – Weather Impact** | Weather × ridership scatter and box charts |
| **Interactive Map** | Folium station imbalance map |
| **5 – Conclusions** | Live-computed stats and key takeaways |

---

## Further Analysis Directions

### A. Origin–Destination (OD) Flow Network
Build a directed graph (nodes = stations, edges = trip counts) to identify dominant corridors, betweenness-central hubs, and asymmetric OD pairs as a rebalancing signal.

### B. Round-Trip Detection as Leisure Proxy
Filter `start_station_id == end_station_id` to isolate recreational trips. Analyze by hour, weekday, and season — expected to cluster near parks and the Hudson River on weekend afternoons.

### C. Demand Forecasting
Predict daily ridership 7–14 days ahead using weather forecast features. Compare Ridge → Random Forest → XGBoost → Prophet with weather regressors, evaluated with rolling 30-day holdout.

### D. Rebalancing Optimization
Frame AM rush imbalance as a minimum-cost flow problem: nodes are stations with supply/demand = net flow 06:00–09:00, edges weighted by distance. Solve with `scipy.optimize` or `OR-Tools`.

### E. Weather Elasticity by Rider Segment
Run separate OLS regressions for member vs. casual rides against weather variables. Hypothesis: casual riders show 2–3× higher rain sensitivity.

### F. Station Behavioral Clustering
Engineer per-station features (AM departure share, PM arrival share, weekend ratio, e-bike share) and cluster with K-means or HDBSCAN. Expected types: Commuter Hub / Residential Feeder / Tourist / Mixed-Use.

### G. Anomaly Detection for Unusual Days
Fit a baseline ridership model, flag days with residual > 2σ, and annotate with the NYC event calendar to quantify event-driven demand lift.

### H. Electric Bike Adoption & Battery Logistics
Proxy trip distance via Haversine coordinates. Compare e-bike share across distance bins and temperature ranges to identify priority charging stations.

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Citi Bike System Data](https://citibikenyc.com/system-data) | Monthly trip CSVs (public S3 bucket) | Free, no auth |
| [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/) | Daily weather observations (GHCND) | Free API token |
| [NYC Open Data](https://opendata.cityofnewyork.us/) | Events, borough boundaries, bike lanes | Free, no auth |
