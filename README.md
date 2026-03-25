# Citi Bike NYC — Exploratory Data Analysis

A full-year (Mar 2025 – Feb 2026) analysis of NYC Citi Bike ridership, covering data acquisition,
cleaning, EDA, and an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset Summary](#dataset-summary)
4. [Project Structure](#project-structure)
5. [Setup & Usage](#setup--usage)
6. [EDA Coverage](#eda-coverage)
7. [Dashboard Pages](#dashboard-pages)
8. [Data Sources](#data-sources)

---

## Project Overview

**Goal:** Analyse how Citi Bike ridership varies by time, weather, geography, and rider segment.

**What this project demonstrates:**
- End-to-end data pipeline from raw CSVs to analytics-ready tables
- Five-section EDA spanning distributions, temporal patterns, spatial analysis, and cross-dimensional analysis
- Interactive Streamlit dashboard with Plotly charts and an LLM-powered Q&A agent

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scipy · Plotly · Streamlit · OpenAI API · NOAA CDO API · Jupyter

---

## Architecture

```
Raw CSVs (Citi Bike S3, one folder per month)
        │
        ▼
build_app_data.py       ← clean, feature-engineer, aggregate
        │
        ├── data/app/daily_rides_weather.csv   (365 rows)
        ├── data/app/station_summary.csv       (2,250 stations)
        └── data/app/trips_sample.csv          (100k trips)
                │
                ▼
            app.py (Streamlit)
```

`data/app/` CSVs are committed to the repo — the dashboard runs immediately after cloning,
no raw data needed.

---

## Dataset Summary

| File | Rows | Columns | Description |
|---|---|---|---|
| `trips_sample.csv` | 100,000 | 24 | Random sample of all trips, with derived features |
| `daily_rides_weather.csv` | 365 | 29 | Daily ridership aggregates merged with NOAA weather |
| `station_summary.csv` | 2,250 | 11 | Per-station totals, net flow, coordinates |

**Key figures:**
- ~28M total rides across the 12-month window
- 2,250 active stations across NYC boroughs
- Temperature range: −11.9 °C to 32.2 °C
- 12 snow days, 90 rain days
- 70%+ of trips on electric bikes
- Monthly demand: ~1.2M (Feb 2026, winter) → ~5.3M (Sep 2025, summer peak)

---

## Project Structure

```
citi_bike/
├── app.py                          # Streamlit dashboard (9 pages, 20+ charts + LLM agent)
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

> Raw monthly CSVs (`data/YYYYMM-citibike-*/`) are gitignored. `data/app/` is the only data
> committed. `.streamlit/secrets.toml` is gitignored and must be created manually.

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

> This file is gitignored and will never be committed. The dashboard runs without it —
> only the Ask the Data page will show an error if the key is missing.

### 3. Run the dashboard *(no raw data needed)*

```bash
streamlit run app.py
```

The pre-built CSVs in `data/app/` are committed to the repo — the dashboard works
immediately after cloning.

### 4. Reproduce `data/app/` from raw CSVs *(optional)*

Download the raw monthly folders from the [Citi Bike system data page](https://citibikenyc.com/system-data)
into `data/`, then run:

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
- Distribution overview: daily rides, avg duration, trip duration — with mean/median reference lines
- Trip duration by rider type: violin + box plot comparison (member vs. casual)

### Section 2 — Temporal Demand Analysis
- **2a** Daily ridership time series with 7-day rolling average and anomaly flags
- **2b** Hourly demand profiles: member vs. casual side-by-side
- **2c** Day-of-week patterns and weekday vs. weekend comparison
- **2d** Seasonal demand decomposition (monthly totals, season averages)
- **2e** Weather × demand: temperature scatter, Clear/Rainy/Snowy box plots, wind
- **2f** Rush-hour commuter analysis (AM vs. PM share)
- **2g** Electric vs. classic bike adoption over time
- **2h** Trip duration distributions by rider type and time of day

### Section 3 — Spatial Demand Analysis
- **3a** Top-20 busiest stations by total departures + departure distribution (log scale)
- **3b** Chronic station imbalance: ranked by `imbalance_ratio = |net_flow| / total_flow`
- **3c** Geographic imbalance map: stations coloured by signed imbalance ratio
- **3d** AM vs. PM rush flow reversal — commuter station detection
- **3e** Station utilization inequality: Lorenz curve & Gini coefficient

### Section 4 — Advanced Analysis
- **4a** Hour × day-of-week demand surface (7×24 heatmap)
- **4b** Weather sensitivity by rider segment: temperature slope and rain penalty for member vs. casual
- **4c** Demand forecasting: seasonal decomposition + Holt-Winters triple exponential smoothing, 60-day holdout vs. seasonal naive benchmark
- **4d** Station behavioural clustering: K-Means (k=4) on log volume, member share, e-bike share, imbalance ratio, avg duration

### Section 5 — Conclusions & Operational Recommendations
- Live-computed summary statistics from all three datasets
- 12 key findings across temporal, spatial, and modelling dimensions
- Operational recommendations for fleet sizing, rebalancing prioritisation, and weather-adaptive scheduling

---

## Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Dataset KPIs, daily ride volume, temperature vs rides |
| **Raw Data Explorer** | Filterable trip-level table with download |
| **1 – Distributions** | Rider type, bike type, duration distributions |
| **2 – Temporal** | 8 time-series and pattern charts (selectbox) |
| **3 – Spatial** | Busiest stations, imbalance ranking, geographic map, AM/PM rush, Lorenz curve |
| **4 – Advanced** | Hour×day heatmap, weather sensitivity, Holt-Winters forecast, K-Means clustering |
| **Interactive Map** | Folium station imbalance map |
| **5 – Conclusions** | Live-computed stats and operational recommendations |
| **💬 Ask the Data** | LLM-powered Q&A agent (requires OpenAI API key) |

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Citi Bike System Data](https://citibikenyc.com/system-data) | Monthly trip CSVs (public S3 bucket) | Free, no auth |
| [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/) | Daily weather observations (GHCND) | Free API token |
| [NYC Open Data](https://opendata.cityofnewyork.us/) | Events, borough boundaries, bike lanes | Free, no auth |
