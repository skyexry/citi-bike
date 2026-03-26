# Citi Bike NYC — Exploratory Data Analysis

Full-year (Mar 2025 – Feb 2026) analysis of NYC Citi Bike ridership: data pipeline, EDA, and an interactive Streamlit dashboard.

**Stack:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scipy · Plotly · Streamlit · OpenAI API · NOAA CDO API

---

## Architecture

```
Raw CSVs (monthly)
       │
       ▼
build_app_data.py  ← clean, feature-engineer, aggregate
       │
       ├── data/app/daily_rides_weather.csv   (365 rows)
       ├── data/app/station_summary.csv       (2,250 stations)
       └── data/app/trips_sample.csv          (100k trips)
               │
               ▼
           app.py (Streamlit)
```

`data/app/` is committed — the dashboard runs immediately after cloning, no raw data needed.

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `trips_sample.csv` | 100,000 | Random trip sample with derived features |
| `daily_rides_weather.csv` | 365 | Daily ridership aggregates + NOAA weather |
| `station_summary.csv` | 2,250 | Per-station totals, net flow, coordinates |

~28M total rides · 2,250 stations · 70%+ electric · range 1.2M (Feb) → 5.3M (Sep)

---

## Project Structure

```
citi_bike/
├── app.py                        # Streamlit dashboard (9 pages, 20+ charts + LLM agent)
├── build_app_data.py             # Reproduce data/app/*.csv from raw CSVs
├── requirements.txt
├── .streamlit/
│   ├── config.toml               # Theme (committed)
│   └── secrets.toml              # API keys — gitignored, create manually
├── files/
│   ├── 01_data_acquisition.ipynb
│   └── 02_eda.ipynb              # Full EDA (Sections 1–5)
├── data/app/                     # Pre-built CSVs (committed)
├── figures/                      # Static PNG exports
└── maps/                         # Folium HTML maps
```

---

## Setup

```bash
git clone https://github.com/skyexry/citi-bike.git
cd citi-bike
pip install -r requirements.txt
streamlit run app.py
```

**LLM agent (optional):** The **💬 Ask the Data** page requires an OpenAI key:

```toml
# .streamlit/secrets.toml  (gitignored)
OPENAI_API_KEY = "sk-..."
```

**Rebuild `data/app/` from raw CSVs (optional):**

```bash
# Download monthly CSVs from citibikenyc.com/system-data into data/, then:
python build_app_data.py --noaa-token YOUR_TOKEN
```

Get a free NOAA token at [ncdc.noaa.gov/cdo-web/token](https://www.ncdc.noaa.gov/cdo-web/token). Optional flags: `--months`, `--sample-size`, `--data-dir`.

---

## EDA Coverage

| Section | Topics |
|---|---|
| **1 – Overview & Quality** | Missing values, summary stats, rider/bike type composition, duration distributions |
| **2 – Temporal Demand** | Daily time series, hourly profiles, day-of-week, seasonal decomposition, weather × demand, rush-hour, e-bike adoption |
| **3 – Spatial Demand** | Top stations, imbalance ranking, geographic map, AM/PM rush reversal, Lorenz curve & Gini |
| **4 – Advanced** | Hour×day heatmap, weather sensitivity by rider segment, Holt-Winters forecast, K-Means station clustering |
| **5 – Conclusions** | 12 key findings, operational recommendations (fleet sizing, rebalancing, weather-adaptive scheduling) |

---

## Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | KPIs, daily volume, temperature vs. rides |
| **Raw Data Explorer** | Filterable trip table with CSV download |
| **1 – Distributions** | Rider/bike type, duration distributions |
| **2 – Temporal** | 8 time-series and pattern charts |
| **3 – Spatial** | Station rankings, imbalance, map, Lorenz curve |
| **4 – Advanced** | Heatmap, weather sensitivity, forecast, clustering |
| **Interactive Map** | Folium station imbalance map |
| **5 – Conclusions** | Live-computed stats and recommendations |
| **💬 Ask the Data** | LLM-powered Q&A agent (requires OpenAI key) |

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Citi Bike System Data](https://citibikenyc.com/system-data) | Monthly trip CSVs | Free, no auth |
| [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/) | Daily weather (GHCND) | Free API token |
| [NYC Open Data](https://opendata.cityofnewyork.us/) | Borough boundaries, bike lanes | Free, no auth |
