# Citi Bike NYC — Exploratory Data Analysis & Dashboard

A full-year (Mar 2025 – Feb 2026) analysis of NYC Citi Bike ridership across all four seasons, combining trip-level data with daily NOAA weather observations. The project includes a reproducible data pipeline, a comprehensive EDA notebook, and an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Summary](#dataset-summary)
3. [Project Structure](#project-structure)
4. [Setup & Usage](#setup--usage)
5. [Current EDA Coverage](#current-eda-coverage)
6. [Dashboard Pages](#dashboard-pages)
7. [Additional Insights & Further Analysis](#additional-insights--further-analysis)
8. [Data Sources](#data-sources)

---

## Project Overview

**Goal:** Understand how Citi Bike ridership patterns vary by time of day, day of week, season, weather, and geography across a complete calendar year — and surface actionable insights for operations, urban planning, and future modeling.

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · Folium · Plotly · Streamlit · NOAA CDO API · Jupyter

---

## Dataset Summary

| Dataset | Rows | Columns | Description |
|---|---|---|---|
| `trips_sample.csv` | 100,000 | 24 | Random sample of all trips, Mar 2025–Feb 2026 |
| `daily_rides_weather.csv` | 365 | 29 | Daily ridership aggregates merged with NOAA weather |
| `station_summary.csv` | 2,352 | 12 | Per-station totals, net flow, and priority score |

**Key figures:**
- ~28M total rides in the 12-month window (extrapolated from sample)
- 2,352 active stations across NYC boroughs
- Temperature range: −11.9 °C to 32.2 °C
- 12 snow days, 90 rain days
- 70%+ of trips taken on electric bikes
- Monthly demand: ~1.2M (Feb 2026 winter low) → ~5.3M (Sep 2025 summer peak)

---

## Project Structure

```
citi_bike/
├── app.py                        # Streamlit dashboard (8 pages, 18 interactive charts)
├── README.md
├── files/
│   ├── 01_data_acquisition.ipynb # Data download: Citi Bike S3 + NOAA CDO API
│   ├── 02_eda.ipynb              # Full EDA notebook (Sections 1–5)
│   └── requirements.txt          # Python dependencies
├── data/
│   └── processed/
│       ├── trips_sample.csv
│       ├── daily_rides_weather.csv
│       └── station_summary.csv
├── figures/                      # Static PNG exports from the EDA notebook
└── maps/                         # Folium HTML interactive maps
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r files/requirements.txt
```

### 2. Run the data pipeline *(optional — processed CSVs already included)*

Open and run `files/01_data_acquisition.ipynb`. Requires a free [NOAA CDO API token](https://www.ncdc.noaa.gov/cdo-web/token).

### 3. Reproduce the EDA notebook

Open and run `files/02_eda.ipynb` in Jupyter Lab or Google Colab.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## Current EDA Coverage

### Section 1 — Ride-Type & Duration Distributions
- Member vs. casual rider split (share of trips and share of ride time)
- Bike-type preference (classic vs. electric) by rider type
- Trip duration distribution (log scale) with percentile markers
- Member/casual duration comparison (violin + box)

### Section 2 — Temporal Patterns
- Daily ridership time series with 7-day rolling average
- Monthly totals bar chart (full seasonal arc)
- Hourly demand profile by weekday vs. weekend
- Day-of-week average ridership
- Heatmap: hour of day × day of week
- AM vs. PM rush hour comparison (counts and direction)
- Holiday vs. regular-day ridership

### Section 3 — Spatial Patterns
- Top-20 departure stations (bar chart)
- Station net flow map (arrivals minus departures, Folium choropleth)
- AM vs. PM rush flow direction (net flow snapshot by time window)
- Station utilization Lorenz curve & Gini coefficient (inequality metric)

### Section 4 — Weather Impact
- Daily rides vs. temperature (scatter + OLS regression)
- Rides on rain days vs. clear days (box + strip)
- Daily rides vs. wind speed (scatter + OLS)
- Rides on snow days vs. non-snow days

### Section 5 — Advanced / Cross-Dimensional Analysis
- Member/casual ratio across seasons (stacked bar)
- Classic vs. electric bike share across seasons
- Composite station priority score: `0.5 × log(departures) + 0.5 × |net_flow|` (top-10 stations needing attention)

---

## Dashboard Pages

The Streamlit app (`app.py`) mirrors the notebook structure with fully interactive Plotly charts:

| Page | Content |
|---|---|
| **Overview** | Dataset KPIs, seasonal ride arc, key findings summary |
| **Raw Data Explorer** | Filterable trip-level table (trips_sample) |
| **1 – Distributions** | Rider type, bike type, duration distributions |
| **2 – Temporal** | 8 time-series and pattern charts (selectbox) |
| **3 – Spatial** | Station map, net flow, AM/PM rush, Lorenz curve |
| **4 – Weather Impact** | 4 weather × ridership scatter/box charts |
| **Interactive Map** | Folium station map embedded in Streamlit |
| **5 – Conclusions** | Live computed stats + key takeaways |

---

## Additional Insights & Further Analysis

The current dataset supports several analyses beyond the existing EDA. Below are eight directions, each with a concrete question, recommended method, and expected output.

---

### A. Origin–Destination (OD) Flow Network

**Question:** Which station *pairs* generate the most trips? Are there dominant corridors?

**Why it matters:** Identifies high-demand routes for targeted infrastructure (protected lanes, additional docks). The sample contains 76,112 unique OD pairs — rich enough for network analysis.

**Approach:**
- Build a directed graph where nodes = stations, edges = trip counts
- Compute betweenness centrality (stations that act as "transfer hubs")
- Visualize top-50 corridors as arc lines on a map (Plotly `scattermapbox` + Kepler.gl)
- Identify asymmetric flows: pairs where outbound >> inbound (rebalancing signal)

**Libraries:** `networkx`, `plotly`, `keplergl`

---

### B. Round-Trip Detection as a Leisure Proxy

**Question:** What fraction of trips start and end at the same station? When and where do round-trips cluster?

**Why it matters:** Round trips (~2.0% of the sample) signal recreational riding rather than commuting. Parks, waterfronts, and tourist corridors likely dominate.

**Approach:**
- Filter `start_station_id == end_station_id`
- Compare round-trip rates by hour, day-of-week, and season
- Map round-trip hotspot stations
- Cross-reference with NYC parks/waterfront shapefiles

**Expected finding:** Round-trip share peaks on weekend afternoons in summer near Hudson River greenway stations.

---

### C. Demand Forecasting Model

**Question:** Can ridership be predicted 7–14 days ahead from weather forecasts?

**Why it matters:** Accurate forecasts let operations pre-position bikes and staff rebalancing trucks before demand spikes.

**Approach:**
- Feature set: `TMAX`, `TMIN`, `PRCP`, `SNWD`, `AWND`, `day_of_week`, `month`, `is_holiday`
- Models to compare: Ridge regression → Random Forest → XGBoost → Prophet (with weather regressors)
- Evaluate with time-series cross-validation (rolling 30-day holdout)
- Target metric: MAPE on daily ride count

**Data available:** 365-row `daily_rides_weather.csv` is small but sufficient for baseline models; augment with 2022–2024 historical data for deep learning.

---

### D. Station Rebalancing Optimization

**Question:** Which stations should bikes be moved *from* and *to* each morning to minimize midday shortage and overflow?

**Why it matters:** Empty docks frustrate departing riders; full docks frustrate arriving riders. Net flow analysis already identifies the most imbalanced stations.

**Approach:**
- Frame as a minimum-cost flow (MCF) or mixed-integer program (MIP):
  - Nodes: stations with supply/demand = `net_flow` during AM rush (06:00–09:00)
  - Edges: truck routes with capacity and cost (distance-based)
- Solve with `scipy.optimize` (LP relaxation) or `PuLP`/`OR-Tools` (exact MIP)
- Output: a daily rebalancing schedule per truck

**Input data:** AM rush net flow from `trips_sample.csv` + station lat/lon from `station_summary.csv`

---

### E. Weather Elasticity by Rider Segment

**Question:** Are casual riders more sensitive to bad weather than members?

**Why it matters:** Different elasticity means weather-based dynamic pricing could shift casual riders to off-peak hours without deterring committed members.

**Approach:**
- Split `daily_rides_weather.csv` by `member_rides` and `casual_rides`
- Run separate OLS regressions: `rides ~ TMAX + PRCP + SNWD + AWND` per segment
- Compare slope coefficients (rain coefficient for casual vs. member)
- Compute rain-day demand elasticity: `(ΔRides%) / (ΔRain%)`

**Expected finding:** Casual riders show 2–3× higher sensitivity to precipitation than members.

---

### F. Station Behavioral Clustering

**Question:** Can stations be grouped by their usage *pattern* (not just volume)?

**Why it matters:** A station with high AM outflow + high PM inflow is a "residential feeder." A station with midday peak + tourist location is a "visitor hub." Different types need different inventory strategies.

**Approach:**
- Engineer per-station features from `trips_sample.csv`:
  - AM rush departure share, PM rush arrival share
  - Weekend vs. weekday ratio
  - Electric bike share, average trip duration, round-trip rate
- Standardize features → K-means (k=4–6) or HDBSCAN
- Validate clusters against known neighborhoods (Chelsea, Midtown, Brooklyn)
- Label cluster types: Commuter Hub / Residential Feeder / Tourist Destination / Mixed-Use

**Libraries:** `sklearn`, `plotly` (radar chart per cluster)

---

### G. Anomaly Detection for Unusual Days

**Question:** Which days had ridership far above or below what weather and season would predict?

**Why it matters:** Unexplained outliers often correspond to events (parades, concerts, subway outages, bike-share promotions) — useful for event-aware forecasting.

**Approach:**
- Fit a baseline model: `rides ~ TMAX + PRCP + day_of_week + month`
- Flag days where `|actual − predicted| > 2σ`
- Annotate with NYC event calendar (NYC Open Data "NYC Events" dataset)
- Quantify the "event lift" in rides

**Data needed:** NYC Open Data events API (free) + existing `daily_rides_weather.csv`

---

### H. Electric Bike Adoption & Battery Logistics

**Question:** How does electric vs. classic bike demand vary by distance, terrain, and time of day?

**Why it matters:** E-bikes need charging infrastructure. Understanding when/where they are preferred helps plan charging station placement and overnight logistics.

**Approach:**
- Proxy trip distance from station lat/lon (Haversine formula)
- Compare e-bike vs. classic share across: distance bins, elevation change (NYC DEM), hour of day, temperature
- Estimate daily battery demand per station (trips × assumed kWh/trip)
- Identify stations where e-bike share > 80% → priority charging nodes

**Data needed:** NYC elevation DEM (free via USGS) + station lat/lon already in `station_summary.csv`

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| [Citi Bike System Data](https://citibikenyc.com/system-data) | Monthly trip CSVs (S3 public bucket) | Free, no auth |
| [NOAA Climate Data Online (CDO)](https://www.ncdc.noaa.gov/cdo-web/) | Daily weather observations (GHCND) | Free API token |
| [NYC Open Data](https://opendata.cityofnewyork.us/) | Events, borough boundaries, bike lanes | Free, no auth |
| [USGS National Map](https://www.usgs.gov/tools/national-map-downloader) | Digital Elevation Model (for terrain analysis) | Free download |
