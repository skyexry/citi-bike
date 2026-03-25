#!/usr/bin/env python3
"""
build_app_data.py
~~~~~~~~~~~~~~~~~
Pure-Pandas pipeline: raw Citi Bike CSVs → data/app/*.csv

Outputs (written to data/app/):
    daily_rides_weather.csv   — one row per day, system-wide stats + weather
    station_summary.csv       — one row per station, departures/arrivals/net_flow
    trips_sample.csv          — 100k random trips with derived features

Usage:
    python build_app_data.py                        # no weather
    python build_app_data.py --noaa-token TOKEN     # with weather
    python build_app_data.py --months 2025-03 2025-04
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
APP_DIR  = ROOT / "data" / "app"

NOAA_STATION_ID = "GHCND:USW00094728"  # NYC Central Park

SAMPLE_COLS = [
    "ride_id", "rideable_type",
    "started_at", "ended_at", "duration_min",
    "start_station_id", "start_station_name",
    "end_station_id",   "end_station_name",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "user_type",
    "date", "hour", "day_of_week", "day_name",
    "month", "year", "season", "is_weekend", "rush_hour",
    "source_month",
]


# ──────────────────────────────────────────────────────────────
# 1. Load raw CSVs
# ──────────────────────────────────────────────────────────────

def load_raw_trips(data_dir: Path, months: list[str] | None) -> pd.DataFrame:
    """Read all monthly CSV folders and concatenate into one DataFrame."""
    frames = []
    failed = []

    folders = sorted(data_dir.glob("??????-citibike-*"))
    if months:
        month_keys = {m.replace("-", "") for m in months}
        folders = [f for f in folders if f.name[:6] in month_keys]

    if not folders:
        raise RuntimeError(f"No matching folders found under {data_dir}")

    for folder in folders:
        source_month = f"{folder.name[:4]}-{folder.name[4:6]}"
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            log.warning("No CSVs in %s", folder)
            failed.append(source_month)
            continue

        log.info("Loading %s  (%d file(s))", folder.name, len(csv_files))
        parts = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df_month = pd.concat(parts, ignore_index=True)
        df_month["source_month"] = source_month
        frames.append(df_month)
        log.info("  → %d rows", len(df_month))

    if failed:
        log.warning("Months not loaded: %s", failed)
    if not frames:
        raise RuntimeError("No data loaded.")

    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# 2. Clean & feature-engineer
# ──────────────────────────────────────────────────────────────

def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"]   = pd.to_datetime(df["ended_at"],   errors="coerce")

    n0 = len(df)
    df.dropna(subset=["started_at", "ended_at",
                       "start_station_id", "end_station_id",
                       "start_lat", "start_lng"], inplace=True)
    log.info("Dropped %d rows (null datetime/station/coords)", n0 - len(df))

    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    n1 = len(df)
    df = df[df["duration_min"].between(1, 180)].copy()
    log.info("Dropped %d rows (duration out of 1–180 min)", n1 - len(df))

    # user_type
    if "member_casual" in df.columns:
        df["user_type"] = df["member_casual"]
    elif "usertype" in df.columns:
        df["user_type"] = (
            df["usertype"].str.lower()
            .str.replace("subscriber", "member", regex=False)
            .str.replace("customer",   "casual",  regex=False)
        )
    else:
        df["user_type"] = "unknown"

    # Temporal features
    df["date"]        = df["started_at"].dt.date
    df["hour"]        = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.dayofweek
    df["day_name"]    = df["started_at"].dt.day_name()
    df["month"]       = df["started_at"].dt.month
    df["year"]        = df["started_at"].dt.year
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["season"]      = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Fall",   10: "Fall",  11: "Fall",
    })
    df["rush_hour"] = (
        (df["is_weekend"] == 0) & df["hour"].isin([7, 8, 17, 18])
    ).astype(int)

    log.info("Clean trips: %d rows", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# 3. NOAA weather fetch
# ──────────────────────────────────────────────────────────────

def _fetch_noaa_month(year_month: str, token: str) -> pd.DataFrame:
    import calendar
    year, month = int(year_month[:4]), int(year_month[5:7])
    last_day = calendar.monthrange(year, month)[1]

    headers = {"token": token}
    params  = {
        "datasetid":  "GHCND",
        "stationid":  NOAA_STATION_ID,
        "startdate":  f"{year_month}-01",
        "enddate":    f"{year_month}-{last_day:02d}",
        "datatypeid": "TMAX,TMIN,PRCP,SNOW,AWND",
        "limit":      1000,
        "units":      "metric",
    }
    try:
        r = requests.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/data",
            headers=headers, params=params, timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("  NOAA %s: request failed — %s", year_month, e)
        return pd.DataFrame()

    results = r.json().get("results", [])
    if not results:
        log.warning("  NOAA %s: no data returned", year_month)
        return pd.DataFrame()

    df = (
        pd.DataFrame(results)
        .assign(date=lambda x: pd.to_datetime(x["date"]).dt.date)
        .pivot_table(index="date", columns="datatype",
                     values="value", aggfunc="first")
        .reset_index()
    )
    df.columns.name = None
    return df


def fetch_weather(start_date: str, end_date: str, token: str) -> pd.DataFrame:
    months = [
        str(p) for p in pd.period_range(start_date[:7], end_date[:7], freq="M")
    ]
    log.info("Fetching NOAA weather (%d months)...", len(months))

    chunks = []
    for ym in months:
        chunk = _fetch_noaa_month(ym, token)
        if not chunk.empty:
            chunks.append(chunk)
            log.info("  ✓ %s: %d days", ym, len(chunk))
        time.sleep(0.3)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    if {"TMAX", "TMIN"}.issubset(df.columns):
        df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2

    df["is_rainy"]       = (df.get("PRCP", 0) > 1.0).astype(float)
    df["is_snowy"]       = (df.get("SNOW", 0) > 0.0).astype(float)
    df["is_bad_weather"] = ((df["is_rainy"] == 1) | (df["is_snowy"] == 1)).astype(float)

    return df


# ──────────────────────────────────────────────────────────────
# 4. Aggregations
# ──────────────────────────────────────────────────────────────

def build_daily(df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date")
        .agg(
            total_rides     = ("ride_id",         "count"),
            member_rides    = ("user_type",        lambda x: (x == "member").sum()),
            casual_rides    = ("user_type",        lambda x: (x == "casual").sum()),
            electric_rides  = ("rideable_type",    lambda x: x.str.contains("electric", case=False, na=False).sum()),
            classic_rides   = ("rideable_type",    lambda x: x.str.contains("classic",  case=False, na=False).sum()),
            rush_hour_rides = ("rush_hour",         "sum"),
            avg_duration    = ("duration_min",      "mean"),
            median_duration = ("duration_min",      "median"),
            unique_stations = ("start_station_id",  "nunique"),
        )
        .reset_index()
    )

    daily["pct_member"]      = (daily["member_rides"]    / daily["total_rides"] * 100).round(2)
    daily["pct_casual"]      = (daily["casual_rides"]    / daily["total_rides"] * 100).round(2)
    daily["pct_electric"]    = (daily["electric_rides"]  / daily["total_rides"] * 100).round(2)
    daily["pct_rush_hour"]   = (daily["rush_hour_rides"] / daily["total_rides"] * 100).round(2)
    daily["avg_duration"]    = daily["avg_duration"].round(2)
    daily["median_duration"] = daily["median_duration"].round(2)

    daily["date"]        = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_name"]    = daily["date"].dt.day_name()
    daily["month"]       = daily["date"].dt.month
    daily["year"]        = daily["date"].dt.year
    daily["is_weekend"]  = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["season"]      = daily["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Fall",   10: "Fall",  11: "Fall",
    })

    if not df_weather.empty:
        daily = pd.merge(daily, df_weather, on="date", how="left")

    return daily.sort_values("date").reset_index(drop=True)


def build_stations(df: pd.DataFrame) -> pd.DataFrame:
    dep = (
        df.groupby(["start_station_id", "start_station_name"])
        .agg(
            total_departures = ("ride_id",      "count"),
            member_dep       = ("user_type",     lambda x: (x == "member").sum()),
            casual_dep       = ("user_type",     lambda x: (x == "casual").sum()),
            electric_dep     = ("rideable_type", lambda x: x.str.contains("electric", case=False, na=False).sum()),
            lat              = ("start_lat",     "median"),
            lng              = ("start_lng",     "median"),
            avg_duration     = ("duration_min",  "mean"),
        )
        .reset_index()
        .rename(columns={"start_station_id": "station_id",
                         "start_station_name": "station_name"})
    )
    arr = (
        df.groupby("end_station_id").size()
        .reset_index(name="total_arrivals")
        .rename(columns={"end_station_id": "station_id"})
    )
    stations = pd.merge(dep, arr, on="station_id", how="left").fillna(0)
    stations["net_flow"]     = (stations["total_arrivals"] - stations["total_departures"]).astype(int)
    stations["pct_member"]   = (stations["member_dep"] / stations["total_departures"] * 100).round(2)
    stations["avg_duration"] = stations["avg_duration"].round(2)
    return stations.sort_values("total_departures", ascending=False).reset_index(drop=True)


def build_sample(df: pd.DataFrame, n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    cols = [c for c in SAMPLE_COLS if c in df.columns]
    return df[cols].sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/app/*.csv from raw Citi Bike CSVs")
    parser.add_argument("--data-dir",    default=str(DATA_DIR),
                        help="Root folder with YYYYMM-citibike-* subfolders (default: data/)")
    parser.add_argument("--out-dir",     default=str(APP_DIR),
                        help="Output folder (default: data/app/)")
    parser.add_argument("--months",      nargs="*", metavar="YYYY-MM",
                        help="Months to process, e.g. --months 2025-03 2025-04")
    parser.add_argument("--noaa-token",  default=os.environ.get("NOAA_TOKEN", ""),
                        help="NOAA CDO API token (or set NOAA_TOKEN env var)")
    parser.add_argument("--sample-size", type=int, default=100_000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & clean ─────────────────────────────────────────
    log.info("Loading raw CSVs from %s ...", data_dir)
    df_raw   = load_raw_trips(data_dir, args.months)
    df_trips = clean_trips(df_raw)
    del df_raw; gc.collect()

    # ── Weather ──────────────────────────────────────────────
    df_weather = pd.DataFrame()
    if args.noaa_token:
        min_date = str(df_trips["date"].min())
        max_date = str(df_trips["date"].max())
        df_weather = fetch_weather(min_date, max_date, args.noaa_token)
    else:
        log.warning("No NOAA token — weather columns will be empty. "
                    "Pass --noaa-token TOKEN or set NOAA_TOKEN env var.")

    # ── Aggregate & export ───────────────────────────────────
    log.info("Building daily_rides_weather.csv ...")
    daily = build_daily(df_trips, df_weather)
    daily_export = daily.copy()
    daily_export["date"] = daily_export["date"].dt.strftime("%Y-%m-%d")
    daily_export.to_csv(out_dir / "daily_rides_weather.csv", index=False)
    log.info("  → %d rows × %d cols", *daily_export.shape)

    log.info("Building station_summary.csv ...")
    stations = build_stations(df_trips)
    stations.to_csv(out_dir / "station_summary.csv", index=False)
    log.info("  → %d rows × %d cols", *stations.shape)

    log.info("Building trips_sample.csv ...")
    sample = build_sample(df_trips, n=args.sample_size)
    sample.to_csv(out_dir / "trips_sample.csv", index=False)
    log.info("  → %d rows × %d cols", *sample.shape)

    del df_trips; gc.collect()

    log.info("=" * 50)
    log.info("Done. Output → %s", out_dir.resolve())
    log.info("=" * 50)


if __name__ == "__main__":
    main()
