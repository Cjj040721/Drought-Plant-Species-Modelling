import httpx
import numpy as np
import pandas as pd

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_weather(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    response = httpx.get(
        OPEN_METEO_ARCHIVE,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        },
        timeout=60,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily")
    if not daily:
        raise ValueError(f"Weather API returned no daily data: {payload}")
    frame = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "precipitation_mm": daily["precipitation_sum"],
        "temperature_max_c": daily["temperature_2m_max"],
        "temperature_min_c": daily["temperature_2m_min"],
    })
    if frame.isna().any().any():
        raise ValueError("Weather data contains missing values")
    return frame


def synthetic_climate(days: int = 1826, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    day = np.arange(days)
    rainfall_probability = np.clip(0.28 + 0.16 * np.sin(2 * np.pi * day / 365 - 0.8), 0.05, 0.7)
    rain_event = rng.random(days) < rainfall_probability
    precipitation = rain_event * rng.gamma(2.0, 4.0, days)
    temperature_mean = 14 + 10 * np.sin(2 * np.pi * day / 365) + rng.normal(0, 1.5, days)
    return pd.DataFrame({
        "date": dates,
        "precipitation_mm": precipitation,
        "temperature_max_c": temperature_mean + 5,
        "temperature_min_c": temperature_mean - 5,
    })


def compute_stress_index(frame: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    frame = frame.sort_values("date").copy().reset_index(drop=True)
    frame["temperature_mean_c"] = (frame["temperature_max_c"] + frame["temperature_min_c"]) / 2
    frame["rolling_precip_30"] = frame["precipitation_mm"].rolling(window, min_periods=window).sum()
    day_of_year = frame["date"].dt.dayofyear
    rain_baseline = frame.groupby(day_of_year)["rolling_precip_30"].transform("mean")
    temperature_baseline = frame.groupby(day_of_year)["temperature_mean_c"].transform("mean")
    temperature_scale = frame.groupby(day_of_year)["temperature_mean_c"].transform("std").replace(0, np.nan)
    rain_ratio = frame["rolling_precip_30"] / rain_baseline.replace(0, np.nan)
    frame["rainfall_deficit"] = (1 - rain_ratio).clip(0, 1).fillna(0)
    frame["heat_anomaly"] = ((frame["temperature_mean_c"] - temperature_baseline) / temperature_scale).clip(0, 1).fillna(0)
    frame["drought_index"] = (0.75 * frame["rainfall_deficit"] + 0.25 * frame["heat_anomaly"]).clip(0, 1)
    return frame.iloc[window - 1:].reset_index(drop=True)
