"""
PVWatts v8 API integration for solar production 8760 generation.
"""

import os
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

_retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))


@dataclass
class PVSystemConfig:
    system_capacity_kw_dc: float
    dc_ac_ratio: float
    array_type: int  # 0 = fixed open rack, 2 = 1-axis tracking
    losses: float = 14.08
    module_type: int = 0  # 0 = standard, 1 = premium, 2 = thin film


def _geocode_google(address: str, api_key: str) -> tuple[float, float]:
    """Geocode via Google Maps Geocoding API."""
    resp = _session.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "OK" or not data.get("results"):
        raise ValueError(
            f"Google geocoding failed (status={data['status']}): {address}"
        )
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def _geocode_nominatim(address: str) -> tuple[float, float]:
    """Geocode via Nominatim (free OSM fallback)."""
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut

    geolocator = Nominatim(user_agent="pv-rate-sim", timeout=10)
    try:
        location = geolocator.geocode(address)
    except GeocoderTimedOut:
        raise ValueError("Geocoding timed out. Please try again.")
    except GeocoderServiceError as e:
        raise ValueError(f"Geocoding service unavailable: {e}")

    if location is None:
        raise ValueError(f"Could not geocode address: {address}")

    return location.latitude, location.longitude  # type: ignore[union-attr]


@lru_cache(maxsize=128)
def geocode_address(address: str) -> tuple[float, float]:
    """Convert address string to (lat, lon).

    Uses Google Maps Geocoding API if GOOGLE_MAPS_API_KEY is set in the
    environment, otherwise falls back to Nominatim (OSM).
    """
    _upper = address.upper()
    if not re.search(r"\bCA\b", _upper) and "CALIFORNIA" not in _upper:
        address = f"{address}, CA"

    try:
        import streamlit as st
        google_key = st.secrets.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    except (ImportError, FileNotFoundError):
        google_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if google_key:
        return _geocode_google(address, google_key)
    return _geocode_nominatim(address)


def fetch_production_8760(
    api_key: str,
    lat: float,
    lon: float,
    config: PVSystemConfig,
    start_year: int = 2026,
) -> tuple[pd.Series, dict]:
    """
    Call PVWatts v8 API and return hourly AC production as a pandas Series.

    Returns:
        (production_series, summary_dict)
        - production_series: 8760-length Series indexed by hourly datetime, values in kWh
        - summary_dict: annual totals from PVWatts (ac_annual, solrad_annual, capacity_factor)
    """
    params = {
        "api_key": api_key,
        "lat": lat,
        "lon": lon,
        "system_capacity": config.system_capacity_kw_dc,
        "dc_ac_ratio": config.dc_ac_ratio,
        "module_type": config.module_type,
        "array_type": config.array_type,
        "losses": config.losses,
        "tilt": 0 if config.array_type == 2 else 20,  # tracker=0, fixed=20 degrees
        "azimuth": 180,
        "timeframe": "hourly",
    }

    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    response = _session.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"PVWatts API error (HTTP {response.status_code}): {response.text}"
        )

    data = response.json()

    if "errors" in data and data["errors"]:
        raise RuntimeError(f"PVWatts API errors: {data['errors']}")

    # AC output is in Wh — convert to kWh
    ac_wh = data["outputs"]["ac"]
    ac_kwh = np.array(ac_wh) / 1000.0

    # Build hourly datetime index — PVWatts always returns 8760 hours (TMY).
    # For leap years the index ends Dec 30 (8760 < 8784); standard TMY practice.
    dt_index = pd.date_range(
        start=f"{start_year}-01-01 00:00", periods=8760, freq="h"
    )

    production = pd.Series(ac_kwh, index=dt_index, name="solar_kwh")

    summary = {
        "ac_annual_kwh": data["outputs"]["ac_annual"],
        "solrad_annual": data["outputs"]["solrad_annual"],
        "capacity_factor": data["outputs"]["capacity_factor"],
    }

    return production, summary


def get_array_type_code(system_type: str) -> int:
    """Map user-friendly system type string to PVWatts array_type code."""
    mapping = {
        "Fixed Tilt (Ground Mount)": 0,
        "Single Axis Tracker": 2,
    }
    return mapping.get(system_type, 0)
