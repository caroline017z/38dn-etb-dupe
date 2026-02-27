"""
OpenEI Utility Rate Database (URDB) integration for tariff schedule lookup and parsing.
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

OPENEI_API_KEY = os.getenv("OPENEI_API_KEY", "")

_retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))


# EIA utility IDs for California IOUs
UTILITY_EIA_IDS = {
    "PG&E": 14328,
    "SCE": 17609,
    "SDG&E": 16609,
}

# Default NEM-2 Non-Bypassable Charge rates ($/kWh) by utility
NBC_DEFAULTS = {
    "PG&E": 0.025,
    "SCE": 0.030,
    "SDG&E": 0.025,
}

# Default Net Surplus Compensation rate ($/kWh)
NSC_DEFAULT_RATE = 0.04


@dataclass
class TariffSchedule:
    """Parsed tariff schedule from OpenEI URDB."""

    label: str
    name: str
    utility: str
    description: str = ""

    # Fixed monthly charge ($/month)
    fixed_monthly_charge: float = 0.0

    # Minimum monthly charge ($/month)
    min_monthly_charge: float = 0.0

    # Energy rate structure: list of periods, each period is list of tiers
    # Each tier: {"rate": float, "adj": float, "max": float or None, "unit": str}
    energy_rate_structure: list = field(default_factory=list)

    # 12x24 matrices mapping (month, hour) -> period index
    energy_weekday_schedule: list = field(default_factory=list)
    energy_weekend_schedule: list = field(default_factory=list)

    # Demand rate structure: list of periods, each period is list of tiers
    # Each tier: {"rate": float, "adj": float, "max": float or None}
    demand_rate_structure: list = field(default_factory=list)

    # 12x24 matrices for demand TOU periods
    demand_weekday_schedule: list = field(default_factory=list)
    demand_weekend_schedule: list = field(default_factory=list)

    # Flat (non-coincident) demand charges: list of tiers
    demand_flat_structure: list = field(default_factory=list)

    # Non-bypassable charge ($/kWh), used for NEM-2 only
    nbc_rate: float = 0.0

    # Raw JSON for inspection
    raw_data: dict = field(default_factory=dict, repr=False)


def fetch_available_rates(utility_name: str) -> list[dict]:
    """
    Fetch available rate schedules for a given utility from OpenEI URDB.

    Returns list of dicts with keys: label, name, startdate, enddate, source
    """
    eia_id = UTILITY_EIA_IDS.get(utility_name)
    if eia_id is None:
        raise ValueError(f"Unknown utility: {utility_name}. Choose from: {list(UTILITY_EIA_IDS.keys())}")

    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 8,
        "format": "json",
        "api_key": OPENEI_API_KEY,
        "eia": eia_id,
        "is_default": "false",
    }

    response = _session.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"OpenEI API error (HTTP {response.status_code}): {response.text}")

    data = response.json()
    items = data.get("items", [])

    rates = []
    for item in items:
        rates.append({
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "startdate": item.get("startdate", ""),
            "enddate": item.get("enddate", ""),
            "source": item.get("source", ""),
            "description": item.get("description", ""),
        })

    # Sort by name for display
    rates.sort(key=lambda x: x["name"])

    return rates


def fetch_tariff_detail(rate_label: str) -> TariffSchedule:
    """
    Fetch and parse the full tariff detail for a given rate label from OpenEI.

    Returns a TariffSchedule dataclass with all parsed fields.
    """
    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 8,
        "format": "json",
        "api_key": OPENEI_API_KEY,
        "detail": "full",
        "getpage": rate_label,
    }

    response = _session.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"OpenEI API error (HTTP {response.status_code}): {response.text}")

    data = response.json()
    items = data.get("items", [])

    if not items:
        raise ValueError(f"No tariff found for label: {rate_label}")

    raw = items[0]

    tariff = TariffSchedule(
        label=raw.get("label", ""),
        name=raw.get("name", ""),
        utility=raw.get("utility", ""),
        description=raw.get("description", ""),
        raw_data=raw,
    )

    # --- Fixed charges ---
    tariff.fixed_monthly_charge = _sum_fixed_charges(raw)
    tariff.min_monthly_charge = raw.get("minmonthlycharge", 0.0) or 0.0

    # --- Energy rate structure ---
    tariff.energy_rate_structure = _parse_rate_structure(
        raw.get("energyratestructure", [])
    )
    tariff.energy_weekday_schedule = raw.get("energyweekdayschedule", [])
    tariff.energy_weekend_schedule = raw.get("energyweekendschedule", [])

    # --- Demand rate structure (TOU) ---
    tariff.demand_rate_structure = _parse_rate_structure(
        raw.get("demandratestructure", [])
    )
    tariff.demand_weekday_schedule = raw.get("demandweekdayschedule", [])
    tariff.demand_weekend_schedule = raw.get("demandweekendschedule", [])

    # --- Flat demand structure ---
    tariff.demand_flat_structure = _parse_rate_structure(
        raw.get("flatdemandstructure", [])
    )

    return tariff


def _sum_fixed_charges(raw: dict) -> float:
    """Sum all fixed monthly charges from raw tariff data."""
    total = 0.0
    raw_fixed = raw.get("fixedmonthlycharge", 0.0) or 0.0
    if raw.get("fixedchargeunits", "") == "$/day":
        total += raw_fixed * 30.44  # avg days/month
    else:
        total += raw_fixed
    # Also check for fixedchargefirstmeter
    total += raw.get("fixedchargefirstmeter", 0.0) or 0.0
    return total


def _parse_rate_structure(structure: list) -> list:
    """
    Parse an OpenEI rate structure into a cleaner format.

    Input: list of periods, each period is list of tier dicts.
    Output: list of periods, each period is list of parsed tier dicts.
    """
    parsed = []
    for period in structure:
        period_tiers = []
        for tier in period:
            period_tiers.append({
                "rate": (tier.get("rate", 0.0) or 0.0),
                "adj": (tier.get("adj", 0.0) or 0.0),
                "max": tier.get("max"),  # None means unlimited
                "unit": tier.get("unit", "kWh"),
                "effective_rate": (tier.get("rate", 0.0) or 0.0) + (tier.get("adj", 0.0) or 0.0),
            })
        parsed.append(period_tiers)
    return parsed


def get_energy_rate(tariff: TariffSchedule, month: int, hour: int, is_weekend: bool) -> float:
    """
    Get the effective energy rate ($/kWh) for a given month, hour, and day type.

    Args:
        month: 0-indexed (0=Jan, 11=Dec)
        hour: 0-23
        is_weekend: True for Saturday/Sunday
    """
    if is_weekend:
        schedule = tariff.energy_weekend_schedule
    else:
        schedule = tariff.energy_weekday_schedule

    if not schedule:
        return 0.0

    period = schedule[month][hour]
    tiers = tariff.energy_rate_structure[period]

    # For simplicity in v1: use first tier rate (most AG rates are flat within period)
    if tiers:
        return tiers[0]["effective_rate"]
    return 0.0


def get_energy_period(tariff: TariffSchedule, month: int, hour: int, is_weekend: bool) -> int:
    """Get the energy TOU period index for a given month/hour/day type."""
    if is_weekend:
        schedule = tariff.energy_weekend_schedule
    else:
        schedule = tariff.energy_weekday_schedule

    if not schedule:
        return 0

    return schedule[month][hour]


def get_demand_period(tariff: TariffSchedule, month: int, hour: int, is_weekend: bool) -> int:
    """Get the demand TOU period index for a given month/hour/day type."""
    if is_weekend:
        schedule = tariff.demand_weekend_schedule
    else:
        schedule = tariff.demand_weekday_schedule

    if not schedule:
        return 0

    return schedule[month][hour]


def format_tariff_summary(tariff: TariffSchedule) -> str:
    """Create a human-readable summary of the tariff for display."""
    lines = []
    lines.append(f"**{tariff.name}** ({tariff.utility})")
    lines.append(f"Label: `{tariff.label}`")

    if tariff.description:
        lines.append(f"Description: {tariff.description}")

    lines.append(f"Fixed Monthly Charge: ${tariff.fixed_monthly_charge:.2f}")

    if tariff.energy_rate_structure:
        lines.append("\n**Energy Rates ($/kWh):**")
        for i, period_tiers in enumerate(tariff.energy_rate_structure):
            for j, tier in enumerate(period_tiers):
                tier_label = f"Tier {j+1}" if len(period_tiers) > 1 else ""
                max_label = f" (up to {tier['max']} kWh)" if tier["max"] else ""
                lines.append(
                    f"- Period {i}: {tier_label} ${tier['effective_rate']:.5f}/kWh{max_label}"
                )

    if tariff.demand_rate_structure:
        lines.append("\n**TOU Demand Charges ($/kW):**")
        for i, period_tiers in enumerate(tariff.demand_rate_structure):
            for j, tier in enumerate(period_tiers):
                lines.append(
                    f"- Period {i}: ${tier['effective_rate']:.4f}/kW"
                )

    if tariff.demand_flat_structure:
        lines.append("\n**Non-Coincident Demand Charges ($/kW):**")
        for i, period_tiers in enumerate(tariff.demand_flat_structure):
            for j, tier in enumerate(period_tiers):
                lines.append(
                    f"- Tier {j+1}: ${tier['effective_rate']:.4f}/kW"
                )

    return "\n".join(lines)
