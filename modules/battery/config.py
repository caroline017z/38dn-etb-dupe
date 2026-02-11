"""
Battery Energy Storage System (BESS) configuration.
"""

from dataclasses import dataclass


@dataclass
class BatteryConfig:
    """Configuration for a co-located BESS."""

    # Sizing — power is derived from capacity_kwh / battery_hours
    battery_hours: float  # hours of storage at rated power (e.g. 4.0)

    # Depth-of-discharge / export-fraction limit
    discharge_limit_pct: float = 80.0  # max fraction of discharge that may be exported

    # Round-trip efficiency (one-way)
    charge_eff: float = 0.95   # charging efficiency (AC -> DC stored)
    discharge_eff: float = 0.95  # discharging efficiency (DC stored -> AC delivered)

    # State-of-charge bounds (% of nameplate capacity)
    min_soc_pct: float = 10.0   # floor SoC
    max_soc_pct: float = 100.0  # ceiling SoC

    # Operating windows (hour-of-day, 0-23 inclusive)
    charge_window_start: int = 10
    charge_window_end: int = 16
    discharge_window_start: int = 16
    discharge_window_end: int = 21
