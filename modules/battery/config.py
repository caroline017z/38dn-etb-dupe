"""
Battery Energy Storage System (BESS) configuration.
"""

from dataclasses import dataclass


@dataclass
class BatteryConfig:
    """Configuration for a co-located BESS."""

    # Sizing — power is derived from capacity_kwh / battery_hours
    battery_hours: float  # hours of storage at rated power (e.g. 4.0)

    def __post_init__(self):
        if self.battery_hours <= 0:
            raise ValueError(f"battery_hours must be > 0, got {self.battery_hours}")

    # Export power cap: max % of rated power that can be exported to grid
    discharge_limit_pct: float = 80.0  # e.g. 80 = export up to 80% of power_kw

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

    # When True, override discharge windows with the most valuable
    # consecutive block per day (length = battery_hours).
    # Charge window becomes all non-discharge hours.
    optimized_discharge: bool = False
