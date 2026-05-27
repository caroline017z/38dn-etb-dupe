"""Regression tests for engine correctness fixes (2026-05 review).

Covers:
  U1 — daily fixed charge on fixedchargefirstmeter is converted to $/month.
  U2 — multi-year export rate keys derive from start_year, not now().year.
  F4 — battery dispatch serves on-site load before exporting to grid (the
       removed BATT_BONUS used to bias discharge to the grid).
"""
import cvxpy as cp
import numpy as np
import pandas as pd

from modules.tariff import _sum_fixed_charges, DAYS_PER_MONTH_AVG
from modules.export_value import parse_multiyear_export_rates
from modules.battery.config import BatteryConfig
from modules.battery.dispatch import dispatch_battery


# --- U1: daily fixed charge ------------------------------------------------
def test_daily_fixed_charge_on_first_meter_converts_to_monthly():
    raw = {"fixedchargefirstmeter": 0.50, "fixedchargeunits": "$/day"}
    assert _sum_fixed_charges(raw) == 0.50 * DAYS_PER_MONTH_AVG


def test_monthly_fixed_charge_unchanged():
    assert _sum_fixed_charges({"fixedchargefirstmeter": 10.0}) == 10.0
    assert _sum_fixed_charges({"fixedmonthlycharge": 10.0}) == 10.0


def test_daily_applies_to_both_fixed_fields():
    raw = {"fixedmonthlycharge": 0.20, "fixedchargefirstmeter": 0.50,
           "fixedchargeunits": "$/day"}
    assert _sum_fixed_charges(raw) == (0.20 + 0.50) * DAYS_PER_MONTH_AVG


# --- U2: multi-year export rate keys --------------------------------------
def test_export_rate_keys_use_start_year_not_current_year():
    df = pd.DataFrame({
        "low_case": np.full(8760, 0.05),
        "high_case": np.full(8760, 0.09),
    })
    result = parse_multiyear_export_rates(df, start_year=2030)
    assert set(result.keys()) == {2030, 2031}


def test_export_rate_year_headers_still_honored():
    df = pd.DataFrame({
        "2040": np.full(8760, 0.05),
        "2041": np.full(8760, 0.06),
    })
    result = parse_multiyear_export_rates(df, start_year=2030)
    assert set(result.keys()) == {2040, 2041}


# --- F4: battery serves load before exporting -----------------------------
def _cfg() -> BatteryConfig:
    return BatteryConfig(
        battery_hours=4.0, discharge_limit_pct=80.0,
        charge_eff=1.0, discharge_eff=1.0,
        min_soc_pct=0.0, max_soc_pct=100.0,
        charge_window_start=0, charge_window_end=23,
        discharge_window_start=0, discharge_window_end=23,
        optimized_discharge=False,
    )


def test_battery_discharges_to_load_not_grid_when_import_dearer():
    """Midday PV surplus, evening load big enough to absorb all discharge,
    import >> export. The battery should serve load (saving import price), not
    dump to the grid for the lower export price."""
    N = 24
    pv = np.zeros(N)
    pv[10:16] = 10.0          # midday surplus
    load = np.zeros(N)
    load[10:16] = 1.0         # small daytime load
    load[18:23] = 5.0         # large evening load (25 kWh > usable battery)
    imp = np.full(N, 0.30)
    exp = np.full(N, 0.05)

    r = dispatch_battery(
        pv_kwh=pv, load_kwh=load, import_price=imp, export_price=exp,
        demand_window_masks={}, demand_prices={},
        battery_config=_cfg(), capacity_kwh=20.0,
    )
    assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    to_load = float(r.batt_discharge_to_load_kwh.sum())
    to_grid = float(r.batt_discharge_to_grid_kwh.sum())
    assert to_load > 0.0
    # essentially nothing should be dumped to grid while load can absorb it
    assert to_grid < 1e-3, f"battery exported {to_grid:.3f} kWh to grid instead of serving load"


def test_battery_energy_conservation_holds_after_bonus_removal():
    N = 24
    pv = np.zeros(N); pv[10:16] = 10.0
    load = np.zeros(N); load[18:23] = 5.0
    imp = np.full(N, 0.30); exp = np.full(N, 0.05)
    r = dispatch_battery(
        pv_kwh=pv, load_kwh=load, import_price=imp, export_price=exp,
        demand_window_masks={}, demand_prices={},
        battery_config=_cfg(), capacity_kwh=20.0,
    )
    # charge in == (discharge to load + discharge to grid) at unit efficiency
    charged = float(r.batt_charge_kwh.sum())
    discharged = float(r.batt_discharge_to_load_kwh.sum() + r.batt_discharge_to_grid_kwh.sum())
    assert discharged <= charged + 1e-6
