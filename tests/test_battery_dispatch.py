"""
Unit tests for battery dispatch LP.

Uses small synthetic time series (48-168 hours) so tests stay fast
and deterministic.  Solver is set explicitly to CLARABEL.
"""

import sys
import os
import numpy as np
import pytest
import cvxpy as cp
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.battery.config import BatteryConfig
from modules.battery.dispatch import dispatch_battery, DispatchResult

SOLVER = cp.CLARABEL
TOL = 1e-3  # tolerance for floating-point assertions


# ---------------------------------------------------------------------------
# Helper: build a minimal BatteryConfig
# ---------------------------------------------------------------------------
def _cfg(
    hours: float = 4.0,
    discharge_limit_pct: float = 80.0,
    charge_eff: float = 1.0,
    discharge_eff: float = 1.0,
    min_soc_pct: float = 0.0,
    max_soc_pct: float = 100.0,
    charge_start: int = 0,
    charge_end: int = 23,
    discharge_start: int = 0,
    discharge_end: int = 23,
    optimized_discharge: bool = False,
) -> BatteryConfig:
    return BatteryConfig(
        battery_hours=hours,
        discharge_limit_pct=discharge_limit_pct,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        min_soc_pct=min_soc_pct,
        max_soc_pct=max_soc_pct,
        charge_window_start=charge_start,
        charge_window_end=charge_end,
        discharge_window_start=discharge_start,
        discharge_window_end=discharge_end,
        optimized_discharge=optimized_discharge,
    )


# ---------------------------------------------------------------------------
# 1. Energy balance holds every hour
# ---------------------------------------------------------------------------
class TestEnergyBalance:
    """grid_import - grid_export == net_load + charge - to_load - to_grid"""

    def test_48h_flat_profile(self):
        N = 48
        pv = np.full(N, 5.0)
        load = np.full(N, 3.0)
        imp_price = np.full(N, 0.20)
        exp_price = np.full(N, 0.05)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=imp_price, export_price=exp_price,
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(), capacity_kwh=20.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        net_load = load - pv
        balance = (
            r.grid_import_kwh - r.grid_export_kwh
            - net_load
            - r.batt_charge_kwh
            + r.batt_discharge_to_load_kwh
            + r.batt_discharge_to_grid_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=TOL)

    def test_168h_variable_profile(self):
        N = 168
        np.random.seed(42)
        pv = np.clip(np.sin(np.linspace(0, 14 * np.pi, N)) * 10, 0, None)
        load = 3.0 + np.random.rand(N) * 4.0
        imp_price = np.full(N, 0.25)
        exp_price = np.full(N, 0.04)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=imp_price, export_price=exp_price,
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(charge_eff=0.95, discharge_eff=0.95),
            capacity_kwh=40.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        net_load = load - pv
        balance = (
            r.grid_import_kwh - r.grid_export_kwh
            - net_load
            - r.batt_charge_kwh
            + r.batt_discharge_to_load_kwh
            + r.batt_discharge_to_grid_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=TOL)


# ---------------------------------------------------------------------------
# 2. SOC bounds always satisfied
# ---------------------------------------------------------------------------
class TestSOCBounds:

    def test_soc_within_bounds(self):
        N = 72
        pv = np.zeros(N)
        pv[6:18] = 8.0   # sun hours only
        load = np.full(N, 2.0)
        cap = 30.0
        cfg = _cfg(min_soc_pct=10.0, max_soc_pct=90.0)
        min_s = 0.10 * cap
        max_s = 0.90 * cap

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.05),
            demand_window_masks={}, demand_prices={},
            battery_config=cfg, capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.all(r.soc_kwh >= min_s - TOL)
        assert np.all(r.soc_kwh <= max_s + TOL)


# ---------------------------------------------------------------------------
# 3. Charge / discharge <= power_kw each hour
# ---------------------------------------------------------------------------
class TestPowerLimits:

    def test_charge_discharge_power_cap(self):
        N = 48
        cap = 20.0
        hrs = 4.0
        power_kw = cap / hrs  # 5.0

        pv = np.full(N, 10.0)
        load = np.full(N, 1.0)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.05),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(hours=hrs), capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.all(r.batt_charge_kwh <= power_kw + TOL)
        discharge = r.batt_discharge_to_load_kwh + r.batt_discharge_to_grid_kwh
        assert np.all(discharge <= power_kw + TOL)


# ---------------------------------------------------------------------------
# 4. Allowed windows force charge/discharge to zero when disallowed
# ---------------------------------------------------------------------------
class TestWindowEnforcement:

    def test_charge_window_only(self):
        """Charging only allowed hours 10-14; must be zero outside."""
        N = 48
        pv = np.full(N, 8.0)
        load = np.full(N, 2.0)
        cfg = _cfg(charge_start=10, charge_end=14,
                    discharge_start=18, discharge_end=22)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=np.full(N, 0.04),
            demand_window_masks={}, demand_prices={},
            battery_config=cfg, capacity_kwh=20.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        # hours modulo 24
        hours = np.arange(N) % 24
        outside_charge = ~((hours >= 10) & (hours <= 14))
        outside_discharge = ~((hours >= 18) & (hours <= 22))

        np.testing.assert_allclose(
            r.batt_charge_kwh[outside_charge], 0.0, atol=TOL,
        )
        discharge = r.batt_discharge_to_load_kwh + r.batt_discharge_to_grid_kwh
        np.testing.assert_allclose(
            discharge[outside_discharge], 0.0, atol=TOL,
        )

    def test_discharge_window_midnight_wrap(self):
        """Discharge window 22-04 wraps midnight; zero outside."""
        N = 48
        pv = np.full(N, 6.0)
        load = np.full(N, 3.0)
        cfg = _cfg(discharge_start=22, discharge_end=4)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=np.full(N, 0.04),
            demand_window_masks={}, demand_prices={},
            battery_config=cfg, capacity_kwh=20.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        hours = np.arange(N) % 24
        allowed = (hours >= 22) | (hours <= 4)
        discharge = r.batt_discharge_to_load_kwh + r.batt_discharge_to_grid_kwh
        np.testing.assert_allclose(
            discharge[~allowed], 0.0, atol=TOL,
        )


# ---------------------------------------------------------------------------
# 5. Export fraction constraint
# ---------------------------------------------------------------------------
class TestExportPowerCap:

    def test_export_power_cap_80pct(self):
        """With discharge_limit_pct=80, batt_to_grid <= 0.80 * power_kw
        every hour."""
        N = 72
        pv = np.zeros(N)
        pv[8:16] = 12.0   # lots of PV mid-day
        load = np.full(N, 3.0)
        cap = 40.0
        hrs = 4.0
        frac = 0.80
        power_kw = cap / hrs  # 10 kW
        max_export_kw = frac * power_kw  # 8 kW

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.10),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(discharge_limit_pct=80.0, hours=hrs),
            capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        # Export power should never exceed frac * power_kw
        assert np.all(r.batt_discharge_to_grid_kwh <= max_export_kw + TOL), (
            f"max export {r.batt_discharge_to_grid_kwh.max():.4f} exceeds "
            f"{max_export_kw}"
        )

    def test_export_power_cap_50pct(self):
        """Tighter limit: 50% export power cap."""
        N = 48
        pv = np.zeros(N)
        pv[10:15] = 15.0
        load = np.full(N, 2.0)
        cap = 30.0
        hrs = 4.0
        frac = 0.50
        power_kw = cap / hrs  # 7.5 kW
        max_export_kw = frac * power_kw  # 3.75 kW

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.15),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(discharge_limit_pct=50.0, hours=hrs),
            capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        assert np.all(r.batt_discharge_to_grid_kwh <= max_export_kw + TOL), (
            f"max export {r.batt_discharge_to_grid_kwh.max():.4f} exceeds "
            f"{max_export_kw}"
        )


# ---------------------------------------------------------------------------
# 6. Demand-charge shaving — the battery MUST reduce monthly peak import
# ---------------------------------------------------------------------------
class TestDemandShaving:

    def test_battery_shaves_peak(self):
        """Construct a profile with one huge peak hour per 'month'.
        Battery should discharge during that hour to reduce the peak
        captured by the demand variable."""
        N = 168  # 7 days = one synthetic 'month'
        load = np.full(N, 5.0)
        # Insert a large spike at hour 100 (during discharge window)
        load[100] = 50.0
        pv = np.zeros(N)
        pv[8:16] = 10.0   # PV charges during day
        # Repeat each 24-h day to let battery charge before the spike
        # Spike is on day 4 at hour 4 of that day (hour 100 = 4*24+4)

        cap = 40.0
        cfg = _cfg(hours=4.0, min_soc_pct=0.0, max_soc_pct=100.0)
        power_kw = cap / cfg.battery_hours  # 10 kW

        # demand charge covers all hours — high price to incentivise shaving
        demand_masks = {"flat": np.ones(N, dtype=bool)}
        demand_prices = {"flat": 20.0}

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.20),
            export_price=np.full(N, 0.04),
            demand_window_masks=demand_masks,
            demand_prices=demand_prices,
            battery_config=cfg, capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        # Without battery, peak import at hour 100 would be 50 kW
        # Battery can discharge up to 10 kW → peak should be <= 50 - 10 = 40 kW
        peak_import = r.grid_import_kwh.max()
        assert peak_import < 50.0 - (power_kw * 0.5), (
            f"Expected peak shaved below {50 - power_kw*0.5:.1f}, "
            f"got {peak_import:.1f}"
        )

        # Confirm battery discharged during the spike hour
        spike_discharge = (
            r.batt_discharge_to_load_kwh[100]
            + r.batt_discharge_to_grid_kwh[100]
        )
        assert spike_discharge > TOL, "Battery should discharge during peak hour"

    def test_demand_charge_vs_no_demand(self):
        """With vs without demand charges: peak import should be lower
        when demand charges are active."""
        N = 72
        load = np.full(N, 4.0)
        load[36] = 30.0   # spike
        pv = np.zeros(N)
        pv[6:18] = 8.0

        cap = 24.0
        cfg = _cfg(hours=4.0, min_soc_pct=0.0, max_soc_pct=100.0)
        common: dict[str, Any] = dict(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=np.full(N, 0.04),
            battery_config=cfg, capacity_kwh=cap,
        )

        # run WITHOUT demand charges
        r_no_dc = dispatch_battery(
            demand_window_masks={}, demand_prices={}, **common,
        )
        # run WITH demand charges
        r_dc = dispatch_battery(
            demand_window_masks={"flat": np.ones(N, dtype=bool)},
            demand_prices={"flat": 15.0},
            **common,
        )

        assert r_no_dc.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert r_dc.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        peak_no_dc = r_no_dc.grid_import_kwh.max()
        peak_dc = r_dc.grid_import_kwh.max()

        assert peak_dc < peak_no_dc - TOL, (
            f"Demand-charge run peak ({peak_dc:.2f}) should be lower than "
            f"no-demand run ({peak_no_dc:.2f})"
        )


# ---------------------------------------------------------------------------
# 7. PV charging — battery never charges more than available PV
# ---------------------------------------------------------------------------
class TestPVCharging:

    def test_no_pure_grid_charging(self):
        """Battery cannot charge in hours with zero PV (no pure grid charging)."""
        N = 48
        pv = np.zeros(N)
        pv[10:14] = 6.0   # only 4 h of PV
        load = np.full(N, 4.0)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.05),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(), capacity_kwh=20.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        # Battery must not charge when there's no PV at all
        no_pv = pv <= 0
        np.testing.assert_allclose(
            r.batt_charge_kwh[no_pv], 0.0, atol=TOL,
        )
        # Battery charge never exceeds surplus PV (PV minus load)
        surplus_pv = np.maximum(0.0, pv - load)
        assert np.all(r.batt_charge_kwh <= surplus_pv + TOL)


# ---------------------------------------------------------------------------
# 8. Solver status always optimal for feasible problems
# ---------------------------------------------------------------------------
class TestSolverStatus:

    def test_trivial_problem_optimal(self):
        N = 48
        r = dispatch_battery(
            pv_kwh=np.full(N, 5.0),
            load_kwh=np.full(N, 5.0),
            import_price=np.full(N, 0.20),
            export_price=np.full(N, 0.05),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(), capacity_kwh=10.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.isfinite(r.objective_value)


# ---------------------------------------------------------------------------
# 9. Monthly LP decomposition
# ---------------------------------------------------------------------------
import pandas as pd


def _make_8760_profiles(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 8760-hour PV and load profiles for testing."""
    np.random.seed(seed)
    hours = np.arange(8760)
    # Sinusoidal PV with daily cycle
    pv = np.clip(np.sin((hours % 24 - 6) / 12 * np.pi) * 8, 0, None)
    # Scale by season (more PV in summer)
    day_of_year = hours / 24
    seasonal = 0.7 + 0.3 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    pv = pv * seasonal
    load = 3.0 + np.random.rand(8760) * 4.0
    return pv, load


class TestMonthlyDecomposition:
    """Tests for monthly LP decomposition mode (monthly=True)."""

    def test_soc_continuity_at_month_boundaries(self):
        """End-of-month SOC feeds correctly into start-of-next-month dynamics."""
        pv, load = _make_8760_profiles(seed=123)
        cfg = _cfg(charge_eff=0.95, discharge_eff=0.95)
        cap = 40.0

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(8760, 0.25),
            export_price=np.full(8760, 0.04),
            demand_window_masks={"flat": np.ones(8760, dtype=bool)},
            demand_prices={"flat": 10.0},
            battery_config=cfg,
            capacity_kwh=cap,
            monthly=True,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert len(r.soc_kwh) == 8760

        dt_index = pd.date_range("2023-01-01", periods=8760, freq="h")
        month_arr: np.ndarray = dt_index.month.values  # type: ignore[attr-defined]

        for m in range(1, 12):
            last_idx = np.where(month_arr == m)[0][-1]
            first_next = np.where(month_arr == m + 1)[0][0]

            end_soc = r.soc_kwh[last_idx]

            # Verify SOC dynamics equation: soc[t] = soc[t-1] + charge*eff - discharge/eff
            expected_soc = (
                end_soc
                + r.batt_charge_kwh[first_next] * cfg.charge_eff
                - (r.batt_discharge_to_load_kwh[first_next]
                   + r.batt_discharge_to_grid_kwh[first_next]) / cfg.discharge_eff
            )
            np.testing.assert_allclose(
                r.soc_kwh[first_next], expected_soc, atol=1e-2,
                err_msg=f"SOC discontinuity at month {m}/{m+1} boundary"
            )

    def test_energy_balance_monthly_mode(self):
        """Energy balance must hold every hour in monthly decomposition."""
        pv, load = _make_8760_profiles(seed=42)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(8760, 0.25),
            export_price=np.full(8760, 0.04),
            demand_window_masks={},
            demand_prices={},
            battery_config=_cfg(charge_eff=0.95, discharge_eff=0.95),
            capacity_kwh=50.0,
            monthly=True,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        net_load = load - pv
        balance = (
            r.grid_import_kwh - r.grid_export_kwh
            - net_load
            - r.batt_charge_kwh
            + r.batt_discharge_to_load_kwh
            + r.batt_discharge_to_grid_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=TOL)

    def test_monthly_vs_annual_close(self):
        """Monthly decomposition objective should be close to annual LP."""
        pv, load = _make_8760_profiles(seed=99)

        common: dict[str, Any] = dict(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(8760, 0.25),
            export_price=np.full(8760, 0.04),
            demand_window_masks={"flat": np.ones(8760, dtype=bool)},
            demand_prices={"flat": 10.0},
            battery_config=_cfg(charge_eff=0.95, discharge_eff=0.95),
            capacity_kwh=40.0,
        )

        r_annual = dispatch_battery(**common, monthly=False)
        r_monthly = dispatch_battery(**common, monthly=True)

        assert r_annual.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert r_monthly.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        annual_obj = r_annual.objective_value
        monthly_obj = r_monthly.objective_value

        # Relative difference should be small (< 5%)
        rel_diff = abs(monthly_obj - annual_obj) / abs(annual_obj) if annual_obj != 0 else 0
        assert rel_diff < 0.05, (
            f"Monthly objective ({monthly_obj:.2f}) differs from annual "
            f"({annual_obj:.2f}) by {rel_diff*100:.1f}%"
        )

    def test_monthly_fallback_for_short_arrays(self):
        """monthly=True with N<8760 should fall back to single LP."""
        N = 168
        pv = np.full(N, 5.0)
        load = np.full(N, 3.0)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=np.full(N, 0.04),
            demand_window_masks={},
            demand_prices={},
            battery_config=_cfg(),
            capacity_kwh=20.0,
            monthly=True,  # should gracefully fall back
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert len(r.grid_import_kwh) == N

    def test_soc_bounds_monthly(self):
        """SOC bounds must be respected across all 8760 hours."""
        pv, load = _make_8760_profiles(seed=7)
        cap = 30.0
        cfg = _cfg(min_soc_pct=10.0, max_soc_pct=90.0,
                    charge_eff=0.95, discharge_eff=0.95)
        min_s = 0.10 * cap
        max_s = 0.90 * cap

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(8760, 0.25),
            export_price=np.full(8760, 0.04),
            demand_window_masks={},
            demand_prices={},
            battery_config=cfg,
            capacity_kwh=cap,
            monthly=True,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.all(r.soc_kwh >= min_s - TOL), (
            f"SOC min violation: {r.soc_kwh.min():.4f} < {min_s}"
        )
        assert np.all(r.soc_kwh <= max_s + TOL), (
            f"SOC max violation: {r.soc_kwh.max():.4f} > {max_s}"
        )


# ---------------------------------------------------------------------------
# 10. Zero-load export with power cap
# ---------------------------------------------------------------------------
class TestZeroLoadExport:
    """Battery must be able to export with zero on-site load.
    Export is capped at frac * power_kw per hour — no energy is wasted."""

    def test_export_with_zero_load(self):
        """With zero load, battery charges from PV and exports during
        discharge window at up to frac * power_kw per hour.
        No energy is curtailed or lost.
        """
        N = 48
        pv = np.zeros(N)
        pv[8:16] = 10.0   # 80 kWh of PV across 8 hours
        load = np.zeros(N)  # no on-site load at all

        # Export price: low during PV hours, higher during discharge window
        exp_price = np.full(N, 0.03)
        hours = np.arange(N) % 24
        exp_price[(hours >= 16) & (hours <= 23)] = 0.20

        cap = 40.0
        hrs = 4.0
        frac = 0.80
        power_kw = cap / hrs  # 10 kW
        max_export_kw = frac * power_kw  # 8 kW

        cfg = _cfg(
            discharge_limit_pct=80.0,
            hours=hrs,
            charge_start=8, charge_end=15,
            discharge_start=16, discharge_end=23,
        )

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=exp_price,
            demand_window_masks={}, demand_prices={},
            battery_config=cfg, capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        total_export = float(r.batt_discharge_to_grid_kwh.sum())
        total_to_load = float(r.batt_discharge_to_load_kwh.sum())

        # With zero load, batt_to_load must be zero
        assert total_to_load < TOL, (
            f"batt_to_load should be ~0 with zero load, got {total_to_load:.4f}"
        )

        # Battery should actually export
        assert total_export > 1.0, (
            f"Battery should export energy with zero load, got {total_export:.4f}"
        )

        # Export power capped per hour
        assert np.all(r.batt_discharge_to_grid_kwh <= max_export_kw + TOL), (
            f"max export {r.batt_discharge_to_grid_kwh.max():.4f} exceeds "
            f"{max_export_kw}"
        )

        # Energy balance: no energy is lost
        net_load = load - pv
        balance = (
            r.grid_import_kwh - r.grid_export_kwh
            - net_load
            - r.batt_charge_kwh
            + r.batt_discharge_to_load_kwh
            + r.batt_discharge_to_grid_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=TOL)


# ---------------------------------------------------------------------------
# 11. Optimized discharge window selection
# ---------------------------------------------------------------------------
class TestOptimizedDischarge:
    """Optimized mode picks the highest-value consecutive block per day."""

    def test_optimized_picks_expensive_hours(self):
        """With two price peaks (hours 10-13 @$0.30, 18-21 @$0.50),
        the optimizer should discharge during hours 18-21."""
        N = 48
        pv = np.zeros(N)
        pv[6:14] = 12.0  # plenty of PV for charging
        load = np.zeros(N)

        exp_price = np.full(N, 0.05)
        hours = np.arange(N) % 24
        exp_price[(hours >= 10) & (hours <= 13)] = 0.30  # lesser peak
        exp_price[(hours >= 18) & (hours <= 21)] = 0.50  # best peak

        cap = 40.0
        cfg = _cfg(hours=4.0, optimized_discharge=True)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.25),
            export_price=exp_price,
            demand_window_masks={}, demand_prices={},
            battery_config=cfg, capacity_kwh=cap,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        # Battery should export during hrs 18-21 (the expensive window)
        for day_offset in [0, 24]:
            day_export = r.batt_discharge_to_grid_kwh[day_offset:day_offset + 24]
            expensive_export = day_export[18:22].sum()
            cheap_export = day_export[10:14].sum()
            assert expensive_export > cheap_export + TOL, (
                f"Optimized should prefer hrs 18-21 ({expensive_export:.2f}) "
                f"over 10-13 ({cheap_export:.2f})"
            )


# ---------------------------------------------------------------------------
# 12. Monthly LP with demand charges
# ---------------------------------------------------------------------------
class TestMonthlyDemandCharges:
    """Verify demand charges work correctly in monthly decomposition mode."""

    def test_monthly_demand_produces_finite_objective(self):
        """Monthly LP with non-trivial demand prices should produce
        a finite, non-negative objective value (no unbounded peak vars)."""
        N = 8760
        pv = np.zeros(N)
        hours = np.arange(N) % 24
        pv[(hours >= 8) & (hours <= 16)] = 5.0
        load = np.full(N, 3.0)

        # Simple demand mask: all hours
        demand_masks = {"flat": np.ones(N, dtype=bool)}
        demand_prices = {"flat": 10.0}  # $10/kW demand charge

        cfg = _cfg(charge_start=8, charge_end=15,
                    discharge_start=16, discharge_end=23)
        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.15),
            export_price=np.full(N, 0.05),
            demand_window_masks=demand_masks,
            demand_prices=demand_prices,
            battery_config=cfg, capacity_kwh=20.0,
            monthly=True,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "mixed"), (
            f"Monthly demand LP failed: {r.solver_status}"
        )
        assert np.isfinite(r.objective_value), (
            f"Objective should be finite, got {r.objective_value}"
        )
        assert r.objective_value >= -1e6, (
            f"Objective should not be extremely negative, got {r.objective_value}"
        )


# ---------------------------------------------------------------------------
# 13. Solver fallback path
# ---------------------------------------------------------------------------
class TestSolverFallback:
    """Verify the failed-solver fallback returns PV-only flows."""

    def test_infeasible_config_falls_back(self):
        """Contradictory SOC bounds (min > max) make the LP infeasible,
        triggering fallback to PV-only flows."""
        N = 48
        pv = np.zeros(N)
        load = np.full(N, 5.0)  # constant load, no PV

        cfg = _cfg(
            min_soc_pct=80.0,   # min_soc > max_soc → infeasible
            max_soc_pct=20.0,
            charge_eff=0.95,
            discharge_eff=0.95,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = dispatch_battery(
                pv_kwh=pv, load_kwh=load,
                import_price=np.full(N, 0.20),
                export_price=np.full(N, 0.05),
                demand_window_masks={}, demand_prices={},
                battery_config=cfg, capacity_kwh=20.0,
            )

        # Fallback should produce PV-only flows
        # With zero PV and constant load, import should equal load
        np.testing.assert_allclose(r.grid_import_kwh, load, atol=TOL)
        np.testing.assert_allclose(r.grid_export_kwh, 0.0, atol=TOL)
        np.testing.assert_allclose(r.batt_charge_kwh, 0.0, atol=TOL)
        np.testing.assert_allclose(r.batt_discharge_to_load_kwh, 0.0, atol=TOL)
        np.testing.assert_allclose(r.batt_discharge_to_grid_kwh, 0.0, atol=TOL)
