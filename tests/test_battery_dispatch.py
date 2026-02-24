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
class TestExportFraction:

    def test_export_fraction_80pct(self):
        """With discharge_limit_pct=80, batt_to_grid/(total_discharge) <= 0.80
        wherever discharge > 0."""
        N = 72
        pv = np.zeros(N)
        pv[8:16] = 12.0   # lots of PV mid-day
        load = np.full(N, 3.0)
        frac = 0.80

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.10),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(discharge_limit_pct=80.0),
            capacity_kwh=40.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        total_disch = (r.batt_discharge_to_load_kwh
                       + r.batt_discharge_to_grid_kwh
                       + r.batt_curtailed_kwh)
        active = total_disch > TOL
        if active.any():
            ratio = r.batt_discharge_to_grid_kwh[active] / total_disch[active]
            assert np.all(ratio <= frac + TOL), (
                f"max export ratio {ratio.max():.4f} exceeds {frac}"
            )

    def test_export_fraction_50pct(self):
        """Tighter limit: 50% export fraction."""
        N = 48
        pv = np.zeros(N)
        pv[10:15] = 15.0
        load = np.full(N, 2.0)
        frac = 0.50

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.15),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(discharge_limit_pct=50.0),
            capacity_kwh=30.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        total_disch = (r.batt_discharge_to_load_kwh
                       + r.batt_discharge_to_grid_kwh
                       + r.batt_curtailed_kwh)
        active = total_disch > TOL
        if active.any():
            ratio = r.batt_discharge_to_grid_kwh[active] / total_disch[active]
            assert np.all(ratio <= frac + TOL), (
                f"max export ratio {ratio.max():.4f} exceeds {frac}"
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
# 7. PV-only charging — battery never charges more than surplus PV
# ---------------------------------------------------------------------------
class TestPVOnlyCharging:

    def test_no_grid_charging(self):
        N = 48
        pv = np.zeros(N)
        pv[10:14] = 6.0   # only 4 h of PV
        load = np.full(N, 4.0)
        surplus = np.maximum(0.0, pv - load)

        r = dispatch_battery(
            pv_kwh=pv, load_kwh=load,
            import_price=np.full(N, 0.30),
            export_price=np.full(N, 0.05),
            demand_window_masks={}, demand_prices={},
            battery_config=_cfg(), capacity_kwh=20.0,
        )
        assert r.solver_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.all(r.batt_charge_kwh <= surplus + TOL)


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
# 10. Zero-load export with curtailment
# ---------------------------------------------------------------------------
class TestZeroLoadExport:
    """Battery must be able to export with zero on-site load, curtailing
    the non-exportable fraction."""

    def test_export_with_zero_load(self):
        """With zero load, battery charges from PV and exports during
        discharge window.  With 80% discharge_limit_pct the LP should
        curtail 20% of discharge and export the remaining 80%.

        Uses a price differential (low export during PV hours, high
        during evening) to give the battery an economic incentive to
        shift energy from daytime to evening export.
        """
        N = 48
        pv = np.zeros(N)
        pv[8:16] = 10.0   # 80 kWh of PV across 8 hours
        load = np.zeros(N)  # no on-site load at all

        # Export price: low during PV hours, higher during discharge window
        # (but still below import_price to prevent unbounded grid arbitrage)
        exp_price = np.full(N, 0.03)
        hours = np.arange(N) % 24
        exp_price[(hours >= 16) & (hours <= 23)] = 0.20

        cap = 40.0
        cfg = _cfg(
            discharge_limit_pct=80.0,
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
        total_curtailed = float(r.batt_curtailed_kwh.sum())
        total_to_load = float(r.batt_discharge_to_load_kwh.sum())

        # With zero load, batt_to_load must be zero
        assert total_to_load < TOL, (
            f"batt_to_load should be ~0 with zero load, got {total_to_load:.4f}"
        )

        # Battery should actually export (the whole point of this fix)
        assert total_export > 1.0, (
            f"Battery should export energy with zero load, got {total_export:.4f}"
        )

        # Curtailment should be non-zero (20% of what's discharged)
        assert total_curtailed > TOL, (
            f"Expected non-zero curtailment, got {total_curtailed:.4f}"
        )

        # Export fraction: grid_export / (grid_export + curtailed) should be ~0.80
        total_disch = total_export + total_curtailed
        if total_disch > TOL:
            export_ratio = total_export / total_disch
            assert abs(export_ratio - 0.80) < 0.02, (
                f"Export ratio should be ~0.80, got {export_ratio:.4f}"
            )

        # Energy balance: curtailed energy is NOT in grid balance
        net_load = load - pv
        balance = (
            r.grid_import_kwh - r.grid_export_kwh
            - net_load
            - r.batt_charge_kwh
            + r.batt_discharge_to_load_kwh
            + r.batt_discharge_to_grid_kwh
        )
        np.testing.assert_allclose(balance, 0.0, atol=TOL)
