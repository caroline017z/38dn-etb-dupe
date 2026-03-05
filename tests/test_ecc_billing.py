"""
Unit tests for ECC billing engine and its integration with projections.

Tests cover:
- BillingResult field population (raw_annual_energy, TOU fields, nem_regime)
- Hourly/monthly energy cost reconciliation
- Demand charge splitting (flat vs TOU)
- Minimum monthly charge floor
- Export credit calculation
- Annual projection with ECC results (_agg_raw_energy correctness)
- Compound vs linear escalation
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.billing_ecc import run_ecc_billing_simulation, _extract_monthly_charges
from modules.billing import BillingResult
from modules.outputs import build_annual_projection

TOL = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_load_series(base_kw: float = 5.0, year: int = 2026) -> pd.Series:
    """Flat load profile at `base_kw` kW for 8760 hours."""
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    return pd.Series(np.full(8760, base_kw), index=idx, name="load_kwh")


def _make_solar_series(peak_kw: float = 3.0, year: int = 2026) -> pd.Series:
    """Simple solar profile: peak_kw during hours 8-16, zero otherwise."""
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    hours = idx.hour
    vals = np.where((hours >= 8) & (hours < 16), peak_kw, 0.0)
    return pd.Series(vals, index=idx, name="solar_kwh")


def _make_export_rates(rate: float = 0.05, year: int = 2026) -> pd.Series:
    """Flat export rate for 8760 hours."""
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    return pd.Series(np.full(8760, rate), index=idx, name="export_rate")


def _mock_cost_calculator(energy_per_month: float = 100.0,
                          demand_per_month: float = 20.0,
                          fixed_per_month: float = 10.0):
    """Create a mock CostCalculator that returns predictable bills."""
    calc = MagicMock()

    def _compute_bill(df, column_data="consumption", monthly_detailed=False):
        idx = df.index
        year = idx[0].year
        if monthly_detailed:
            result = {}
            for m in range(1, 13):
                key = f"{year}-{m:02d}"
                result[key] = {
                    "fix_charge_1": (1, fixed_per_month),
                    "energy_charge_1": (100, energy_per_month),
                    "demand_charge_1": {
                        str(demand_per_month / 10): {"max-demand": 10.0}
                    },
                }
            return result
        else:
            return {"total": (energy_per_month + demand_per_month + fixed_per_month) * 12}

    calc.compute_bill = _compute_bill

    def _print_aggregated(bill, verbose=False):
        total = (energy_per_month + demand_per_month + fixed_per_month) * 12
        return total, {"energy": energy_per_month * 12, "demand": demand_per_month * 12}, None

    calc.print_aggregated_bill = _print_aggregated
    return calc


# ---------------------------------------------------------------------------
# _extract_monthly_charges tests
# ---------------------------------------------------------------------------

class TestExtractMonthlyCharges:
    """Tests for the ECC bill dict parser."""

    def test_empty_bill(self):
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges({})
        assert fixed == 0.0
        assert energy == 0.0
        assert flat_d == 0.0
        assert tou_d == 0.0
        assert peak == 0.0

    def test_energy_and_fixed(self):
        bill = {
            "fix_charge_1": (1, 12.50),
            "energy_charge_1": (500, 75.00),
            "energy_charge_2": (200, 30.00),
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(fixed - 12.50) < TOL
        assert abs(energy - 105.00) < TOL
        assert flat_d == 0.0
        assert tou_d == 0.0

    def test_demand_dict_tou(self):
        """TOU demand charges (no 'flat' or 'max' in label)."""
        bill = {
            "demand_charge_tou_1": {
                "5.0": {"max-demand": 10.0},
            },
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(tou_d - 50.0) < TOL
        assert flat_d == 0.0
        assert abs(peak - 10.0) < TOL

    def test_demand_dict_flat(self):
        """Flat demand charges (label contains 'flat')."""
        bill = {
            "demand_charge_flat_1": {
                "3.0": {"max-demand": 8.0},
            },
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(flat_d - 24.0) < TOL
        assert tou_d == 0.0
        assert abs(peak - 8.0) < TOL

    def test_demand_tuple_fallback(self):
        """Demand as tuple (not dict), TOU path."""
        bill = {
            "demand_charge_1": (10, 45.0),
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(tou_d - 45.0) < TOL
        assert flat_d == 0.0

    def test_demand_max_label_is_flat(self):
        """Label containing 'max' should route to flat demand."""
        bill = {
            "demand_charge_max_1": (10, 30.0),
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(flat_d - 30.0) < TOL
        assert tou_d == 0.0

    def test_returns_5_tuple(self):
        result = _extract_monthly_charges({})
        assert len(result) == 5


# ---------------------------------------------------------------------------
# BillingResult field population tests
# ---------------------------------------------------------------------------

class TestEccBillingResultFields:
    """Verify BillingResult from ECC has all projection-critical fields."""

    @pytest.fixture
    def ecc_result(self):
        load = _make_load_series(5.0)
        solar = _make_solar_series(3.0)
        export = _make_export_rates(0.05)
        calc = _mock_cost_calculator()
        return run_ecc_billing_simulation(load, solar, calc, export)

    def test_raw_annual_energy_nonzero(self, ecc_result):
        """Bug 1 fix: raw_annual_energy should equal annual_energy_cost."""
        assert ecc_result.raw_annual_energy > 0
        assert abs(ecc_result.raw_annual_energy - ecc_result.annual_energy_cost) < TOL

    def test_tou_annual_fields_populated(self, ecc_result):
        """Bug 2 fix: TOU fields should be populated for projection."""
        assert ecc_result.tou_annual_energy > 0
        assert abs(ecc_result.tou_annual_energy - ecc_result.annual_energy_cost) < TOL
        # Export credit may be zero if export rates are zero in some hours
        assert ecc_result.tou_annual_credit >= 0

    def test_tou_monthly_dicts_populated(self, ecc_result):
        """Bug 2 fix: Monthly TOU dicts should have 12 entries."""
        assert ecc_result.tou_monthly_energy is not None
        assert ecc_result.tou_monthly_credit is not None
        assert len(ecc_result.tou_monthly_energy) == 12
        assert len(ecc_result.tou_monthly_credit) == 12

    def test_nem_regime_is_nem3(self, ecc_result):
        """Bug 4 fix: ECC always produces NEM-3 billing."""
        assert ecc_result.nem_regime == "NEM-3"

    def test_monthly_summary_has_12_rows(self, ecc_result):
        assert len(ecc_result.monthly_summary) == 12

    def test_monthly_baseline_details(self, ecc_result):
        assert ecc_result.monthly_baseline_details is not None
        assert len(ecc_result.monthly_baseline_details) == 12

    def test_hourly_detail_shape(self, ecc_result):
        assert len(ecc_result.hourly_detail) == 8760

    def test_agg_raw_energy_nonnegative(self, ecc_result):
        """Critical: _agg_raw_energy in projection should be >= 0 (Bug 1)."""
        _gen_raw = float(ecc_result.hourly_detail["energy_cost"].sum())
        _agg_raw = ecc_result.raw_annual_energy - _gen_raw
        # For single-meter ECC, agg should be ~0 (not negative)
        assert _agg_raw >= -TOL, (
            f"_agg_raw_energy is {_agg_raw}, should be >= 0"
        )


# ---------------------------------------------------------------------------
# Hourly/monthly reconciliation tests
# ---------------------------------------------------------------------------

class TestEccReconciliation:
    """Verify hourly energy_cost is reconciled with monthly ECC totals."""

    def test_hourly_monthly_energy_match(self):
        load = _make_load_series(5.0)
        solar = _make_solar_series(3.0)
        export = _make_export_rates(0.05)
        calc = _mock_cost_calculator(energy_per_month=150.0)
        result = run_ecc_billing_simulation(load, solar, calc, export)

        # Sum hourly energy_cost per month and compare to monthly_summary
        hd = result.hourly_detail
        ms = result.monthly_summary
        for m in range(1, 13):
            hourly_sum = float(hd.loc[hd.index.month == m, "energy_cost"].sum())
            monthly_sum = float(ms.loc[ms["month"] == m, "energy_cost"].values[0])
            assert abs(hourly_sum - monthly_sum) < 0.02, (
                f"Month {m}: hourly={hourly_sum:.2f} vs monthly={monthly_sum:.2f}"
            )


# ---------------------------------------------------------------------------
# Minimum monthly charge tests
# ---------------------------------------------------------------------------

class TestMinMonthlyCharge:
    """Verify min monthly charge floor from tariff_data."""

    def test_min_charge_applied(self):
        """Net bill should not go below min monthly charge."""
        load = _make_load_series(1.0)  # small load
        solar = _make_solar_series(5.0)  # large solar -> big export credit
        export = _make_export_rates(0.10)
        calc = _mock_cost_calculator(energy_per_month=5.0, demand_per_month=0.0, fixed_per_month=2.0)
        tariff_data = [{"minmonthlycharge": 15.0}]
        result = run_ecc_billing_simulation(load, solar, calc, export, tariff_data=tariff_data)

        # Each month's net bill should be >= 15.0
        for _, row in result.monthly_summary.iterrows():
            assert row["net_bill"] >= 15.0 - TOL, (
                f"Month {row['month']}: net_bill={row['net_bill']} < min=15.0"
            )

    def test_no_tariff_data_floor_is_zero(self):
        """Without tariff_data, floor should be 0."""
        load = _make_load_series(1.0)
        solar = _make_solar_series(5.0)
        export = _make_export_rates(0.10)
        calc = _mock_cost_calculator(energy_per_month=5.0, demand_per_month=0.0, fixed_per_month=2.0)
        result = run_ecc_billing_simulation(load, solar, calc, export, tariff_data=None)

        # Bills can be 0 (export credit exceeds costs)
        has_zero = any(row["net_bill"] == 0.0 for _, row in result.monthly_summary.iterrows())
        # At least some months should have net_bill = 0 since solar >> load
        assert has_zero or all(
            row["net_bill"] >= 0 for _, row in result.monthly_summary.iterrows()
        )


# ---------------------------------------------------------------------------
# Export credit tests
# ---------------------------------------------------------------------------

class TestExportCredits:
    """Verify export credits are correctly computed."""

    def test_zero_export_rates_zero_credits(self):
        load = _make_load_series(5.0)
        solar = _make_solar_series(3.0)
        export = _make_export_rates(0.0)  # zero rates
        calc = _mock_cost_calculator()
        result = run_ecc_billing_simulation(load, solar, calc, export)
        assert abs(result.annual_export_credit) < TOL

    def test_positive_export_rates_positive_credits(self):
        load = _make_load_series(2.0)  # small load so there's export
        solar = _make_solar_series(5.0)
        export = _make_export_rates(0.08)
        calc = _mock_cost_calculator()
        result = run_ecc_billing_simulation(load, solar, calc, export)
        assert result.annual_export_credit > 0

    def test_export_credit_equals_monthly_sum(self):
        load = _make_load_series(2.0)
        solar = _make_solar_series(5.0)
        export = _make_export_rates(0.08)
        calc = _mock_cost_calculator()
        result = run_ecc_billing_simulation(load, solar, calc, export)
        monthly_sum = float(result.monthly_summary["export_credit"].sum())
        assert abs(result.annual_export_credit - monthly_sum) < TOL


# ---------------------------------------------------------------------------
# Projection integration tests
# ---------------------------------------------------------------------------

class TestProjectionWithEcc:
    """Verify annual projection works correctly with ECC BillingResult."""

    @pytest.fixture
    def ecc_result(self):
        load = _make_load_series(5.0)
        solar = _make_solar_series(3.0)
        export = _make_export_rates(0.05)
        calc = _mock_cost_calculator()
        return run_ecc_billing_simulation(load, solar, calc, export)

    def test_projection_runs_without_error(self, ecc_result):
        proj = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=10,
        )
        assert len(proj) == 10
        assert "Annual Savings ($)" in proj.columns

    def test_projection_year1_matches_billing(self, ecc_result):
        proj = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=0.0,
            years=5,
            degradation_pct=0.0,
            compound_escalation=False,
        )
        # Year 1 energy should match annual_energy_cost
        y1_row = proj.iloc[0]
        y1_energy = y1_row.get("Energy ($)", y1_row.get("TOU Energy ($)", 0))
        assert abs(y1_energy - ecc_result.annual_energy_cost) < 1.0

    def test_projection_savings_nonnegative(self, ecc_result):
        proj = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=0.0,
            years=5,
        )
        # With solar, savings should be non-negative each year
        # (may be 0 if mock baseline == solar bill after rounding)
        for _, row in proj.iterrows():
            assert row["Annual Savings ($)"] >= 0

    def test_compound_escalation_higher_than_linear(self, ecc_result):
        """Compound escalation should produce higher costs over 25 years."""
        proj_compound = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=0.0,
            years=25,
            compound_escalation=True,
        )
        proj_linear = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=0.0,
            years=25,
            compound_escalation=False,
        )
        # Year 25 baseline should be higher with compound
        y25_compound = proj_compound.iloc[-1]["Bill w/o Solar ($)"]
        y25_linear = proj_linear.iloc[-1]["Bill w/o Solar ($)"]
        assert y25_compound > y25_linear, (
            f"Compound={y25_compound} should be > Linear={y25_linear}"
        )

    def test_compound_year1_same_as_linear(self, ecc_result):
        """Year 1 should be identical for both escalation modes."""
        proj_c = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=5,
            compound_escalation=True,
        )
        proj_l = build_annual_projection(
            result=ecc_result,
            system_cost=20000,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=5,
            compound_escalation=False,
        )
        # Year 1: factor = 1.0 for both modes
        for col in ["Bill w/ Solar ($)", "Bill w/o Solar ($)"]:
            if col in proj_c.columns:
                assert abs(proj_c.iloc[0][col] - proj_l.iloc[0][col]) < 1.0


# ---------------------------------------------------------------------------
# Demand charge splitting tests
# ---------------------------------------------------------------------------

class TestDemandChargeSplitting:
    """Verify flat vs TOU demand split in monthly summary."""

    def test_flat_demand_from_label(self):
        bill = {
            "demand_charge_flat_summer": {
                "4.0": {"max-demand": 12.0},
            },
            "demand_charge_tou_peak": {
                "8.0": {"max-demand": 10.0},
            },
        }
        fixed, energy, flat_d, tou_d, peak = _extract_monthly_charges(bill)
        assert abs(flat_d - 48.0) < TOL  # 4 * 12
        assert abs(tou_d - 80.0) < TOL   # 8 * 10
        assert abs(peak - 12.0) < TOL     # max of 12 and 10

    def test_monthly_summary_has_split_demand(self):
        load = _make_load_series(5.0)
        solar = _make_solar_series(3.0)
        export = _make_export_rates(0.05)

        # Mock with demand charges that include flat label
        calc = MagicMock()

        def _compute_bill(df, column_data="consumption", monthly_detailed=False):
            year = df.index[0].year
            if monthly_detailed:
                result = {}
                for m in range(1, 13):
                    key = f"{year}-{m:02d}"
                    result[key] = {
                        "fix_charge_1": (1, 10.0),
                        "energy_charge_1": (100, 80.0),
                        "demand_charge_flat_1": {"3.0": {"max-demand": 5.0}},
                        "demand_charge_tou_1": {"6.0": {"max-demand": 8.0}},
                    }
                return result
            else:
                return {"total": 1800}

        calc.compute_bill = _compute_bill
        calc.print_aggregated_bill = lambda bill, verbose=False: (1800, {}, None)

        result = run_ecc_billing_simulation(load, solar, calc, export)

        for _, row in result.monthly_summary.iterrows():
            assert abs(row["flat_demand_charge"] - 15.0) < TOL   # 3 * 5
            assert abs(row["tou_demand_charge"] - 48.0) < TOL    # 6 * 8
            assert abs(row["total_demand_charge"] - 63.0) < TOL  # sum
