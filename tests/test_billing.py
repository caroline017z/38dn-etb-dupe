"""
Unit tests for the core billing engine (modules/billing.py).

Covers NEM-3, NEM-1/2 TOU netting, NBC charges, MBO/ABO billing options,
NSC true-up, baseline bill calculation, and edge cases.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.tariff import TariffSchedule
from modules.billing import run_billing_simulation, BillingResult, _calc_baseline_bill

TOL = 0.01  # tolerance for dollar comparisons


# ---------------------------------------------------------------------------
# Helpers: tariff builders
# ---------------------------------------------------------------------------
def _make_flat_tariff(
    rate: float = 0.20,
    fixed_monthly: float = 10.0,
    min_monthly: float = 0.0,
) -> TariffSchedule:
    """Single-period flat energy rate, no demand charges."""
    energy_rate_structure = [
        [{"rate": rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": rate}],
    ]
    schedule = [[0] * 24 for _ in range(12)]
    return TariffSchedule(
        label="test_flat",
        name="Flat Test",
        utility="PG&E",
        fixed_monthly_charge=fixed_monthly,
        min_monthly_charge=min_monthly,
        energy_rate_structure=energy_rate_structure,
        energy_weekday_schedule=schedule,
        energy_weekend_schedule=schedule,
    )


def _make_tou_tariff(
    peak_rate: float = 0.30,
    offpeak_rate: float = 0.10,
    fixed_monthly: float = 10.0,
) -> TariffSchedule:
    """Two-period TOU: peak (hours 16-20, period 0) and off-peak (period 1)."""
    energy_rate_structure = [
        [{"rate": peak_rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": peak_rate}],
        [{"rate": offpeak_rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": offpeak_rate}],
    ]
    row = [1] * 24  # default off-peak
    for h in range(16, 21):
        row[h] = 0  # peak
    schedule = [list(row) for _ in range(12)]
    return TariffSchedule(
        label="test_tou",
        name="TOU Test",
        utility="PG&E",
        fixed_monthly_charge=fixed_monthly,
        energy_rate_structure=energy_rate_structure,
        energy_weekday_schedule=schedule,
        energy_weekend_schedule=schedule,
    )


# ---------------------------------------------------------------------------
# Helpers: 8760 series builders
# ---------------------------------------------------------------------------
def _make_dt_index(year: int = 2025) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{year}-01-01", periods=8760, freq="h")


def _make_series(values, year: int = 2025) -> pd.Series:
    dt = _make_dt_index(year)
    return pd.Series(np.broadcast_to(np.asarray(values, dtype=float), 8760).copy(), index=dt)


def _const_series(val: float, year: int = 2025) -> pd.Series:
    return _make_series(np.full(8760, val), year)


# ---------------------------------------------------------------------------
# 1. NEM-3 basic billing
# ---------------------------------------------------------------------------
class TestNEM3Basic:
    """NEM-3/NVBT: hourly settlement with separate import cost and export credit."""

    def test_energy_cost_equals_import_times_rate(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)    # 10 kWh/hr constant load
        solar = _const_series(3.0)    # 3 kWh/hr constant solar
        export_rates = _const_series(0.05)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3",
        )
        # Net import = 7 kWh/hr every hour
        expected_annual_import = 7.0 * 8760
        assert abs(result.annual_import_kwh - expected_annual_import) < TOL
        assert result.annual_export_kwh < TOL  # no export

        # Energy cost = 7 * 0.20 * 8760
        expected_cost = 7.0 * 0.20 * 8760
        assert abs(result.annual_energy_cost - expected_cost) < TOL

    def test_export_credit_positive_when_net_exporter(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)   # surplus 7 kWh/hr
        export_rates = _const_series(0.08)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3",
        )
        expected_export = 7.0 * 8760
        assert abs(result.annual_export_kwh - expected_export) < TOL
        assert result.annual_import_kwh < TOL

        expected_credit = 7.0 * 0.08 * 8760
        assert abs(result.annual_export_credit - expected_credit) < TOL

    def test_monthly_summary_has_12_rows(self):
        tariff = _make_flat_tariff()
        load = _const_series(5.0)
        solar = _const_series(2.0)
        export_rates = _const_series(0.05)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3",
        )
        assert len(result.monthly_summary) == 12
        assert list(result.monthly_summary["month"]) == list(range(1, 13))

    def test_net_bill_includes_fixed_charge(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=15.0)
        load = _const_series(10.0)
        solar = _const_series(0.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3",
        )
        assert abs(result.annual_fixed_cost - 15.0 * 12) < TOL

    def test_min_monthly_charge_applied(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=0.0, min_monthly=5.0)
        load = _const_series(0.001)   # near-zero load
        solar = _const_series(0.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3",
        )
        # Each month's net bill should be at least min_monthly_charge
        for _, row in result.monthly_summary.iterrows():
            assert row["net_bill"] >= 5.0 - TOL


# ---------------------------------------------------------------------------
# 2. NEM-1/2 TOU period netting
# ---------------------------------------------------------------------------
class TestNEM12TouNetting:
    """NEM-1/NEM-2: exports offset imports within same TOU period."""

    def test_tou_netting_reduces_bill(self):
        """When solar exports in peak hours offset peak imports, bill should be lower
        than if valued at hourly settlement."""
        tariff = _make_tou_tariff(peak_rate=0.30, offpeak_rate=0.10)
        dt = _make_dt_index()

        # Build load that's constant 5 kWh/hr
        load = _const_series(5.0)

        # Solar produces only during peak hours (16-20) at 10 kWh/hr
        solar_vals = np.zeros(8760)
        for h in range(8760):
            if dt[h].hour >= 16 and dt[h].hour < 21:
                solar_vals[h] = 10.0  # export 5 kWh net during peak
        solar = pd.Series(solar_vals, index=dt)
        export_rates = _const_series(0.0)  # not used in NEM-1/2

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", billing_option="MBO",
        )
        # With TOU netting, peak exports offset peak imports within the month
        assert result.annual_bill_with_solar < result.annual_bill_without_solar

    def test_nem2_same_netting_as_nem1(self):
        """NEM-1 and NEM-2 should use same TOU netting logic (NEM-2 just adds NBC)."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(5.0)
        export_rates = _const_series(0.0)

        r1 = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nbc_rate=0.0, billing_option="MBO",
        )
        r2 = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=0.0, billing_option="MBO",
        )
        # With zero NBC, NEM-1 and NEM-2 should produce the same bill
        assert abs(r1.annual_bill_with_solar - r2.annual_bill_with_solar) < TOL


# ---------------------------------------------------------------------------
# 3. NEM-2 NBC charges
# ---------------------------------------------------------------------------
class TestNEM2NBC:
    """Non-Bypassable Charges apply to net consumption under NEM-2."""

    def test_nbc_increases_bill(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(5.0)
        export_rates = _const_series(0.0)

        r_no_nbc = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=0.0, billing_option="MBO",
        )
        r_with_nbc = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=0.03, billing_option="MBO",
        )
        assert r_with_nbc.annual_bill_with_solar > r_no_nbc.annual_bill_with_solar
        assert r_with_nbc.annual_nbc_cost > 0

    def test_nbc_zero_when_nem1(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(5.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nbc_rate=0.03, billing_option="MBO",
        )
        assert result.annual_nbc_cost < TOL

    def test_nbc_proportional_to_net_import(self):
        """NBC = net_import_kwh * nbc_rate (since load > solar every hour)."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(3.0)  # net import = 7 every hour
        export_rates = _const_series(0.0)
        nbc_rate = 0.025

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=nbc_rate, billing_option="MBO",
        )
        expected_nbc = 7.0 * 8760 * nbc_rate
        assert abs(result.annual_nbc_cost - expected_nbc) < TOL


# ---------------------------------------------------------------------------
# 4. MBO credit carryover
# ---------------------------------------------------------------------------
class TestMBOCreditCarryover:
    """Monthly Billing Option: negative months bank credit for future months."""

    def test_credit_carries_forward(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=0.0)
        dt = _make_dt_index()

        # High solar in summer (months 6-8), moderate load year-round
        load_vals = np.full(8760, 5.0)
        solar_vals = np.zeros(8760)
        for h in range(8760):
            m = dt[h].month
            if m in (6, 7, 8):
                solar_vals[h] = 20.0  # big surplus
            else:
                solar_vals[h] = 2.0   # small production

        load = pd.Series(load_vals, index=dt)
        solar = pd.Series(solar_vals, index=dt)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", billing_option="MBO",
        )
        # Under MBO, negative months should show net_bill = 0 (credit banked)
        summer_bills = result.monthly_summary[
            result.monthly_summary["month"].isin([6, 7, 8])
        ]["net_bill"]
        assert (summer_bills >= 0).all()  # floored at 0


# ---------------------------------------------------------------------------
# 5. ABO energy deferral to month 12
# ---------------------------------------------------------------------------
class TestABODeferral:
    """Annual Billing Option: energy charges deferred, only demand+fixed paid monthly."""

    def test_months_1_to_11_no_energy_cost_displayed(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)
        solar = _const_series(3.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=0.0, billing_option="ABO",
        )
        # Months 1-11: displayed energy_cost should be 0
        for _, row in result.monthly_summary.iterrows():
            if row["month"] < 12:
                assert abs(row["energy_cost"]) < TOL

    def test_month_12_has_deferred_energy(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)
        solar = _const_series(3.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nbc_rate=0.0, billing_option="ABO",
        )
        dec_row = result.monthly_summary[result.monthly_summary["month"] == 12].iloc[0]
        # Month 12 energy_cost should contain all deferred energy
        assert dec_row["energy_cost"] > 0

    def test_abo_annual_total_matches_mbo(self):
        """ABO and MBO should produce the same annual total for a net consumer."""
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)
        solar = _const_series(3.0)  # always net consumer
        export_rates = _const_series(0.0)

        r_abo = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nbc_rate=0.0, billing_option="ABO",
        )
        r_mbo = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nbc_rate=0.0, billing_option="MBO",
        )
        # For a pure net consumer (no negative months), ABO and MBO annual total should match
        assert abs(r_abo.annual_bill_with_solar - r_mbo.annual_bill_with_solar) < TOL


# ---------------------------------------------------------------------------
# 6. Baseline bill (no solar)
# ---------------------------------------------------------------------------
class TestBaselineBill:
    """_calc_baseline_bill: load-only bill for savings comparison."""

    def test_baseline_matches_zero_solar_bill(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, _const_series(0.0), tariff, export_rates, nem_regime="NEM-3",
        )
        # With zero solar, bill_with_solar should equal bill_without_solar
        assert abs(result.annual_bill_with_solar - result.annual_bill_without_solar) < TOL

    def test_baseline_monthly_details_sum_to_annual(self):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)

        total, monthly = _calc_baseline_bill(load, tariff)
        assert len(monthly) == 12
        summed = sum(d["total"] for d in monthly)
        assert abs(summed - total) < TOL

    def test_baseline_monthly_components(self):
        tariff = _make_flat_tariff(rate=0.25, fixed_monthly=12.0)
        load = _const_series(8.0)

        total, monthly = _calc_baseline_bill(load, tariff)
        # Each month should have energy + fixed (no demand for this tariff)
        for d in monthly:
            assert d["energy"] > 0
            assert abs(d["fixed"] - 12.0) < TOL
            assert abs(d["demand"]) < TOL  # no demand structure


# ---------------------------------------------------------------------------
# 7. Edge case: zero solar = bill equals baseline
# ---------------------------------------------------------------------------
class TestZeroSolarEdgeCase:
    """Zero solar production should yield bill == baseline for all regimes."""

    @pytest.mark.parametrize("regime", ["NEM-1", "NEM-2", "NEM-3"])
    def test_zero_solar_all_regimes(self, regime):
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(10.0)
        solar = _const_series(0.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime=regime, nbc_rate=0.0, billing_option="MBO",
        )
        assert abs(result.annual_savings) < TOL
        assert abs(result.annual_export_kwh) < TOL
        assert abs(result.annual_solar_kwh) < TOL


# ---------------------------------------------------------------------------
# 8. Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_wrong_length_raises_value_error(self):
        tariff = _make_flat_tariff()
        dt = pd.date_range("2025-01-01", periods=100, freq="h")
        load = pd.Series(np.ones(100), index=dt)
        solar = pd.Series(np.zeros(100), index=dt)
        export_rates = pd.Series(np.zeros(100), index=dt)

        with pytest.raises(ValueError, match="Expected 8760"):
            run_billing_simulation(load, solar, tariff, export_rates)


# ---------------------------------------------------------------------------
# 9. NEM regime field on result
# ---------------------------------------------------------------------------
class TestResultMetadata:
    def test_nem_regime_stored(self):
        tariff = _make_flat_tariff()
        load = _const_series(5.0)
        solar = _const_series(2.0)
        export_rates = _const_series(0.0)

        for regime in ("NEM-1", "NEM-2", "NEM-3"):
            result = run_billing_simulation(
                load, solar, tariff, export_rates, nem_regime=regime,
                billing_option="MBO",
            )
            assert result.nem_regime == regime


# ---------------------------------------------------------------------------
# 10. NSC true-up
# ---------------------------------------------------------------------------
class TestNSCTrueUp:
    """Net Surplus Compensation: reduces export credit when annual net surplus."""

    def test_no_nsc_when_net_consumer(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(5.0)  # always net consumer
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nsc_rate=0.04, billing_option="MBO",
        )
        assert abs(result.annual_nsc_adjustment) < TOL

    def test_nsc_applied_when_net_exporter(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)  # big net exporter
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nsc_rate=0.04, billing_option="MBO",
        )
        # NSC adjustment should be positive (reduces credit)
        assert result.annual_nsc_adjustment > 0

    def test_nsc_adjustment_in_month_12(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nsc_rate=0.04, billing_option="MBO",
        )
        dec = result.monthly_summary[result.monthly_summary["month"] == 12].iloc[0]
        # NSC adjustment should appear in month 12
        assert dec["nsc_adjustment"] > 0
