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
from modules.billing import (
    run_billing_simulation,
    _calc_baseline_bill,
    simulate_year_under_billing_option,
)

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


def _diurnal_series(day_val: float, night_val: float,
                    day_hours=range(9, 17), year: int = 2025) -> pd.Series:
    """8760 series: day_val during day_hours, night_val otherwise. Used to build
    a net-exporter-that-also-imports profile (midday export banks credit; night
    import draws it down) — the realistic case where NSC fires and the
    consumed-credit cap is exercised."""
    dt = _make_dt_index(year)
    vals = np.where(np.isin(dt.hour, list(day_hours)), float(day_val), float(night_val))
    return pd.Series(vals, index=dt)


def _seasonal_series(summer_val: float, winter_val: float,
                     summer_months=(4, 5, 6, 7, 8, 9), year: int = 2025) -> pd.Series:
    """8760 series: summer_val in summer_months, winter_val otherwise. Used to
    test cross-month credit banking (summer surplus offsetting winter import)."""
    dt = _make_dt_index(year)
    vals = np.where(np.isin(dt.month, list(summer_months)), float(summer_val), float(winter_val))
    return pd.Series(vals, index=dt)


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

        # nsc_rate=0 isolates gross export credit; NEM-3 NSC mechanics are
        # exercised separately in TestNEM3NscTrueUp below.
        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3", nsc_rate=0.0,
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
        # NEM-1/2 net MONTHLY, so a net-export-every-month profile never has a
        # positive bill to consume banked credit. Use a seasonal profile: summer
        # banks credit, winter imports and draws it down — so credit is genuinely
        # consumed and the consumed-cap doesn't zero the clawback.
        tariff = _make_flat_tariff(rate=0.20)
        solar = _seasonal_series(40.0, 0.0)
        load = _seasonal_series(5.0, 20.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nsc_rate=0.04, billing_option="MBO",
        )
        # NSC adjustment should be positive (reduces credit)
        assert result.annual_nsc_adjustment > 0

    def test_nsc_adjustment_in_month_12(self):
        tariff = _make_flat_tariff(rate=0.20)
        solar = _seasonal_series(40.0, 0.0)
        load = _seasonal_series(5.0, 20.0)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nsc_rate=0.04, billing_option="MBO",
        )
        dec = result.monthly_summary[result.monthly_summary["month"] == 12].iloc[0]
        # NSC adjustment should appear in month 12
        assert dec["nsc_adjustment"] > 0

    def test_no_clawback_when_credit_never_consumed_mbo(self):
        """F3: an all-export MBO customer banks credit but never draws it down
        (every month already floors at 0), so nothing is consumed and the NSC
        clawback caps at $0 — the customer isn't charged for credit they never
        realized."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)   # always exporting, no positive-bill months
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", nsc_rate=0.04, billing_option="MBO",
        )
        assert abs(result.annual_nsc_adjustment) < TOL


# ---------------------------------------------------------------------------
# 10b. NEM-3 / NBT NSC true-up (per CPUC D.22-12-056 + AB 920)
# ---------------------------------------------------------------------------
class TestNEM3NscTrueUp:
    """NEM-3 / NBT: avg ACC export rate re-priced to wholesale NSC rate
    on net surplus electricity at year-end true-up."""

    def test_no_nsc_when_net_consumer(self):
        """Imports > exports → no surplus → no adjustment."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(10.0)
        solar = _const_series(5.0)
        export_rates = _const_series(0.06)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-3", nsc_rate=0.03,
        )
        assert abs(result.annual_nsc_adjustment) < TOL

    def test_no_nsc_when_nsc_above_avg_acc(self):
        """When NSC rate ≥ avg ACC rate, the customer would be made WHOLE
        (or better) by the NSC payout — no clawback applies."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)
        export_rates = _const_series(0.03)  # avg ACC = 0.03

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-3", nsc_rate=0.03,  # NSC = avg ACC
        )
        assert abs(result.annual_nsc_adjustment) < TOL

    def test_nsc_applied_when_net_exporter(self):
        """A pure net exporter (consumes almost nothing) gets the full surplus
        haircut: surplus × (avg ACC − NSC). The leftover banked surplus is large,
        so the cap does not bind."""
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)   # always exporting; surplus 7 kWh/hr
        export_rates = _const_series(0.08)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-3", nsc_rate=0.03,
        )
        surplus_kwh = result.annual_export_kwh - result.annual_import_kwh
        assert surplus_kwh > 0
        expected_adj = surplus_kwh * (0.08 - 0.03)
        assert result.annual_nsc_adjustment == pytest.approx(expected_adj, rel=0.03)

    def test_nsc_lands_on_month_12(self):
        tariff = _make_flat_tariff(rate=0.20)
        load = _const_series(3.0)
        solar = _const_series(10.0)
        export_rates = _const_series(0.08)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-3", nsc_rate=0.03,
        )
        dec = result.monthly_summary[result.monthly_summary["month"] == 12].iloc[0]
        assert dec["nsc_adjustment"] > 0
        # No NSC in months 1–11
        for m in range(1, 12):
            row = result.monthly_summary[result.monthly_summary["month"] == m].iloc[0]
            assert row["nsc_adjustment"] == 0.0

    def test_clawback_excludes_consumed_credit(self):
        """Consumed credit is never clawed back: when banked surplus is partly
        drawn down by import, the clawback is bounded by the LEFTOVER surplus —
        below the naive kWh haircut. (Profile: heavy night import draws down the
        midday-banked credit, but the account stays in $ surplus annually.)"""
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=0.0, min_monthly=0.0)
        load = _const_series(10.0)
        solar = _diurnal_series(80.0, 0.0)   # big midday export, heavy night import
        export_rates = _const_series(0.08)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-3", nsc_rate=0.03,
        )
        surplus_kwh = result.annual_export_kwh - result.annual_import_kwh
        naive_haircut = surplus_kwh * (0.08 - 0.03)
        adj = result.annual_nsc_adjustment
        # Clawback is positive but strictly below the naive kWh haircut because
        # the consumed (drawn-down) credit is excluded.
        assert 0 < adj < naive_haircut

    def test_nem3_banks_monthly_surplus_to_offset_later_months(self):
        """F1: surplus-month export credit carries forward and offsets a later
        deficit month, rather than being discarded at the monthly floor."""
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=0.0, min_monthly=0.0)
        # Summer: big solar, tiny load → bank credit. Winter: no solar, big load.
        solar = _seasonal_series(40.0, 0.0)
        load = _seasonal_series(5.0, 20.0)
        export_rates = _const_series(0.08)

        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3", nsc_rate=0.0,
        )
        ms = result.monthly_summary
        # October (first post-summer month) imports heavily, but its bill is
        # offset by banked summer credit — well below its raw import charge.
        oct_row = ms[ms["month"] == 10].iloc[0]
        raw_import_charge = oct_row["import_kwh"] * 0.20
        assert raw_import_charge > TOL
        assert oct_row["net_bill"] < raw_import_charge - TOL


# ---------------------------------------------------------------------------
# 11. Monthly ↔ annual projection tie-out (export-credit escalation)
# ---------------------------------------------------------------------------
class TestProjectionMonthlyAnnualTieOut:
    """The monthly projection must reconcile to the annual projection. The
    NEM-3 export-credit fallback previously escalated monthly export credit at
    the retail rate_factor while the annual path scaled by volume only, so the
    two views drifted apart (growing with the escalator) for flat-export NEM-3
    deals."""

    def test_nem3_flat_export_net_bill_ties_out(self):
        from modules.outputs import build_annual_projection, _build_multiyear_monthly_df

        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=10.0)
        load = _const_series(40.0)
        solar = _diurnal_series(120.0, 0.0)   # heavy midday export → material credit
        export_rates = _const_series(0.06)
        r = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3", nsc_rate=0.0,
        )
        YEARS, ESC = 10, 3.0
        ann = build_annual_projection(
            r, system_cost=0.0, rate_escalator_pct=ESC, load_escalator_pct=0.0,
            years=YEARS, nem_regime_1="NEM-3",
        )
        mon = _build_multiyear_monthly_df(
            r, rate_escalator_pct=ESC, load_escalator_pct=0.0,
            years=YEARS, nem_regime_1="NEM-3",
        )
        for y in range(1, YEARS + 1):
            annual_bill = float(ann[ann["Year"] == y]["Bill w/ Solar ($)"].iloc[0])
            monthly_sum = float(mon[mon["Year"] == y]["Net Bill ($)"].sum())
            # Only cents-vs-dollars rounding should remain (12 monthly roundings
            # vs one annual). Pre-fix this drifted by tens of dollars by Y10.
            assert abs(annual_bill - monthly_sum) < 1.0, (
                f"year {y}: annual {annual_bill:.2f} vs monthly {monthly_sum:.2f}"
            )


# ---------------------------------------------------------------------------
# 12. Credit offsets ENERGY only — demand / fixed / NBC never offset
#     (PG&E NEM2 Special Conditions 2.c/2.d; NBT Special Condition 2.d)
# ---------------------------------------------------------------------------
def _make_demand_tariff(rate=0.20, demand_rate=15.0, fixed_monthly=0.0,
                        min_monthly=0.0):
    """Flat energy rate + flat (non-coincident) demand charge."""
    ers = [[{"rate": rate, "adj": 0.0, "max": None, "unit": "kWh",
             "effective_rate": rate}]]
    sched = [[0] * 24 for _ in range(12)]
    dfs = [[{"rate": demand_rate, "adj": 0.0, "max": None, "unit": "kW",
             "effective_rate": demand_rate}]]
    return TariffSchedule(
        label="test_demand", name="Demand Test", utility="PG&E",
        fixed_monthly_charge=fixed_monthly, min_monthly_charge=min_monthly,
        energy_rate_structure=ers,
        energy_weekday_schedule=sched, energy_weekend_schedule=sched,
        demand_flat_structure=dfs, demand_flat_months=list(range(1, 13)),
    )


def _empty_components():
    return [{"energy": 0.0, "export_credit": 0.0, "demand": 0.0, "fixed": 0.0,
             "nbc": 0.0, "nsc_adj": 0.0} for _ in range(12)]


class TestCreditOffsetsEnergyOnly:
    """Banked / export credit must reduce only the volumetric energy charge."""

    def test_mbo_banked_credit_does_not_offset_demand_or_nbc(self):
        # Month 1: $500 energy credit banked. Month 2: $300 demand + $100 NBC,
        # no energy. The bank must NOT touch demand/NBC -> month 2 owes $400.
        comps = _empty_components()
        comps[0]["export_credit"] = 500.0          # net energy credit
        comps[1]["demand"] = 300.0
        comps[1]["nbc"] = 100.0
        nb = simulate_year_under_billing_option(comps, "NEM-2", "MBO", 0.0)
        assert nb[0] == pytest.approx(0.0, abs=TOL)
        assert nb[1] == pytest.approx(400.0, abs=TOL)

    def test_mbo_banked_credit_offsets_later_energy(self):
        # Sanity: the bank DOES draw down a later energy charge (just not demand).
        comps = _empty_components()
        comps[0]["export_credit"] = 500.0
        comps[1]["energy"] = 200.0                  # pure energy charge
        comps[1]["demand"] = 300.0
        nb = simulate_year_under_billing_option(comps, "NEM-2", "MBO", 0.0)
        # $200 energy fully offset by bank; $300 demand still due.
        assert nb[1] == pytest.approx(300.0, abs=TOL)

    def test_nem3_export_credit_does_not_offset_demand(self):
        comps = _empty_components()
        comps[0]["export_credit"] = 500.0
        comps[1]["demand"] = 300.0
        nb = simulate_year_under_billing_option(comps, "NEM-3", "MBO", 0.0)
        assert nb[1] == pytest.approx(300.0, abs=TOL)

    def test_abo_dec_energy_credit_does_not_offset_demand(self):
        # Every month carries $300 demand; a large summer energy credit defers
        # to December. Demand is paid all 12 months; the Dec net-energy credit
        # cannot pull December below its demand charge.
        comps = _empty_components()
        for i in range(12):
            comps[i]["demand"] = 300.0
        comps[5]["export_credit"] = 5000.0          # big June energy credit
        nb = simulate_year_under_billing_option(comps, "NEM-2", "ABO", 0.0)
        assert all(b == pytest.approx(300.0, abs=TOL) for b in nb)

    def test_integration_mbo_demand_paid_despite_banked_credit(self):
        # Summer: huge solar banks energy credit. Winter: heavy load with a
        # demand spike. Each winter month's bill must be >= its demand charge.
        tariff = _make_demand_tariff(rate=0.20, demand_rate=15.0)
        solar = _seasonal_series(40.0, 0.0)         # summer export
        load = _seasonal_series(2.0, 10.0)          # winter import
        # Add a sharp January demand spike (hour 0 of Jan 1 region handled by max).
        load_vals = load.values.copy()
        dt = load.index
        load_vals[(dt.month == 1) & (dt.hour == 18)] = 100.0   # 100 kW Jan spike
        load = pd.Series(load_vals, index=dt)
        export_rates = _const_series(0.0)

        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-1", billing_option="MBO",
        )
        jan = result.monthly_summary[result.monthly_summary["month"] == 1].iloc[0]
        assert jan["total_demand_charge"] > TOL
        # The banked summer credit cannot erase the January demand charge.
        assert jan["net_bill"] >= jan["total_demand_charge"] - TOL

    def test_integration_nem2_nbc_charged_despite_banked_credit(self):
        tariff = _make_demand_tariff(rate=0.20, demand_rate=0.0)
        solar = _seasonal_series(40.0, 0.0)
        load = _seasonal_series(2.0, 10.0)
        export_rates = _const_series(0.0)
        result = run_billing_simulation(
            load, solar, tariff, export_rates,
            nem_regime="NEM-2", nbc_rate=0.03, billing_option="MBO",
        )
        # Some winter month imports -> nonzero NBC that survives credit banking.
        nbc_total = result.monthly_summary["nbc_charge"].sum()
        assert nbc_total > TOL

    def test_nem3_projection_demand_paid_despite_export_credit(self):
        # NEM-3 Y>1 projection: a huge export credit must not erase demand.
        from modules.outputs import build_annual_projection, _build_multiyear_monthly_df
        tariff = _make_demand_tariff(rate=0.20, demand_rate=15.0)
        load = _const_series(3.0)
        solar = _diurnal_series(60.0, 0.0)       # heavy midday export
        export_rates = _const_series(0.10)       # large export credit
        r = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3", nsc_rate=0.0,
        )
        ann = build_annual_projection(
            r, system_cost=0.0, rate_escalator_pct=3.0, load_escalator_pct=0.0,
            years=5, nem_regime_1="NEM-3",
        )
        for y in range(2, 6):
            yr_demand = float(ann[ann["Year"] == y]["Demand ($)"].iloc[0])
            yr_bill = float(ann[ann["Year"] == y]["Bill w/ Solar ($)"].iloc[0])
            assert yr_demand > TOL
            # Energy is fully offset by the credit, but demand survives intact.
            assert yr_bill >= yr_demand - 1.0
        mon = _build_multiyear_monthly_df(
            r, rate_escalator_pct=3.0, load_escalator_pct=0.0,
            years=5, nem_regime_1="NEM-3",
        )
        y2 = mon[mon["Year"] == 2]
        for _, row in y2.iterrows():
            assert row["Net Bill ($)"] >= row["Demand ($)"] - TOL

    def test_tou_demand_not_offset_by_credit(self):
        # Same rule must hold for TOU (coincident) demand, not just flat demand.
        comps = _empty_components()
        comps[0]["export_credit"] = 800.0
        comps[1]["demand"] = 450.0       # TOU demand flows through the demand bucket
        nb = simulate_year_under_billing_option(comps, "NEM-2", "MBO", 0.0)
        assert nb[1] == pytest.approx(450.0, abs=TOL)

    def test_min_charge_floor_with_banked_credit_not_inflated(self):
        # Flat tariff, min charge > 0, no demand. A big summer export banks
        # credit; near-zero winter usage must pay exactly the min charge — and
        # credit that would breach the floor must NOT be consumed (no bank
        # inflation), so summer months also floor at the min charge.
        tariff = _make_flat_tariff(rate=0.20, fixed_monthly=0.0, min_monthly=15.0)
        solar = _seasonal_series(50.0, 0.0)
        load = _seasonal_series(1.0, 0.5)
        export_rates = _const_series(0.0)
        result = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-1",
            billing_option="MBO", nsc_rate=0.0,
        )
        # Every month floors at exactly the min charge (never below, never a
        # phantom negative from an inflated bank).
        for _, row in result.monthly_summary.iterrows():
            assert row["net_bill"] >= 15.0 - TOL


class TestNem3SeasonalProjectionTieOut:
    """NEM-3 Y>1: a seasonal net-annual-CONSUMER (summer surplus banks, winter
    deficit) must keep the monthly projection tied to the annual projection, and
    the banked export credit must never erase demand. This is the case the naive
    per-month energy floor broke (annual floored on totals, monthly per-month)."""

    def test_nem3_seasonal_net_consumer_ties_and_preserves_demand(self):
        from modules.outputs import build_annual_projection, _build_multiyear_monthly_df
        tariff = _make_demand_tariff(rate=0.20, demand_rate=12.0)
        solar = _seasonal_series(12.0, 0.0)   # summer surplus banks credit
        load = _seasonal_series(3.0, 20.0)     # heavy winter import, net annual consumer
        export_rates = _const_series(0.08)
        r = run_billing_simulation(
            load, solar, tariff, export_rates, nem_regime="NEM-3", nsc_rate=0.0,
        )
        assert r.annual_import_kwh > r.annual_export_kwh   # net annual consumer
        YEARS, ESC = 8, 3.0
        ann = build_annual_projection(
            r, system_cost=0.0, rate_escalator_pct=ESC, load_escalator_pct=0.0,
            years=YEARS, nem_regime_1="NEM-3",
        )
        mon = _build_multiyear_monthly_df(
            r, rate_escalator_pct=ESC, load_escalator_pct=0.0,
            years=YEARS, nem_regime_1="NEM-3",
        )
        for y in range(1, YEARS + 1):
            annual_bill = float(ann[ann["Year"] == y]["Bill w/ Solar ($)"].iloc[0])
            monthly_sum = float(mon[mon["Year"] == y]["Net Bill ($)"].sum())
            assert abs(annual_bill - monthly_sum) < 1.0, (
                f"year {y}: annual {annual_bill:.2f} vs monthly {monthly_sum:.2f}"
            )
        for _, row in mon[mon["Year"] == 3].iterrows():
            assert row["Net Bill ($)"] >= row["Demand ($)"] - TOL
