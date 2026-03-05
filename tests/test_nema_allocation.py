"""
Unit tests for NEM-A (NEM Aggregation) billing module.

Tests allocation algorithm, credit valuation, effective export price,
aggregate BillingResult construction, and fee calculations.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.billing_aggregation import (
    MeterConfig,
    NemAProfile,
    AllocationResult,
    NEMA_FEES,
    compute_nema_fees,
    compute_monthly_allocation,
    value_allocation_at_retail_rates,
    compute_effective_export_price,
    run_aggregation_simulation,
)
from modules.tariff import TariffSchedule
from modules.billing import BillingResult


# ---------------------------------------------------------------------------
# Helper: build a minimal TariffSchedule for testing
# ---------------------------------------------------------------------------
def _make_tariff(
    name: str = "Test Tariff",
    fixed_monthly: float = 10.0,
    energy_rates: list[float] | None = None,
) -> TariffSchedule:
    """Create a simple TariffSchedule with flat energy rate (single period)."""
    if energy_rates is None:
        energy_rates = [0.20]

    energy_rate_structure = [
        [{"rate": r, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": r}]
        for r in energy_rates
    ]
    # Single-period schedule: all hours map to period 0
    schedule = [[0] * 24 for _ in range(12)]

    return TariffSchedule(
        label="test",
        name=name,
        utility="PG&E",
        fixed_monthly_charge=fixed_monthly,
        energy_rate_structure=energy_rate_structure,
        energy_weekday_schedule=schedule,
        energy_weekend_schedule=schedule,
    )


def _make_two_period_tariff(
    name: str = "TOU Tariff",
    peak_rate: float = 0.30,
    offpeak_rate: float = 0.10,
    fixed_monthly: float = 10.0,
) -> TariffSchedule:
    """Create a TOU tariff with peak (period 0, hours 16-20) and off-peak (period 1)."""
    energy_rate_structure = [
        [{"rate": peak_rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": peak_rate}],
        [{"rate": offpeak_rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": offpeak_rate}],
    ]
    # Peak = hours 16-20, off-peak = all others
    row = [1] * 24  # default off-peak
    for h in range(16, 21):
        row[h] = 0  # peak
    schedule = [list(row) for _ in range(12)]

    return TariffSchedule(
        label="test_tou",
        name=name,
        utility="PG&E",
        fixed_monthly_charge=fixed_monthly,
        energy_rate_structure=energy_rate_structure,
        energy_weekday_schedule=schedule,
        energy_weekend_schedule=schedule,
    )


def _make_load_series(
    base_kwh: float = 100.0,
    start_year: int = 2026,
) -> pd.Series:
    """Create a flat 8760 load series."""
    dt_index = pd.date_range(start=f"{start_year}-01-01 00:00", periods=8760, freq="h")
    return pd.Series(np.full(8760, base_kwh), index=dt_index, name="load_kwh")


def _make_seasonal_load(
    summer_kwh: float = 200.0,
    winter_kwh: float = 50.0,
    start_year: int = 2026,
) -> pd.Series:
    """Create a load series that varies by season."""
    dt_index = pd.date_range(start=f"{start_year}-01-01 00:00", periods=8760, freq="h")
    values = np.array([
        summer_kwh if dt.month in (6, 7, 8, 9) else winter_kwh
        for dt in dt_index
    ])
    return pd.Series(values, index=dt_index, name="load_kwh")


# ---------------------------------------------------------------------------
# 1. Fee calculation
# ---------------------------------------------------------------------------
class TestFeeCalculation:
    def test_pge_fees_basic(self):
        fees = compute_nema_fees("PG&E", 3)
        assert fees["setup_cost"] == 75.0  # 25 * 3
        assert fees["monthly_admin"] == 15.0  # 5 * 3
        assert fees["annual_admin"] == 180.0  # 15 * 12

    def test_pge_fees_cap(self):
        # 25 * 30 = 750, capped at 500
        fees = compute_nema_fees("PG&E", 30)
        assert fees["setup_cost"] == 500.0

    def test_sce_fees_no_cap(self):
        fees = compute_nema_fees("SCE", 5)
        assert fees["setup_cost"] == 80.0  # 16 * 5
        assert fees["monthly_admin"] == 13.5  # 2.70 * 5
        assert fees["annual_admin"] == pytest.approx(162.0, abs=0.01)

    def test_sdge_fees(self):
        fees = compute_nema_fees("SDG&E", 2)
        assert fees["setup_cost"] == 40.0  # 20 * 2
        assert fees["monthly_admin"] == 6.0  # 3.00 * 2
        assert fees["annual_admin"] == 72.0

    def test_unknown_utility_defaults_to_pge(self):
        fees = compute_nema_fees("Unknown", 1)
        assert fees["setup_cost"] == 25.0  # Falls back to PG&E


# ---------------------------------------------------------------------------
# 2. Two-meter proportional allocation
# ---------------------------------------------------------------------------
class TestTwoMeterAllocation:
    def test_basic_allocation(self):
        """1 generating meter + 1 aggregated meter, all exports allocated."""
        tariff = _make_tariff()
        agg_load = _make_load_series(100.0)  # 100 kWh every hour

        agg_meter = MeterConfig(
            name="Warehouse", load_8760=agg_load, tariff=tariff, is_generating=False
        )

        # 500 kWh exports per month (well under consumption cap)
        gen_monthly_exports = {m: 500.0 for m in range(1, 13)}

        result = compute_monthly_allocation(gen_monthly_exports, [agg_meter])

        # All exports should be allocated (500 < monthly consumption)
        assert result.annual_allocated_kwh == pytest.approx(6000.0, abs=1.0)
        assert result.annual_unallocated_kwh == pytest.approx(0.0, abs=1.0)

    def test_cap_enforcement(self):
        """Allocation cannot exceed a meter's monthly consumption."""
        tariff = _make_tariff()
        # Very small load: ~10 kWh/hr * ~730 hrs/month = ~7300 kWh/month
        agg_load = _make_load_series(10.0)

        agg_meter = MeterConfig(
            name="Small Meter", load_8760=agg_load, tariff=tariff, is_generating=False
        )

        # Huge exports: 50000 kWh/month — way more than consumption
        gen_monthly_exports = {m: 50000.0 for m in range(1, 13)}

        result = compute_monthly_allocation(gen_monthly_exports, [agg_meter])

        # Allocated should be roughly equal to annual consumption
        annual_consumption = float(agg_load.sum())
        assert result.annual_allocated_kwh <= annual_consumption + 1.0
        assert result.annual_unallocated_kwh > 0

    def test_zero_exports_month(self):
        """No allocation when generating meter has no surplus."""
        tariff = _make_tariff()
        agg_load = _make_load_series(100.0)

        agg_meter = MeterConfig(
            name="Warehouse", load_8760=agg_load, tariff=tariff, is_generating=False
        )

        gen_monthly_exports = {m: 0.0 for m in range(1, 13)}

        result = compute_monthly_allocation(gen_monthly_exports, [agg_meter])

        assert result.annual_allocated_kwh == 0.0
        assert result.annual_unallocated_kwh == 0.0


# ---------------------------------------------------------------------------
# 3. Three-meter proportional split
# ---------------------------------------------------------------------------
class TestThreeMeterAllocation:
    def test_proportional_split(self):
        """Two aggregated meters split credits proportional to consumption."""
        tariff = _make_tariff()
        load_a = _make_load_series(200.0)  # 2x consumption
        load_b = _make_load_series(100.0)  # 1x consumption

        meter_a = MeterConfig(name="Big", load_8760=load_a, tariff=tariff)
        meter_b = MeterConfig(name="Small", load_8760=load_b, tariff=tariff)

        # Moderate exports — well below total consumption
        gen_monthly_exports = {m: 1000.0 for m in range(1, 13)}

        result = compute_monthly_allocation(gen_monthly_exports, [meter_a, meter_b])

        total_alloc = result.annual_allocated_kwh
        assert total_alloc == pytest.approx(12000.0, abs=1.0)  # 1000 * 12

        # Check that Big gets ~2/3 and Small gets ~1/3
        annual_a = sum(result.monthly_allocation[m]["Big"] for m in range(1, 13))
        annual_b = sum(result.monthly_allocation[m]["Small"] for m in range(1, 13))

        # Big should get approximately 2x what Small gets
        assert annual_a / annual_b == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# 4. Effective export price
# ---------------------------------------------------------------------------
class TestEffectiveExportPrice:
    def test_single_meter_matches_tariff(self):
        """With one aggregated meter, effective price = that meter's TOU rate."""
        tariff = _make_tariff(energy_rates=[0.25])
        agg_load = _make_load_series(100.0)
        dt_index = agg_load.index

        meters = [
            MeterConfig(name="Gen", load_8760=_make_load_series(50.0), tariff=tariff, is_generating=True),
            MeterConfig(name="Agg", load_8760=agg_load, tariff=tariff),
        ]

        eff_price = compute_effective_export_price(meters, dt_index)

        # Should be flat 0.25 everywhere (flat tariff, constant load)
        assert np.allclose(eff_price, 0.25, atol=1e-6)

    def test_load_weighted_average(self):
        """Two meters with different rates, weighted by load."""
        tariff_cheap = _make_tariff(energy_rates=[0.10])
        tariff_expensive = _make_tariff(energy_rates=[0.30])

        load_cheap = _make_load_series(100.0)    # 100 kWh at $0.10
        load_expensive = _make_load_series(100.0) # 100 kWh at $0.30
        dt_index = load_cheap.index

        meters = [
            MeterConfig(name="Gen", load_8760=_make_load_series(50.0),
                       tariff=tariff_cheap, is_generating=True),
            MeterConfig(name="Cheap", load_8760=load_cheap, tariff=tariff_cheap),
            MeterConfig(name="Expensive", load_8760=load_expensive, tariff=tariff_expensive),
        ]

        eff_price = compute_effective_export_price(meters, dt_index)

        # Equal loads → simple average: (0.10 + 0.30) / 2 = 0.20
        assert np.allclose(eff_price, 0.20, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Credit valuation
# ---------------------------------------------------------------------------
class TestCreditValuation:
    def test_basic_valuation(self):
        """Credits valued at flat rate."""
        tariff = _make_tariff(energy_rates=[0.20])
        agg_load = _make_load_series(100.0)

        agg_meter = MeterConfig(name="Agg", load_8760=agg_load, tariff=tariff)

        # Simple allocation: 500 kWh/month to single meter
        monthly_alloc = {m: {"Agg": 500.0} for m in range(1, 13)}
        allocation = AllocationResult(
            monthly_allocation=monthly_alloc,
            annual_fees=0.0,
            annual_allocated_kwh=6000.0,
            annual_unallocated_kwh=0.0,
        )

        monthly_credits = value_allocation_at_retail_rates(allocation, [agg_meter])

        # 6000 kWh * $0.20/kWh = $1200 total across all months
        annual_credit = sum(monthly_credits["Agg"].values())
        assert annual_credit == pytest.approx(1200.0, rel=0.01)
        # Each month: 500 kWh * $0.20 = $100
        for month in range(1, 13):
            assert monthly_credits["Agg"][month] == pytest.approx(100.0, rel=0.01)


# ---------------------------------------------------------------------------
# 6. Aggregate BillingResult (integration test via orchestrator)
# ---------------------------------------------------------------------------
class TestAggregateResult:
    def test_annual_totals_sum(self):
        """Aggregate result's annual_load_kwh = sum of all meters' loads."""
        gen_tariff = _make_tariff(energy_rates=[0.20], fixed_monthly=10.0)
        agg_tariff = _make_tariff(energy_rates=[0.25], fixed_monthly=15.0)

        gen_load = _make_load_series(100.0)
        agg_load = _make_load_series(80.0)

        # Production that exceeds generating meter load some hours
        dt_index = gen_load.index
        production = pd.Series(np.full(8760, 150.0), index=dt_index, name="ac_watts")
        export_rates = pd.Series(np.zeros(8760), index=dt_index, name="export_rate_per_kwh")

        profile = NemAProfile(
            utility="PG&E",
            meters=[
                MeterConfig(name="Gen", load_8760=gen_load, tariff=gen_tariff, is_generating=True),
                MeterConfig(name="Agg", load_8760=agg_load, tariff=agg_tariff),
            ],
            nem_regime="NEM-1",
            nbc_rate=0.0,
            nsc_rate=0.04,
            billing_option="ABO",
        )

        result = run_aggregation_simulation(
            profile=profile,
            production_8760=production,
            export_rates_8760=export_rates,
        )

        # Annual load should be sum of both meters
        expected_load = float(gen_load.sum()) + float(agg_load.sum())
        assert result.annual_load_kwh == pytest.approx(expected_load, rel=0.01)

        # Regime string should indicate NEM-A
        assert result.nem_regime == "NEM-A (NEM-1)"

    def test_nem_a_regime_string_nem2(self):
        """Verify regime string for NEM-2."""
        tariff = _make_tariff()
        gen_load = _make_load_series(100.0)
        agg_load = _make_load_series(50.0)

        dt_index = gen_load.index
        production = pd.Series(np.full(8760, 120.0), index=dt_index, name="ac_watts")
        export_rates = pd.Series(np.zeros(8760), index=dt_index, name="export_rate_per_kwh")

        profile = NemAProfile(
            utility="SCE",
            meters=[
                MeterConfig(name="Gen", load_8760=gen_load, tariff=tariff, is_generating=True),
                MeterConfig(name="Agg", load_8760=agg_load, tariff=tariff),
            ],
            nem_regime="NEM-2",
            nbc_rate=0.03,
            nsc_rate=0.04,
            billing_option="ABO",
        )

        result = run_aggregation_simulation(
            profile=profile,
            production_8760=production,
            export_rates_8760=export_rates,
        )

        assert result.nem_regime == "NEM-A (NEM-2)"

    def test_savings_positive_with_solar(self):
        """NEM-A simulation should show positive savings when solar > gen meter load."""
        tariff = _make_tariff(energy_rates=[0.20])
        gen_load = _make_load_series(50.0)
        agg_load = _make_load_series(100.0)

        dt_index = gen_load.index
        # Production well exceeds generating meter's load
        production = pd.Series(np.full(8760, 120.0), index=dt_index, name="ac_watts")
        export_rates = pd.Series(np.zeros(8760), index=dt_index, name="export_rate_per_kwh")

        profile = NemAProfile(
            utility="PG&E",
            meters=[
                MeterConfig(name="Gen", load_8760=gen_load, tariff=tariff, is_generating=True),
                MeterConfig(name="Agg", load_8760=agg_load, tariff=tariff),
            ],
            nem_regime="NEM-1",
            nbc_rate=0.0,
            nsc_rate=0.04,
            billing_option="ABO",
        )

        result = run_aggregation_simulation(
            profile=profile,
            production_8760=production,
            export_rates_8760=export_rates,
        )

        assert result.annual_savings > 0
        assert result.annual_bill_with_solar < result.annual_bill_without_solar


# ---------------------------------------------------------------------------
# 7. NEM-A regime string works with build_annual_projection
# ---------------------------------------------------------------------------
class TestProjectionCompatibility:
    def test_nema_regime_treated_as_tou_netted(self):
        """build_annual_projection should use TOU-netted path for NEM-A regimes."""
        from modules.outputs import build_annual_projection

        tariff = _make_tariff(energy_rates=[0.20])
        gen_load = _make_load_series(100.0)
        agg_load = _make_load_series(50.0)

        dt_index = gen_load.index
        production = pd.Series(np.full(8760, 130.0), index=dt_index, name="ac_watts")
        export_rates = pd.Series(np.zeros(8760), index=dt_index, name="export_rate_per_kwh")

        profile = NemAProfile(
            utility="PG&E",
            meters=[
                MeterConfig(name="Gen", load_8760=gen_load, tariff=tariff, is_generating=True),
                MeterConfig(name="Agg", load_8760=agg_load, tariff=tariff),
            ],
            nem_regime="NEM-1",
            nbc_rate=0.0,
            nsc_rate=0.04,
            billing_option="ABO",
        )

        result = run_aggregation_simulation(
            profile=profile,
            production_8760=production,
            export_rates_8760=export_rates,
        )

        # Should not raise — NEM-A regime handled
        proj_df = build_annual_projection(
            result=result,
            system_cost=100000.0,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=5,
            nem_regime_1="NEM-A (NEM-1)",
        )

        assert len(proj_df) == 5
        assert "Annual Savings ($)" in proj_df.columns
