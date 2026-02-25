"""
ECC (electricitycostcalculator) billing adapter — alternative billing engine
using OpenEI tariff data via the CostCalculator class.

Provides the same BillingResult output as the custom engine so all downstream
code (projections, charts, downloads) works unchanged.
"""

import pandas as pd
import numpy as np

from electricitycostcalculator.openei_tariff.openei_tariff_analyzer import (
    OpenEI_tariff,
    tariff_struct_from_openei_data,
)
from electricitycostcalculator.cost_calculator.cost_calculator import CostCalculator
from electricitycostcalculator.cost_calculator.rate_structure import ChargeType

from .billing import BillingResult


def _build_tou_arrays(
    dt_index: pd.DatetimeIndex, tariff_data: list,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build 8760 energy_period and energy_rate arrays from OpenEI tariff data.

    Args:
        dt_index: Naive datetime index (8760 hours).
        tariff_data: OpenEI data_openei list (from JSON or API).

    Returns:
        (energy_period, energy_rate, peak_periods_frozenset)
    """
    tariff = tariff_data[0] if isinstance(tariff_data, list) else tariff_data
    weekday_sched = tariff.get("energyweekdayschedule")
    weekend_sched = tariff.get("energyweekendschedule")
    rate_struct = tariff.get("energyratestructure")

    n = len(dt_index)
    if not weekday_sched or not weekend_sched or not rate_struct:
        return np.full(n, -1, dtype=int), np.zeros(n), 0

    energy_period = np.empty(n, dtype=int)
    energy_rate = np.empty(n)

    for i in range(n):
        ts = dt_index[i]
        month_idx = ts.month - 1
        hour_idx = ts.hour
        sched = weekend_sched if ts.dayofweek >= 5 else weekday_sched
        period = sched[month_idx][hour_idx]
        energy_period[i] = period
        if period < len(rate_struct) and rate_struct[period]:
            energy_rate[i] = rate_struct[period][0].get("rate", 0.0)
        else:
            energy_rate[i] = 0.0

    # Identify peak periods from the weekday schedule:
    # Within each month, the period with fewer hours is "peak."
    peak_periods: set[int] = set()
    for month_row in weekday_sched:
        period_counts: dict[int, int] = {}
        for p in month_row:
            period_counts[p] = period_counts.get(p, 0) + 1
        if len(period_counts) >= 2:
            min_hours = min(period_counts.values())
            for p, cnt in period_counts.items():
                if cnt == min_hours:
                    peak_periods.add(p)

    # Fallback: if no peak found, use the highest-rate period
    if not peak_periods:
        max_rate = 0.0
        best = 0
        for idx, tiers in enumerate(rate_struct):
            if tiers and tiers[0].get("rate", 0.0) > max_rate:
                max_rate = tiers[0]["rate"]
                best = idx
        peak_periods = {best}

    return energy_period, energy_rate, frozenset(peak_periods)


def fetch_and_populate_ecc_tariff(
    utility_id: int,
    sector: str = "commercial",
    tariff_rate_filter: str = "",
    distrib_level: str = "Secondary",
    phase_wiring: str = "Single",
    tou: bool = True,
    pdp: bool = False,
) -> tuple[CostCalculator, list]:
    """
    Fetch tariff data from OpenEI API and populate a CostCalculator.

    Returns (cost_calculator, tariff_data_list) where tariff_data_list is
    the raw OpenEI tariff entries for display/metadata purposes.
    """
    openei_tariff = OpenEI_tariff(
        utility_id=utility_id,
        sector=sector.lower(),
        tariff_rate_of_interest=tariff_rate_filter.lower() if tariff_rate_filter else "",
        distrib_level_of_interest=distrib_level,
        phasewing=phase_wiring if phase_wiring != "None" else None,
        tou=tou,
        pdp=pdp,
    )
    openei_tariff.call_api()

    if not openei_tariff.data_openei:
        raise ValueError(
            "No tariff data returned from OpenEI. "
            "Try adjusting the sector, rate filter, or distribution level."
        )

    calculator = CostCalculator()
    tariff_struct_from_openei_data(openei_tariff, calculator)

    return calculator, openei_tariff.data_openei


def load_ecc_tariff_from_json(json_path: str) -> tuple[CostCalculator, list]:
    """
    Load tariff data from a local JSON file and populate a CostCalculator.

    Returns (cost_calculator, tariff_data_list).
    """
    openei_tariff = OpenEI_tariff()
    result = openei_tariff.read_from_json(filename=json_path)
    if result != 0:
        raise ValueError(
            f"Failed to read tariff JSON (code {result}). "
            "Check that the file is a valid OpenEI tariff JSON."
        )

    if not openei_tariff.data_openei:
        raise ValueError("JSON file loaded but contains no tariff data.")

    calculator = CostCalculator()
    tariff_struct_from_openei_data(openei_tariff, calculator)

    return calculator, openei_tariff.data_openei


def run_ecc_billing_simulation(
    load_8760: pd.Series,
    production_8760: pd.Series,
    cost_calculator: CostCalculator,
    export_rates_8760: pd.Series,
    tariff_data: list | None = None,
) -> BillingResult:
    """
    Run billing simulation using the ECC CostCalculator engine.

    Produces a BillingResult compatible with all downstream code.
    Export credits are calculated separately (ECC does not handle exports).
    """
    load = load_8760.values
    solar = np.asarray(production_8760)
    export_rates = export_rates_8760.values
    n_hours = len(load)
    assert n_hours == 8760, f"Expected 8760 hours, got {n_hours}"

    # --- Hour-by-hour netting ---
    net_kwh = load - solar
    import_kwh = np.maximum(net_kwh, 0.0)
    export_kwh = np.maximum(-net_kwh, 0.0)
    export_credit = export_kwh * export_rates

    # --- Build tz-aware datetime index for ECC (US/Pacific) ---
    _start_year = load_8760.index[0].year
    dt_index = pd.date_range(
        start=f"{_start_year}-01-01", periods=8760, freq="h", tz="US/Pacific"
    )

    # --- ECC expects Wh, not kWh ---
    import_wh = import_kwh * 1000.0
    load_wh = load * 1000.0

    # --- With-solar bill (monthly detailed) ---
    df_solar = pd.DataFrame({"consumption": import_wh}, index=dt_index)
    bill_solar = cost_calculator.compute_bill(
        df_solar, column_data="consumption", monthly_detailed=True
    ) or {}

    # --- Baseline bill (no solar — full load) ---
    df_baseline = pd.DataFrame({"consumption": load_wh}, index=dt_index)
    bill_baseline = cost_calculator.compute_bill(
        df_baseline, column_data="consumption", monthly_detailed=False
    )

    # --- Build TOU arrays early so monthly summary can split peak/off-peak ---
    dt_naive = pd.date_range(start=f"{_start_year}-01-01", periods=8760, freq="h")
    if tariff_data:
        energy_period, energy_rate, peak_periods = _build_tou_arrays(
            dt_naive, tariff_data,
        )
    else:
        energy_period = np.full(n_hours, -1, dtype=int)
        energy_rate = np.zeros(n_hours)
        peak_periods = frozenset()

    # Pre-compute per-hour is_peak boolean mask
    if isinstance(peak_periods, (set, frozenset)) and peak_periods:
        _is_peak = np.isin(energy_period, list(peak_periods))
    else:
        _is_peak = np.zeros(n_hours, dtype=bool)

    # --- Parse monthly ECC bills into our monthly_summary format ---
    monthly_rows = []
    for month_num in range(1, 13):
        month_key = f"{_start_year}-{month_num:02d}"
        month_mask = dt_index.month == month_num
        month_mask_naive = dt_naive.month == month_num

        m_load = float(load[month_mask].sum())
        m_solar = float(solar[month_mask].sum())
        m_import = float(import_kwh[month_mask].sum())
        m_export = float(export_kwh[month_mask].sum())
        m_export_credit = float(export_credit[month_mask].sum())

        # Split export into peak / off-peak using TOU periods
        _m_peak = month_mask_naive & _is_peak
        _m_offpeak = month_mask_naive & ~_is_peak
        m_export_peak = float(export_kwh[_m_peak].sum())
        m_export_offpeak = float(export_kwh[_m_offpeak].sum())

        # Extract ECC charges for this month
        m_energy_cost = 0.0
        m_demand_cost = 0.0
        m_fixed_cost = 0.0
        m_peak_kw = 0.0

        if month_key in bill_solar:
            month_bill = bill_solar[month_key]
            m_fixed_cost, m_energy_cost, m_demand_cost, m_peak_kw = (
                _extract_monthly_charges(month_bill)
            )

        m_net_bill = m_energy_cost + m_demand_cost + m_fixed_cost - m_export_credit
        m_net_bill = max(m_net_bill, 0.0)

        monthly_rows.append({
            "month": month_num,
            "load_kwh": round(m_load, 1),
            "solar_kwh": round(m_solar, 1),
            "import_kwh": round(m_import, 1),
            "export_kwh": round(m_export, 1),
            "peak_demand_kw": round(m_peak_kw, 2),
            "export_peak_kwh": round(m_export_peak, 1),
            "export_offpeak_kwh": round(m_export_offpeak, 1),
            "energy_cost": round(m_energy_cost, 2),
            "flat_demand_charge": 0.0,
            "tou_demand_charge": round(m_demand_cost, 2),
            "total_demand_charge": round(m_demand_cost, 2),
            "fixed_charge": round(m_fixed_cost, 2),
            "export_credit": round(m_export_credit, 2),
            "nbc_charge": 0.0,
            "nsc_adjustment": 0.0,
            "net_bill": round(m_net_bill, 2),
        })

    monthly_summary = pd.DataFrame(monthly_rows)

    # --- Baseline (no-solar) annual totals ---
    baseline_total, baseline_by_type, _ = cost_calculator.print_aggregated_bill(
        bill_baseline, verbose=False
    )

    # --- Parse monthly baseline bill for Indexed Tariff support ---
    # Re-compute baseline with monthly detail
    bill_baseline_monthly = cost_calculator.compute_bill(
        df_baseline, column_data="consumption", monthly_detailed=True
    ) or {}
    monthly_baseline_details = []
    for month_num in range(1, 13):
        month_key = f"{_start_year}-{month_num:02d}"
        b_fixed = 0.0
        b_energy = 0.0
        b_demand = 0.0
        if month_key in bill_baseline_monthly:
            b_fixed, b_energy, b_demand, _ = _extract_monthly_charges(
                bill_baseline_monthly[month_key]
            )
        monthly_baseline_details.append({
            "energy": b_energy,
            "demand": b_demand,
            "fixed": b_fixed,
            "total": b_energy + b_demand + b_fixed,
        })

    # --- Build hourly detail DataFrame ---
    # dt_naive and energy_period/energy_rate already built above (before monthly loop).
    # For the no-tariff fallback, backfill energy_rate from monthly averages.
    if not tariff_data:
        for row in monthly_rows:
            month_mask = dt_naive.month == row["month"]
            if row["import_kwh"] > 0:
                energy_rate[month_mask] = row["energy_cost"] / row["import_kwh"]

    hourly_detail = pd.DataFrame({
        "load_kwh": load,
        "solar_kwh": solar,
        "net_kwh": net_kwh,
        "import_kwh": import_kwh,
        "export_kwh": export_kwh,
        "energy_period": energy_period,
        "energy_rate": energy_rate,
        "energy_cost": import_kwh * energy_rate,
        "export_credit": export_credit,
    }, index=dt_naive)
    hourly_detail.index.name = "datetime"

    # --- Annual totals ---
    annual_load = float(load.sum())
    annual_solar = float(solar.sum())
    annual_import = float(import_kwh.sum())
    annual_export = float(export_kwh.sum())
    annual_energy_cost = float(monthly_summary["energy_cost"].sum())
    annual_demand_cost = float(monthly_summary["total_demand_charge"].sum())
    annual_fixed_cost = float(monthly_summary["fixed_charge"].sum())
    annual_export_credit = float(monthly_summary["export_credit"].sum())
    annual_bill_solar = float(monthly_summary["net_bill"].sum())
    annual_savings = baseline_total - annual_bill_solar
    savings_pct = (annual_savings / baseline_total * 100) if baseline_total > 0 else 0.0

    return BillingResult(
        hourly_detail=hourly_detail,
        monthly_summary=monthly_summary,
        annual_load_kwh=annual_load,
        annual_solar_kwh=annual_solar,
        annual_import_kwh=annual_import,
        annual_export_kwh=annual_export,
        annual_energy_cost=annual_energy_cost,
        annual_demand_cost=annual_demand_cost,
        annual_fixed_cost=annual_fixed_cost,
        annual_export_credit=annual_export_credit,
        annual_bill_with_solar=annual_bill_solar,
        annual_bill_without_solar=baseline_total,
        annual_savings=annual_savings,
        savings_pct=savings_pct,
        monthly_baseline_details=monthly_baseline_details,
    )


def _extract_monthly_charges(month_bill: dict) -> tuple[float, float, float, float]:
    """
    Extract (fixed_cost, energy_cost, demand_cost, peak_kw) from a single
    month's ECC bill dict.
    """
    fixed_cost = 0.0
    energy_cost = 0.0
    demand_cost = 0.0
    peak_kw = 0.0

    for label, value in month_bill.items():
        if "fix_charge" in label:
            # Fixed charges: (metric, cost) tuple
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                fixed_cost += float(value[1])

        elif "energy_charge" in label or "energy_credit" in label:
            # Energy charges: (kwh, cost) tuple
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                energy_cost += float(value[1])

        elif "demand_charge" in label or "demand_credit" in label:
            # Demand charges: dict keyed by price_per_kw
            if isinstance(value, dict):
                for price_per_kw, demand_info in value.items():
                    if isinstance(demand_info, dict):
                        kw = float(demand_info.get("max-demand", 0))
                        demand_cost += float(price_per_kw) * kw
                        peak_kw = max(peak_kw, kw)
            elif isinstance(value, (tuple, list)) and len(value) >= 2:
                demand_cost += float(value[1])

    return fixed_cost, energy_cost, demand_cost, peak_kw
