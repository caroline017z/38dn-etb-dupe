"""
NEM-A (NEM Aggregation) billing module.

Supports Legacy NEM-A (NEM-1/NEM-2 retail TOU netting) where one generating
meter shares excess export credits across multiple aggregated consuming meters
on the same or contiguous property. Each meter may have a different tariff.

The orchestrator runs billing per-meter independently, computes NEM-A allocation,
then collapses results into a synthetic aggregate BillingResult that all
downstream code (outputs, projections, proposal, Excel export) consumes unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import cast

from .tariff import TariffSchedule
from .billing import (
    BillingResult,
    _assemble_billing_result,
    run_billing_simulation,
    _build_hourly_energy_rates,
    simulate_year_under_billing_option,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class MeterConfig:
    """Configuration for a single meter in a NEM-A aggregation group."""
    name: str                    # e.g. "Main Office", "Warehouse"
    load_8760: pd.Series         # hourly consumption (kWh)
    tariff: TariffSchedule       # this meter's rate schedule
    is_generating: bool = False  # True for the PV/ESS meter


@dataclass
class NemAProfile:
    """Complete NEM-A aggregation profile."""
    utility: str                 # "PG&E" | "SCE" | "SDG&E"
    meters: list[MeterConfig]
    nem_regime: str              # "NEM-1" or "NEM-2"
    nbc_rate: float              # NEM-2 only
    nsc_rate: float
    billing_option: str          # "ABO" or "MBO"


@dataclass
class AllocationResult:
    """Result of NEM-A credit allocation across meters."""
    monthly_allocation: dict[int, dict[str, float]]  # month -> {meter_name: kwh}
    annual_fees: float
    annual_allocated_kwh: float
    annual_unallocated_kwh: float


# ---------------------------------------------------------------------------
# NEM-A Fee Constants (per utility)
# ---------------------------------------------------------------------------
NEMA_FEES = {
    "PG&E":  {"setup_per_meter": 25.0, "setup_cap": 500.0, "monthly_per_meter": 5.0},
    "SCE":   {"setup_per_meter": 16.0, "setup_cap": None,   "monthly_per_meter": 2.70},
    "SDG&E": {"setup_per_meter": 20.0, "setup_cap": None,   "monthly_per_meter": 3.00},
}


def compute_nema_fees(utility: str, num_aggregated_meters: int) -> dict:
    """Compute NEM-A setup and annual admin fees for a utility.

    Args:
        utility: Utility name (PG&E, SCE, SDG&E)
        num_aggregated_meters: Number of aggregated (non-generating) meters

    Returns:
        dict with keys: setup_cost, monthly_admin, annual_admin
    """
    fee_info = NEMA_FEES.get(utility, NEMA_FEES["PG&E"])
    setup_per = fee_info["setup_per_meter"]
    setup_cap = fee_info["setup_cap"]
    monthly_per = fee_info["monthly_per_meter"]

    setup_cost = setup_per * num_aggregated_meters
    if setup_cap is not None:
        setup_cost = min(setup_cost, setup_cap)

    monthly_admin = monthly_per * num_aggregated_meters
    annual_admin = monthly_admin * 12

    return {
        "setup_cost": setup_cost,
        "monthly_admin": monthly_admin,
        "annual_admin": annual_admin,
    }


# ---------------------------------------------------------------------------
# Allocation Algorithm
# ---------------------------------------------------------------------------
def compute_monthly_allocation(
    gen_monthly_exports: dict[int, float],
    agg_meters: list[MeterConfig],
) -> AllocationResult:
    """Allocate generating meter's monthly exports to aggregated meters.

    Uses cumulative proportional allocation: each month, the total cumulative
    exports are split proportionally by each meter's cumulative consumption,
    then incremental allocation for the month is derived.

    Args:
        gen_monthly_exports: {month_num (1-12): export_kwh}
        agg_meters: List of aggregated (non-generating) MeterConfig objects

    Returns:
        AllocationResult with monthly allocations per meter
    """
    monthly_allocation: dict[int, dict[str, float]] = {}
    cumulative_exports = 0.0
    cumulative_consumption = {m.name: 0.0 for m in agg_meters}
    cumulative_allocated = {m.name: 0.0 for m in agg_meters}

    total_allocated = 0.0
    total_unallocated = 0.0

    for month in range(1, 13):
        month_exports = gen_monthly_exports.get(month, 0.0)
        cumulative_exports += month_exports

        # Compute each meter's consumption for this month
        month_consumption = {}
        for m in agg_meters:
            dt_index = cast(pd.DatetimeIndex, m.load_8760.index)
            month_mask = dt_index.month == month
            month_consumption[m.name] = float(m.load_8760[month_mask].sum())
            cumulative_consumption[m.name] += month_consumption[m.name]

        total_cum_consumption = sum(cumulative_consumption.values())

        month_alloc: dict[str, float] = {}
        month_total_alloc = 0.0

        if total_cum_consumption > 0 and cumulative_exports > 0:
            for m in agg_meters:
                # Target cumulative allocation
                target_cum = cumulative_exports * cumulative_consumption[m.name] / total_cum_consumption
                # Incremental for this month
                delta = max(0.0, target_cum - cumulative_allocated[m.name])
                # Cap: no more than this meter's monthly consumption
                delta = min(delta, month_consumption[m.name])
                month_alloc[m.name] = delta
                cumulative_allocated[m.name] += delta
                month_total_alloc += delta
        else:
            for m in agg_meters:
                month_alloc[m.name] = 0.0

        monthly_allocation[month] = month_alloc
        total_allocated += month_total_alloc
        total_unallocated += max(0.0, month_exports - month_total_alloc)

    total_exports = sum(gen_monthly_exports.values())
    return AllocationResult(
        monthly_allocation=monthly_allocation,
        annual_fees=0.0,  # Set by caller
        annual_allocated_kwh=total_allocated,
        annual_unallocated_kwh=max(0.0, total_exports - total_allocated),
    )


# ---------------------------------------------------------------------------
# Credit Valuation
# ---------------------------------------------------------------------------
def value_allocation_at_retail_rates(
    allocation: AllocationResult,
    agg_meters: list[MeterConfig],
) -> dict[str, dict[int, float]]:
    """Value allocated kWh at each receiving meter's consumption-weighted average TOU rate.

    For each meter and month, compute the consumption-weighted average TOU rate,
    then multiply by the allocated kWh for that month.

    Args:
        allocation: AllocationResult from compute_monthly_allocation()
        agg_meters: List of aggregated MeterConfig objects

    Returns:
        dict mapping meter_name -> {month (1-12) -> credit value ($)}
    """
    monthly_credit_values: dict[str, dict[int, float]] = {}

    for m in agg_meters:
        dt_index = cast(pd.DatetimeIndex, m.load_8760.index)
        hourly_rates = _build_hourly_energy_rates(m.tariff, dt_index)
        load_arr = np.asarray(m.load_8760.values)

        meter_monthly: dict[int, float] = {}
        for month in range(1, 13):
            alloc_kwh = allocation.monthly_allocation[month].get(m.name, 0.0)
            if alloc_kwh <= 0:
                meter_monthly[month] = 0.0
                continue

            month_mask = dt_index.month == month
            month_load = load_arr[month_mask]
            month_rates = hourly_rates[month_mask]

            # Consumption-weighted average rate for this month
            total_load = month_load.sum()
            if total_load > 0:
                avg_rate = float(np.dot(month_load, month_rates) / total_load)
            else:
                avg_rate = float(month_rates.mean()) if len(month_rates) > 0 else 0.0

            meter_monthly[month] = alloc_kwh * avg_rate

        monthly_credit_values[m.name] = meter_monthly

    return monthly_credit_values


# ---------------------------------------------------------------------------
# Effective Export Price (for battery dispatch)
# ---------------------------------------------------------------------------
def compute_effective_export_price(
    meters: list[MeterConfig],
    dt_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Compute 8760 hourly export price as load-weighted average of all
    aggregated meters' TOU rates.

    This gives the battery the correct optimization signal — exports are
    worth more when they offset expensive peak consumption across the portfolio.

    Args:
        meters: All MeterConfig objects in the NEM-A group
        dt_index: DatetimeIndex for the simulation year

    Returns:
        np.ndarray of shape (8760,) with effective $/kWh export prices
    """
    agg_meters = [m for m in meters if not m.is_generating]
    if not agg_meters:
        return np.zeros(len(dt_index))

    total_load = np.zeros(len(dt_index))
    weighted_rate = np.zeros(len(dt_index))

    for m in agg_meters:
        load_arr = np.asarray(m.load_8760.values)
        rates = _build_hourly_energy_rates(m.tariff, dt_index)
        total_load += load_arr
        weighted_rate += load_arr * rates

    return np.where(total_load > 0, weighted_rate / total_load, 0.0)


# ---------------------------------------------------------------------------
# Aggregate BillingResult Builder
# ---------------------------------------------------------------------------
def _build_aggregate_result(
    gen_result: BillingResult,
    agg_results: dict[str, BillingResult],
    allocation: AllocationResult,
    monthly_credit_values: dict[str, dict[int, float]],
    nema_fees: dict,
    nem_regime: str,
) -> BillingResult:
    """Construct a synthetic aggregate BillingResult from per-meter results.

    Sums costs across all meters and subtracts allocated credits from
    receiving meters, producing a single BillingResult that downstream
    code (outputs, projections, export) can consume unchanged.

    Args:
        gen_result: BillingResult for the generating meter (with PV + optional battery)
        agg_results: {meter_name: BillingResult} for each aggregated meter (load-only)
        allocation: AllocationResult from compute_monthly_allocation()
        monthly_credit_values: {meter_name: {month: credit_$}} from value_allocation_at_retail_rates()
        nema_fees: Fee dict from compute_nema_fees()
        nem_regime: "NEM-1" or "NEM-2" (will be prefixed with "NEM-A")

    Returns:
        Synthetic aggregate BillingResult
    """
    # --- Hourly detail: use the generating meter's (it has solar/battery columns) ---
    hourly_detail = gen_result.hourly_detail.copy()

    # --- Monthly summary: sum across all meters, subtract allocated credits ---
    gen_monthly = gen_result.monthly_summary.copy()
    all_monthly_rows = []

    for _, gen_row in gen_monthly.iterrows():
        month = int(gen_row["month"])
        row = dict(gen_row)

        # Add aggregated meter costs
        for name, agg_res in agg_results.items():
            agg_month_row = agg_res.monthly_summary[
                agg_res.monthly_summary["month"] == month
            ].iloc[0]
            row["load_kwh"] += agg_month_row["load_kwh"]
            row["energy_cost"] += agg_month_row["energy_cost"]
            row["flat_demand_charge"] += agg_month_row["flat_demand_charge"]
            row["tou_demand_charge"] += agg_month_row["tou_demand_charge"]
            row["total_demand_charge"] += agg_month_row["total_demand_charge"]
            row["fixed_charge"] += agg_month_row["fixed_charge"]
            if "nbc_charge" in row and "nbc_charge" in agg_month_row:
                row["nbc_charge"] += agg_month_row.get("nbc_charge", 0.0)

        # Add NEM-A monthly admin fee
        row["fixed_charge"] += nema_fees["monthly_admin"]

        # Subtract allocated credits for this month (valued at receiving meter rates)
        month_credit = 0.0
        for meter_name in allocation.monthly_allocation[month]:
            if meter_name in monthly_credit_values:
                month_credit += monthly_credit_values[meter_name].get(month, 0.0)

        row["export_credit"] = round(row.get("export_credit", 0.0) + month_credit, 2)

        # Recompute net bill
        row["net_bill"] = round(
            row["energy_cost"]
            + row["total_demand_charge"]
            + row["fixed_charge"]
            + row.get("nbc_charge", 0.0)
            - row["export_credit"],
            2,
        )

        all_monthly_rows.append(row)

    monthly_summary = pd.DataFrame(all_monthly_rows)

    regime_str = f"NEM-A ({nem_regime})"

    # --- Apply aggregate-level MBO/ABO floors + min_monthly_charge to per-month
    # net_bills. The recomputation above sums gross components and ignores
    # billing-option floors at the aggregate level — for over-sized PV under
    # MBO that produces deeply negative monthly bills which don't match what
    # the customer actually pays. The simulator applies the same floor +
    # banking + NSC-trueup logic single-meter NEM-1/2 uses, lifted to the
    # aggregate level. Y1 and the Y>1 projection now share methodology.
    _agg_min_charge = gen_result.min_monthly_charge
    _agg_billing_option = gen_result.billing_option

    # NSC must be assessed on the AGGREGATE net surplus, not the generating
    # meter alone. The gen meter (small load, large PV) shows a large standalone
    # surplus, but in NEM-A its exports offset the other meters' load — so the
    # group frequently nets to a deficit with no surplus to true up. NSC is
    # linear in surplus kWh at the same ACC rate, so scale the gen meter's NSC
    # by aggregate-surplus / gen-surplus. Only the gen meter exports (the
    # aggregated meters are load-only), so this ratio is in [0, 1]: it zeroes
    # the clawback when the aggregate is a net importer and reduces it
    # proportionally when the group retains partial surplus.
    _gen_surplus_kwh = max(0.0, gen_result.annual_export_kwh - gen_result.annual_import_kwh)
    _agg_import_kwh = gen_result.annual_import_kwh + sum(
        r.annual_import_kwh for r in agg_results.values()
    )
    _agg_surplus_kwh = max(0.0, gen_result.annual_export_kwh - _agg_import_kwh)
    _nsc_scale = (_agg_surplus_kwh / _gen_surplus_kwh) if _gen_surplus_kwh > 0 else 0.0

    _agg_components = []
    for _, _r in monthly_summary.iterrows():
        _agg_components.append({
            "energy": float(_r["energy_cost"]),
            "export_credit": float(_r["export_credit"]),
            "demand": float(_r["total_demand_charge"]),
            "fixed": float(_r["fixed_charge"]),
            "nbc": float(_r.get("nbc_charge", 0.0)),
            # NSC clawback rides on month-12 (carried over from gen_result via
            # `row = dict(gen_row)`); the simulator layers it over the floored
            # month-12 bill, matching _build_monthly_nem12. Scaled to the
            # aggregate surplus (see above).
            "nsc_adj": float(_r.get("nsc_adjustment", 0.0)) * _nsc_scale,
        })
    _agg_net_bills = simulate_year_under_billing_option(
        _agg_components, regime_str, _agg_billing_option, _agg_min_charge,
    )
    monthly_summary["net_bill"] = [round(v, 2) for v in _agg_net_bills]

    # --- Annual totals ---
    all_results = [gen_result] + list(agg_results.values())

    annual_load = sum(r.annual_load_kwh for r in all_results)
    annual_solar = gen_result.annual_solar_kwh
    annual_import = sum(r.annual_import_kwh for r in all_results)
    annual_export = gen_result.annual_export_kwh

    annual_nbc = float(monthly_summary["nbc_charge"].sum()) if "nbc_charge" in monthly_summary.columns else 0.0

    # Baseline: sum of all meters' no-solar bills
    annual_bill_no_solar = sum(r.annual_bill_without_solar for r in all_results)

    # NSC adjustment from the generating meter, scaled to the aggregate net
    # surplus (see _nsc_scale above) so a net-importing group is not charged NSC.
    annual_nsc_adj = gen_result.annual_nsc_adjustment * _nsc_scale

    # Monthly baseline details: sum across all meters
    monthly_baseline = None
    if gen_result.monthly_baseline_details is not None:
        monthly_baseline = []
        for month_idx in range(12):
            _gen_bd = gen_result.monthly_baseline_details[month_idx]
            combined = {
                "energy": _gen_bd["energy"],
                "demand": _gen_bd["demand"],
                "fixed": _gen_bd["fixed"],
                "peak_demand_kw": _gen_bd.get("peak_demand_kw", 0),
                "total": _gen_bd["total"],
            }
            for agg_res in agg_results.values():
                if agg_res.monthly_baseline_details:
                    _agg_bd = agg_res.monthly_baseline_details[month_idx]
                    combined["energy"] += _agg_bd["energy"]
                    combined["demand"] += _agg_bd["demand"]
                    combined["fixed"] += _agg_bd["fixed"]
                    combined["peak_demand_kw"] = max(
                        combined["peak_demand_kw"], _agg_bd.get("peak_demand_kw", 0)
                    )
                    combined["total"] += _agg_bd["total"]
            # NEM-A admin fees are a cost of being on NEM-A (with solar),
            # NOT part of the no-solar baseline — do not add them here.
            monthly_baseline.append(combined)

    # Aggregate TOU energy/credit across all meters for projection use.
    # Generating meter has the solar TOU netting; aggregated meters are load-only
    # (their tou_annual_energy = full energy cost, tou_annual_credit = 0).
    # The allocated credits (from value_allocation_at_retail_rates) are added
    # to the credit side since they represent TOU-valued export credits
    # distributed to receiving meters.
    agg_tou_energy = gen_result.tou_annual_energy
    agg_tou_credit = gen_result.tou_annual_credit
    for agg_res in agg_results.values():
        agg_tou_energy += agg_res.tou_annual_energy
        agg_tou_credit += agg_res.tou_annual_credit
    # Add allocated credits (valued at retail TOU rates) to the credit side
    for meter_credits in monthly_credit_values.values():
        agg_tou_credit += sum(meter_credits.values())

    # Aggregate raw import energy cost across all meters for NEM-3 regime-switch.
    # Each meter's raw_annual_energy = sum(import_kwh * energy_rate) from hourly data.
    agg_raw_energy = gen_result.raw_annual_energy
    for agg_res in agg_results.values():
        agg_raw_energy += agg_res.raw_annual_energy

    # Per-month aggregate TOU split, derived from the aggregate monthly_summary.
    # signed = gross imports across meters − (gen exports + cross-meter allocated
    # credits). Splitting by sign gives the helper the same shape it gets from
    # single-meter NEM-1/2 (positive side, negative side), so Y>1 projection
    # can apply MBO/ABO floors at the aggregate level. Note: aggregate Y1
    # net_bill above doesn't itself apply aggregate-level MBO; Y>1 may diverge
    # from Y1 for portfolios where aggregate monthly net would have floored.
    agg_tou_monthly_energy: dict[int, float] = {}
    agg_tou_monthly_credit: dict[int, float] = {}
    for _, ms_row in monthly_summary.iterrows():
        m = int(ms_row["month"])
        signed = float(ms_row["energy_cost"]) - float(ms_row["export_credit"])
        if signed >= 0:
            agg_tou_monthly_energy[m] = signed
            agg_tou_monthly_credit[m] = 0.0
        else:
            agg_tou_monthly_energy[m] = 0.0
            agg_tou_monthly_credit[m] = abs(signed)

    return _assemble_billing_result(
        hourly_detail=hourly_detail,
        monthly_summary=monthly_summary,
        load=np.array([annual_load]),
        solar=np.array([annual_solar]),
        import_kwh=np.array([annual_import]),
        export_kwh=np.array([annual_export]),
        baseline_bill=annual_bill_no_solar,
        nem_regime=regime_str,
        tou_annual_energy=agg_tou_energy,
        tou_annual_credit=agg_tou_credit,
        tou_monthly_energy=agg_tou_monthly_energy,
        tou_monthly_credit=agg_tou_monthly_credit,
        monthly_baseline_details=monthly_baseline,
        raw_annual_energy=agg_raw_energy,
        annual_nbc=annual_nbc,
        annual_nsc_adj=annual_nsc_adj,
        billing_option=gen_result.billing_option,
        min_monthly_charge=gen_result.min_monthly_charge,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_aggregation_simulation(
    profile: NemAProfile,
    production_8760: pd.Series,
    export_rates_8760: pd.Series,
    battery_config=None,
    capacity_kwh: float = 0.0,
    monthly_dispatch: bool = False,
) -> BillingResult:
    """Run a full NEM-A aggregation simulation.

    Steps:
    1. Run billing for the generating meter (with PV + optional battery)
    2. Run billing for each aggregated meter (load only, no solar)
    3. Compute monthly allocation of generating meter's exports
    4. Value allocated credits at each receiving meter's retail rates
    5. Build and return a synthetic aggregate BillingResult

    Args:
        profile: NemAProfile with all meter configs and NEM parameters
        production_8760: Hourly solar production (kWh)
        export_rates_8760: Hourly export rates ($/kWh) — used as placeholder for NEM-1/2
        battery_config: Optional BatteryConfig for dispatch
        capacity_kwh: Battery nameplate capacity (kWh)
        monthly_dispatch: Whether to use monthly LP dispatch

    Returns:
        Synthetic aggregate BillingResult
    """
    # Identify generating and aggregated meters
    gen_meter = None
    agg_meters = []
    for m in profile.meters:
        if m.is_generating:
            gen_meter = m
        else:
            agg_meters.append(m)

    if gen_meter is None:
        raise ValueError("NEM-A profile must have exactly one generating meter")

    # NEM params
    nem_nbc = profile.nbc_rate if profile.nem_regime == "NEM-2" else 0.0
    nem_nsc = profile.nsc_rate if profile.nem_regime in ("NEM-1", "NEM-2") else 0.0
    nem_billing = profile.billing_option if profile.nem_regime in ("NEM-1", "NEM-2") else "ABO"

    # Step 1: Run billing for generating meter
    gen_result = run_billing_simulation(
        load_8760=gen_meter.load_8760,
        production_8760=production_8760,
        tariff=gen_meter.tariff,
        export_rates_8760=export_rates_8760,
        battery_config=battery_config,
        capacity_kwh=capacity_kwh,
        monthly_dispatch=monthly_dispatch,
        nem_regime=profile.nem_regime,
        nbc_rate=nem_nbc,
        nsc_rate=nem_nsc,
        billing_option=nem_billing,
    )

    # Step 2: Run billing for each aggregated meter (load only, no solar)
    agg_results: dict[str, BillingResult] = {}
    zero_production = pd.Series(
        np.zeros(8760), index=gen_meter.load_8760.index, name="ac_watts"
    )
    zero_export = pd.Series(
        np.zeros(8760), index=gen_meter.load_8760.index, name="export_rate_per_kwh"
    )

    for m in agg_meters:
        agg_results[m.name] = run_billing_simulation(
            load_8760=m.load_8760,
            production_8760=zero_production,
            tariff=m.tariff,
            export_rates_8760=zero_export,
            nem_regime=profile.nem_regime,
            nbc_rate=0.0,   # NBC only applies to the generating NEM meter
            nsc_rate=0.0,   # NSC only applies to the generating NEM meter
            billing_option=nem_billing,
        )

    # Step 3: Extract generating meter's monthly exports and compute allocation
    gen_monthly_exports = {}
    gen_ms = gen_result.monthly_summary
    for _, row in gen_ms.iterrows():
        gen_monthly_exports[int(row["month"])] = float(row["export_kwh"])

    allocation = compute_monthly_allocation(gen_monthly_exports, agg_meters)

    # Compute fees
    fees = compute_nema_fees(profile.utility, len(agg_meters))
    allocation.annual_fees = fees["annual_admin"]

    # Step 4: Value credits at receiving meter rates (monthly breakdown per meter)
    monthly_credit_values = value_allocation_at_retail_rates(allocation, agg_meters)

    # Step 5: Build aggregate result
    return _build_aggregate_result(
        gen_result=gen_result,
        agg_results=agg_results,
        allocation=allocation,
        monthly_credit_values=monthly_credit_values,
        nema_fees=fees,
        nem_regime=profile.nem_regime,
    )
