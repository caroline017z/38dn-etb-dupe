"""
Core billing engine — hourly import/export calculation and monthly bill aggregation.

Performs hour-by-hour netting of solar production against load,
optionally runs battery dispatch (LP) to reshape grid flows,
applies energy charges per TOU period, export credits per ACC rates,
and combines with demand charges for total bill calculation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import cast
from .tariff import TariffSchedule, get_energy_rate, get_energy_period
from .demand import calculate_monthly_demand_charges


@dataclass
class BillingResult:
    """Complete billing simulation results."""

    # Hourly detail (8760 rows)
    hourly_detail: pd.DataFrame

    # Monthly summary (12 rows)
    monthly_summary: pd.DataFrame

    # Annual totals
    annual_load_kwh: float
    annual_solar_kwh: float
    annual_import_kwh: float
    annual_export_kwh: float
    annual_energy_cost: float
    annual_demand_cost: float
    annual_fixed_cost: float
    annual_export_credit: float
    annual_bill_with_solar: float
    annual_bill_without_solar: float
    annual_savings: float
    savings_pct: float

    # NEM regime fields
    annual_nbc_cost: float = 0.0           # Total NBC charges (NEM-2 only)
    annual_nsc_adjustment: float = 0.0     # NSC reduction at true-up (NEM-1/NEM-2)
    nem_regime: str = "NEM-3"              # Which regime was used

    # TOU-netted annual values (NEM-1/2 only; 0.0 for NEM-3)
    tou_annual_energy: float = 0.0   # positive side of TOU per-period netting
    tou_annual_credit: float = 0.0   # negative side of TOU per-period netting

    # Raw import energy cost: sum(import_kwh * energy_rate) across all hours.
    # Always represents gross import cost regardless of TOU netting or billing option.
    # Used by projection for NEM-3 regime-switch energy cost baseline.
    raw_annual_energy: float = 0.0

    # Monthly baseline breakdown (no-solar bill components per month)
    monthly_baseline_details: list[dict] | None = None  # 12 dicts: {energy, demand, fixed, total}


def _build_schedule_arrays(
    tariff: TariffSchedule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2-D NumPy schedule and rate arrays from tariff for vectorised lookups.

    Returns:
        (weekday_sched, weekend_sched, period_rates)
        - weekday_sched / weekend_sched: int arrays of shape (12, 24) — period index
        - period_rates: 1-D float array mapping period index -> $/kWh
    """
    wkday = tariff.energy_weekday_schedule
    wkend = tariff.energy_weekend_schedule or wkday
    weekday_sched = np.asarray(wkday, dtype=int)   # (12, 24)
    weekend_sched = np.asarray(wkend, dtype=int)    # (12, 24)

    n_periods = max(int(weekday_sched.max()), int(weekend_sched.max())) + 1
    period_rates = np.zeros(n_periods)
    if tariff.energy_rate_structure:
        for pidx, tiers in enumerate(tariff.energy_rate_structure):
            if pidx < n_periods and tiers:
                period_rates[pidx] = tiers[0]["effective_rate"]

    return weekday_sched, weekend_sched, period_rates


def _vectorized_period_and_rate(
    tariff: TariffSchedule, dt_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised energy-period and energy-rate lookup for every hour.

    Returns:
        (energy_period, energy_rate) — both 1-D arrays of length len(dt_index)
    """
    weekday_sched, weekend_sched, period_rate_map = _build_schedule_arrays(tariff)

    months = dt_index.month.values - 1        # 0-indexed
    hours = dt_index.hour.values
    is_weekend = dt_index.weekday.values >= 5

    # Fancy-index both schedules, then select per-hour
    periods_wkday = weekday_sched[months, hours]
    periods_wkend = weekend_sched[months, hours]
    energy_period = np.where(is_weekend, periods_wkend, periods_wkday)

    energy_rate = period_rate_map[energy_period]
    return energy_period, energy_rate


def _build_hourly_energy_rates(
    tariff: TariffSchedule, dt_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Return an array of $/kWh energy rates for every hour in dt_index."""
    _, energy_rate = _vectorized_period_and_rate(tariff, dt_index)
    return energy_rate


def _build_demand_lp_inputs(
    tariff: TariffSchedule, dt_index: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Extract demand-charge masks and prices from tariff for the dispatch LP.

    Returns (demand_window_masks, demand_prices) where keys are
    ``"flat"`` and ``"tou_<period_idx>"`` strings.
    """
    n = len(dt_index)
    masks: dict[str, np.ndarray] = {}
    prices: dict[str, float] = {}

    # ---- flat (non-coincident) demand ----
    if tariff.demand_flat_structure:
        masks["flat"] = np.ones(n, dtype=bool)
        # Use the maximum flat demand rate across all seasonal periods as LP price.
        # The LP needs a single scalar price for flat demand; using the max ensures
        # the optimizer doesn't under-value demand reduction in any season.
        max_rate = 0.0
        for period_tiers in tariff.demand_flat_structure:
            if period_tiers:
                r = period_tiers[0].get("effective_rate", 0.0)
                if r > max_rate:
                    max_rate = r
        if max_rate > 0:
            prices["flat"] = max_rate

    # ---- TOU demand periods ----
    if tariff.demand_rate_structure and tariff.demand_weekday_schedule:
        # Pre-compute demand period assignment for every hour
        month_idx = dt_index.month.values - 1
        hours_arr = dt_index.hour.values
        is_wknd = dt_index.weekday.values >= 5

        period_arr = np.zeros(n, dtype=int)
        for i in range(n):
            if is_wknd[i] and tariff.demand_weekend_schedule:
                period_arr[i] = tariff.demand_weekend_schedule[month_idx[i]][hours_arr[i]]
            else:
                period_arr[i] = tariff.demand_weekday_schedule[month_idx[i]][hours_arr[i]]

        for pidx, tiers in enumerate(tariff.demand_rate_structure):
            if not tiers:
                continue
            rate = tiers[0].get("effective_rate", 0.0)
            if rate <= 0:
                continue
            mask = period_arr == pidx
            if mask.any():
                key = f"tou_{pidx}"
                masks[key] = mask
                prices[key] = rate

    return masks, prices


def run_billing_simulation(
    load_8760: pd.Series,
    production_8760: pd.Series,
    tariff: TariffSchedule,
    export_rates_8760: pd.Series,
    battery_config=None,
    capacity_kwh: float = 0.0,
    monthly_dispatch: bool = False,
    nem_regime: str = "NEM-3",
    nbc_rate: float = 0.0,
    nsc_rate: float = 0.04,
    billing_option: str = "ABO",
) -> BillingResult:
    """
    Run the full billing simulation.

    Args:
        load_8760: Hourly load profile (kWh), 8760 values with datetime index
        production_8760: Hourly solar production (kWh), 8760 values with datetime index
        tariff: Parsed TariffSchedule
        export_rates_8760: Hourly export compensation rates ($/kWh), 8760 values
        battery_config: Optional BatteryConfig; when provided, runs LP dispatch
        capacity_kwh: Battery nameplate capacity (kWh); ignored when battery_config is None

    Returns:
        BillingResult with all hourly and monthly data
    """
    # Ensure aligned indices
    load = np.asarray(load_8760.values)
    solar = np.asarray(production_8760)
    export_rates = (
        np.asarray(export_rates_8760.values)
        if export_rates_8760 is not None
        else np.zeros(len(load))
    )
    dt_index = cast(pd.DatetimeIndex, load_8760.index)

    n_hours = len(load)
    if n_hours != 8760:
        raise ValueError(f"Expected 8760 hours, got {n_hours}")

    # --- Vectorised hour-by-hour calculation ---
    net_kwh = load - solar
    import_kwh = np.maximum(net_kwh, 0.0)
    export_kwh = np.maximum(-net_kwh, 0.0)

    energy_period, energy_rate = _vectorized_period_and_rate(tariff, dt_index)
    energy_cost = import_kwh * energy_rate
    export_credit = export_kwh * export_rates

    # --- Optional battery dispatch ---
    batt_dispatch = None
    if battery_config is not None and capacity_kwh > 0:
        from .battery.dispatch import dispatch_battery

        demand_masks, demand_prices = _build_demand_lp_inputs(tariff, dt_index)

        batt_dispatch = dispatch_battery(
            pv_kwh=solar,
            load_kwh=load,
            import_price=energy_rate,
            export_price=export_rates,
            demand_window_masks=demand_masks,
            demand_prices=demand_prices,
            battery_config=battery_config,
            capacity_kwh=capacity_kwh,
            monthly=monthly_dispatch,
            dt_index=dt_index,
        )

        # Replace grid-exchange arrays with post-battery values
        import_kwh = batt_dispatch.grid_import_kwh
        export_kwh = batt_dispatch.grid_export_kwh
        net_kwh = import_kwh - export_kwh

        # Recompute energy cost and export credit on new arrays
        energy_cost = import_kwh * energy_rate
        export_credit = export_kwh * export_rates

    # For NEM-1/2, value exports at retail TOU rates for hourly reporting.
    # (Monthly summary computes TOU-netted credit independently.)
    if nem_regime in ("NEM-1", "NEM-2"):
        export_credit = export_kwh * energy_rate

    # Build hourly detail DataFrame
    detail_dict = {
        "datetime": dt_index,
        "load_kwh": load,
        "solar_kwh": solar,
        "net_kwh": net_kwh,
        "import_kwh": import_kwh,
        "export_kwh": export_kwh,
        "energy_period": energy_period,
        "energy_rate": energy_rate,
        "energy_cost": energy_cost,
        "export_credit": export_credit,
    }
    if batt_dispatch is not None:
        detail_dict.update({
            "batt_charge_kwh": batt_dispatch.batt_charge_kwh,
            "batt_to_load_kwh": batt_dispatch.batt_discharge_to_load_kwh,
            "batt_to_grid_kwh": batt_dispatch.batt_discharge_to_grid_kwh,
            "soc_kwh": batt_dispatch.soc_kwh,
        })

    hourly_detail = pd.DataFrame(detail_dict)
    hourly_detail.set_index("datetime", inplace=True)

    # --- Demand charges (monthly) ---
    import_series = pd.Series(import_kwh, index=dt_index, name="import_kwh")
    demand_df = calculate_monthly_demand_charges(import_series, tariff)

    # --- Identify peak vs off-peak energy periods ---
    # The period with the highest effective rate is "peak"; all others are "off-peak"
    peak_period_idx = 0
    if tariff.energy_rate_structure:
        max_rate = 0.0
        for idx, tiers in enumerate(tariff.energy_rate_structure):
            if tiers and tiers[0]["effective_rate"] > max_rate:
                max_rate = tiers[0]["effective_rate"]
                peak_period_idx = idx

    # --- Monthly aggregation (regime-dependent) ---
    tou_annual_energy = 0.0
    tou_annual_credit = 0.0
    if nem_regime in ("NEM-1", "NEM-2"):
        monthly_rows, tou_annual_energy, tou_annual_credit = _build_monthly_nem12(
            load, solar, import_kwh, export_kwh, energy_period, energy_rate,
            dt_index, tariff, demand_df, peak_period_idx,
            nem_regime, nbc_rate, nsc_rate, billing_option,
        )
    else:
        # NEM-3/NVBT: existing hourly settlement logic
        monthly_rows = _build_monthly_nem3(
            load, solar, import_kwh, export_kwh, export_credit,
            energy_period, energy_cost, dt_index, tariff, demand_df, peak_period_idx,
        )

    monthly_summary = pd.DataFrame(monthly_rows)

    # --- Baseline bill (no solar) ---
    baseline_bill, monthly_baseline_list = _calc_baseline_bill(load_8760, tariff)

    # --- Annual totals ---
    annual_load = float(load.sum())
    annual_solar = float(solar.sum())
    annual_import = float(import_kwh.sum())
    annual_export = float(export_kwh.sum())
    annual_energy_cost = float(monthly_summary["energy_cost"].sum())
    annual_demand_cost = float(monthly_summary["total_demand_charge"].sum())
    annual_fixed_cost = float(monthly_summary["fixed_charge"].sum())
    annual_export_credit = float(monthly_summary["export_credit"].sum())
    annual_nbc = float(monthly_summary["nbc_charge"].sum()) if "nbc_charge" in monthly_summary.columns else 0.0
    annual_bill_solar = float(monthly_summary["net_bill"].sum())
    annual_savings = baseline_bill - annual_bill_solar
    savings_pct = (annual_savings / baseline_bill * 100) if baseline_bill > 0 else 0.0

    # NSC adjustment (stored in monthly rows already applied)
    _nsc_col = monthly_summary.get("nsc_adjustment")
    annual_nsc_adj = float(_nsc_col.sum()) if _nsc_col is not None else 0.0

    # Raw import energy cost: sum(import_kwh * energy_rate) from hourly arrays.
    # This is always the gross import cost before any TOU netting or billing
    # option adjustments.  The energy_cost column in hourly_detail holds these
    # raw values (set at line 216 before NEM-1/2 export_credit override).
    raw_annual_energy = float(hourly_detail["energy_cost"].sum())

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
        annual_bill_without_solar=baseline_bill,
        annual_savings=annual_savings,
        savings_pct=savings_pct,
        annual_nbc_cost=annual_nbc,
        annual_nsc_adjustment=annual_nsc_adj,
        nem_regime=nem_regime,
        tou_annual_energy=tou_annual_energy,
        tou_annual_credit=tou_annual_credit,
        raw_annual_energy=raw_annual_energy,
        monthly_baseline_details=monthly_baseline_list,
    )


def _build_monthly_nem3(
    load, solar, import_kwh, export_kwh, export_credit,
    energy_period, energy_cost, dt_index, tariff, demand_df, peak_period_idx,
) -> list[dict]:
    """NEM-3/NVBT: hourly settlement (original behavior)."""
    energy_cost_arr = energy_cost

    monthly_rows = []
    for month_num in range(1, 13):
        month_mask = dt_index.month == month_num
        m_load = load[month_mask].sum()
        m_solar = solar[month_mask].sum()
        m_import = import_kwh[month_mask].sum()
        m_export = export_kwh[month_mask].sum()
        m_energy_cost = energy_cost_arr[month_mask].sum()
        m_export_credit = export_credit[month_mask].sum()

        demand_row = demand_df[demand_df["month"] == month_num].iloc[0]
        m_demand_cost = demand_row["total_demand_charge"]
        m_flat_demand = demand_row["flat_demand_charge"]
        m_tou_demand = demand_row["tou_demand_charge"]
        m_peak_kw = demand_row["flat_demand_kw"]

        month_export = export_kwh[month_mask]
        month_periods = energy_period[month_mask]
        peak_mask = month_periods == peak_period_idx
        offpeak_mask = ~peak_mask

        m_export_peak = float(month_export[peak_mask].sum()) if peak_mask.any() else 0.0
        m_export_offpeak = float(month_export[offpeak_mask].sum()) if offpeak_mask.any() else 0.0

        m_fixed = tariff.fixed_monthly_charge
        m_net_bill = m_energy_cost + m_demand_cost + m_fixed - m_export_credit
        m_net_bill = max(m_net_bill, tariff.min_monthly_charge)

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
            "flat_demand_charge": round(m_flat_demand, 2),
            "tou_demand_charge": round(m_tou_demand, 2),
            "total_demand_charge": round(m_demand_cost, 2),
            "fixed_charge": round(m_fixed, 2),
            "export_credit": round(m_export_credit, 2),
            "nbc_charge": 0.0,
            "nsc_adjustment": 0.0,
            "net_bill": round(m_net_bill, 2),
        })
    return monthly_rows


def _build_monthly_nem12(
    load, solar, import_kwh, export_kwh, energy_period, energy_rate,
    dt_index, tariff, demand_df, peak_period_idx,
    nem_regime, nbc_rate, nsc_rate, billing_option,
) -> tuple[list[dict], float, float]:
    """NEM-1 / NEM-2: TOU-period monthly netting with NBC and NSC true-up.

    Returns:
        (monthly_rows, tou_annual_energy, tou_annual_credit)
    """
    n_hours = len(dt_index)

    # Collect unique TOU period indices and their rates
    period_rates = {}
    if tariff.energy_rate_structure:
        for pidx, tiers in enumerate(tariff.energy_rate_structure):
            if tiers:
                period_rates[pidx] = tiers[0]["effective_rate"]

    monthly_rows = []
    credit_bank = 0.0        # MBO credit carryover
    deferred_energy = 0.0    # ABO deferred energy charges
    tou_annual_energy = 0.0  # positive side of TOU netting (across all months)
    tou_annual_credit = 0.0  # negative side of TOU netting (across all months)

    for month_num in range(1, 13):
        month_mask = dt_index.month == month_num
        m_load = load[month_mask].sum()
        m_solar = solar[month_mask].sum()
        m_import_raw = import_kwh[month_mask].sum()
        m_export_raw = export_kwh[month_mask].sum()

        # Demand charges
        demand_row = demand_df[demand_df["month"] == month_num].iloc[0]
        m_demand_cost = demand_row["total_demand_charge"]
        m_flat_demand = demand_row["flat_demand_charge"]
        m_tou_demand = demand_row["tou_demand_charge"]
        m_peak_kw = demand_row["flat_demand_kw"]

        m_fixed = tariff.fixed_monthly_charge

        # --- TOU-period netting ---
        # For each TOU period in this month, net imports vs exports
        month_import = import_kwh[month_mask]
        month_export = export_kwh[month_mask]
        month_periods = energy_period[month_mask]
        month_rates = energy_rate[month_mask]

        # Gross import energy cost and gross export credit (for display)
        m_energy_cost = float((month_import * month_rates).sum())
        m_export_credit = float((month_export * month_rates).sum())

        # TOU-netted energy charge (for net bill calculation)
        monthly_energy_charge = 0.0
        for pidx, rate in period_rates.items():
            period_mask = month_periods == pidx
            if not period_mask.any():
                continue
            net_kwh_p = float(month_import[period_mask].sum() - month_export[period_mask].sum())
            energy_charge_p = net_kwh_p * rate
            monthly_energy_charge += energy_charge_p

        # Accumulate TOU-netted energy/credit split for projection use
        if monthly_energy_charge >= 0:
            tou_annual_energy += monthly_energy_charge
        else:
            tou_annual_credit += abs(monthly_energy_charge)

        # Export energy split by TOU period (peak vs off-peak) — for reporting
        peak_mask_ep = month_periods == peak_period_idx
        offpeak_mask_ep = ~peak_mask_ep
        m_export_peak = float(month_export[peak_mask_ep].sum()) if peak_mask_ep.any() else 0.0
        m_export_offpeak = float(month_export[offpeak_mask_ep].sum()) if offpeak_mask_ep.any() else 0.0

        # --- NEM-2 NBC: interval-level non-bypassable charges ---
        m_nbc_charge = 0.0
        if nem_regime == "NEM-2" and nbc_rate > 0:
            # NBC applies to net consumption per hour (hours where import > export).
            # Post-battery dispatch, mutual exclusivity cleanup ensures import and
            # export are never simultaneously positive, so this is equivalent to
            # summing import_kwh for import-only hours.
            month_net = month_import - month_export
            nbc_kwh = float(np.maximum(month_net, 0).sum())
            m_nbc_charge = float(nbc_kwh * nbc_rate)

        # --- Net bill (pre-true-up) ---
        # Use TOU-netted energy charge (not gross) for the actual bill
        m_net_bill = monthly_energy_charge + m_demand_cost + m_fixed + m_nbc_charge

        # --- Billing option logic ---
        if billing_option == "MBO":
            # Monthly Billing Option: credits carry forward, bills floor at 0
            if m_net_bill < 0:
                credit_bank += abs(m_net_bill)
                m_net_bill = 0.0
            elif m_net_bill > 0 and credit_bank > 0:
                reduction = min(credit_bank, m_net_bill)
                m_net_bill -= reduction
                credit_bank -= reduction
        elif billing_option == "ABO":
            # Annual Billing Option: only demand + fixed + NBC paid monthly,
            # energy charges deferred to month 12
            if month_num < 12:
                deferred_energy += monthly_energy_charge
                m_net_bill = m_demand_cost + m_fixed + m_nbc_charge
            else:
                # True-up month: pay deferred energy + this month's energy
                m_net_bill = monthly_energy_charge + deferred_energy + m_demand_cost + m_fixed + m_nbc_charge

        m_net_bill = max(m_net_bill, tariff.min_monthly_charge)

        # For ABO, adjust displayed energy_cost to match net_bill accounting
        if billing_option == "ABO":
            if month_num < 12:
                _display_energy = 0.0
            else:
                _display_energy = monthly_energy_charge + deferred_energy
        else:
            _display_energy = m_energy_cost

        monthly_rows.append({
            "month": month_num,
            "load_kwh": round(m_load, 1),
            "solar_kwh": round(m_solar, 1),
            "import_kwh": round(m_import_raw, 1),
            "export_kwh": round(m_export_raw, 1),
            "peak_demand_kw": round(m_peak_kw, 2),
            "export_peak_kwh": round(m_export_peak, 1),
            "export_offpeak_kwh": round(m_export_offpeak, 1),
            "energy_cost": round(_display_energy, 2),
            "flat_demand_charge": round(m_flat_demand, 2),
            "tou_demand_charge": round(m_tou_demand, 2),
            "total_demand_charge": round(m_demand_cost, 2),
            "fixed_charge": round(m_fixed, 2),
            "export_credit": round(m_export_credit, 2),
            "nbc_charge": round(m_nbc_charge, 2),
            "nsc_adjustment": 0.0,
            "net_bill": round(m_net_bill, 2),
        })

    # --- NSC true-up ---
    _apply_nsc_true_up(
        monthly_rows, import_kwh, export_kwh, energy_rate, energy_period,
        period_rates, nsc_rate,
    )

    return monthly_rows, tou_annual_energy, tou_annual_credit


def _apply_nsc_true_up(
    monthly_rows: list[dict],
    import_kwh: np.ndarray,
    export_kwh: np.ndarray,
    energy_rate: np.ndarray,
    energy_period: np.ndarray,
    period_rates: dict,
    nsc_rate: float,
) -> None:
    """Apply Net Surplus Compensation true-up to month 12 if annual net surplus.

    Uses per-TOU-period annual netting to compute the value of the surplus,
    matching the same netting logic used in _build_monthly_nem12.

    Modifies monthly_rows in place.
    """
    annual_net_energy = float(import_kwh.sum() - export_kwh.sum())
    if annual_net_energy >= 0:
        # No net surplus — customer consumed more than exported annually
        return

    surplus_kwh = abs(annual_net_energy)

    # Compute what TOU netting valued the surplus at using per-period netting
    # (same math as _build_monthly_nem12: net each TOU period, sum negative nets)
    tou_credit_for_surplus = 0.0
    for pidx, rate in period_rates.items():
        period_mask = energy_period == pidx
        if not period_mask.any():
            continue
        net_p = float(import_kwh[period_mask].sum() - export_kwh[period_mask].sum())
        if net_p < 0:
            # This period has net surplus; TOU netting credits it at this rate
            tou_credit_for_surplus += abs(net_p) * rate

    nsc_credit = surplus_kwh * nsc_rate
    nsc_adjustment = tou_credit_for_surplus - nsc_credit  # positive = credit reduction

    if nsc_adjustment <= 0:
        return

    # Apply adjustment to month 12 (true-up month)
    row_12 = monthly_rows[11]
    row_12["nsc_adjustment"] = round(nsc_adjustment, 2)
    # Reduce export credit and increase net bill
    row_12["export_credit"] = round(max(row_12["export_credit"] - nsc_adjustment, 0), 2)
    row_12["net_bill"] = round(row_12["net_bill"] + nsc_adjustment, 2)


def _calc_baseline_bill(load_8760: pd.Series, tariff: TariffSchedule) -> tuple[float, list[dict]]:
    """
    Calculate annual bill without solar (baseline) for savings comparison.
    Uses the same tariff but with zero production.

    Returns (annual_total, monthly_details) where monthly_details is a list
    of 12 dicts with keys: energy, demand, fixed, total.
    """
    dt_index = load_8760.index
    load = load_8760.values
    n_hours = len(load)

    # Vectorised hourly energy cost
    _, rates = _vectorized_period_and_rate(tariff, dt_index)
    energy_by_hour = load * rates

    # Demand charges (all load = import when no solar)
    demand_df = calculate_monthly_demand_charges(load_8760, tariff)

    # Build monthly breakdown
    monthly_details = []
    for month_num in range(1, 13):
        month_mask = dt_index.month == month_num
        m_energy = float(energy_by_hour[month_mask].sum())
        demand_row = demand_df[demand_df["month"] == month_num].iloc[0]
        m_demand = float(demand_row["total_demand_charge"])
        m_fixed = tariff.fixed_monthly_charge
        monthly_details.append({
            "energy": m_energy,
            "demand": m_demand,
            "fixed": m_fixed,
            "total": m_energy + m_demand + m_fixed,
        })

    total_baseline = sum(d["total"] for d in monthly_details)
    return float(total_baseline), monthly_details
