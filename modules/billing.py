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
from .tariff import TariffSchedule
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

    # Per-month TOU-netted breakdowns (NEM-1/2 only; empty for NEM-3)
    tou_monthly_energy: dict[int, float] | None = None  # month -> positive energy $
    tou_monthly_credit: dict[int, float] | None = None  # month -> credit $

    # Raw import energy cost: sum(import_kwh * energy_rate) across all hours.
    # Always represents gross import cost regardless of TOU netting or billing option.
    # Used by projection for NEM-3 regime-switch energy cost baseline.
    raw_annual_energy: float = 0.0

    # Monthly baseline breakdown (no-solar bill components per month)
    monthly_baseline_details: list[dict] | None = None  # 12 dicts: {energy, demand, fixed, total}

    # Rate shift analysis fields (old tariff baseline)
    old_rate_annual_baseline: float | None = None          # Annual cost on old tariff, no solar
    old_rate_monthly_baselines: list[float] | None = None   # 12 monthly costs on old tariff
    rate_shift_annual_savings: float | None = None          # old_rate_baseline - new_rate_baseline

    # Billing-option context (used by Y>1 projection to re-apply MBO/ABO floors).
    # MBO banks negative monthly bills as credits; ABO defers energy to month 12;
    # both clamp each month at min_monthly_charge.
    billing_option: str = "MBO"
    min_monthly_charge: float = 0.0


def _assemble_billing_result(
    hourly_detail: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    load: np.ndarray,
    solar: np.ndarray,
    import_kwh: np.ndarray,
    export_kwh: np.ndarray,
    baseline_bill: float,
    nem_regime: str = "NEM-3",
    tou_annual_energy: float = 0.0,
    tou_annual_credit: float = 0.0,
    tou_monthly_energy: dict | None = None,
    tou_monthly_credit: dict | None = None,
    monthly_baseline_details: list | None = None,
    raw_annual_energy: float | None = None,
    annual_nbc: float = 0.0,
    annual_nsc_adj: float = 0.0,
    billing_option: str = "MBO",
    min_monthly_charge: float = 0.0,
) -> "BillingResult":
    """Assemble a BillingResult from monthly_summary and supporting arrays.

    Computes standard annual totals (energy_cost, demand, fixed, export_credit)
    from monthly_summary, derives annual_bill_solar / annual_savings / savings_pct,
    and returns the populated dataclass.
    """
    annual_energy_cost = float(monthly_summary["energy_cost"].sum())
    annual_demand_cost = float(monthly_summary["total_demand_charge"].sum())
    annual_fixed_cost = float(monthly_summary["fixed_charge"].sum())
    annual_export_credit = float(monthly_summary["export_credit"].sum())
    annual_bill_solar = float(monthly_summary["net_bill"].sum())
    annual_savings = baseline_bill - annual_bill_solar
    savings_pct = (annual_savings / baseline_bill * 100) if baseline_bill > 0 else 0.0

    if raw_annual_energy is None:
        raw_annual_energy = float(hourly_detail["energy_cost"].sum())

    return BillingResult(
        hourly_detail=hourly_detail,
        monthly_summary=monthly_summary,
        annual_load_kwh=float(load.sum()),
        annual_solar_kwh=float(solar.sum()),
        annual_import_kwh=float(import_kwh.sum()),
        annual_export_kwh=float(export_kwh.sum()),
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
        tou_monthly_energy=tou_monthly_energy,
        tou_monthly_credit=tou_monthly_credit,
        raw_annual_energy=raw_annual_energy,
        monthly_baseline_details=monthly_baseline_details,
        billing_option=billing_option,
        min_monthly_charge=min_monthly_charge,
    )


def _draw_credit_against_energy(
    gross_energy: float,
    available_credit: float,
    nonoffsettable: float,
    min_charge: float,
) -> tuple[float, float]:
    """Apply a credit balance to the ENERGY bucket only.

    A generation/export credit may reduce the volumetric energy charge but never
    the demand, fixed/customer, or non-bypassable (NBC) charges, and never below
    the minimum-charge floor (PG&E NEM2 SC 2.c/2.d; NBT SC 2.d). Credit that
    would breach the floor is not consumed — it stays available to bank forward.

    Returns ``(energy_due, credit_used)`` where ``credit_used`` is in
    ``[0, available_credit]`` (never negative, so the bank can't be inflated).
    """
    energy_floor = max(min_charge - nonoffsettable, 0.0)
    reducible = max(gross_energy - energy_floor, 0.0)
    credit_used = min(available_credit, reducible)
    return gross_energy - credit_used, credit_used


def simulate_year_under_billing_option(
    monthly_components: list[dict],
    nem_regime: str,
    billing_option: str = "MBO",
    min_monthly_charge: float = 0.0,
) -> list[float]:
    """Apply the same MBO/ABO + min-charge floor + NEM-1/2 NSC trueup logic that
    ``run_billing_simulation`` uses for Y1, but to scaled monthly components for
    any year. Returns 12 net_bill values.

    monthly_components is a list of 12 dicts with keys: ``energy`` (positive side
    of TOU netting for NEM-1/2, or gross import cost for NEM-3), ``export_credit``
    (negative side / NEM-3 export comp, as a positive number), ``demand``,
    ``fixed``, ``nbc``, and ``nsc_adj`` (only month 12 carries a non-zero value).

    The TOU-netted-vs-NEM-3 distinction matters because:
      - NEM-1/2 (and NEM-A) lets monthly net energy go negative → MBO banks it
        as a credit, ABO defers it to month 12. Year-end NSC trueup adds a
        separate clawback to month 12 before the final min-charge re-clamp.
      - NEM-3 has no monthly netting; export credit just offsets the bill
        and there's no inter-month banking.

    Lives here (not in outputs.py) so both projection (outputs.py) and the
    NEM-A aggregate Y1 reconciliation (billing_aggregation.py) can use the
    same floor logic.
    """
    is_tou_netted = nem_regime in ("NEM-1", "NEM-2") or nem_regime.startswith("NEM-A")
    net_bills = [0.0] * 12
    credit_bank = 0.0
    deferred_energy = 0.0

    for i in range(12):
        c = monthly_components[i]
        m_energy = c.get("energy", 0.0)
        m_credit = c.get("export_credit", 0.0)
        m_demand = c.get("demand", 0.0)
        m_fixed = c.get("fixed", 0.0)
        m_nbc = c.get("nbc", 0.0)
        m_nsc_adj = c.get("nsc_adj", 0.0) if i == 11 else 0.0

        # Credit (banked NEM-1/2 credit or NEM-3 export credit) offsets the
        # VOLUMETRIC ENERGY charge only. Demand, fixed/customer, and NBC are
        # billed monthly regardless and are never reduced by a credit
        # (PG&E NEM2 Special Conditions 2.c/2.d; NBT Special Condition 2.d).
        nonoffsettable = m_demand + m_fixed + m_nbc
        if is_tou_netted and billing_option == "ABO":
            if i < 11:
                # Energy deferred to month 12; only demand/fixed/NBC due now.
                deferred_energy += (m_energy - m_credit)
                m_net = nonoffsettable
            else:
                # Year-end true-up. Net energy may be a credit, but a credit
                # cannot offset demand/fixed/NBC — floor the energy bucket at 0.
                total_energy = deferred_energy + (m_energy - m_credit)
                m_net = max(total_energy, 0.0) + nonoffsettable
        else:
            # MBO and NEM-3/NBT share one rule: net energy that goes negative
            # banks as a credit; the bank then draws down ENERGY in later months
            # only. (NEM-3 has no monthly netting, but its export credit banks
            # forward the same way and is likewise energy-only.)
            month_energy = (m_energy - m_credit)
            if month_energy < 0:
                credit_bank += -month_energy
                gross_energy = 0.0
            else:
                gross_energy = month_energy
            energy_due, credit_used = _draw_credit_against_energy(
                gross_energy, credit_bank, nonoffsettable, min_monthly_charge
            )
            credit_bank -= credit_used
            m_net = energy_due + nonoffsettable

        m_net = max(m_net, min_monthly_charge)

        # NSC clawback (NEM-1/2 only): added to month 12 raw bill, re-clamped at floor
        if m_nsc_adj > 0:
            m_net = max(m_net + m_nsc_adj, min_monthly_charge)

        net_bills[i] = m_net

    return net_bills


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

        # For NEM-1/NEM-2, exports are valued at retail TOU rates.  If the
        # caller passed zeros (placeholder), use the energy_rate array so the
        # LP has proper incentive to discharge.  When the caller provides real
        # export prices (e.g. NEM-A effective export), respect those.
        _export_all_zero = not np.any(export_rates > 0)
        _lp_export_price = (
            energy_rate
            if nem_regime in ("NEM-1", "NEM-2") and _export_all_zero
            else export_rates
        )

        batt_dispatch = dispatch_battery(
            pv_kwh=solar,
            load_kwh=load,
            import_price=energy_rate,
            export_price=_lp_export_price,
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
    tou_monthly_energy: dict[int, float] | None = None
    tou_monthly_credit: dict[int, float] | None = None
    if nem_regime in ("NEM-1", "NEM-2"):
        monthly_rows, tou_annual_energy, tou_annual_credit, tou_monthly_energy, tou_monthly_credit = _build_monthly_nem12(
            load, solar, import_kwh, export_kwh, energy_period, energy_rate,
            dt_index, tariff, demand_df, peak_period_idx,
            nem_regime, nbc_rate, nsc_rate, billing_option,
        )
    else:
        # NEM-3/NVBT: hourly settlement at ACC export rates with year-end NSC
        # true-up per PG&E Schedule NBT / D.22-12-056. NEM-3 NSC re-prices any
        # net surplus electricity (kWh) from the rolling-12-mo avg ACC rate
        # down to the wholesale NSC rate; smaller magnitude than NEM-1/2 NSC
        # but non-zero whenever annual exports exceed annual imports and
        # nsc_rate < avg ACC rate.
        monthly_rows, nem3_leftover_bank = _build_monthly_nem3(
            load, solar, import_kwh, export_kwh, export_credit,
            energy_period, energy_cost, dt_index, tariff, demand_df, peak_period_idx,
        )
        if nsc_rate > 0:
            _apply_nbt_nsc_true_up(
                monthly_rows, import_kwh, export_kwh, nsc_rate,
                tariff.min_monthly_charge,
                banked_surplus=nem3_leftover_bank,
            )

    monthly_summary = pd.DataFrame(monthly_rows)

    # --- Baseline bill (no solar) ---
    baseline_bill, monthly_baseline_list = _calc_baseline_bill(load_8760, tariff)

    # --- Annual totals via shared helper ---
    annual_nbc = float(monthly_summary["nbc_charge"].sum()) if "nbc_charge" in monthly_summary.columns else 0.0
    _nsc_col = monthly_summary.get("nsc_adjustment")
    annual_nsc_adj = float(_nsc_col.sum()) if _nsc_col is not None else 0.0

    return _assemble_billing_result(
        hourly_detail=hourly_detail,
        monthly_summary=monthly_summary,
        load=load,
        solar=solar,
        import_kwh=import_kwh,
        export_kwh=export_kwh,
        baseline_bill=baseline_bill,
        nem_regime=nem_regime,
        tou_annual_energy=tou_annual_energy,
        tou_annual_credit=tou_annual_credit,
        tou_monthly_energy=tou_monthly_energy,
        tou_monthly_credit=tou_monthly_credit,
        monthly_baseline_details=monthly_baseline_list,
        annual_nbc=annual_nbc,
        annual_nsc_adj=annual_nsc_adj,
        billing_option=billing_option,
        min_monthly_charge=tariff.min_monthly_charge,
    )


def _build_monthly_nem3(
    load, solar, import_kwh, export_kwh, export_credit,
    energy_period, energy_cost, dt_index, tariff, demand_df, peak_period_idx,
) -> list[dict]:
    """NEM-3/NVBT: hourly settlement (original behavior)."""
    energy_cost_arr = energy_cost

    monthly_rows = []
    # True-NBT credit banking: a month's surplus export credit carries forward
    # and offsets later months' bills (the customer always pays >= the minimum
    # charge). This is a no-op for net-import months — the credit is fully
    # consumed and the bank stays 0 — so net-importer bills are identical to the
    # prior floor-only behavior. The leftover bank after month 12 is the genuine
    # unconsumed net surplus (in $), returned to bound the year-end NSC clawback:
    # consumed credit offset real bills and is never clawed back; only the
    # leftover surplus is trued down to the NSC rate.
    credit_bank = 0.0
    min_charge = tariff.min_monthly_charge
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
        # Export credit offsets the VOLUMETRIC ENERGY charge only; demand and
        # fixed/customer charges are billed regardless and are never reduced by
        # an export credit (NBT Special Condition 2.d). Surplus credit banks
        # forward to offset later months' energy. The min charge still applies,
        # and credit that would breach it is not consumed (stays banked).
        nonoffsettable = m_demand_cost + m_fixed  # NEM-3 has no NBC line
        available_credit = m_export_credit + credit_bank
        energy_due, credit_used = _draw_credit_against_energy(
            m_energy_cost, available_credit, nonoffsettable, min_charge
        )
        credit_bank = available_credit - credit_used
        m_net_bill = max(energy_due + nonoffsettable, min_charge)

        monthly_rows.append({
            "month": month_num,
            "load_kwh": float(m_load),
            "solar_kwh": float(m_solar),
            "import_kwh": float(m_import),
            "export_kwh": float(m_export),
            "peak_demand_kw": round(m_peak_kw, 2),
            "export_peak_kwh": float(m_export_peak),
            "export_offpeak_kwh": float(m_export_offpeak),
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
    return monthly_rows, credit_bank


def _build_monthly_nem12(
    load, solar, import_kwh, export_kwh, energy_period, energy_rate,
    dt_index, tariff, demand_df, peak_period_idx,
    nem_regime, nbc_rate, nsc_rate, billing_option,
) -> tuple[list[dict], float, float, dict[int, float], dict[int, float]]:
    """NEM-1 / NEM-2: TOU-period monthly netting with NBC and NSC true-up.

    Returns:
        (monthly_rows, tou_annual_energy, tou_annual_credit,
         tou_monthly_energy, tou_monthly_credit)
    """
    # Collect unique TOU period indices and their rates
    period_rates = {}
    if tariff.energy_rate_structure:
        for pidx, tiers in enumerate(tariff.energy_rate_structure):
            if tiers:
                period_rates[pidx] = tiers[0]["effective_rate"]

    # NEM-2: the URDB retail rate already bundles the non-bypassable charge
    # components, so the TOU-netted energy must be valued NET of nbc_rate —
    # otherwise NBC is double-counted (once inside the netted energy, once in the
    # explicit non-bypassable line below). Exports likewise do not earn the
    # non-bypassable portion (PG&E NEM2 Special Condition 2.c: NBCs "may not be
    # reduced by any credits for exports to the grid"). NEM-1 has no NBC
    # (nbc_rate == 0), so this is a no-op there. Adjusting the locals here
    # propagates consistently to the gross display, the TOU netting, the
    # per-month projection inputs, and the NSC true-up (#27).
    if nem_regime == "NEM-2" and nbc_rate > 0:
        energy_rate = np.maximum(energy_rate - nbc_rate, 0.0)
        period_rates = {p: max(r - nbc_rate, 0.0) for p, r in period_rates.items()}

    monthly_rows = []
    credit_bank = 0.0        # MBO credit carryover
    credit_consumed_mbo = 0.0  # MBO credit actually drawn down against positive bills
    deferred_energy = 0.0    # ABO deferred energy charges
    tou_annual_energy = 0.0  # positive side of TOU netting (across all months)
    tou_annual_credit = 0.0  # negative side of TOU netting (across all months)
    tou_monthly_energy: dict[int, float] = {}  # per-month positive energy $
    tou_monthly_credit: dict[int, float] = {}  # per-month credit $

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
            tou_monthly_energy[month_num] = monthly_energy_charge
            tou_monthly_credit[month_num] = 0.0
            tou_annual_energy += monthly_energy_charge
        else:
            tou_monthly_energy[month_num] = 0.0
            tou_monthly_credit[month_num] = abs(monthly_energy_charge)
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

        # --- Net bill ---
        # A credit (banked TOU-netted energy credit) offsets the ENERGY charge
        # only. Demand, fixed/customer, and NBC are billed monthly regardless
        # and are never reduced by a credit (PG&E NEM2 Special Conditions
        # 2.c/2.d). monthly_energy_charge is the TOU-netted energy and may be
        # negative (a credit).
        nonoffsettable = m_demand_cost + m_fixed + m_nbc_charge
        if billing_option == "MBO":
            # Monthly: net energy credit banks forward and draws down later
            # months' ENERGY only; the bill floors at the minimum charge.
            if monthly_energy_charge < 0:
                credit_bank += -monthly_energy_charge
                gross_energy = 0.0
            else:
                gross_energy = monthly_energy_charge
            energy_due, credit_used = _draw_credit_against_energy(
                gross_energy, credit_bank, nonoffsettable, tariff.min_monthly_charge
            )
            credit_bank -= credit_used
            credit_consumed_mbo += credit_used  # bounds the NSC clawback
            m_net_bill = energy_due + nonoffsettable
        else:  # ABO
            # Annual: only demand + fixed + NBC paid monthly; energy deferred
            # to month 12. At true-up the net energy may be a credit, but a
            # credit cannot offset demand/fixed/NBC — floor the energy bucket
            # at 0 (surplus is settled separately via NSC, not as a bill offset).
            if month_num < 12:
                deferred_energy += monthly_energy_charge
                energy_due = 0.0
            else:
                energy_due = max(monthly_energy_charge + deferred_energy, 0.0)
            m_net_bill = energy_due + nonoffsettable

        m_net_bill = max(m_net_bill, tariff.min_monthly_charge)

        # Displayed energy_cost matches the energy bucket actually billed.
        if billing_option == "ABO":
            _display_energy = energy_due
        else:
            _display_energy = m_energy_cost

        monthly_rows.append({
            "month": month_num,
            "load_kwh": float(m_load),
            "solar_kwh": float(m_solar),
            "import_kwh": float(m_import_raw),
            "export_kwh": float(m_export_raw),
            "peak_demand_kw": round(m_peak_kw, 2),
            "export_peak_kwh": float(m_export_peak),
            "export_offpeak_kwh": float(m_export_offpeak),
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
    # Under MBO, cap the clawback at credit actually consumed (drawn down against
    # positive-bill months); unconsumed bank evaporates at year-end, so clawing
    # back beyond consumed would over-charge. ABO keeps the banked-credit cap.
    _nsc_consumed_cap = credit_consumed_mbo if billing_option == "MBO" else None
    _apply_nsc_true_up(
        monthly_rows, import_kwh, export_kwh, energy_rate, energy_period,
        period_rates, nsc_rate, tariff.min_monthly_charge,
        tou_annual_credit=tou_annual_credit,
        credit_consumed=_nsc_consumed_cap,
    )

    return monthly_rows, tou_annual_energy, tou_annual_credit, tou_monthly_energy, tou_monthly_credit


def _apply_nsc_true_up(
    monthly_rows: list[dict],
    import_kwh: np.ndarray,
    export_kwh: np.ndarray,
    energy_rate: np.ndarray,
    energy_period: np.ndarray,
    period_rates: dict,
    nsc_rate: float,
    min_monthly_charge: float = 0.0,
    tou_annual_credit: float = 0.0,
    credit_consumed: float | None = None,
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

    # Cap adjustment at the credit actually banked through monthly TOU netting
    # (sum over months of negative monthly_energy_charge values). When TOU
    # period imbalance lets annual per-period netting credit MORE than the sum
    # of monthly per-period nettings, the customer never actually banked the
    # excess — so the clawback can't exceed what was banked.
    # MBO caps at credit actually consumed (passed in, may be 0 → no clawback);
    # ABO keeps the banked-credit cap (credit_consumed is None). See
    # _build_monthly_nem12.
    if credit_consumed is not None:
        nsc_adjustment = min(nsc_adjustment, credit_consumed)
    elif tou_annual_credit > 0:
        nsc_adjustment = min(nsc_adjustment, tou_annual_credit)

    if nsc_adjustment <= 0:
        return

    # Apply adjustment to month 12 (true-up month).
    # NSC modifies the bill BEFORE the min_monthly_charge floor, so we
    # recompute the floor after adding the adjustment.
    row_12 = monthly_rows[11]
    row_12["nsc_adjustment"] = round(nsc_adjustment, 2)
    # Reduce export credit and increase net bill
    row_12["export_credit"] = round(max(row_12["export_credit"] - nsc_adjustment, 0), 2)
    # Remove the previous min_monthly_charge clamp, add NSC, then re-clamp
    raw_bill = row_12["net_bill"] + nsc_adjustment
    row_12["net_bill"] = round(max(raw_bill, min_monthly_charge), 2)


def _apply_nbt_nsc_true_up(
    monthly_rows: list[dict],
    import_kwh: np.ndarray,
    export_kwh: np.ndarray,
    nsc_rate: float,
    min_monthly_charge: float = 0.0,
    banked_surplus: float = 0.0,
) -> None:
    """Apply NEM-3 / NBT Net Surplus Compensation true-up to month 12.

    Per PG&E Schedule NBT and CPUC D.22-12-056: at year-end, any Net Surplus
    Electricity (kWh) is debited from the customer's account at the utility's
    rolling 12-month average ACC export compensation rate and re-credited at
    the NSC rate (DLAP wholesale ~ $0.02–0.03/kWh per AB 920). Net effect on
    the bill is positive (increases bill) when avg ACC > NSC rate.

    Modeling proxy: avg ACC rate = total annual export credit $ / total annual
    export kWh from the simulated hourly dispatch. The customer's own simulated
    ACC compensation stands in for the utility-wide rolling average.

    Modifies monthly_rows in place.
    """
    annual_net_energy = float(import_kwh.sum() - export_kwh.sum())
    if annual_net_energy >= 0:
        return  # No surplus: customer consumed more than exported annually

    surplus_kwh = abs(annual_net_energy)

    total_export_kwh = float(export_kwh.sum())
    total_export_credit = sum(row["export_credit"] for row in monthly_rows)
    if total_export_kwh <= 0 or total_export_credit <= 0:
        return  # Can't price surplus without an effective ACC rate

    avg_acc_rate = total_export_credit / total_export_kwh

    nsc_credit = surplus_kwh * nsc_rate
    nsc_adjustment = surplus_kwh * avg_acc_rate - nsc_credit  # positive = clawback

    if nsc_adjustment <= 0:
        return  # NSC rate >= avg ACC: no adjustment

    # Cap the clawback at the LEFTOVER banked surplus — the genuine unconsumed
    # net surplus (in $) after monthly banking. Consumed credit offset real bills
    # and is never clawed back; only the leftover surplus is trued down from the
    # ACC value to the NSC rate. For a deeply net-export customer that consumes
    # little, the leftover is large, so the full surplus haircut applies (the
    # stricter, economically-correct NBT settlement). For a $-balanced customer
    # whose bank is exhausted by imports, the leftover is ~0 and no clawback
    # applies. (avg_acc_rate above is the per-kWh ACC price, not a cap.)
    nsc_adjustment = min(nsc_adjustment, banked_surplus)

    if nsc_adjustment <= 0:
        return

    # Apply to month 12 (true-up month). Mirrors NEM-1/2 NSC mechanics:
    # reduce export credit, add to bill, re-clamp at min_monthly_charge.
    row_12 = monthly_rows[11]
    row_12["nsc_adjustment"] = round(nsc_adjustment, 2)
    row_12["export_credit"] = round(max(row_12["export_credit"] - nsc_adjustment, 0), 2)
    raw_bill = row_12["net_bill"] + nsc_adjustment
    row_12["net_bill"] = round(max(raw_bill, min_monthly_charge), 2)


def _calc_baseline_bill(load_8760: pd.Series, tariff: TariffSchedule) -> tuple[float, list[dict]]:
    """
    Calculate annual bill without solar (baseline) for savings comparison.
    Uses the same tariff but with zero production.

    Returns (annual_total, monthly_details) where monthly_details is a list
    of 12 dicts with keys: energy, demand, fixed, total.
    """
    dt_index = load_8760.index
    load = load_8760.values

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
        m_peak_kw = float(demand_row["flat_demand_kw"])
        m_fixed = tariff.fixed_monthly_charge
        monthly_details.append({
            "energy": m_energy,
            "demand": m_demand,
            "fixed": m_fixed,
            "peak_demand_kw": m_peak_kw,
            "total": m_energy + m_demand + m_fixed,
        })

    total_baseline = sum(d["total"] for d in monthly_details)
    return float(total_baseline), monthly_details


def compute_old_rate_baseline(
    load_8760: pd.Series, old_tariff: TariffSchedule,
) -> dict:
    """Compute baseline bill on the old (pre-switch) tariff for rate shift analysis.

    Returns dict with keys: annual_cost, monthly_costs (list of 12 floats).
    """
    annual_total, monthly_details = _calc_baseline_bill(load_8760, old_tariff)
    return {
        "annual_cost": annual_total,
        "monthly_costs": [d["total"] for d in monthly_details],
    }
