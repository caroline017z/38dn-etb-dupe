"""
Demand charge calculation module.

Computes both non-coincident (flat) demand charges and TOU-period demand charges
on a monthly basis, based on net import kW values.
"""

import warnings

import pandas as pd
import numpy as np
from .tariff import TariffSchedule


def calculate_monthly_demand_charges(
    import_kwh: pd.Series,
    tariff: TariffSchedule,
) -> pd.DataFrame:
    """
    Calculate monthly demand charges from hourly net import data.

    Since intervals are 1-hour, import_kwh ≈ import_kW for demand purposes.

    Args:
        import_kwh: 8760-length Series of hourly import energy (kWh). Index is datetime.
        tariff: Parsed TariffSchedule object.

    Returns:
        DataFrame with columns:
            month (1-12), flat_demand_kw, flat_demand_charge,
            tou_demand_charges (dict by period), total_demand_charge
    """
    results = []

    for month_num in range(1, 13):
        month_mask = import_kwh.index.month == month_num
        month_data = import_kwh[month_mask]

        if month_data.empty:
            results.append({
                "month": month_num,
                "flat_demand_kw": 0.0,
                "flat_demand_charge": 0.0,
                "tou_demand_details": {},
                "tou_demand_charge": 0.0,
                "total_demand_charge": 0.0,
            })
            continue

        # --- Non-coincident (flat) demand charge ---
        flat_demand_kw = month_data.max()
        flat_demand_charge = _calc_flat_demand(flat_demand_kw, month_num, tariff)

        # --- TOU-period demand charges ---
        tou_demand_charge, tou_details = _calc_tou_demand(
            month_data, month_num, tariff
        )

        results.append({
            "month": month_num,
            "flat_demand_kw": flat_demand_kw,
            "flat_demand_charge": flat_demand_charge,
            "tou_demand_details": tou_details,
            "tou_demand_charge": tou_demand_charge,
            "total_demand_charge": flat_demand_charge + tou_demand_charge,
        })

    return pd.DataFrame(results)


def _calc_flat_demand(peak_kw: float, month_num: int, tariff: TariffSchedule) -> float:
    """Calculate non-coincident demand charge using flat demand structure.

    Uses ``flatdemandmonths`` (month-to-period mapping) to select the correct
    seasonal period from ``flatdemandstructure``.  Falls back to period 0 when
    the mapping is absent.
    """
    if not tariff.demand_flat_structure:
        return 0.0

    # Determine which period applies to this month
    month_idx = month_num - 1  # 0-indexed
    if tariff.demand_flat_months and month_idx < len(tariff.demand_flat_months):
        period_idx = tariff.demand_flat_months[month_idx]
    else:
        period_idx = 0

    if period_idx >= len(tariff.demand_flat_structure):
        period_idx = 0

    period_tiers = tariff.demand_flat_structure[period_idx]

    if len(period_tiers) > 1:
        warnings.warn(
            f"Flat demand period {period_idx} has {len(period_tiers)} tiers; "
            "only first-tier pricing is used. Tiered demand pricing is not yet supported.",
            stacklevel=2,
        )

    total = 0.0
    remaining_kw = peak_kw

    for tier in period_tiers:
        tier_max = tier.get("max")
        rate = tier["effective_rate"]

        if tier_max is not None and tier_max > 0:
            applicable_kw = min(remaining_kw, tier_max)
        else:
            applicable_kw = remaining_kw

        total += applicable_kw * rate
        remaining_kw -= applicable_kw

        if remaining_kw <= 0:
            break

    return total


def _calc_tou_demand(
    month_data: pd.Series,
    month_num: int,
    tariff: TariffSchedule,
) -> tuple[float, dict]:
    """
    Calculate TOU-period demand charges for a given month.

    Finds the peak kW in each TOU demand period and applies the period's rate.

    Returns:
        (total_tou_charge, details_dict)
        details_dict maps period_index -> {"peak_kw": float, "rate": float, "charge": float}
    """
    if not tariff.demand_rate_structure or not tariff.demand_weekday_schedule:
        return 0.0, {}

    # Build period assignment for each hour in the month (vectorized)
    hours = month_data.index.hour
    month_indices = month_data.index.month - 1  # 0-indexed for OpenEI schedule
    is_weekend = month_data.index.weekday >= 5

    weekday_arr = np.array(tariff.demand_weekday_schedule)
    periods = weekday_arr[month_indices, hours]

    if tariff.demand_weekend_schedule:
        weekend_arr = np.array(tariff.demand_weekend_schedule)
        periods = np.where(is_weekend, weekend_arr[month_indices, hours], periods)

    period_series = pd.Series(periods, index=month_data.index)

    # Find peak kW in each unique period
    unique_periods = period_series.unique()
    total_charge = 0.0
    details = {}

    for period_idx in unique_periods:
        period_mask = period_series == period_idx
        peak_kw = month_data[period_mask].max()

        # Get rate for this period
        if period_idx < len(tariff.demand_rate_structure):
            period_tiers = tariff.demand_rate_structure[period_idx]
            if period_tiers:
                if len(period_tiers) > 1:
                    warnings.warn(
                        f"TOU demand period {period_idx} has {len(period_tiers)} tiers; "
                        "only the first tier rate is used. Tiered demand pricing is not yet supported.",
                        stacklevel=2,
                    )
                rate = period_tiers[0]["effective_rate"]
            else:
                rate = 0.0
        else:
            rate = 0.0

        charge = peak_kw * rate
        total_charge += charge

        details[int(period_idx)] = {
            "peak_kw": peak_kw,
            "rate": rate,
            "charge": charge,
        }

    return total_charge, details
