"""
Demand charge calculation module.

Computes both non-coincident (flat) demand charges and TOU-period demand charges
on a monthly basis, based on net import kW values.
"""

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
        flat_demand_charge = _calc_flat_demand(flat_demand_kw, tariff)

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


def _calc_flat_demand(peak_kw: float, tariff: TariffSchedule) -> float:
    """Calculate non-coincident demand charge using flat demand structure."""
    if not tariff.demand_flat_structure:
        return 0.0

    total = 0.0
    remaining_kw = peak_kw

    # Flat demand can have multiple tiers — typically month-grouped.
    # Use the first period's tiers (most common structure).
    # OpenEI flatdemandstructure is a list of month-groups, each with tiers.
    for period_tiers in tariff.demand_flat_structure:
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
        break  # Use first period only for flat demand

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

    # Build period assignment for each hour in the month
    periods = []
    for dt in month_data.index:
        hour = dt.hour
        month_idx = dt.month - 1  # 0-indexed for OpenEI schedule
        is_weekend = dt.weekday() >= 5

        if is_weekend and tariff.demand_weekend_schedule:
            period = tariff.demand_weekend_schedule[month_idx][hour]
        elif tariff.demand_weekday_schedule:
            period = tariff.demand_weekday_schedule[month_idx][hour]
        else:
            period = 0
        periods.append(period)

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
