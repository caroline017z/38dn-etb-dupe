"""
Battery sizing sweep — evaluate candidate capacities and pick the best.

For each candidate size the dispatch LP is solved and the resulting
total annual bill is computed.  The cheapest option wins.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import BatteryConfig
from .dispatch import dispatch_battery, DispatchResult


@dataclass
class SizingRow:
    """One row of the sizing results table."""
    size_kwh: float
    power_kw: float
    energy_charge: float
    demand_charge: float
    export_credit: float
    net_bill: float


@dataclass
class SizingResult:
    """Complete output of the sizing sweep."""
    best_size_kwh: float
    best_dispatch: DispatchResult
    table: pd.DataFrame          # columns match SizingRow fields


def optimize_capacity_kwh(
    candidate_sizes_kwh: list[float],
    pv_kwh: np.ndarray,
    load_kwh: np.ndarray,
    import_price: np.ndarray,
    export_price: np.ndarray,
    demand_window_masks: dict[str, np.ndarray],
    demand_prices: dict[str, float],
    battery_config: BatteryConfig,
    monthly: bool = False,
    dt_index: "pd.DatetimeIndex | None" = None,
) -> SizingResult:
    """Run dispatch for each candidate battery size and return the best.

    Parameters
    ----------
    candidate_sizes_kwh : list[float]
        Candidate nameplate capacities to evaluate (kWh).
    pv_kwh, load_kwh, import_price, export_price :
        Same arrays passed to ``dispatch_battery``.
    demand_window_masks, demand_prices :
        Same dicts passed to ``dispatch_battery``.
    battery_config : BatteryConfig
        Shared BESS config (``battery_hours``, efficiencies, windows, etc.).
        ``capacity_kwh`` is overridden by each candidate.

    Returns
    -------
    SizingResult
        Contains ``best_size_kwh``, ``best_dispatch``, and a summary
        ``table`` (DataFrame) with one row per candidate.
    """
    N = len(pv_kwh)
    if dt_index is None:
        dt_index = pd.date_range("2023-01-01", periods=N, freq="h")
    month_arr = dt_index.month.values

    rows: list[dict] = []
    best_bill = float("inf")
    best_size = candidate_sizes_kwh[0]
    best_dr: DispatchResult | None = None

    for size in candidate_sizes_kwh:
        power_kw = size / battery_config.battery_hours

        dr = dispatch_battery(
            pv_kwh=pv_kwh,
            load_kwh=load_kwh,
            import_price=import_price,
            export_price=export_price,
            demand_window_masks=demand_window_masks,
            demand_prices=demand_prices,
            battery_config=battery_config,
            capacity_kwh=size,
            monthly=monthly,
        )

        # --- Compute bill components from dispatch arrays ---
        energy_charge = float(np.sum(dr.grid_import_kwh * import_price))
        export_credit = float(np.sum(dr.grid_export_kwh * export_price))

        # Demand charges: monthly peak import × $/kW, summed over periods
        demand_total = 0.0
        for pname, price in demand_prices.items():
            if price <= 0 or pname not in demand_window_masks:
                continue
            mask = demand_window_masks[pname]
            for m in range(1, 13):
                sel = (month_arr == m) & mask
                if sel.any():
                    monthly_peak = float(dr.grid_import_kwh[sel].max())
                    demand_total += monthly_peak * price

        net_bill = energy_charge + demand_total - export_credit

        rows.append({
            "size_kwh": size,
            "power_kw": round(power_kw, 2),
            "energy_charge": round(energy_charge, 2),
            "demand_charge": round(demand_total, 2),
            "export_credit": round(export_credit, 2),
            "net_bill": round(net_bill, 2),
        })

        if net_bill < best_bill:
            best_bill = net_bill
            best_size = size
            best_dr = dr

    table = pd.DataFrame(rows)
    assert best_dr is not None
    return SizingResult(
        best_size_kwh=best_size,
        best_dispatch=best_dr,
        table=table,
    )
