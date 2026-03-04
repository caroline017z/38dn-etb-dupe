"""
Load adjustment for existing solar (decommission) scenarios.

When a customer is decommissioning an existing solar system, the utility's
interval data shows NET consumption (true_load - existing_solar). These
functions recover the true gross load by adding the existing solar production
back to the load data.
"""

import pandas as pd
import numpy as np


def adjust_load_single_meter(
    raw_load_8760: pd.Series,
    existing_solar_8760: pd.Series,
) -> pd.Series:
    """Add existing solar production back to net-metered load to recover gross load.

    Parameters
    ----------
    raw_load_8760 : pd.Series
        Hourly net consumption from utility interval data (8760 values).
    existing_solar_8760 : pd.Series
        Hourly production estimate for the old solar system (8760 values).

    Returns
    -------
    pd.Series
        Adjusted gross load, floored at zero.
    """
    adjusted = raw_load_8760 + existing_solar_8760.values
    return adjusted.clip(lower=0)


def adjust_loads_nema(
    meter_loads: dict[int, pd.Series],
    existing_solar_8760: pd.Series,
    selected_meter_indices: list[int],
) -> dict[int, pd.Series]:
    """Distribute existing solar production across selected NEM-A meters proportionally.

    At each hour, the existing solar is split among selected meters based on
    their share of total selected-meter consumption. If all selected loads are
    zero at a given hour, equal shares (1/N) are used as fallback.

    Parameters
    ----------
    meter_loads : dict[int, pd.Series]
        Mapping of meter index to hourly load Series.
    existing_solar_8760 : pd.Series
        Hourly production estimate for the old solar system (8760 values).
    selected_meter_indices : list[int]
        Indices of meters that the existing system was offsetting.

    Returns
    -------
    dict[int, pd.Series]
        Updated meter loads — selected meters adjusted, unselected unchanged.
    """
    if not selected_meter_indices:
        return meter_loads

    solar_vals = np.asarray(existing_solar_8760.values, dtype=float)
    n_selected = len(selected_meter_indices)

    # Stack selected meter loads into a 2D array (meters x hours)
    selected_arrays = {}
    for idx in selected_meter_indices:
        selected_arrays[idx] = np.asarray(meter_loads[idx].values, dtype=float)

    # Compute total selected load at each hour
    total_selected = np.zeros(len(solar_vals))
    for arr in selected_arrays.values():
        total_selected += arr

    result = dict(meter_loads)  # shallow copy

    for idx in selected_meter_indices:
        load_arr = selected_arrays[idx]
        # Compute hourly share: load_m / total_selected, with equal fallback
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(
                total_selected > 0,
                load_arr / total_selected,
                1.0 / n_selected,
            )
        adjusted = load_arr + solar_vals * share
        adjusted = np.clip(adjusted, 0, None)
        result[idx] = pd.Series(
            adjusted, index=meter_loads[idx].index, name=meter_loads[idx].name
        )

    return result
