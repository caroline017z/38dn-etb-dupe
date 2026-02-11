"""
Battery dispatch optimizer — CVXPY linear programme.

Solves for the hourly charge/discharge schedule that minimises total
electricity cost  =  energy_import_cost
                   - export_revenue
                   + demand_charges
                   + epsilon * throughput          (tie-breaker)

Supports two modes:
  - **annual** (default): single LP over all N hours
  - **monthly**: 12 smaller LPs with SOC stitched at month boundaries
"""

import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from .config import BatteryConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DispatchResult:
    """Hourly battery dispatch results."""

    grid_import_kwh: np.ndarray           # net grid import after battery
    grid_export_kwh: np.ndarray           # total export (PV + battery)
    pv_export_kwh: np.ndarray             # PV-only portion of export
    batt_export_kwh: np.ndarray           # battery portion of export
    soc_kwh: np.ndarray                   # state of charge (end-of-hour)
    batt_charge_kwh: np.ndarray           # energy into battery (from PV)
    batt_discharge_to_load_kwh: np.ndarray  # battery -> load
    batt_discharge_to_grid_kwh: np.ndarray  # battery -> grid
    solver_status: str
    objective_value: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_window_mask(
    start_hour: int,
    end_hour: int,
    hours: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for an inclusive hour-of-day window.

    Handles midnight-crossing windows (e.g. start=22, end=4).
    """
    if start_hour <= end_hour:
        return (hours >= start_hour) & (hours <= end_hour)
    # wraps midnight
    return (hours >= start_hour) | (hours <= end_hour)


def _aggregate_status(results: list[DispatchResult]) -> str:
    """Combine solver statuses from multiple monthly LPs."""
    statuses = [r.solver_status for r in results]
    if all(s == cp.OPTIMAL for s in statuses):
        return cp.OPTIMAL
    if all(s in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) for s in statuses):
        return cp.OPTIMAL_INACCURATE
    return "mixed: " + ", ".join(f"m{i+1}={s}" for i, s in enumerate(statuses))


def _concatenate_results(results: list[DispatchResult]) -> DispatchResult:
    """Stitch monthly DispatchResults into one annual result."""
    return DispatchResult(
        grid_import_kwh=np.concatenate([r.grid_import_kwh for r in results]),
        grid_export_kwh=np.concatenate([r.grid_export_kwh for r in results]),
        pv_export_kwh=np.concatenate([r.pv_export_kwh for r in results]),
        batt_export_kwh=np.concatenate([r.batt_export_kwh for r in results]),
        soc_kwh=np.concatenate([r.soc_kwh for r in results]),
        batt_charge_kwh=np.concatenate([r.batt_charge_kwh for r in results]),
        batt_discharge_to_load_kwh=np.concatenate(
            [r.batt_discharge_to_load_kwh for r in results]
        ),
        batt_discharge_to_grid_kwh=np.concatenate(
            [r.batt_discharge_to_grid_kwh for r in results]
        ),
        solver_status=_aggregate_status(results),
        objective_value=sum(r.objective_value for r in results),
    )


# ---------------------------------------------------------------------------
# Core LP solver (single chunk — a month or the full year)
# ---------------------------------------------------------------------------
def _solve_single_lp(
    pv_kwh: np.ndarray,
    load_kwh: np.ndarray,
    import_price: np.ndarray,
    export_price: np.ndarray,
    demand_window_masks: dict[str, np.ndarray],
    demand_prices: dict[str, float],
    battery_config: BatteryConfig,
    capacity_kwh: float,
    initial_soc: float,
    dt_index: pd.DatetimeIndex,
) -> DispatchResult:
    """Solve the battery dispatch LP for a single contiguous time chunk.

    This is the inner workhorse.  ``dispatch_battery`` calls it once
    (annual mode) or twelve times (monthly mode).
    """
    N = len(pv_kwh)
    cfg = battery_config

    # ----- derived parameters -----
    power_kw = capacity_kwh / cfg.battery_hours
    min_soc = cfg.min_soc_pct / 100.0 * capacity_kwh
    max_soc = cfg.max_soc_pct / 100.0 * capacity_kwh

    frac = min(cfg.discharge_limit_pct / 100.0, 0.9999)
    alpha = frac / (1.0 - frac)

    # pre-computed constants
    net_load = load_kwh - pv_kwh               # positive  = deficit
    surplus_pv = np.maximum(0.0, -net_load)     # available PV for charging
    load_deficit = np.maximum(0.0, net_load)    # unmet load after PV

    # hour-of-day masks
    hours = dt_index.hour.values
    month_arr = dt_index.month.values           # 1-12

    charge_mask = _build_window_mask(
        cfg.charge_window_start, cfg.charge_window_end, hours,
    ).astype(float)
    discharge_mask = _build_window_mask(
        cfg.discharge_window_start, cfg.discharge_window_end, hours,
    ).astype(float)

    # ================================================================
    # CVXPY variables
    # ================================================================
    batt_charge   = cp.Variable(N, nonneg=True, name="charge")
    batt_to_load  = cp.Variable(N, nonneg=True, name="to_load")
    batt_to_grid  = cp.Variable(N, nonneg=True, name="to_grid")
    soc           = cp.Variable(N, nonneg=True, name="soc")
    grid_import   = cp.Variable(N, nonneg=True, name="g_imp")
    grid_export   = cp.Variable(N, nonneg=True, name="g_exp")

    # monthly peak demand vars — one per (period, month)
    peak_kw: dict[str, cp.Variable] = {}
    for pname in demand_prices:
        if demand_prices[pname] > 0 and pname in demand_window_masks:
            peak_kw[pname] = cp.Variable(12, nonneg=True, name=f"pk_{pname}")

    discharge_total = batt_to_load + batt_to_grid

    # ================================================================
    # Constraints
    # ================================================================
    constraints: list[cp.Constraint] = []

    # 1. nodal energy balance
    constraints.append(
        grid_import - grid_export
        == net_load + batt_charge - batt_to_load - batt_to_grid
    )

    # 2. PV-only charging (battery may only absorb surplus PV)
    constraints.append(batt_charge <= surplus_pv)

    # 3. battery-to-load cannot exceed unmet load
    constraints.append(batt_to_load <= load_deficit)

    # 4. power limits gated by allowed windows
    constraints.append(batt_charge <= power_kw * charge_mask)
    constraints.append(discharge_total <= power_kw * discharge_mask)

    # 5. SOC dynamics
    #    SOC[t] = SOC[t-1] + charge*η_c − discharge/η_d
    constraints.append(
        soc[0] == initial_soc
        + batt_charge[0] * cfg.charge_eff
        - discharge_total[0] / cfg.discharge_eff
    )
    if N > 1:
        constraints.append(
            soc[1:] == soc[:-1]
            + batt_charge[1:] * cfg.charge_eff
            - discharge_total[1:] / cfg.discharge_eff
        )

    # 6. SOC bounds
    constraints.append(soc >= min_soc)
    constraints.append(soc <= max_soc)

    # 7. export-fraction limit:
    #    batt_to_grid <= alpha * batt_to_load
    constraints.append(batt_to_grid <= alpha * batt_to_load)

    # 8. demand-charge peak constraints
    for pname, pk_var in peak_kw.items():
        mask = demand_window_masks[pname]
        for m_idx in range(12):
            month_num = m_idx + 1
            sel = np.where((month_arr == month_num) & mask)[0]
            if sel.size > 0:
                constraints.append(grid_import[sel] <= pk_var[m_idx])

    # ================================================================
    # Objective
    # ================================================================
    EPSILON = 1e-5

    energy_cost   = cp.sum(cp.multiply(import_price, grid_import))
    export_rev    = cp.sum(cp.multiply(export_price, grid_export))
    demand_cost   = sum(
        demand_prices[pname] * cp.sum(pk_var)
        for pname, pk_var in peak_kw.items()
    ) if peak_kw else 0
    throughput    = EPSILON * cp.sum(batt_charge + batt_to_load + batt_to_grid)

    objective = cp.Minimize(energy_cost - export_rev + demand_cost + throughput)

    # ================================================================
    # Solve
    # ================================================================
    prob = cp.Problem(objective, constraints)

    # Try solvers in preference order for LP
    _SOLVERS = [cp.CLARABEL, cp.ECOS, cp.HIGHS, cp.SCS]
    solved = False
    for solver in _SOLVERS:
        try:
            prob.solve(solver=solver)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solved = True
                break
        except (cp.SolverError, Exception):
            continue

    if not solved:
        # last-resort: let CVXPY pick
        prob.solve()

    # ================================================================
    # Extract results
    # ================================================================
    def _val(var: cp.Variable) -> np.ndarray:
        v = var.value
        if v is None:
            return np.zeros(var.shape[0])
        arr = np.asarray(v).flatten()
        # Replace NaN with 0 for numerical robustness
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # Check if solver actually found a solution
    solver_ok = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if not solver_ok:
        warnings.warn(
            f"Battery dispatch LP failed (status={prob.status}). "
            "Falling back to PV-only grid flows for this period.",
            stacklevel=2,
        )
        # Fall back to PV-only: no battery action
        pv_import = np.maximum(0.0, net_load)
        pv_export = np.maximum(0.0, -net_load)
        return DispatchResult(
            grid_import_kwh=pv_import,
            grid_export_kwh=pv_export,
            pv_export_kwh=pv_export,
            batt_export_kwh=np.zeros(N),
            soc_kwh=np.full(N, min_soc),
            batt_charge_kwh=np.zeros(N),
            batt_discharge_to_load_kwh=np.zeros(N),
            batt_discharge_to_grid_kwh=np.zeros(N),
            solver_status=prob.status or "failed",
            objective_value=float("inf"),
        )

    g_imp = _val(grid_import)
    g_exp = _val(grid_export)
    b_chg = _val(batt_charge)
    b_tl  = _val(batt_to_load)
    b_tg  = _val(batt_to_grid)
    s     = _val(soc)

    # ---- Post-solve cleanup: numerical robustness ----
    # 1. Clip solver noise to zero (all variables are non-negative)
    g_imp = np.maximum(0.0, g_imp)
    g_exp = np.maximum(0.0, g_exp)
    b_chg = np.maximum(0.0, b_chg)
    b_tl  = np.maximum(0.0, b_tl)
    b_tg  = np.maximum(0.0, b_tg)
    s     = np.clip(s, min_soc, max_soc)

    # 2. Enforce mutual exclusivity: battery cannot charge and discharge
    #    in the same hour.  The LP cost structure (round-trip losses +
    #    throughput penalty) already prevents this, but solver noise can
    #    leave tiny residuals.  Zero out the smaller side.
    discharge = b_tl + b_tg
    both_mask = (b_chg > 1e-6) & (discharge > 1e-6)
    if both_mask.any():
        for i in np.where(both_mask)[0]:
            if b_chg[i] >= discharge[i]:
                b_chg[i] -= discharge[i]
                b_tl[i] = 0.0
                b_tg[i] = 0.0
            else:
                ratio = b_tl[i] / discharge[i] if discharge[i] > 0 else 1.0
                b_tl[i] -= b_chg[i] * ratio
                b_tg[i] -= b_chg[i] * (1.0 - ratio)
                b_chg[i] = 0.0
        b_chg = np.maximum(0.0, b_chg)
        b_tl  = np.maximum(0.0, b_tl)
        b_tg  = np.maximum(0.0, b_tg)

    # 3. Enforce no grid charging: battery only charges from surplus PV
    #    (LP constraint: batt_charge <= surplus_pv; clip any residual)
    b_chg = np.minimum(b_chg, surplus_pv)

    # derive PV-only export = total export − battery export
    pv_exp = np.maximum(0.0, g_exp - b_tg)

    return DispatchResult(
        grid_import_kwh=g_imp,
        grid_export_kwh=g_exp,
        pv_export_kwh=pv_exp,
        batt_export_kwh=b_tg,
        soc_kwh=s,
        batt_charge_kwh=b_chg,
        batt_discharge_to_load_kwh=b_tl,
        batt_discharge_to_grid_kwh=b_tg,
        solver_status=prob.status or "unknown",
        objective_value=float(prob.value) if prob.value is not None else float("inf"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def dispatch_battery(
    pv_kwh: np.ndarray,
    load_kwh: np.ndarray,
    import_price: np.ndarray,
    export_price: np.ndarray,
    demand_window_masks: dict[str, np.ndarray],
    demand_prices: dict[str, float],
    battery_config: BatteryConfig,
    capacity_kwh: float,
    monthly: bool = False,
) -> DispatchResult:
    """Solve the optimal battery dispatch LP.

    Parameters
    ----------
    pv_kwh : array (N,)
        Hourly PV production in kWh.
    load_kwh : array (N,)
        Hourly site load in kWh.
    import_price : array (N,)
        Hourly grid import price ($/kWh).
    export_price : array (N,)
        Hourly export compensation rate ($/kWh).
    demand_window_masks : dict[str, array(N,) bool]
        For each demand-charge period name (e.g. ``"flat"``, ``"onpeak"``),
        a boolean mask indicating which hours belong to that period.
    demand_prices : dict[str, float]
        $/kW demand charge for each period name.  Keys must be a subset
        of *demand_window_masks*.
    battery_config : BatteryConfig
        BESS parameters (efficiency, SOC limits, windows, etc.).
    capacity_kwh : float
        Nameplate energy capacity of the battery in kWh.
    monthly : bool
        When *True* and ``N == 8760``, decompose into 12 monthly LPs
        with SOC stitched at month boundaries.  Falls back to a single
        LP for non-8760 arrays.

    Returns
    -------
    DispatchResult
        Hourly arrays + solver metadata.
    """
    N = len(pv_kwh)
    assert len(load_kwh) == N
    assert len(import_price) == N
    assert len(export_price) == N

    cfg = battery_config
    min_soc = cfg.min_soc_pct / 100.0 * capacity_kwh
    dt_index = pd.date_range("2023-01-01", periods=N, freq="h")

    # ---- Monthly decomposition path ----
    if monthly and N == 8760:
        month_arr = dt_index.month.values
        month_slices: list[tuple[int, int]] = []
        for m in range(1, 13):
            indices = np.where(month_arr == m)[0]
            month_slices.append((int(indices[0]), int(indices[-1]) + 1))

        current_soc = min_soc
        monthly_results: list[DispatchResult] = []

        for start, end in month_slices:
            sliced_masks = {
                k: v[start:end] for k, v in demand_window_masks.items()
            }
            result_m = _solve_single_lp(
                pv_kwh=pv_kwh[start:end],
                load_kwh=load_kwh[start:end],
                import_price=import_price[start:end],
                export_price=export_price[start:end],
                demand_window_masks=sliced_masks,
                demand_prices=demand_prices,
                battery_config=cfg,
                capacity_kwh=capacity_kwh,
                initial_soc=current_soc,
                dt_index=dt_index[start:end],
            )
            monthly_results.append(result_m)
            # Stitch: end-of-month SOC → start-of-next-month
            end_soc = float(result_m.soc_kwh[-1])
            # Guard against invalid SOC from failed solvers
            if not (min_soc <= end_soc <= capacity_kwh) or np.isnan(end_soc):
                current_soc = min_soc
            else:
                current_soc = end_soc

        return _concatenate_results(monthly_results)

    # ---- Warn if monthly requested but N != 8760 ----
    if monthly and N != 8760:
        warnings.warn(
            f"monthly=True requires N=8760 but got N={N}; "
            "falling back to single LP.",
            stacklevel=2,
        )

    # ---- Single-LP path (annual or short arrays) ----
    return _solve_single_lp(
        pv_kwh=pv_kwh,
        load_kwh=load_kwh,
        import_price=import_price,
        export_price=export_price,
        demand_window_masks=demand_window_masks,
        demand_prices=demand_prices,
        battery_config=cfg,
        capacity_kwh=capacity_kwh,
        initial_soc=min_soc,
        dt_index=dt_index,
    )
