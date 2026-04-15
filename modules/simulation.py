"""Pure-function simulation facade.

Wraps the existing billing + battery engines in a dataclass-in/dataclass-out
API that is independent of Streamlit's ``session_state``. Existence rationale:

* Monte Carlo / tornado sensitivity (Phase 2) needs to re-run billing many
  times with mutated inputs; building every run from session_state is awkward
  and makes parallelism via joblib impractical because ``ScriptRunContext`` is
  not picklable.
* Tests and AI agents want a callable engine without spinning up Streamlit.

Scope: covers the **single-meter custom-billing path** (NEM-1 / NEM-2 / NEM-3
single-meter). ECC and NEM-A aggregation remain wired through ``app.py`` for
now and will migrate in follow-up work — ``BillingEngine`` is defined here so
they can slot in without re-plumbing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import pandas as pd

from .billing import BillingResult, run_billing_simulation
from .tariff import TariffSchedule


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationInputs:
    """Every input the single-meter custom engine needs. Frozen so callers
    can safely share a base instance across Monte-Carlo samples and mutate
    copies via :meth:`dataclasses.replace`.

    Large arrays (load/production/export_rates) are stored by reference —
    callers vectorising over many samples should pre-construct the Series
    once and use ``replace(base, load_8760=...)`` to swap.
    """

    load_8760: pd.Series
    production_8760: pd.Series
    tariff: TariffSchedule
    export_rates_8760: pd.Series  # zeros placeholder is valid for NEM-1/2

    nem_regime: str = "NEM-3"
    nbc_rate: float = 0.0
    nsc_rate: float = 0.04
    billing_option: str = "ABO"

    # Optional battery dispatch. When ``battery_config`` is None or
    # ``battery_capacity_kwh`` <= 0 the run is PV-only.
    battery_config: object | None = None
    battery_capacity_kwh: float = 0.0
    monthly_dispatch: bool = False


@dataclass(frozen=True)
class SimulationResult:
    """Output bundle. ``billing_result`` is the canonical answer
    (PV + battery if enabled, else PV-only). ``pv_only_result`` is always
    populated — downstream savings/projection views need both.
    """

    billing_result: BillingResult
    pv_only_result: BillingResult
    has_battery: bool


class BillingEngine(Protocol):
    """Structural contract for a billing engine.

    Existing implementations (``run_billing_simulation`` and
    ``run_ecc_billing_simulation``) already conform by signature — this
    Protocol documents the expectation so Monte-Carlo / AI callers can
    depend on the contract rather than a specific function.
    """

    def run(self, inputs: SimulationInputs) -> BillingResult: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def run_simulation(inputs: SimulationInputs) -> SimulationResult:
    """Run the single-meter custom billing pipeline.

    Always computes the PV-only baseline. If battery_config and a positive
    capacity are supplied, runs a second pass with LP dispatch and returns
    that as the canonical ``billing_result``.
    """
    pv_only = _run_billing(inputs, with_battery=False)

    has_battery = (
        inputs.battery_config is not None and inputs.battery_capacity_kwh > 0
    )
    if has_battery:
        with_batt = _run_billing(inputs, with_battery=True)
        return SimulationResult(
            billing_result=with_batt,
            pv_only_result=pv_only,
            has_battery=True,
        )
    return SimulationResult(
        billing_result=pv_only,
        pv_only_result=pv_only,
        has_battery=False,
    )


def _run_billing(inputs: SimulationInputs, *, with_battery: bool) -> BillingResult:
    return run_billing_simulation(
        load_8760=inputs.load_8760,
        production_8760=inputs.production_8760,
        tariff=inputs.tariff,
        export_rates_8760=inputs.export_rates_8760,
        battery_config=inputs.battery_config if with_battery else None,
        capacity_kwh=inputs.battery_capacity_kwh if with_battery else 0.0,
        monthly_dispatch=inputs.monthly_dispatch,
        nem_regime=inputs.nem_regime,
        nbc_rate=inputs.nbc_rate,
        nsc_rate=inputs.nsc_rate,
        billing_option=inputs.billing_option,
    )


# ---------------------------------------------------------------------------
# Streamlit adapter
# ---------------------------------------------------------------------------
def inputs_from_session_state(
    session_state,
    *,
    nem_regime: str,
    nbc_rate: float,
    nsc_rate: float,
    billing_option: str,
    export_rates_placeholder: pd.Series,
    include_battery: bool = True,
) -> SimulationInputs:
    """Build a :class:`SimulationInputs` from Streamlit session_state.

    Keeps the session_state-reading logic in one place so tests and the
    Monte-Carlo harness can build inputs directly without touching this
    function. ``export_rates_placeholder`` is a zero-filled 8760 Series
    the caller supplies for the NEM-1/2 case where session_state has no
    export rates loaded.
    """
    export = session_state.get("export_rates") or export_rates_placeholder
    battery_config = session_state.get("battery_config") if include_battery else None
    battery_capacity = (
        float(session_state.get("battery_capacity_kwh", 0.0))
        if include_battery and session_state.get("battery_enabled")
        else 0.0
    )
    monthly_dispatch = bool(session_state.get("battery_fast_dispatch", False))

    return SimulationInputs(
        load_8760=session_state["load_8760"],
        production_8760=session_state["production_8760"],
        tariff=session_state["tariff"],
        export_rates_8760=export,
        nem_regime=nem_regime,
        nbc_rate=nbc_rate,
        nsc_rate=nsc_rate,
        billing_option=billing_option,
        battery_config=battery_config,
        battery_capacity_kwh=battery_capacity,
        monthly_dispatch=monthly_dispatch,
    )


__all__ = [
    "SimulationInputs",
    "SimulationResult",
    "BillingEngine",
    "run_simulation",
    "inputs_from_session_state",
]
