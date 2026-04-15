"""Equivalence tests for ``modules.simulation.run_simulation``.

The wrapper must be behavior-preserving: its output must match a direct
``run_billing_simulation`` call dollar-for-dollar. Monte Carlo / AI features
build on this contract.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.billing import run_billing_simulation
from modules.simulation import SimulationInputs, run_simulation
from tests.golden_helpers import (
    build_export_rate,
    build_load,
    build_solar,
    build_tariff,
)

TOL = 0.01  # $/yr


def _inputs_for(scenario: dict) -> SimulationInputs:
    return SimulationInputs(
        load_8760=build_load(scenario["load"]),
        production_8760=build_solar(scenario["solar"]),
        tariff=build_tariff(scenario["tariff"]),
        export_rates_8760=build_export_rate(scenario["export_rate"]),
        nem_regime=scenario["nem_regime"],
        nbc_rate=scenario.get("nbc_rate", 0.0),
        nsc_rate=scenario.get("nsc_rate", 0.04),
        billing_option=scenario.get("billing_option", "ABO"),
    )


@pytest.mark.parametrize("nem_regime,nbc", [("NEM-3", 0.0), ("NEM-2", 0.025)])
def test_run_simulation_matches_direct_call(nem_regime, nbc):
    scenario = {
        "tariff": {"kind": "flat", "rate": 0.20, "fixed_monthly": 10.0},
        "load": {"kind": "flat", "value": 10.0},
        "solar": {"kind": "bell", "peak": 5.0},
        "export_rate": {"kind": "flat", "value": 0.05},
        "nem_regime": nem_regime,
        "nbc_rate": nbc,
    }
    inputs = _inputs_for(scenario)

    direct = run_billing_simulation(
        load_8760=inputs.load_8760,
        production_8760=inputs.production_8760,
        tariff=inputs.tariff,
        export_rates_8760=inputs.export_rates_8760,
        nem_regime=inputs.nem_regime,
        nbc_rate=inputs.nbc_rate,
        nsc_rate=inputs.nsc_rate,
        billing_option=inputs.billing_option,
    )

    wrapped = run_simulation(inputs)

    assert abs(wrapped.pv_only_result.annual_bill_with_solar - direct.annual_bill_with_solar) <= TOL
    assert abs(wrapped.billing_result.annual_bill_with_solar - direct.annual_bill_with_solar) <= TOL
    assert wrapped.has_battery is False
    assert wrapped.billing_result is wrapped.pv_only_result


def test_run_simulation_frozen_inputs_support_replace():
    """Monte Carlo relies on ``dataclasses.replace`` producing a valid new
    ``SimulationInputs`` — this pins that behaviour so the frozen dataclass
    doesn't regress into something mutable or non-replace-able."""
    from dataclasses import replace

    scenario = {
        "tariff": {"kind": "flat", "rate": 0.20},
        "load": {"kind": "flat", "value": 5.0},
        "solar": {"kind": "zero"},
        "export_rate": {"kind": "zero"},
        "nem_regime": "NEM-3",
    }
    base = _inputs_for(scenario)
    mutated = replace(base, nbc_rate=0.05)

    assert base.nbc_rate == 0.0
    assert mutated.nbc_rate == 0.05
    # Large arrays should be shared by reference (no unnecessary copy).
    assert mutated.load_8760 is base.load_8760
