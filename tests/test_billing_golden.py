"""Golden regression tests for the billing engine.

Each scenario under ``tests/fixtures/golden/scenarios/`` is loaded,
run through ``run_billing_simulation``, and the summary compared against
a frozen expected JSON.

Seeding baselines (first time, or after an intentional change):

    PV_SIM_REGEN_GOLDEN=1 pytest tests/test_billing_golden.py
    # …review git diff of tests/fixtures/golden/expected/*.json…
    pytest tests/test_billing_golden.py  # confirm stability

Tolerances are tight (±$0.05 on annual totals) — any drift indicates a
billing-engine change that deserves a changelog entry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.billing import run_billing_simulation
from tests.golden_helpers import (
    EXPECTED_DIR,
    build_export_rate,
    build_load,
    build_solar,
    build_tariff,
    list_scenarios,
    load_scenario,
    summarize_result,
)

ANNUAL_TOLERANCE = 0.05  # $/yr
MONTHLY_TOLERANCE = 0.05  # $/mo


def _regen_enabled() -> bool:
    return os.environ.get("PV_SIM_REGEN_GOLDEN") == "1"


def _expected_path(name: str) -> Path:
    return EXPECTED_DIR / f"{name}.json"


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_billing_golden(scenario_name: str):
    scenario = load_scenario(scenario_name)
    load = build_load(scenario["load"])
    solar = build_solar(scenario["solar"])
    export_rate = build_export_rate(scenario["export_rate"])
    tariff = build_tariff(scenario["tariff"])

    result = run_billing_simulation(
        load_8760=load,
        production_8760=solar,
        tariff=tariff,
        export_rates_8760=export_rate,
        nem_regime=scenario["nem_regime"],
        nbc_rate=scenario.get("nbc_rate", 0.0),
        nsc_rate=scenario.get("nsc_rate", 0.04),
        billing_option=scenario.get("billing_option", "ABO"),
    )
    actual = summarize_result(result)

    expected_path = _expected_path(scenario_name)
    if _regen_enabled() or not expected_path.exists():
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        if not _regen_enabled():
            pytest.fail(
                f"Seeded baseline for {scenario_name}. "
                "Re-run without PV_SIM_REGEN_GOLDEN=1 to confirm."
            )
        return

    with expected_path.open() as f:
        expected = json.load(f)

    _compare(actual, expected, scenario_name)


def _compare(actual: dict, expected: dict, name: str) -> None:
    annual_keys = ["annual_total_bill", "annual_energy_cost", "annual_export_credit", "old_rate_annual_baseline"]
    for key in annual_keys:
        a, e = actual.get(key), expected.get(key)
        if a is None and e is None:
            continue
        assert a is not None and e is not None, f"{name}: {key} presence mismatch"
        assert abs(a - e) <= ANNUAL_TOLERANCE, f"{name}: {key} drift — expected {e}, got {a}"

    for col, exp_vals in expected.get("monthly", {}).items():
        act_vals = actual["monthly"].get(col)
        assert act_vals is not None, f"{name}: monthly column {col} missing"
        assert len(act_vals) == len(exp_vals), f"{name}: monthly {col} length mismatch"
        for i, (a, e) in enumerate(zip(act_vals, exp_vals)):
            assert abs(a - e) <= MONTHLY_TOLERANCE, (
                f"{name}: monthly {col} month {i+1} drift — expected {e}, got {a}"
            )
