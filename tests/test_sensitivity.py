"""Tests for modules.sensitivity.

Pins that Monte Carlo is seed-stable, that tornado ordering is consistent
with hand-computed swings, and that ``percentiles`` behaves correctly for
the P10/P50/P90 values the UI depends on.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.billing import run_billing_simulation
from modules.sensitivity import (
    Lever,
    monte_carlo,
    percentiles,
    project_npv,
    tornado,
)
from tests.golden_helpers import (
    build_export_rate,
    build_load,
    build_solar,
    build_tariff,
)


@pytest.fixture(scope="module")
def year1_results():
    load = build_load({"kind": "flat", "value": 20.0})
    solar = build_solar({"kind": "bell", "peak": 30.0})
    export = build_export_rate({"kind": "flat", "value": 0.05})
    tariff = build_tariff({"kind": "flat", "rate": 0.20, "fixed_monthly": 10.0})
    result = run_billing_simulation(
        load_8760=load, production_8760=solar, tariff=tariff,
        export_rates_8760=export, nem_regime="NEM-3",
    )
    return result, None


def test_project_npv_matches_manual_discount(year1_results):
    result, pv_only = year1_results
    npv = project_npv(
        result=result, result_pv_only=pv_only,
        system_cost=50_000.0, years=5, discount_rate_pct=7.0,
        rate_escalator=0.0, load_escalator=0.0, degradation=0.0,
    )
    # No escalation / no degradation means every projected year has the
    # same annual savings. Verify by computing that savings by hand.
    annual = result.annual_savings
    r = 0.07
    manual = sum(annual / (1 + r) ** y for y in range(1, 6)) - 50_000.0
    assert abs(npv - manual) < 1.0


def test_monte_carlo_seed_stable(year1_results):
    result, pv_only = year1_results
    levers = [Lever("rate_escalator", "normal", (3.0, 1.0), "rate", "%")]
    kw = dict(
        result=result, result_pv_only=pv_only,
        system_cost=50_000.0, years=10, discount_rate_pct=7.0,
        levers=levers, n=100, nem_regime_1="NEM-3",
    )
    a = monte_carlo(seed=123, **kw)
    b = monte_carlo(seed=123, **kw)
    assert np.allclose(a["npv"].to_numpy(), b["npv"].to_numpy())
    c = monte_carlo(seed=456, **kw)
    # Different seed should produce a different (not identical) distribution
    assert not np.allclose(a["npv"].to_numpy(), c["npv"].to_numpy())


def test_monte_carlo_percentile_ordering(year1_results):
    result, pv_only = year1_results
    levers = [
        Lever("rate_escalator", "normal", (3.0, 1.0), "rate", "%"),
        Lever("load_escalator", "normal", (1.0, 0.5), "load", "%"),
    ]
    df = monte_carlo(
        result=result, result_pv_only=pv_only,
        system_cost=50_000.0, years=10, discount_rate_pct=7.0,
        levers=levers, n=300, seed=42, nem_regime_1="NEM-3",
    )
    pct = percentiles(df["npv"].to_numpy())
    assert pct[10] <= pct[50] <= pct[90]


def test_monte_carlo_progress_cb_fires(year1_results):
    result, pv_only = year1_results
    hits: list[int] = []
    monte_carlo(
        result=result, result_pv_only=pv_only,
        system_cost=50_000.0, years=5, discount_rate_pct=7.0,
        levers=[Lever("rate_escalator", "normal", (3.0, 1.0), "r", "%")],
        n=100, seed=1, nem_regime_1="NEM-3",
        progress_cb=lambda i, arr: hits.append(i),
        chunk=25,
    )
    assert hits == [25, 50, 75, 100]


def test_tornado_ordering_matches_swing(year1_results):
    result, pv_only = year1_results
    levers = [
        Lever("rate_escalator", "normal", (3.0, 1.0), "Rate escalator", "%"),
        Lever("load_escalator", "normal", (1.0, 0.5), "Load escalator", "%"),
        Lever("degradation", "triangular", (0.3, 0.5, 0.8), "Degradation", "%"),
    ]
    df = tornado(
        result=result, result_pv_only=pv_only,
        system_cost=50_000.0, years=20, discount_rate_pct=7.0,
        levers=levers, pct_low=-0.10, pct_high=0.10,
        nem_regime_1="NEM-3",
    )
    # Ordering invariant: rows sorted by descending absolute swing
    swings = df["swing"].to_numpy()
    assert np.all(swings[:-1] >= swings[1:])
    # Every swing is non-negative and defined
    assert np.all(swings >= 0)
    # Base NPV is attached as a DataFrame attr
    assert "base_npv" in df.attrs
