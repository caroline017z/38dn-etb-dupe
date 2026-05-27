"""Regression tests for the indexed-tariff PPA backsolve.

Locks in the per-year flat-%-of-offset semantics (the 2026-05 regime-switch
fix) and the monthly↔annual reconciliation. These builders previously had no
test coverage.
"""
import pandas as pd
import pytest

from modules.outputs import (
    build_indexed_tariff_annual,
    build_indexed_tariff_monthly,
)

NUM_YEARS_1 = 5
TERM = 20


def _annual_proj(nem3_offset_frac: float = 0.22) -> pd.DataFrame:
    """NEM-2 (big offset) for 5 yrs, then NEM-3 (crushed offset)."""
    rows = []
    for yr in range(1, TERM + 1):
        bill_no = 3000.0 * (1.03 ** (yr - 1))
        frac = 0.55 if yr <= NUM_YEARS_1 else nem3_offset_frac
        util_sav = frac * bill_no
        rows.append({
            "Year": yr, "Calendar Year": 2026 + yr - 1,
            "Bill w/o Solar ($)": bill_no,
            "Bill w/ Solar ($)": bill_no - util_sav,
            "Solar (kWh)": 12000.0,
        })
    return pd.DataFrame(rows)


def _monthly_proj() -> pd.DataFrame:
    """2 years × 12 months, seasonal solar (winter months near zero),
    NEM-2 in yr 1 → NEM-3 in yr 2."""
    rows = []
    for yr in (1, 2):
        for m in range(1, 13):
            # seasonal solar: low in winter, peak in summer; Jan/Dec ~0
            seasonal = max(0.0, 1500.0 * (1 - abs(m - 7) / 6.0))
            base_bill = 300.0
            frac = 0.55 if yr == 1 else 0.22
            util_sav = frac * base_bill * (seasonal / 1000.0)
            rows.append({
                "Year": yr, "Month": m, "Calendar Year": 2025 + yr,
                "Baseline Bill ($)": base_bill,
                "Net Bill ($)": base_bill - util_sav,
                "Solar (kWh)": seasonal,
            })
    return pd.DataFrame(rows)


def test_annual_flat_pct_of_offset_each_year():
    """Every year — across the NEM-2→NEM-3 switch — delivers the savings
    target as a flat % of that year's offset."""
    df = build_indexed_tariff_annual(
        _annual_proj(), base_savings_pct=10.0,
        nem_regime_2="NEM-3", num_years_1=NUM_YEARS_1,
    )
    for _, r in df.iterrows():
        offset = r["Utility Savings ($)"]
        assert offset > 0
        assert r["Customer Savings ($)"] / offset == pytest.approx(0.10, abs=1e-3)


def test_annual_floor_when_offset_below_target():
    """When the offset can't reach the target, PPA floors at $0 and the
    customer keeps the full (smaller) offset."""
    # Target 30% but NEM-3 offset is only ~10% of the bill in absolute terms;
    # for the rate to floor we need offset*(1-frac) < 0 → impossible here, so
    # instead use a target so high the rate would go negative: pick a tiny
    # offset year and a 99% target is still feasible. Force infeasibility by
    # making bill_with_solar exceed bill_no (negative offset).
    proj = _annual_proj()
    proj.loc[proj["Year"] == 10, "Bill w/ Solar ($)"] = (
        proj.loc[proj["Year"] == 10, "Bill w/o Solar ($)"] + 50.0
    )
    df = build_indexed_tariff_annual(
        proj, base_savings_pct=10.0,
        nem_regime_2="NEM-3", num_years_1=NUM_YEARS_1,
    )
    row10 = df[df["Year"] == 10].iloc[0]
    assert row10["PPA Rate ($/kWh)"] == 0.0
    # customer keeps the full (negative) offset; no PPA piled on top
    assert row10["Customer Savings ($)"] == pytest.approx(row10["Utility Savings ($)"])


def test_monthly_reconciles_to_annual_and_constant_rate_within_year():
    """The monthly table applies one rate per year, so monthly customer
    savings sum to the year's target % of the year's offset, and every month
    in a year shows the same PPA rate."""
    mdf = build_indexed_tariff_monthly(
        _monthly_proj(), base_savings_pct=10.0,
        nem_regime_2="NEM-3", num_years_1=1,
    )
    for yr in (1, 2):
        yr_rows = mdf[mdf["Year"] == yr]
        # one rate per year
        assert yr_rows["PPA Rate ($/kWh)"].nunique() == 1
        # monthly savings reconcile to 10% of the year's aggregate offset
        offset = yr_rows["Utility Savings ($)"].sum()
        cust = yr_rows["Customer Savings ($)"].sum()
        assert cust / offset == pytest.approx(0.10, abs=1e-3)


def test_no_escalator_kwargs_accepted():
    """The dead escalator params are gone — passing them should now raise."""
    with pytest.raises(TypeError):
        build_indexed_tariff_annual(
            _annual_proj(), base_savings_pct=10.0, ppa_escalator_pct=2.9,
        )
