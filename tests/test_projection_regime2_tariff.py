"""
Tests for the regime-2 tariff swap in build_annual_projection
(modules/outputs.py).

Feature: after a NEM regime switch year, the multi-year projection can re-bill
post-transition years on a SECOND tariff (already billed into a second
BillingResult by the caller) rather than carrying tariff #1's costs.

Contract:
  - `result_regime2: BillingResult | None = None` kwarg.
  - When result_regime2 is provided AND nem_regime_2 / num_years_1 are set,
    projection years with yr > num_years_1 derive their YEAR-1 BASE billing
    components from result_regime2 (escalated from year 1 with the same
    rate_mult, degradation, export-regime context).
  - When result_regime2 is None, behavior is byte-identical to today.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.tariff import TariffSchedule
from modules.billing import run_billing_simulation
from modules.outputs import build_annual_projection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dt_index(year: int = 2025) -> pd.DatetimeIndex:
    return pd.date_range(start=f"{year}-01-01", periods=8760, freq="h")


def _const_series(val: float, year: int = 2025) -> pd.Series:
    return pd.Series(np.full(8760, float(val)), index=_make_dt_index(year))


def _make_flat_tariff(rate: float = 0.20, fixed_monthly: float = 10.0) -> TariffSchedule:
    """Single-period flat energy rate, no demand charges."""
    energy_rate_structure = [
        [{"rate": rate, "adj": 0.0, "max": None, "unit": "kWh", "effective_rate": rate}],
    ]
    schedule = [[0] * 24 for _ in range(12)]
    return TariffSchedule(
        label="test_flat",
        name="Flat Test",
        utility="PG&E",
        fixed_monthly_charge=fixed_monthly,
        min_monthly_charge=0.0,
        energy_rate_structure=energy_rate_structure,
        energy_weekday_schedule=schedule,
        energy_weekend_schedule=schedule,
    )


def _run(rate: float):
    """A simple net-importer NEM-3 run on a flat tariff at the given rate."""
    load = _const_series(10.0)     # 10 kWh/hr load
    solar = _const_series(3.0)     # 3 kWh/hr solar -> always net importer
    export_rates = _const_series(0.05)
    return run_billing_simulation(
        load, solar, _make_flat_tariff(rate=rate), export_rates,
        nem_regime="NEM-3", nsc_rate=0.0,
    )


# ---------------------------------------------------------------------------
# 1. result_regime2=None -> identical to baseline (single + dual-regime cases)
# ---------------------------------------------------------------------------
class TestRegime2NoneByteIdentical:
    def test_single_regime_identical(self):
        """No regime switch, no result_regime2 -> unchanged output."""
        result = _run(0.20)
        kwargs = dict(
            result=result,
            system_cost=100000.0,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=10,
            degradation_pct=0.5,
        )
        baseline = build_annual_projection(**kwargs)
        with_none = build_annual_projection(**kwargs, result_regime2=None)
        pd.testing.assert_frame_equal(baseline, with_none)

    def test_dual_regime_no_tariff2_identical(self):
        """Regime switch configured but no result_regime2 -> unchanged output.

        This is the critical golden-preserving case: a regime switch alone must
        NOT change billing-base selection when result_regime2 is omitted.
        """
        result = _run(0.20)
        kwargs = dict(
            result=result,
            system_cost=100000.0,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=10,
            nem_regime_1="NEM-3 / NVBT",
            nem_regime_2="NEM-2",
            num_years_1=4,
            degradation_pct=0.5,
        )
        baseline = build_annual_projection(**kwargs)
        with_none = build_annual_projection(**kwargs, result_regime2=None)
        pd.testing.assert_frame_equal(baseline, with_none)


# ---------------------------------------------------------------------------
# 2. With a differing result_regime2, post-transition years flip the base
# ---------------------------------------------------------------------------
class TestRegime2Swap:
    def test_post_transition_years_use_regime2_base(self):
        """Tariff #2 has a higher energy rate; post-switch years' energy/bill
        must reflect regime-2's (escalated) base, while pre-switch years match
        regime-1's base.
        """
        result = _run(0.20)            # tariff #1: $0.20/kWh
        result_regime2 = _run(0.40)    # tariff #2: $0.40/kWh (higher energy cost)

        num_years_1 = 4
        common = dict(
            system_cost=100000.0,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=8,
            nem_regime_1="NEM-3 / NVBT",
            nem_regime_2="NEM-3 / NVBT",   # same export regime; only base bill differs
            num_years_1=num_years_1,
            degradation_pct=0.5,
        )

        # Reference: regime-1 only (what today's single-tariff code produces).
        ref = build_annual_projection(result=result, **common)
        # Reference: what a pure regime-2 projection would look like.
        ref2 = build_annual_projection(result=result_regime2, **common)

        # New: swap to regime 2 after the transition.
        swapped = build_annual_projection(
            result=result, result_regime2=result_regime2, **common,
        )

        col = "Energy ($)"

        # Pre-transition years (yr <= num_years_1) match regime 1.
        for yr in range(num_years_1):
            assert swapped[col].iloc[yr] == pytest.approx(ref[col].iloc[yr]), \
                f"pre-transition year {yr+1} energy should match regime 1"

        # Post-transition years (yr > num_years_1) match regime 2's escalated base.
        for yr in range(num_years_1, common["years"]):
            assert swapped[col].iloc[yr] == pytest.approx(ref2[col].iloc[yr]), \
                f"post-transition year {yr+1} energy should match regime 2"
            # And must differ from regime 1 (since tariff #2 is more expensive).
            assert swapped[col].iloc[yr] != pytest.approx(ref[col].iloc[yr]), \
                f"post-transition year {yr+1} energy should differ from regime 1"

    def test_switch_year_boundary_flips(self):
        """The flip happens exactly at yr > num_years_1 (year num_years_1+1)."""
        result = _run(0.20)
        result_regime2 = _run(0.40)
        num_years_1 = 3

        common = dict(
            system_cost=100000.0,
            rate_escalator_pct=0.0,    # no escalation -> isolate the base swap
            load_escalator_pct=0.0,
            years=6,
            nem_regime_1="NEM-3 / NVBT",
            nem_regime_2="NEM-3 / NVBT",
            num_years_1=num_years_1,
            degradation_pct=0.0,
        )
        ref = build_annual_projection(result=result, **common)
        ref2 = build_annual_projection(result=result_regime2, **common)
        swapped = build_annual_projection(
            result=result, result_regime2=result_regime2, **common,
        )

        col = "Energy ($)"

        # Last regime-1 year still regime 1, first regime-2 year is regime 2.
        assert swapped[col].iloc[num_years_1 - 1] == pytest.approx(ref[col].iloc[num_years_1 - 1])
        assert swapped[col].iloc[num_years_1] == pytest.approx(ref2[col].iloc[num_years_1])
        assert swapped[col].iloc[num_years_1] != pytest.approx(ref[col].iloc[num_years_1])


# ---------------------------------------------------------------------------
# 3. Net-importer / no-switch sanity (regime-2 base never consulted)
# ---------------------------------------------------------------------------
class TestNoSwitchIgnoresRegime2:
    def test_no_num_years_1_means_regime2_unused(self):
        """If num_years_1 is None, result_regime2 must be ignored entirely."""
        result = _run(0.20)
        result_regime2 = _run(0.40)
        kwargs = dict(
            result=result,
            system_cost=100000.0,
            rate_escalator_pct=3.0,
            load_escalator_pct=2.0,
            years=10,
            degradation_pct=0.5,
        )
        baseline = build_annual_projection(**kwargs)
        with_r2_but_no_switch = build_annual_projection(
            **kwargs, result_regime2=result_regime2,
        )
        pd.testing.assert_frame_equal(baseline, with_r2_but_no_switch)


# ---------------------------------------------------------------------------
# 6. Monthly view re-bills on tariff #2 too — monthly ↔ annual tie-out
# ---------------------------------------------------------------------------
class TestRegime2MonthlyAnnualTieOut:
    """The monthly builder must also re-bill post-transition years on tariff #2,
    so the monthly view stays consistent with the annual projection."""

    def _build(self):
        from modules.outputs import build_annual_projection, _build_multiyear_monthly_df
        r1 = _run(0.20)
        r2 = _run(0.40)   # post-transition tariff: double the energy rate
        kw = dict(nem_regime_1="NEM-3", nem_regime_2="NEM-3", num_years_1=5)
        ann = build_annual_projection(
            r1, system_cost=0.0, rate_escalator_pct=3.0, load_escalator_pct=0.0,
            years=10, result_regime2=r2, **kw,
        )
        mon = _build_multiyear_monthly_df(
            r1, rate_escalator_pct=3.0, load_escalator_pct=0.0, years=10,
            result_regime2=r2, **kw,
        )
        mon_no_r2 = _build_multiyear_monthly_df(
            r1, rate_escalator_pct=3.0, load_escalator_pct=0.0, years=10, **kw,
        )
        return ann, mon, mon_no_r2

    def test_monthly_ties_to_annual_with_regime2(self):
        ann, mon, _ = self._build()
        for y in range(1, 11):
            a = float(ann[ann["Year"] == y]["Bill w/ Solar ($)"].iloc[0])
            m = float(mon[mon["Year"] == y]["Net Bill ($)"].sum())
            assert abs(a - m) < 2.0, f"year {y}: annual {a:.2f} vs monthly {m:.2f}"

    def test_post_transition_months_rebill_on_tariff2(self):
        _, mon, mon_no_r2 = self._build()
        # Pre-transition (yr<=5) identical; post-transition (yr>5) higher bill on tariff #2.
        pre = mon[mon["Year"] == 3]["Net Bill ($)"].sum()
        pre0 = mon_no_r2[mon_no_r2["Year"] == 3]["Net Bill ($)"].sum()
        assert abs(pre - pre0) < 1.0
        post = mon[mon["Year"] == 8]["Net Bill ($)"].sum()
        post0 = mon_no_r2[mon_no_r2["Year"] == 8]["Net Bill ($)"].sum()
        assert post > post0 + 1.0, "post-transition months should re-bill on the higher tariff #2"


# ---------------------------------------------------------------------------
# 7. Downloads re-bill on tariff #2 too (screen-vs-download parity)
# ---------------------------------------------------------------------------
class TestRegime2DownloadParity:
    """The Monthly CSV download must thread result_regime2 so post-transition
    years re-bill on tariff #2 exactly like the on-screen monthly view —
    otherwise the downloaded file silently diverges from the app."""

    def test_monthly_csv_ties_to_screen_with_regime2(self):
        import io
        import pandas as pd
        from modules.outputs import generate_monthly_csv, _build_multiyear_monthly_df

        r1, r2 = _run(0.20), _run(0.40)
        kw = dict(rate_escalator_pct=3.0, years=10,
                  nem_regime_1="NEM-3", nem_regime_2="NEM-3", num_years_1=5)
        screen = _build_multiyear_monthly_df(r1, result_regime2=r2, **kw)
        dl = pd.read_csv(io.StringIO(generate_monthly_csv(r1, result_regime2=r2, **kw)))
        no_r2 = pd.read_csv(io.StringIO(generate_monthly_csv(r1, **kw)))
        for y in range(1, 11):
            a = dl[dl["Year"] == y]["Net Bill ($)"].sum()
            b = screen[screen["Year"] == y]["Net Bill ($)"].sum()
            assert abs(a - b) < 0.05, f"yr {y}: download {a:.2f} vs screen {b:.2f}"
        # regime-2 actually re-bills post-transition (differs from no-regime2)
        y8 = dl[dl["Year"] == 8]["Net Bill ($)"].sum()
        y8_no = no_r2[no_r2["Year"] == 8]["Net Bill ($)"].sum()
        assert abs(y8 - y8_no) > 1.0, "result_regime2 had no effect on the download"
