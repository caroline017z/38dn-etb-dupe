"""Monte Carlo + tornado sensitivity harness.

Varies projection-level levers (escalators, degradation) around a base case
and reports the distribution of the chosen NPV metric. Year-1 billing is
computed once from the caller's ``SimulationResult`` and re-used across
every sample, so a 500-sample Monte Carlo is cheap even for complex TOU
scenarios.

The sensitivity model is intentionally *projection-only* in this first cut.
Levers that would require re-solving the battery LP (e.g. round-trip
efficiency, system losses driving a different 8760) are reserved for a
follow-up that re-uses the same ``Lever`` abstraction over
``SimulationInputs``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .billing import BillingResult


Distribution = Literal["normal", "triangular", "uniform"]

LEVER_KEYS = ("rate_escalator", "load_escalator", "degradation")


@dataclass(frozen=True)
class Lever:
    """One input dimension to vary.

    ``key`` must be one of :data:`LEVER_KEYS` — the projection kwargs
    understood by :func:`project_npv`. Additional levers require extending
    the projector.

    Parameters for the supported distributions:
      - ``"normal"``:      (mean, std)
      - ``"triangular"``:  (low, mode, high)
      - ``"uniform"``:     (low, high)
    """

    key: str
    distribution: Distribution
    params: tuple[float, ...]
    display_name: str = ""
    unit: str = "%"
    # Optional absolute swing (in the lever's unit, e.g. percentage points)
    # used by :func:`tornado`. When set > 0, tornado ignores its global
    # pct_low / pct_high and instead perturbs this lever by base ± abs_swing.
    abs_swing: float = 0.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.distribution == "normal":
            mean, std = self.params
            return rng.normal(mean, std, n)
        if self.distribution == "triangular":
            low, mode, high = self.params
            return rng.triangular(low, mode, high, n)
        if self.distribution == "uniform":
            low, high = self.params
            return rng.uniform(low, high, n)
        raise ValueError(f"Unknown distribution: {self.distribution}")

    def bounds(self, pct_low: float, pct_high: float) -> tuple[float, float]:
        """Base±range bounds used for tornado analysis."""
        base = self.base()
        return base * (1.0 + pct_low), base * (1.0 + pct_high)

    def base(self) -> float:
        """Best-guess central value. ``mean`` for normal, ``mode`` for
        triangular, midpoint for uniform."""
        if self.distribution == "normal":
            return float(self.params[0])
        if self.distribution == "triangular":
            return float(self.params[1])
        if self.distribution == "uniform":
            return (float(self.params[0]) + float(self.params[1])) / 2.0
        raise ValueError(self.distribution)


DEFAULT_LEVERS: tuple[Lever, ...] = (
    Lever("rate_escalator", "normal", (3.0, 1.0), "Rate escalator", "%/yr"),
    Lever("load_escalator", "normal", (1.0, 0.5), "Load escalator", "%/yr"),
    Lever("degradation", "triangular", (0.3, 0.5, 0.8), "PV degradation", "%/yr"),
)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_npv(
    *,
    result: BillingResult,
    result_pv_only: BillingResult | None,
    system_cost: float,
    years: int,
    discount_rate_pct: float,
    rate_escalator: float,
    load_escalator: float,
    degradation: float,
    nem_regime_1: str = "NEM-3 / NVBT",
) -> float:
    """Year-N NPV of customer savings (no solar bill minus with-solar bill)
    net of the up-front system cost.

    Wraps :func:`modules.outputs.build_annual_projection`. Returning just
    the scalar NPV keeps the Monte Carlo hot loop free of DataFrame
    allocation overhead for metrics we don't need.
    """
    from .outputs import build_annual_projection  # late import: cycle avoidance

    projection = build_annual_projection(
        result=result,
        system_cost=system_cost,
        rate_escalator_pct=rate_escalator,
        load_escalator_pct=load_escalator,
        years=years,
        result_pv_only=result_pv_only,
        nem_regime_1=nem_regime_1,
        degradation_pct=degradation,
    )

    # "Annual Savings ($)" is produced by build_annual_projection for every
    # year (see modules/outputs.py). Discount to present value and net
    # system cost.
    if "Annual Savings ($)" not in projection.columns:
        raise RuntimeError("projection missing 'Annual Savings ($)' column")

    r = discount_rate_pct / 100.0
    years_idx = np.arange(1, len(projection) + 1)
    discount = (1.0 + r) ** years_idx
    annual_savings = projection["Annual Savings ($)"].to_numpy(dtype=float)
    pv_savings = float(np.sum(annual_savings / discount))
    return pv_savings - float(system_cost)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(
    *,
    result: BillingResult,
    result_pv_only: BillingResult | None,
    system_cost: float,
    years: int,
    discount_rate_pct: float,
    levers: Iterable[Lever],
    n: int,
    seed: int = 42,
    nem_regime_1: str = "NEM-3 / NVBT",
    progress_cb: Callable[[int, np.ndarray], None] | None = None,
    chunk: int = 25,
) -> pd.DataFrame:
    """Run ``n`` samples and return a DataFrame with one row per sample.

    ``progress_cb(i, npv_array_so_far)`` fires every ``chunk`` samples so the
    caller can update a live chart. The callback receives a view of all NPVs
    computed so far (length ``i``).

    Runs serially — the per-sample cost is ~ms because ``build_annual_projection``
    is pure pandas arithmetic. Parallelism (joblib) is reserved for the
    future case where levers require re-solving the billing engine.
    """
    levers = list(levers)
    if not levers:
        raise ValueError("at least one lever is required")

    rng = np.random.default_rng(seed)
    samples = {lev.key: lev.sample(rng, n) for lev in levers}
    base_values = {k: _base_value(k) for k in LEVER_KEYS}

    npvs = np.empty(n, dtype=float)
    for i in range(n):
        kwargs = dict(base_values)
        for lev in levers:
            kwargs[lev.key] = float(samples[lev.key][i])
        npvs[i] = project_npv(
            result=result,
            result_pv_only=result_pv_only,
            system_cost=system_cost,
            years=years,
            discount_rate_pct=discount_rate_pct,
            nem_regime_1=nem_regime_1,
            **kwargs,
        )
        if progress_cb is not None and (i + 1) % chunk == 0:
            progress_cb(i + 1, npvs[: i + 1])

    rows = {"npv": npvs}
    for lev in levers:
        rows[lev.key] = samples[lev.key]
    return pd.DataFrame(rows)


def _base_value(key: str) -> float:
    """Defaults matching the existing UI assumptions; individual levers
    override these when supplied."""
    return {
        "rate_escalator": 3.0,
        "load_escalator": 0.0,
        "degradation": 0.5,
    }[key]


def percentiles(npvs: np.ndarray, ps: Iterable[float] = (10, 50, 90)) -> dict[float, float]:
    return {p: float(np.percentile(npvs, p)) for p in ps}


# ---------------------------------------------------------------------------
# Tornado
# ---------------------------------------------------------------------------
def tornado(
    *,
    result: BillingResult,
    result_pv_only: BillingResult | None,
    system_cost: float,
    years: int,
    discount_rate_pct: float,
    levers: Iterable[Lever],
    pct_low: float = -0.10,
    pct_high: float = 0.10,
    nem_regime_1: str = "NEM-3 / NVBT",
) -> pd.DataFrame:
    """One-at-a-time sensitivity: for each lever, perturb by
    ``pct_low`` / ``pct_high`` around its base value and record the
    resulting NPV. Returns a DataFrame sorted by |swing| descending.
    """
    levers = list(levers)
    base_values = {k: _base_value(k) for k in LEVER_KEYS}
    for lev in levers:
        base_values[lev.key] = lev.base()

    base_npv = project_npv(
        result=result,
        result_pv_only=result_pv_only,
        system_cost=system_cost,
        years=years,
        discount_rate_pct=discount_rate_pct,
        nem_regime_1=nem_regime_1,
        **base_values,
    )

    rows = []
    for lev in levers:
        base = lev.base()
        # Prefer the lever's absolute swing when set — lets the UI give each
        # lever its own pp-swing (e.g. rate esc ±1%, degradation ±0.5%)
        # rather than a uniform relative percentage.
        if lev.abs_swing and lev.abs_swing > 0:
            low = base - lev.abs_swing
            high = base + lev.abs_swing
        else:
            low = base * (1.0 + pct_low)
            high = base * (1.0 + pct_high)

        low_kwargs = dict(base_values, **{lev.key: low})
        high_kwargs = dict(base_values, **{lev.key: high})
        low_npv = project_npv(
            result=result,
            result_pv_only=result_pv_only,
            system_cost=system_cost,
            years=years,
            discount_rate_pct=discount_rate_pct,
            nem_regime_1=nem_regime_1,
            **low_kwargs,
        )
        high_npv = project_npv(
            result=result,
            result_pv_only=result_pv_only,
            system_cost=system_cost,
            years=years,
            discount_rate_pct=discount_rate_pct,
            nem_regime_1=nem_regime_1,
            **high_kwargs,
        )
        rows.append({
            "lever": lev.display_name or lev.key,
            "key": lev.key,
            "base": base,
            "low": low,
            "high": high,
            "low_npv": low_npv,
            "high_npv": high_npv,
            "swing": abs(high_npv - low_npv),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("swing", ascending=False).reset_index(drop=True)
    df.attrs["base_npv"] = base_npv
    return df


__all__ = [
    "Lever",
    "DEFAULT_LEVERS",
    "LEVER_KEYS",
    "project_npv",
    "monte_carlo",
    "tornado",
    "percentiles",
]
