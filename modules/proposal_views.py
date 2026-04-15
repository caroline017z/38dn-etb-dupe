"""Rendering helpers for Customer Proposal views: side-by-side comparison
chart, metric grid, and XLSX export.

Kept in its own module so ``modules/proposals.py`` stays I/O-agnostic and
``modules/proposal.py`` (the PPTX deck generator) stays focused on a single
artifact type.

38DN palette is duplicated here intentionally — keeps this module
dependency-free on ``modules/outputs.py``.
"""

from __future__ import annotations

import io
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go

from .proposals import PPASnapshot, Proposal
from .ui.components import sparkline_svg

_NAVY = "#0E2841"
_GREEN = "#45A750"
_BLUE = "#1D6FA9"
_TEAL = "#518484"
_AMBER = "#D48A1A"
_INK = "#1A1A1A"
_FONT = "Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif"

# Distinct colours for up to 4 PPAs (primary + 3 comparisons).
_PPA_PALETTE = (_GREEN, _BLUE, _TEAL, _AMBER)


# ---------------------------------------------------------------------------
# Metric grid
# ---------------------------------------------------------------------------
def build_comparison_table(proposal: Proposal) -> pd.DataFrame:
    """Side-by-side metric grid: rows = metrics, cols = PPAs (Primary + comps).

    Metrics (per the Phase 4 product review):
      - Yr-1 PPA Rate (NEM-1)
      - Yr-1 PPA Rate (NEM-2) when a regime switch exists
      - PPA Escalator (NEM-1 / NEM-2)
      - Savings Target %
      - Lifetime Savings ($)
      - Effective $/kWh  (derived: mean of rate_per_year)
    """
    snaps: tuple[PPASnapshot, ...] = (proposal.primary_ppa, *proposal.comparison_ppas)

    labels = [f"{s.name} (Primary)" if i == 0 else s.name for i, s in enumerate(snaps)]

    def _fmt_rate(v):
        return f"${v:.4f}/kWh" if v not in (None, 0) else "—"

    def _fmt_pct(v):
        return f"{v:.1f}%" if v is not None else "—"

    def _fmt_usd(v):
        return f"${v:,.0f}" if v is not None else "—"

    any_regime_2 = any(s.nem_regime_2 for s in snaps)

    rows: list[dict] = []

    def _row(metric: str, values: list) -> None:
        rows.append({"Metric": metric, **dict(zip(labels, values, strict=False))})

    _row("Yr-1 PPA Rate (NEM-1)", [_fmt_rate(s.year1_rate_r1) for s in snaps])
    if any_regime_2:
        _row(
            "Yr-1 PPA Rate (NEM-2)",
            [_fmt_rate(s.year1_rate_r2) for s in snaps],
        )
    _row(
        "PPA Escalator (NEM-1)",
        [f"{s.escalator_r1_pct:.1f}%/yr" for s in snaps],
    )
    if any_regime_2:
        _row(
            "PPA Escalator (NEM-2)",
            [f"{s.escalator_r2_pct:.1f}%/yr" if s.escalator_r2_pct is not None else "—"
             for s in snaps],
        )
    _row("Savings Target", [_fmt_pct(s.savings_pct) for s in snaps])
    _row("Lifetime Savings", [_fmt_usd(s.lifetime_savings_usd) for s in snaps])

    # Effective $/kWh — the avg rate weighted by solar kWh, when available.
    eff_rates = []
    for s in snaps:
        if s.rate_per_year and s.solar_kwh_per_year and \
                len(s.rate_per_year) == len(s.solar_kwh_per_year):
            total_kwh = sum(s.solar_kwh_per_year)
            total_spend = sum(r * k for r, k in zip(s.rate_per_year, s.solar_kwh_per_year))
            eff_rates.append(f"${total_spend / total_kwh:.4f}/kWh" if total_kwh > 0 else "—")
        elif s.rate_per_year:
            avg = sum(s.rate_per_year) / max(len(s.rate_per_year), 1)
            eff_rates.append(f"${avg:.4f}/kWh (unwtd.)")
        else:
            eff_rates.append("—")
    _row("Effective PPA $/kWh", eff_rates)

    _row("Term", [f"{s.term_years} yrs" for s in snaps])

    # Inline sparkline of PPA rate trajectory — lets the reader eyeball
    # the shape (flat vs escalating vs regime-switch drop) without opening
    # the chart. Rendered as SVG so it embeds directly into the HTML table.
    _row("Rate trajectory", [
        sparkline_svg(list(s.rate_per_year), width=110, height=22)
        if s.rate_per_year else "—"
        for s in snaps
    ])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison chart (overlay + grouped-bar toggle)
# ---------------------------------------------------------------------------
def build_comparison_chart(proposal: Proposal, *, mode: str = "overlay") -> go.Figure:
    """Return a Plotly figure for the Proposal's PPAs.

    ``mode``:
      - ``"overlay"`` — line chart, one trace per PPA + utility-only baseline
        of the primary (the utility curve is the same for every PPA since
        bills-without-solar don't depend on the PPA rate).
      - ``"grouped"`` — grouped-bar chart of year-N total bill for a handful
        of snapshot years (Y1, Y5, Y10, Y20), side-by-side per PPA.

    When source arrays are missing, the chart falls back to a text
    annotation explaining what's needed.
    """
    snaps: tuple[PPASnapshot, ...] = (proposal.primary_ppa, *proposal.comparison_ppas)

    if mode == "grouped":
        return _build_grouped_bar(snaps)
    return _build_overlay(snaps)


def _build_overlay(snaps: tuple[PPASnapshot, ...]) -> go.Figure:
    fig = go.Figure()

    util = _pick_utility_series(snaps)
    if util:
        x, y = util
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y),
            name="Utility Only",
            mode="lines",
            line=dict(color=_NAVY, width=2.5, dash="dot"),
            hovertemplate="%{x}<br><b>$%{y:.1f}K</b><extra>Utility Only</extra>",
        ))

    for i, s in enumerate(snaps):
        if not s.calendar_years or not s.total_ppa_bill_k_per_year:
            continue
        label = f"{s.name}" + (" (Primary)" if i == 0 else "")
        fig.add_trace(go.Scatter(
            x=list(s.calendar_years),
            y=list(s.total_ppa_bill_k_per_year),
            name=label,
            mode="lines+markers",
            line=dict(color=_PPA_PALETTE[i % len(_PPA_PALETTE)], width=3),
            marker=dict(size=6, color=_PPA_PALETTE[i % len(_PPA_PALETTE)]),
            hovertemplate=f"%{{x}}<br><b>${{y:.1f}}K</b><extra>{label}</extra>".replace("{y", "%{y"),
        ))

    if len(fig.data) == 0:
        fig.add_annotation(
            text="Not enough data to render comparison chart.<br>"
                 "Save PPAs on the PPA Rate tab first.",
            showarrow=False,
            font=dict(color=_INK, size=13),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )

    fig.update_layout(
        title=dict(text="Annual Bill — All Proposal PPAs vs Utility Only",
                   font=dict(size=15, color=_NAVY)),
        xaxis_title="Year",
        yaxis_title="Annual Cost ($K)",
        yaxis=dict(rangemode="tozero", gridcolor="#E5E7EB"),
        xaxis=dict(gridcolor="#E5E7EB"),
        template="plotly_white",
        height=440,
        margin=dict(l=60, r=30, t=70, b=55),
        font=dict(family=_FONT, size=12, color=_INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=_INK, size=11)),
        hovermode="x unified",
    )
    return fig


def _build_grouped_bar(snaps: tuple[PPASnapshot, ...]) -> go.Figure:
    """Year-N total-bill-in-$K bars grouped by snapshot year.

    The bar chart reads better when the customer wants to see "how does
    each PPA hit in Year 1, Year 5, Year 10, Year 20?" without the
    visual noise of overlapping continuous lines.
    """
    fig = go.Figure()
    snapshot_years = (1, 5, 10, 20)

    x_labels = [f"Y{y}" for y in snapshot_years]

    for i, s in enumerate(snaps):
        if not s.total_ppa_bill_k_per_year:
            continue
        vals = []
        for yr in snapshot_years:
            idx = yr - 1
            if 0 <= idx < len(s.total_ppa_bill_k_per_year):
                vals.append(s.total_ppa_bill_k_per_year[idx])
            else:
                vals.append(None)
        label = s.name + (" (Primary)" if i == 0 else "")
        fig.add_trace(go.Bar(
            x=x_labels, y=vals, name=label,
            marker_color=_PPA_PALETTE[i % len(_PPA_PALETTE)], opacity=0.9,
            hovertemplate=f"{label}<br>%{{x}}: $%{{y:.1f}}K<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Annual Bill at Snapshot Years — Proposal PPAs",
                   font=dict(size=15, color=_NAVY)),
        xaxis_title="Projection Year",
        yaxis_title="Annual Cost ($K)",
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=30, t=70, b=55),
        font=dict(family=_FONT, size=12, color=_INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=_INK, size=11)),
    )
    return fig


def _pick_utility_series(snaps):
    """Any snapshot can supply the utility-only curve — they are all based
    on the same simulation. Pick the first non-empty one."""
    for s in snaps:
        if s.calendar_years and s.utility_only_bill_k_per_year and \
                len(s.calendar_years) == len(s.utility_only_bill_k_per_year):
            return (s.calendar_years, s.utility_only_bill_k_per_year)
    return None


# ---------------------------------------------------------------------------
# XLSX export: one sheet per PPA + a Summary sheet with the metric grid.
# ---------------------------------------------------------------------------
def export_comparison_xlsx(proposal: Proposal) -> bytes:
    """Return an XLSX workbook containing:

    - **Summary** — the metric grid from :func:`build_comparison_table`.
    - **Proposal** — customer / site / account / notes.
    - one sheet per PPA (Primary + comparisons) with per-year rate, solar
      kWh, utility-only bill, PPA bill, cumulative savings.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary = build_comparison_table(proposal)
        summary.to_excel(xl, sheet_name="Summary", index=False)

        meta = pd.DataFrame([
            ("Proposal", proposal.name),
            ("Customer", proposal.customer_name),
            ("Site", proposal.site_address),
            ("Utility Account", proposal.utility_account),
            ("Simulation", proposal.simulation_name or ""),
            ("Term (years)", proposal.term_years),
            ("Created", proposal.created_at),
            ("Updated", proposal.updated_at),
            ("Notes", proposal.notes),
        ], columns=["Field", "Value"])
        meta.to_excel(xl, sheet_name="Proposal", index=False)

        for i, snap in enumerate((proposal.primary_ppa, *proposal.comparison_ppas)):
            sheet = _safe_sheet_name(
                f"{'Primary' if i == 0 else 'Alt'}_{snap.name or f'PPA_{i}'}"
            )
            _ppa_sheet(snap).to_excel(xl, sheet_name=sheet, index=False)

    return buf.getvalue()


def _ppa_sheet(snap: PPASnapshot) -> pd.DataFrame:
    term = snap.term_years
    cols: dict[str, list] = {"Year": list(range(1, term + 1))}

    if snap.calendar_years and len(snap.calendar_years) == term:
        cols["Calendar Year"] = list(snap.calendar_years)
    if snap.rate_per_year and len(snap.rate_per_year) == term:
        cols["PPA Rate ($/kWh)"] = list(snap.rate_per_year)
    if snap.solar_kwh_per_year and len(snap.solar_kwh_per_year) == term:
        cols["Solar (kWh)"] = list(snap.solar_kwh_per_year)
    if snap.utility_only_bill_k_per_year and len(snap.utility_only_bill_k_per_year) == term:
        cols["Utility Only ($K)"] = list(snap.utility_only_bill_k_per_year)
    if snap.total_ppa_bill_k_per_year and len(snap.total_ppa_bill_k_per_year) == term:
        cols["Solar + PPA ($K)"] = list(snap.total_ppa_bill_k_per_year)
        if "Utility Only ($K)" in cols:
            cols["Annual Savings ($K)"] = [
                u - p for u, p in zip(cols["Utility Only ($K)"], cols["Solar + PPA ($K)"])
            ]
            running = 0.0
            cum: list[float] = []
            for s in cols["Annual Savings ($K)"]:
                running += float(s or 0.0)
                cum.append(running)
            cols["Cumulative Savings ($K)"] = cum

    return pd.DataFrame(cols)


def _safe_sheet_name(name: str) -> str:
    # Excel caps at 31 chars and forbids : \ / ? * [ ]
    bad = set(r":\/?*[]")
    cleaned = "".join("_" if c in bad else c for c in name)
    return cleaned[:31]


__all__ = [
    "build_comparison_table",
    "build_comparison_chart",
    "export_comparison_xlsx",
]
