"""
PV Solar Rate Simulator — Streamlit Application

Simulates annual electricity bills for California agricultural and commercial
customers with solar PV systems under Net Value Billing Tariff (NVBT).
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import glob
import logging
from dataclasses import asdict
from datetime import date
from typing import cast
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from modules.pvwatts import (
    PVSystemConfig,
    geocode_address,
    fetch_production_8760,
    get_array_type_code,
)
from modules.tariff import (
    UTILITY_EIA_IDS,
    NBC_DEFAULTS,
    NSC_DEFAULT_RATE,
    TariffSchedule,
    fetch_available_rates,
    fetch_tariff_detail,
    format_tariff_summary,
)
from modules.export_value import (
    load_acc_from_upload,
    create_flat_export_rates,
    parse_multiyear_export_rates,
)
from modules.billing import BillingResult, compute_old_rate_baseline
from modules.simulation import (
    run_simulation,
    inputs_from_session_state,
)
from modules.sensitivity import (
    Lever,
    monte_carlo as _sens_monte_carlo,
    percentiles as _sens_percentiles,
    tornado as _sens_tornado,
)
from dataclasses import replace as _dc_replace

# AI features (lazy-used so app still runs without ANTHROPIC_API_KEY)
from modules.ai.proposal_narrative import (
    ProposalContext as _AIProposalContext,
    generate_executive_summary as _ai_generate_exec_summary,
)
from modules.ai.bill_ingest import extract_bill as _ai_extract_bill
from modules.ai.tariff_qa import ask as _ai_tariff_ask
from modules.ai.client import AnthropicCreditError as _AnthropicCreditError


def _render_ai_error(exc: Exception, action_label: str) -> None:
    """Show a friendly error message for AI failures. Handles the common
    depleted-credit case with a direct call-out; falls back to a generic
    message with the underlying exception text for everything else."""
    if isinstance(exc, _AnthropicCreditError):
        st.markdown(
            '<div style="background:#FDF3E2;border:1px solid #F0D7A8;'
            'border-left:3px solid #D48A1A;border-radius:6px;'
            'padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.55;'
            'color:#0E2841;">'
            '<div style="font-size:10px;font-weight:600;color:#D48A1A;'
            'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
            'Anthropic API credits depleted</div>'
            f"Top up your workspace credit balance at "
            '<a href="https://console.anthropic.com/settings/billing" '
            'target="_blank" style="color:#1D6FA9;font-weight:600;">'
            'console.anthropic.com &rarr; Plans &amp; Billing</a> '
            f"to re-enable {action_label}. The rest of the app continues "
            "to work without AI features."
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(f"{action_label} failed: {exc}")

# Phase 4: Proposals (named PPA bundles per simulation, with comparison).
# See modules/proposals.py for the data model; modules/proposal_views.py
# supplies the comparison chart + XLSX export helpers.
from modules import proposals as _proposals
from modules.proposals import (
    MAX_COMPARISON_PPAS as _PROP_MAX_COMPARISONS,
    Proposal as _ProposalObj,
    create_proposal as _create_proposal_obj,
    update_proposal as _update_proposal_obj,
    snapshot_from_saved as _snapshot_from_saved,
    snapshot_drift_reasons as _snapshot_drift_reasons,
    save_proposal_to_session as _save_proposal_session,
    delete_proposal_from_session as _delete_proposal_session,
    list_proposals_in_session as _list_proposals_session,
    get_active_proposal as _get_active_proposal,
    persist_proposal as _persist_proposal_gcs,
    load_proposals_for_simulation as _load_proposals_gcs,
    delete_persisted_proposal as _delete_proposal_gcs,
)

# Background-persist a Proposal to GCS so the UI thread doesn't block on a
# ~1–2s round-trip. Session state is the source of truth for the current
# page render; GCS is the durable store for cross-session reloads. On
# thread exit the GCS blob is committed; if the user closes the tab in
# under a second the save may not reach GCS, so the caller also runs a
# synchronous fallback when the last-ditch save matters (e.g. explicit
# "save and close" flows).
import threading as _threading
import logging as _prop_logging
_bg_logger = _prop_logging.getLogger("proposals.bg_persist")


def _persist_proposal_async(proposal: _ProposalObj) -> None:
    """Fire-and-forget GCS persist; exceptions are logged, not raised."""
    def _worker() -> None:
        try:
            _persist_proposal_gcs(proposal)
        except Exception:
            _bg_logger.warning(
                "background GCS persist of Proposal %s failed",
                proposal.id, exc_info=True,
            )
    _threading.Thread(target=_worker, daemon=True).start()
from modules.proposal_views import (
    build_comparison_chart as _build_prop_chart,
    build_comparison_table as _build_prop_table,
    export_comparison_xlsx as _export_prop_xlsx,
)
from modules.billing_aggregation import (
    MeterConfig,
    NemAProfile,
    NEMA_FEES,
    compute_nema_fees,
    compute_effective_export_price,
    run_aggregation_simulation,
)
from modules.proposal import generate_proposal_pptx
from modules.billing_ecc import (
    fetch_and_populate_ecc_tariff,
    load_ecc_tariff_from_json,
    run_ecc_billing_simulation,
    compute_old_rate_baseline_ecc,
)
from modules.rate_extractor import (
    extract_text_from_pdf,
    extract_tariff_from_text,
    validate_tariff_structure,
    save_custom_tariff,
)
from modules.load_adjustment import adjust_load_single_meter, adjust_loads_nema
from modules.battery import BatteryConfig
from modules.battery.sizing import optimize_capacity_kwh
from modules.billing import _build_demand_lp_inputs, _build_hourly_energy_rates
from modules.outputs import (
    MONTH_NAMES,
    build_monthly_summary_display,
    build_savings_summary,
    build_annual_projection,
    build_battery_kpi_summary,
    build_grid_exchange_summary,
    build_indexed_tariff_annual,
    build_indexed_tariff_monthly,
    _build_multiyear_monthly_df,
    create_production_vs_load_chart,
    create_monthly_bill_chart,
    generate_hourly_csv,
    generate_monthly_csv,
    generate_annual_csv,
    generate_simulation_excel,
    _negate_outflow_columns,
    fmt_num,
    fmt_dollar,
    fmt_rate,
    render_styled_table,
)

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets (Cloud) first, then fall back to env vars (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# =============================================================================
# DIRECTORIES
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SIMULATIONS_DIR = os.path.join(DATA_DIR, "simulations")
LOAD_PROFILES_DIR = os.path.join(DATA_DIR, "load_profiles")
EXPORT_PROFILES_DIR = os.path.join(DATA_DIR, "export_profiles")
ECC_TARIFFS_DIR = os.path.join(DATA_DIR, "ecc_tariffs")
SYSTEM_PROFILES_DIR = os.path.join(DATA_DIR, "system_profiles")
NEMA_PROFILES_DIR = os.path.join(DATA_DIR, "nema_profiles")

for d in [SIMULATIONS_DIR, LOAD_PROFILES_DIR, EXPORT_PROFILES_DIR, ECC_TARIFFS_DIR, SYSTEM_PROFILES_DIR, NEMA_PROFILES_DIR]:
    os.makedirs(d, exist_ok=True)

_DIR_TO_GCS_PREFIX = {
    LOAD_PROFILES_DIR: "load_profiles/",
    NEMA_PROFILES_DIR: "nema_profiles/",
    EXPORT_PROFILES_DIR: "export_profiles/",
    SYSTEM_PROFILES_DIR: "system_profiles/",
    ECC_TARIFFS_DIR: "ecc_tariffs/",
}


# =============================================================================
# HELPER FUNCTIONS — Simulations (shared module)
# =============================================================================
from sim_helpers import (
    list_saved_simulations,
    load_simulation as _load_simulation,
    save_simulation as _save_simulation,
    delete_simulation,
    touch_simulation_mtime,
    get_simulation_metadata,
    populate_session_from_simulation,
    sanitize_filename,
    list_profile_files,
    load_profile_bytes,
    save_profile_bytes,
    delete_profile_file,
    gcs_diagnostic,
)


def _list_saved(directory: str, ext: str = ".json") -> list[str]:
    """Generic file lister with GCS backing."""
    gcs_prefix = _DIR_TO_GCS_PREFIX.get(directory)
    if gcs_prefix:
        return list_profile_files(directory, gcs_prefix, ext)
    # Fallback: local-only for unknown directories
    files = glob.glob(os.path.join(directory, f"*{ext}"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


@st.cache_data(ttl=30)
def _list_all_load_profiles() -> list[tuple[str, str]]:
    """Return unified (name, type) list of CSV + NEM-A profiles (GCS + local)."""
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[float, str, str]] = []  # (mtime, name, type)
    # Local files (have mtime for sorting)
    for f in glob.glob(os.path.join(LOAD_PROFILES_DIR, "*.csv")):
        name = os.path.splitext(os.path.basename(f))[0]
        entries.append((os.path.getmtime(f), name, "csv"))
        seen.add((name, "csv"))
    for f in glob.glob(os.path.join(NEMA_PROFILES_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(f))[0]
        entries.append((os.path.getmtime(f), name, "nema"))
        seen.add((name, "nema"))
    # GCS-only files (no mtime — append at end with mtime=0)
    from sim_helpers import gcs_list_files
    for gcs_names, ext, typ in [
        (gcs_list_files("load_profiles/", ".csv"), ".csv", "csv"),
        (gcs_list_files("nema_profiles/", ".json"), ".json", "nema"),
    ]:
        if gcs_names:
            for n in gcs_names:
                if (n, typ) not in seen:
                    entries.append((0, n, typ))
    entries.sort(key=lambda x: x[0], reverse=True)
    return [(name, typ) for _, name, typ in entries]


def _delete_file(directory, name, ext):
    """Generic file deleter with GCS backing."""
    gcs_prefix = _DIR_TO_GCS_PREFIX.get(directory)
    if gcs_prefix:
        delete_profile_file(directory, gcs_prefix, name, ext)
    else:
        fp = os.path.join(directory, f"{name}{ext}")
        if os.path.exists(fp):
            os.remove(fp)


# =============================================================================
# HELPER — Battery Solver Check
# =============================================================================
def _check_battery_solver(result: "BillingResult"):
    """Warn the user if the battery dispatch solver failed."""
    hd = result.hourly_detail
    if "batt_to_load_kwh" in hd.columns:
        total_discharge = hd["batt_to_load_kwh"].sum() + hd.get("batt_to_grid_kwh", pd.Series([0])).sum()
        if total_discharge < 0.1:
            st.warning(
                "Battery dispatch produced near-zero discharge. "
                "Check that export rates are loaded and charge/discharge windows are configured correctly."
            )


def _render_savings_dashboard(
    *,
    result,
    pv_only_result,
    pv_batt_result,
    system_cost: float,
    system_life_years: int,
    has_battery: bool,
    main_projection,
) -> None:
    """Dashboard-style Savings & Payback view.

    Headline KPIs top, a view toggle to switch the contextual chart between
    Financial Impact / Energy Flow / Scenario Comparison / Cumulative Payback,
    and a unified Scenario Comparison table + optional Rate Shift block.
    """
    from modules.outputs import (
        build_savings_summary,
    )

    NAVY, GREEN, BLUE, TEAL, AMBER = "#0E2841", "#45A750", "#1D6FA9", "#518484", "#D48A1A"
    INK = "#1A1A1A"
    FONT = "Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif"

    st.subheader("Savings & Payback")

    summary = build_savings_summary(result, system_cost)
    annual_savings = float(summary["annual_savings"])
    savings_pct = float(summary["savings_pct"])
    payback_yrs = summary.get("simple_payback_years")

    # Cumulative 20-yr savings from the projection (more accurate than annual * N
    # because it reflects escalators / degradation).
    try:
        lifetime_savings = float(main_projection["Cumulative Savings ($)"].iloc[-1])
    except Exception:
        lifetime_savings = annual_savings * int(system_life_years)

    # ── Headline KPI row ─────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Year-1 Savings", f"${annual_savings:,.0f}", delta=f"{savings_pct:.1f}%")
    k2.metric(f"{system_life_years}-yr Cumulative Savings",
              f"${lifetime_savings:,.0f}")
    k3.metric("Simple Payback",
              f"{payback_yrs:.1f} yrs" if payback_yrs is not None else "N/A")
    k4.metric("System Cost", f"${float(system_cost):,.0f}")

    st.markdown("")  # visual breather

    # ── Stacked sections (no view-toggle radio — each lens scrollable) ─────
    st.markdown("#### 💰 Financial Impact")
    _sp_financial_view(result, pv_only_result, has_battery,
                       NAVY, GREEN, BLUE, TEAL, AMBER, INK, FONT)
    st.divider()

    st.markdown("#### ⚡ Energy Flow")
    _sp_energy_view(result, NAVY, GREEN, BLUE, TEAL, INK, FONT)
    st.divider()

    st.markdown("#### 📈 Cumulative Payback")
    _sp_payback_view(main_projection, system_cost, NAVY, GREEN, BLUE, INK, FONT)

    if has_battery and pv_only_result is not None and pv_batt_result is not None:
        st.divider()
        st.markdown("#### 📊 Scenario Comparison")
        _sp_scenario_view(result, pv_only_result, pv_batt_result, has_battery)

    # ── Rate shift (conditional, always visible when applicable) ────────
    if summary.get("rate_shift_annual_savings") is not None:
        st.divider()
        st.markdown("##### Rate Shift Analysis")
        rs1, rs2, rs3 = st.columns(3)
        rs1.metric("Old Rate Baseline",
                   f"${float(result.old_rate_annual_baseline):,.0f}")
        rs2.metric("Rate Shift Savings",
                   f"${float(summary['rate_shift_annual_savings']):,.0f}/yr")
        rs3.metric("Total Combined Savings",
                   f"${float(summary['total_annual_savings']):,.0f}/yr")


def _sp_financial_view(result, pv_only_result, has_battery, NAVY, GREEN, BLUE, TEAL, AMBER, INK, FONT) -> None:
    """Monthly bill: no-solar baseline vs with-solar stacked components.
    Savings shown as a filled band between the two."""
    import plotly.graph_objects as go
    df = result.monthly_summary
    months = MONTH_NAMES
    fig = go.Figure()

    # With-solar stacked components (positive)
    fig.add_trace(go.Bar(x=months, y=df["energy_cost"], name="Energy",
                         marker_color=NAVY, opacity=0.92))
    fig.add_trace(go.Bar(x=months, y=df["total_demand_charge"], name="Demand",
                         marker_color=BLUE, opacity=0.92))
    fig.add_trace(go.Bar(x=months, y=df["fixed_charge"], name="Fixed",
                         marker_color=TEAL, opacity=0.92))
    if "nbc_charge" in df.columns and df["nbc_charge"].sum() > 0:
        fig.add_trace(go.Bar(x=months, y=df["nbc_charge"], name="NBC",
                             marker_color=AMBER, opacity=0.92))
    fig.add_trace(go.Bar(x=months, y=-df["export_credit"], name="Export Credit",
                         marker_color=GREEN, opacity=0.92))

    # Baseline (no-solar) line
    if result.monthly_baseline_details is not None:
        baseline = [d["total"] for d in result.monthly_baseline_details]
        fig.add_trace(go.Scatter(
            x=months, y=baseline, name="Bill w/o Solar (baseline)",
            mode="lines+markers",
            line=dict(color="#1A1A1A", width=2.5, dash="dash"),
            marker=dict(size=7, color="#1A1A1A"),
        ))

    fig.update_layout(
        title=dict(text="Monthly Bill: With-Solar Components vs No-Solar Baseline",
                   font=dict(size=15, color=NAVY)),
        xaxis_title="Month", yaxis_title="Cost ($)",
        barmode="relative", template="plotly_white", height=420,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=60, r=30, t=70, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=INK, size=11)),
        xaxis=dict(tickfont=dict(color=INK), title_font=dict(color=INK)),
        yaxis=dict(tickfont=dict(color=INK), title_font=dict(color=INK),
                   gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True, key="sp_financial_chart")

    if has_battery and pv_only_result is not None:
        batt_value = (pv_only_result.annual_bill_with_solar
                      - result.annual_bill_with_solar)
        st.caption(
            f"Battery contributes **${batt_value:,.0f}/yr** of additional savings "
            f"beyond PV-only (demand reduction + export arbitrage)."
        )


def _sp_energy_view(result, NAVY, GREEN, BLUE, TEAL, INK, FONT) -> None:
    """Monthly load / solar / import / export — kWh view of the same year."""
    import plotly.graph_objects as go
    df = result.monthly_summary
    months = MONTH_NAMES
    fig = go.Figure()

    fig.add_trace(go.Bar(x=months, y=df["load_kwh"], name="Load",
                         marker_color=NAVY, opacity=0.88))
    fig.add_trace(go.Bar(x=months, y=df["solar_kwh"], name="Solar Production",
                         marker_color=GREEN, opacity=0.88))
    fig.add_trace(go.Scatter(x=months, y=df["import_kwh"], name="Net Import",
                             mode="lines+markers",
                             line=dict(color=BLUE, width=2.5),
                             marker=dict(size=7, color=BLUE)))
    fig.add_trace(go.Scatter(x=months, y=df["export_kwh"], name="Net Export",
                             mode="lines+markers",
                             line=dict(color=TEAL, width=2.5, dash="dot"),
                             marker=dict(size=7, color=TEAL)))

    # Self-consumption fraction overlay (secondary axis)
    self_cons = (df["solar_kwh"] - df["export_kwh"]).clip(lower=0)
    self_cons_pct = (self_cons / df["solar_kwh"].replace(0, np.nan) * 100).fillna(0)
    fig.add_trace(go.Scatter(
        x=months, y=self_cons_pct, name="Self-Consumption %",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#D48A1A", width=2, dash="dashdot"),
        marker=dict(size=6, color="#D48A1A"),
    ))
    fig.update_layout(
        title=dict(text="Monthly Energy Flow: Load, Solar, Grid Exchange",
                   font=dict(size=15, color=NAVY)),
        xaxis_title="Month", yaxis_title="Energy (kWh)",
        yaxis2=dict(title="Self-Consumption (%)", overlaying="y", side="right",
                    range=[0, 100], tickfont=dict(color="#D48A1A"),
                    title_font=dict(color="#D48A1A")),
        barmode="group", template="plotly_white", height=440,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=60, r=70, t=70, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=INK, size=11)),
        xaxis=dict(tickfont=dict(color=INK)),
        yaxis=dict(tickfont=dict(color=INK), gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True, key="sp_energy_chart")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual Load", f"{result.annual_load_kwh:,.0f} kWh")
    k2.metric("Annual Solar", f"{result.annual_solar_kwh:,.0f} kWh")
    solar_offset_pct = (
        result.annual_solar_kwh / result.annual_load_kwh * 100
        if result.annual_load_kwh > 0 else 0.0
    )
    k3.metric("Solar Offset", f"{solar_offset_pct:.1f}%")
    annual_self_pct = (
        (result.annual_solar_kwh - result.annual_export_kwh)
        / result.annual_solar_kwh * 100
        if result.annual_solar_kwh > 0 else 0.0
    )
    k4.metric("Self-Consumption", f"{annual_self_pct:.1f}%")


def _sp_scenario_view(result, pv_only_result, pv_batt_result, has_battery) -> None:
    """Side-by-side No-Solar / PV-Only / PV+Battery comparison. The existing
    detailed table surfaces the per-component breakdown; this view is the
    primary way to see the battery's incremental value."""
    if not has_battery or pv_only_result is None or pv_batt_result is None:
        st.info(
            "Scenario comparison is most useful when a battery is enabled — "
            "it shows the PV-only vs PV+Battery delta. With PV only, the "
            "primary savings number is the headline above."
        )
        pv_only_result = pv_only_result or result

    cmp_data = {
        "Metric": [
            "Annual Bill",
            "Energy Charges",
            "Demand Charges",
            "Export Credit",
            "Savings vs No-Solar",
        ],
        "No Solar": [
            fmt_dollar(-result.annual_bill_without_solar),
            "—", "—", "—", "—",
        ],
        "PV Only": [
            fmt_dollar(-pv_only_result.annual_bill_with_solar),
            fmt_dollar(-pv_only_result.annual_energy_cost),
            fmt_dollar(-pv_only_result.annual_demand_cost),
            fmt_dollar(pv_only_result.annual_export_credit),
            fmt_dollar(pv_only_result.annual_savings),
        ],
    }
    if has_battery and pv_batt_result is not None:
        cmp_data["PV + Battery"] = [
            fmt_dollar(-pv_batt_result.annual_bill_with_solar),
            fmt_dollar(-pv_batt_result.annual_energy_cost),
            fmt_dollar(-pv_batt_result.annual_demand_cost),
            fmt_dollar(pv_batt_result.annual_export_credit),
            fmt_dollar(pv_batt_result.annual_savings),
        ]
        battery_value = (pv_only_result.annual_bill_with_solar
                         - pv_batt_result.annual_bill_with_solar)
        cmp_data["Metric"].append("Battery Incremental Value")
        cmp_data["No Solar"].append("—")
        cmp_data["PV Only"].append("—")
        cmp_data["PV + Battery"].append(fmt_dollar(battery_value))

    if (st.session_state.get("rate_shift_enabled")
            and pv_only_result.rate_shift_annual_savings is not None):
        cmp_data["Metric"].append("Rate Shift Savings")
        cmp_data["No Solar"].append("—")
        cmp_data["PV Only"].append(fmt_dollar(pv_only_result.rate_shift_annual_savings))
        if has_battery and pv_batt_result is not None:
            cmp_data["PV + Battery"].append(fmt_dollar(pv_batt_result.rate_shift_annual_savings))

    st.markdown(
        render_styled_table(pd.DataFrame(cmp_data), bold_cols=["Metric"]),
        unsafe_allow_html=True,
    )
    st.caption(
        "Bill components shown as accounting negatives; Export Credit and Savings "
        "are positive. Battery Incremental Value = PV-only annual bill − PV+Battery annual bill."
    )


def _sp_payback_view(projection, system_cost, NAVY, GREEN, BLUE, INK, FONT) -> None:
    """Cumulative savings vs system cost — crosspoint visualises payback."""
    import plotly.graph_objects as go
    if projection is None or "Cumulative Savings ($)" not in projection.columns:
        st.info("Projection not available.")
        return

    x = projection["Year"]
    cum = projection["Cumulative Savings ($)"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=cum, name="Cumulative Solar Savings",
        mode="lines+markers",
        line=dict(color=GREEN, width=3),
        marker=dict(size=7, color=GREEN),
        fill="tozeroy",
        fillcolor="rgba(69,167,80,0.12)",
    ))
    fig.add_hline(
        y=system_cost, line_dash="dash", line_color=NAVY, line_width=2.5,
        annotation_text=f"<b>System Cost</b>: ${system_cost:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color=NAVY, size=12),
    )
    # Find approximate payback year (first year cum >= cost)
    crossed = cum[cum >= system_cost]
    if len(crossed):
        xp = projection.loc[crossed.index[0], "Year"]
        fig.add_vline(
            x=xp, line_dash="dot", line_color=BLUE, line_width=2,
            annotation_text=f"<b>Payback</b> yr {int(xp)}",
            annotation_position="top",
            annotation_font=dict(color=BLUE, size=12),
        )
    fig.update_layout(
        title=dict(text="Cumulative Savings vs System Cost",
                   font=dict(size=15, color=NAVY)),
        xaxis_title="Year", yaxis_title="$",
        template="plotly_white", height=420,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=60, r=30, t=70, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=INK, size=11)),
        xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color=INK)),
        yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color=INK)),
    )
    st.plotly_chart(fig, use_container_width=True, key="sp_payback_chart")


def _realized_cagr(series: list[float]) -> float:
    """CAGR (%) of the positive values in ``series``; 0.0 if fewer than 2 points.

    Used to read the effective escalator off the solved per-year PPA rate now
    that there is no fixed escalator input.
    """
    pts = [v for v in series if v and v > 0]
    if len(pts) < 2:
        return 0.0
    return ((pts[-1] / pts[0]) ** (1.0 / (len(pts) - 1)) - 1.0) * 100.0


def _proposal_comparison_payload(source: "_ProposalObj") -> list[dict]:
    """Shape the primary + comparison PPASnapshots into the dict list that
    ``generate_proposal_pptx``'s ``comparison_ppas`` kwarg expects for the
    Alternatives Considered appendix slide.
    """
    return [
        {
            "name": "Recommended",
            "year1_rate_r1": source.primary_ppa.year1_rate_r1,
            "year1_rate_r2": source.primary_ppa.year1_rate_r2,
            "escalator_r1_pct": source.primary_ppa.escalator_r1_pct,
            "escalator_r2_pct": source.primary_ppa.escalator_r2_pct,
            "savings_pct": source.primary_ppa.savings_pct,
            "lifetime_savings_usd": source.primary_ppa.lifetime_savings_usd,
            "term_years": source.primary_ppa.term_years,
        },
        *[
            {
                "name": s.name,
                "year1_rate_r1": s.year1_rate_r1,
                "year1_rate_r2": s.year1_rate_r2,
                "escalator_r1_pct": s.escalator_r1_pct,
                "escalator_r2_pct": s.escalator_r2_pct,
                "savings_pct": s.savings_pct,
                "lifetime_savings_usd": s.lifetime_savings_usd,
                "term_years": s.term_years,
            }
            for s in source.comparison_ppas
        ],
    ]


def _build_proposal_deck_bytes(
    *,
    source: "_ProposalObj",
    include_appendix: bool,
    result,
    pv_only_result,
    system_cost: float,
    rate_escalator: float,
    load_escalator: float,
    compound_escalation: bool,
    rs_old_baseline,
    es_offset_annual,
    common_nem_kw: dict,
    utility_name: str,
    selected_rate_name: str | None,
    system_size_kw: float,
    dc_ac_ratio: float,
    battery_cap_kwh: float,
    nem_regime_1: str,
    nem_regime_2: str | None,
    num_years_1: int | None,
) -> bytes:
    """Assemble the proposal-term projection, inject PPA cost per year, and
    hand off to generate_proposal_pptx. Used from both the Proposals tab
    export buttons and the Downloads tab builder so outputs converge.
    """
    primary = source.primary_ppa
    base_proj = build_annual_projection(
        result=result,
        system_cost=system_cost,
        rate_escalator_pct=rate_escalator,
        load_escalator_pct=load_escalator,
        years=source.term_years,
        export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
        result_pv_only=pv_only_result,
        compound_escalation=compound_escalation,
        rate_shift_old_baseline=rs_old_baseline,
        existing_solar_offset_kwh=es_offset_annual,
        **common_nem_kw,
    )

    # Align snapshot rate-per-year to the proposal term — extrapolate at the
    # regime-2 escalator when short, truncate when long.
    rates = list(primary.rate_per_year)
    if len(rates) < source.term_years:
        tail_esc = (primary.escalator_r2_pct or primary.escalator_r1_pct) / 100.0
        last = rates[-1] if rates else primary.year1_rate_r1
        for _ in range(source.term_years - len(rates)):
            last = max(0.0, last * (1.0 + tail_esc))  # PPA rate can't go negative
            rates.append(round(last, 5))
    elif len(rates) > source.term_years:
        rates = rates[: source.term_years]

    proj_df = base_proj.copy()
    rate_lookup = dict(enumerate(rates, start=1))
    for idx, row in proj_df.iterrows():
        yr = int(row["Year"])
        rate_yr = float(rate_lookup.get(yr, 0.0))
        solar_kwh = row["Solar (kWh)"]
        ppa_cost = max(rate_yr, 0.0) * solar_kwh
        util_residual = row["Bill w/ Solar ($)"]
        total_cost = util_residual + ppa_cost
        bill_no = row["Bill w/o Solar ($)"]
        proj_df.at[idx, "PPA Cost ($)"] = round(ppa_cost, 2)
        proj_df.at[idx, "Bill w/ Solar ($)"] = round(total_cost, 2)
        proj_df.at[idx, "Annual Savings ($)"] = round(bill_no - total_cost, 2)
    proj_df["Cumulative Savings ($)"] = proj_df["Annual Savings ($)"].cumsum().round(2)

    return generate_proposal_pptx(
        customer_name=source.customer_name or "Customer",
        address=source.site_address,
        utility_account=source.utility_account,
        utility_name=utility_name,
        tariff_name=selected_rate_name or "",
        date_str=date.today().strftime("%B %Y"),
        system_size_kw=float(system_size_kw or 0.0),
        dc_ac_ratio=float(dc_ac_ratio or 1.0),
        battery_kwh=float(battery_cap_kwh or 0.0),
        battery_kw=0.0,
        ppa_rate=primary.year1_rate_r1 or None,
        ppa_escalator_pct=primary.escalator_r1_pct,
        ppa_escalator_pct_2=primary.escalator_r2_pct,
        term_years=source.term_years,
        rate_escalator_pct=rate_escalator,
        result=result,
        annual_proj_df=proj_df,
        nem_regime_1=nem_regime_1,
        nem_regime_2=nem_regime_2,
        num_years_1=num_years_1,
        customer_savings_pct=primary.savings_pct,
        customer_savings_pct_2=(primary.savings_pct_r2
                                if primary.savings_pct_r2 is not None
                                else primary.savings_pct),
        ppa_rate_regime_2=primary.year1_rate_r2,
        annual_proj_df_original=base_proj,
        narrative_bullets=list(source.narrative_bullets) or None,
        comparison_ppas=(
            _proposal_comparison_payload(source)
            if include_appendix and source.comparison_ppas else None
        ),
    )


def _render_proposals_tab(
    *,
    simulation_name,
    result,
    pv_only_result,
    main_projection,
    system_size_kw,
    dc_ac_ratio,
    battery_cap_kwh,
    system_cost,
    system_life_years,
    nem_regime_1,
    nem_regime_2,
    num_years_1,
    utility_name,
    selected_rate_name,
    rate_escalator,
    load_escalator,
    compound_escalation,
    cod_date,
    annual_degradation_pct,
    common_nem_kw,
    rs_old_baseline,
    es_offset_annual,
) -> None:
    """Named-Proposal workspace. Two-pane split:

    Left (Builder): create/edit; pick primary + up to 3 comparison PPAs from
    ``saved_ppa_scenarios``; customer/site/account/term fields; narrative toggle.

    Right (Preview & Export): comparison metric grid, chart (overlay / grouped
    bar toggle), and three export buttons (Deck PPTX, Deck + Appendix PPTX,
    Comparison XLSX). All exports draw from the current Proposal snapshots,
    so they stay in sync with what's saved — not the live PPA library.
    """
    st.subheader("Proposals")
    st.caption(
        "A Proposal bundles a **primary PPA** and up to three **comparison PPAs** "
        "for a single simulation, plus customer/site metadata. PPAs are snapshot "
        "into the Proposal — later edits on the PPA Rate tab don't silently "
        "mutate a saved deal."
    )

    saved_ppas = st.session_state.get("saved_ppa_scenarios") or {}
    if not saved_ppas:
        st.info(
            "No saved PPA structures yet. Open the **PPA Rate** tab, configure "
            "a PPA, and click **💾 Save PPA** — then come back to bundle them "
            "into a Proposal."
        )
        return

    existing_proposals = _list_proposals_session(
        st.session_state, simulation_name=simulation_name,
    )
    active_id = st.session_state.get("active_proposal_id")
    is_new_mode = bool(st.session_state.pop("_proposals_tab_new", False)) or (
        not existing_proposals
    )
    active_proposal = None
    if not is_new_mode and active_id:
        active_proposal = _get_active_proposal(st.session_state)

    cfg_col, out_col = st.columns([0.42, 0.58], gap="large")

    # ── Left: Builder ────────────────────────────────────────────────────
    with cfg_col:
        _mode_label = "Create new Proposal" if (is_new_mode or not active_proposal) else "Edit Proposal"
        st.markdown(f"**{_mode_label}**")

        defaults = {
            "name": (active_proposal.name if active_proposal else ""),
            "customer_name": (active_proposal.customer_name if active_proposal else ""),
            "site_address": (active_proposal.site_address if active_proposal else ""),
            "utility_account": (active_proposal.utility_account if active_proposal else ""),
            "term_years": (active_proposal.term_years if active_proposal else min(25, int(system_life_years))),
            "notes": (active_proposal.notes if active_proposal else ""),
            "primary_name": (active_proposal.primary_ppa.name
                             if active_proposal and active_proposal.primary_ppa.name in saved_ppas
                             else list(saved_ppas.keys())[0]),
            "comparison_names": tuple(
                s.name for s in (active_proposal.comparison_ppas if active_proposal else ())
                if s.name in saved_ppas
            ),
            "narrative_on": bool(active_proposal and active_proposal.narrative_bullets),
        }

        prop_name = st.text_input(
            "Proposal name", value=defaults["name"],
            placeholder="e.g. West Island Cotton — Q1 Standard",
            key="proposals_tab_name",
        )
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input(
                "Customer / Facility", value=defaults["customer_name"],
                key="proposals_tab_customer",
            )
            site_address = st.text_input(
                "Site address", value=defaults["site_address"],
                key="proposals_tab_address",
            )
        with c2:
            utility_account = st.text_input(
                "Utility account (optional)", value=defaults["utility_account"],
                key="proposals_tab_account",
            )
            term_years = st.number_input(
                "Term (years)", min_value=1, max_value=40,
                value=int(defaults["term_years"]), step=1,
                key="proposals_tab_term",
            )

        st.markdown("**PPA selection**")
        ppa_names = list(saved_ppas.keys())
        primary_name = st.selectbox(
            "Primary PPA",
            options=ppa_names,
            index=ppa_names.index(defaults["primary_name"]) if defaults["primary_name"] in ppa_names else 0,
            key="proposals_tab_primary",
            help="The PPA you'd present as the recommended offer.",
        )
        comparison_candidates = [n for n in ppa_names if n != primary_name]
        _default_comps = [c for c in defaults["comparison_names"] if c in comparison_candidates]
        comparison_names = st.multiselect(
            f"Comparison PPAs (up to {_PROP_MAX_COMPARISONS})",
            options=comparison_candidates,
            default=_default_comps,
            max_selections=_PROP_MAX_COMPARISONS,
            key="proposals_tab_comparisons",
            help="Alternative PPAs to show side-by-side with the primary offer.",
        )

        narrative_on = st.checkbox(
            "Include AI-generated executive summary bullets",
            value=defaults["narrative_on"],
            key="proposals_tab_narrative",
            help=(
                "Generated from the current simulation when the Proposal is saved. "
                "Requires ANTHROPIC_API_KEY."
            ),
        )

        notes = st.text_area(
            "Internal notes (not included in customer deck)",
            value=defaults["notes"], height=70, key="proposals_tab_notes",
        )

        persist_on_save = st.toggle(
            "Persist to GCS on save",
            value=True, key="proposals_tab_persist",
            help="Writes the Proposal JSON to both local disk and GCS so it "
                 "survives session reloads and is visible to other analysts.",
        )

        save_col, delete_col = st.columns([0.7, 0.3])
        with save_col:
            save_clicked = st.button(
                "💾 Save Proposal",
                type="primary", key="proposals_tab_save_btn",
                width="stretch",
                disabled=not prop_name.strip(),
            )
        with delete_col:
            delete_clicked = st.button(
                "Delete",
                key="proposals_tab_delete_btn",
                width="stretch",
                disabled=active_proposal is None,
            )

        if save_clicked:
            try:
                primary_snap = _snapshot_from_saved(
                    primary_name, saved_ppas[primary_name], term_years=int(term_years),
                )
                comparison_snaps = tuple(
                    _snapshot_from_saved(n, saved_ppas[n], term_years=int(term_years))
                    for n in comparison_names
                )
                bullets: tuple[str, ...] = ()
                if narrative_on:
                    try:
                        _ctx = _AIProposalContext(
                            customer_name=customer_name or "Customer",
                            site_address=site_address or "",
                            system_size_kw=float(system_size_kw or 0.0),
                            battery_capacity_kwh=float(battery_cap_kwh or 0.0),
                            nem_regime=nem_regime_1 or "NEM-3",
                            year1_savings_usd=float(getattr(result, "annual_savings", 0.0) or 0.0),
                            year1_bill_without_solar_usd=float(getattr(result, "annual_bill_without_solar", 0.0) or 0.0),
                            year1_bill_with_solar_usd=float(getattr(result, "annual_bill_with_solar", 0.0) or 0.0),
                            savings_pct=float(getattr(result, "savings_pct", 0.0) or 0.0),
                            horizon_years=int(term_years),
                            total_projected_savings_usd=float(primary_snap.lifetime_savings_usd or 0.0),
                            ppa_rate_usd_per_kwh=(primary_snap.year1_rate_r1 or None),
                        )
                        bullets = tuple(_ai_generate_exec_summary(_ctx) or ())
                    except Exception as exc:
                        st.warning(f"AI narrative skipped: {exc}")

                if active_proposal is not None:
                    updated = _update_proposal_obj(
                        active_proposal,
                        name=prop_name.strip(),
                        customer_name=customer_name,
                        site_address=site_address,
                        utility_account=utility_account,
                        term_years=int(term_years),
                        primary_ppa=primary_snap,
                        comparison_ppas=comparison_snaps,
                        narrative_bullets=bullets,
                        notes=notes,
                    )
                else:
                    updated = _create_proposal_obj(
                        name=prop_name.strip(),
                        simulation_name=simulation_name,
                        customer_name=customer_name,
                        site_address=site_address,
                        utility_account=utility_account,
                        term_years=int(term_years),
                        primary_ppa=primary_snap,
                        comparison_ppas=comparison_snaps,
                        narrative_bullets=bullets,
                        notes=notes,
                    )
                _save_proposal_session(st.session_state, updated)
                if persist_on_save:
                    try:
                        _persist_proposal_gcs(updated)
                    except Exception as exc:
                        st.warning(f"GCS persistence skipped: {exc}")
                st.success(f"Saved Proposal: {updated.name}")
                st.session_state["_focus_proposals_tab"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

        if delete_clicked and active_proposal is not None:
            try:
                if persist_on_save:
                    try:
                        _delete_proposal_gcs(active_proposal)
                    except Exception:
                        pass
                _delete_proposal_session(st.session_state, active_proposal.id)
                st.success(f"Deleted Proposal: {active_proposal.name}")
                st.session_state["_focus_proposals_tab"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")

    # ── Right: Preview + Export ──────────────────────────────────────────
    with out_col:
        # Use the just-saved active Proposal to drive the preview; fall back to
        # a live-preview built from the current form state.
        preview_source: _ProposalObj | None = _get_active_proposal(st.session_state)
        if preview_source is None and primary_name in saved_ppas:
            try:
                _live_primary = _snapshot_from_saved(
                    primary_name, saved_ppas[primary_name], term_years=int(term_years),
                )
                _live_comps = tuple(
                    _snapshot_from_saved(n, saved_ppas[n], term_years=int(term_years))
                    for n in comparison_names[:_PROP_MAX_COMPARISONS]
                )
                preview_source = _create_proposal_obj(
                    name=prop_name.strip() or "Preview",
                    simulation_name=simulation_name,
                    customer_name=customer_name or "",
                    site_address=site_address or "",
                    utility_account=utility_account or "",
                    term_years=int(term_years),
                    primary_ppa=_live_primary,
                    comparison_ppas=_live_comps,
                    notes="",
                )
            except Exception:
                preview_source = None

        if preview_source is None:
            st.info("Pick a primary PPA on the left to render the preview.")
            return

        _ui_section_header(
            f"Preview · {preview_source.name or 'Unsaved'}",
            caption=f"Primary: {preview_source.primary_ppa.name} · "
                    f"{len(preview_source.comparison_ppas)} comparison PPA"
                    f"{'s' if len(preview_source.comparison_ppas) != 1 else ''}",
        )

        # Snapshot-drift check: walk every snapshot in the Proposal and see
        # whether the underlying simulation has drifted since each was
        # captured (system size, rate/load escalator). Stale snapshots
        # produce misleading exports — surface the mismatch in an amber
        # callout and offer a one-click resnap via the PPA Rate tab.
        _all_snaps = [preview_source.primary_ppa, *preview_source.comparison_ppas]
        _stale = []
        for _snap in _all_snaps:
            reasons = _snapshot_drift_reasons(
                _snap,
                current_system_size_kw=float(system_size_kw or 0.0) or None,
                current_rate_escalator_pct=float(rate_escalator or 0.0) or None,
                current_load_escalator_pct=float(load_escalator or 0.0) or None,
            )
            if reasons:
                _stale.append((_snap.name, reasons))
        if _stale:
            _reason_items = "".join(
                f"<li><strong>{_name}</strong>: {'; '.join(_reasons)}</li>"
                for _name, _reasons in _stale
            )
            st.markdown(
                '<div style="background:var(--38dn-warning-bg);'
                'border:1px solid #F0D7A8;border-left:3px solid var(--38dn-amber);'
                'border-radius:var(--38dn-radius-md);'
                'padding:10px 14px;margin:6px 0 12px 0;'
                'font-size:var(--38dn-fs-body);color:var(--38dn-ink);">'
                '<div class="eyebrow-38dn" style="color:var(--38dn-amber);'
                'margin-bottom:4px;">Snapshots may be stale</div>'
                "The simulation has changed since the following PPA "
                "snapshots were captured. Exports will use the captured "
                "numbers — resave the PPA on the PPA Rate tab to refresh."
                f'<ul style="margin:6px 0 0 18px;padding:0;">{_reason_items}</ul>'
                '</div>',
                unsafe_allow_html=True,
            )

        _cmp_df = _build_prop_table(preview_source)
        st.markdown(
            render_styled_table(_cmp_df, bold_cols=["Metric"]),
            unsafe_allow_html=True,
        )

        _chart_mode = st.radio(
            "Chart view",
            options=["Overlay", "Grouped bars (Y1 / Y5 / Y10 / Y20)"],
            horizontal=True, key="proposals_tab_chart_mode",
            label_visibility="collapsed",
        )
        _mode = "overlay" if _chart_mode.startswith("Overlay") else "grouped"
        st.plotly_chart(
            _build_prop_chart(preview_source, mode=_mode),
            use_container_width=True,
            key=f"proposals_chart_{preview_source.id}_{_mode}",
        )

        if preview_source.narrative_bullets:
            with st.expander("Executive summary bullets", expanded=False):
                for b in preview_source.narrative_bullets:
                    st.markdown(f"- {b}")

        # ── Export controls ────────────────────────────────────────
        # Use the shared section_header component for consistent visual
        # rhythm with other Proposals-tab section boundaries.
        st.divider()
        _ui_section_header(
            "Export", caption="Customer-facing deliverables for this Proposal",
        )
        # Weighted columns: primary Deck action gets 40% visual width; the
        # two secondary exports take 30% each so the primary action reads
        # as the dominant CTA.
        ex1, ex2, ex3 = st.columns([0.4, 0.3, 0.3])

        def _comparison_ppas_payload(source: _ProposalObj) -> list[dict]:
            """Shape the PPASnapshots for the PPTX alternatives-considered slide."""
            return [
                {
                    "name": "Recommended",
                    "year1_rate_r1": source.primary_ppa.year1_rate_r1,
                    "year1_rate_r2": source.primary_ppa.year1_rate_r2,
                    "escalator_r1_pct": source.primary_ppa.escalator_r1_pct,
                    "escalator_r2_pct": source.primary_ppa.escalator_r2_pct,
                    "savings_pct": source.primary_ppa.savings_pct,
                    "lifetime_savings_usd": source.primary_ppa.lifetime_savings_usd,
                    "term_years": source.primary_ppa.term_years,
                },
                *[
                    {
                        "name": s.name,
                        "year1_rate_r1": s.year1_rate_r1,
                        "year1_rate_r2": s.year1_rate_r2,
                        "escalator_r1_pct": s.escalator_r1_pct,
                        "escalator_r2_pct": s.escalator_r2_pct,
                        "savings_pct": s.savings_pct,
                        "lifetime_savings_usd": s.lifetime_savings_usd,
                        "term_years": s.term_years,
                    }
                    for s in source.comparison_ppas
                ],
            ]

        def _build_deck(include_appendix: bool) -> bytes:
            """Single-line delegation to the module-level helper so
            Proposals-tab and any future Downloads-tab builder produce
            byte-identical decks from the same code path."""
            return _build_proposal_deck_bytes(
                source=preview_source,
                include_appendix=include_appendix,
                result=result,
                pv_only_result=pv_only_result,
                system_cost=system_cost,
                rate_escalator=rate_escalator,
                load_escalator=load_escalator,
                compound_escalation=compound_escalation,
                rs_old_baseline=rs_old_baseline,
                es_offset_annual=es_offset_annual,
                common_nem_kw=common_nem_kw,
                utility_name=utility_name,
                selected_rate_name=selected_rate_name,
                system_size_kw=system_size_kw,
                dc_ac_ratio=dc_ac_ratio,
                battery_cap_kwh=battery_cap_kwh,
                nem_regime_1=nem_regime_1,
                nem_regime_2=nem_regime_2,
                num_years_1=num_years_1,
            )

        _safe_name = (preview_source.customer_name or "Customer").replace(" ", "_")[:30]
        _date = date.today().strftime("%Y-%m-%d")

        with ex1:
            deck_bytes: bytes | None = None
            if st.button("📄 Build Deck (PPTX)", key="proposals_tab_deck_btn",
                         width="stretch", type="primary"):
                try:
                    with st.spinner("Building customer proposal deck..."):
                        deck_bytes = _build_deck(include_appendix=False)
                except Exception as exc:
                    st.error(f"Deck build failed: {exc}")
            if deck_bytes:
                st.download_button(
                    "Download Deck (.pptx)",
                    data=deck_bytes,
                    file_name=f"{_safe_name}_Deck_{_date}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="proposals_tab_deck_dl",
                    width="stretch",
                )

        with ex2:
            appendix_bytes: bytes | None = None
            _has_comps = bool(preview_source.comparison_ppas)
            if st.button(
                "📊 Deck + Comparison Appendix",
                key="proposals_tab_appendix_btn",
                width="stretch",
                disabled=not _has_comps,
                help="Adds a final slide listing the primary + comparison PPAs side-by-side.",
            ):
                try:
                    with st.spinner("Building deck with comparison appendix..."):
                        appendix_bytes = _build_deck(include_appendix=True)
                except Exception as exc:
                    st.error(f"Deck+appendix build failed: {exc}")
            if appendix_bytes:
                st.download_button(
                    "Download Deck + Appendix (.pptx)",
                    data=appendix_bytes,
                    file_name=f"{_safe_name}_Deck_with_Alternatives_{_date}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="proposals_tab_appendix_dl",
                    width="stretch",
                )

        with ex3:
            if st.button("📈 Comparison Workbook (XLSX)",
                         key="proposals_tab_xlsx_btn", width="stretch"):
                try:
                    xlsx_bytes = _export_prop_xlsx(preview_source)
                    st.session_state["_proposals_tab_xlsx"] = xlsx_bytes
                except Exception as exc:
                    st.error(f"XLSX export failed: {exc}")
            if st.session_state.get("_proposals_tab_xlsx"):
                st.download_button(
                    "Download Comparison (.xlsx)",
                    data=st.session_state["_proposals_tab_xlsx"],
                    file_name=f"{_safe_name}_Proposal_Comparison_{_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="proposals_tab_xlsx_dl",
                    width="stretch",
                )


def _render_ai_assistant_tab(
    *,
    result,
    customer_name: str,
    address: str,
    system_size_kw: float,
    battery_capacity_kwh: float,
    nem_regime: str,
    horizon_years: int,
    ppa_rate: float | None,
    tariff,
) -> None:
    """Narrative generation + bill ingestion + tariff Q&A.

    Each sub-panel is independent — the tab degrades gracefully when the
    underlying input (simulation result, uploaded PDF, loaded tariff) is
    absent. Outbound Anthropic calls only fire when the user clicks a
    button, so the tab renders cheaply on idle.
    """
    st.subheader("AI Assistant")
    st.caption(
        "Ask a question about the selected system, tariff, NEM regime, or "
        "billing structure — or upload a utility bill for auto-extraction, "
        "or generate proposal narrative bullets. Requires `ANTHROPIC_API_KEY`."
    )

    narrative_col, bill_col = st.columns(2)

    # -- Executive-summary narrative --------------------------------------
    with narrative_col:
        st.markdown("**Proposal executive summary**")
        st.caption("3–5 factual bullets generated from the current simulation results.")
        disabled = result is None
        if st.button("Generate bullets", key="ai_gen_narrative", disabled=disabled):
            try:
                ctx = _AIProposalContext(
                    customer_name=customer_name or "Customer",
                    site_address=address or "",
                    system_size_kw=system_size_kw,
                    battery_capacity_kwh=battery_capacity_kwh,
                    nem_regime=nem_regime,
                    year1_savings_usd=float(getattr(result, "annual_savings", 0.0) or 0.0),
                    year1_bill_without_solar_usd=float(
                        getattr(result, "annual_bill_without_solar", 0.0) or 0.0),
                    year1_bill_with_solar_usd=float(
                        getattr(result, "annual_bill_with_solar", 0.0) or 0.0),
                    savings_pct=float(getattr(result, "savings_pct", 0.0) or 0.0),
                    horizon_years=int(horizon_years),
                    total_projected_savings_usd=float(
                        getattr(result, "annual_savings", 0.0) or 0.0) * int(horizon_years),
                    ppa_rate_usd_per_kwh=float(ppa_rate) if ppa_rate else None,
                )
                bullets = _ai_generate_exec_summary(ctx)
                st.session_state["ai_narrative_bullets"] = bullets
            except Exception as exc:
                _render_ai_error(exc, "Narrative generation")

        if st.session_state.get("ai_narrative_bullets"):
            for b in st.session_state["ai_narrative_bullets"]:
                st.markdown(f"- {b}")

    # -- Bill PDF ingestion -----------------------------------------------
    with bill_col:
        st.markdown("**Extract data from a utility bill**")
        st.caption("Upload a recent PDF bill — fields below are pre-filled suggestions you can copy into the sidebar.")
        up = st.file_uploader("Utility bill (PDF)", type=["pdf"], key="ai_bill_upload")
        if up is not None and st.button("Extract", key="ai_bill_extract"):
            try:
                extraction = _ai_extract_bill(up.getvalue())
                st.session_state["ai_bill_extraction"] = extraction
            except Exception as exc:
                _render_ai_error(exc, "Bill extraction")

        extraction = st.session_state.get("ai_bill_extraction")
        if extraction is not None:
            fields = {
                "Utility": extraction.utility,
                "Rate schedule": extraction.rate_schedule,
                "Billing period": (
                    f"{extraction.billing_period_start} → {extraction.billing_period_end}"
                    if extraction.billing_period_start else None
                ),
                "Total kWh": (
                    f"{extraction.total_kwh:,.0f}" if extraction.total_kwh is not None else None
                ),
                "Peak demand (kW)": (
                    f"{extraction.peak_demand_kw:.1f}"
                    if extraction.peak_demand_kw is not None else None
                ),
                "Total charges": (
                    f"${extraction.total_charges_usd:,.2f}"
                    if extraction.total_charges_usd is not None else None
                ),
                "NEM true-up": "Yes" if extraction.nem_true_up else "No",
            }
            for label, value in fields.items():
                if value is not None:
                    st.markdown(f"- **{label}:** {value}")
            if extraction.notes:
                st.info(extraction.notes)

    # -- System / tariff / NEM / billing Q&A ------------------------------
    st.divider()
    st.markdown("**Ask a question about the selected system, tariff, NEM regime, or billing structure**")
    if tariff is None:
        st.info("Load a tariff in Section 4 of the sidebar to enable Q&A.")
    else:
        st.caption(
            "The assistant answers from the current URDB tariff JSON plus the "
            "system / NEM context shown below. For questions about tariff terms "
            "it quotes verbatim; for NEM regime / billing structure it cites the "
            "rule mechanics from the tariff rate structure."
        )
        q = st.text_input(
            "Question", key="ai_tariff_q",
            placeholder=(
                "e.g. What are the peak TOU hours in summer? "
                "What demand charges apply? How does NEM-3 settle exports?"
            ),
        )
        if q and st.button("Ask", key="ai_tariff_ask"):
            try:
                # Supplement the URDB JSON with the simulation context so the
                # assistant can answer system / NEM / billing-structure questions
                # beyond what the raw tariff JSON contains.
                ctx = {
                    "urdb_tariff": getattr(tariff, "raw_data", {}) or {},
                    "system_context": {
                        "system_size_kw": system_size_kw,
                        "battery_capacity_kwh": battery_capacity_kwh,
                        "nem_regime": nem_regime,
                        "utility": getattr(tariff, "utility", ""),
                        "rate_schedule_label": getattr(tariff, "label", ""),
                        "rate_schedule_name": getattr(tariff, "name", ""),
                    },
                }
                answer = _ai_tariff_ask(
                    q,
                    tariff_label=getattr(tariff, "label", ""),
                    urdb_json=ctx,
                )
                st.markdown(answer)
            except Exception as exc:
                _render_ai_error(exc, "Q&A")


@st.fragment
def _render_sensitivity_tab(
    *,
    result,
    result_pv_only,
    system_cost: float,
    rate_escalator: float,
    load_escalator: float,
    degradation_pct: float,
    system_life_years: int,
    nem_regime_1: str,
) -> None:
    """Monte Carlo + tornado sensitivity view.

    Decorated with ``@st.fragment`` so interacting with the config inputs
    (σ sliders, sample-count, seed, horizon, discount) only re-runs this
    fragment instead of the whole page — otherwise pressing Enter in any
    input would bounce the user back to the first top-level tab.

    Lets the user select projection-level levers (rate escalator, load
    escalator, PV degradation), pick a sample count, and see the NPV
    distribution update live as samples accumulate.
    """
    import plotly.graph_objects as go

    # 38DN palette
    NAVY = "#0E2841"
    GREEN = "#45A750"
    BLUE = "#1D6FA9"

    st.subheader("Sensitivity Analysis")
    st.markdown(
        "Projection-level Monte Carlo and tornado — **year-1 billing is held fixed**; "
        "escalators and degradation are perturbed around the base case. "
        "The reported metric is **NPV of Customer Savings** over the horizon, "
        "discounted at the chosen rate, net of the up-front system cost. "
        "Positive values = customer comes out ahead."
    )
    st.markdown("")  # one-line visual breather

    cfg_col, out_col = st.columns([0.38, 0.62], gap="large")

    with cfg_col:
        years = st.number_input(
            "Projection horizon (years)", 5, max(system_life_years, 5), min(20, system_life_years), 1,
            key="sens_years",
        )
        discount = st.number_input(
            "Discount rate (%)", 0.0, 20.0, 7.0, 0.5, key="sens_discount",
        )
        seed = st.number_input("Seed", 0, 9999, 42, 1, key="sens_seed")
        n_samples = st.slider("Samples", 50, 2000, 500, 50, key="sens_n")

        st.markdown(
            "**Annual levers** — all values below are expressed in **% per year**. "
            "σ controls the Monte Carlo spread around the base value; "
            "the Tornado swing controls how far the ± bars walk each lever "
            "in the tornado-chart sensitivity sweep."
        )

        st.markdown("*Utility rate escalator*")
        _rs_col1, _rs_col2 = st.columns(2)
        with _rs_col1:
            rate_sigma = st.number_input(
                "σ (%/yr)", 0.0, 5.0, 1.0, 0.1, key="sens_rate_sigma",
            )
        with _rs_col2:
            rate_swing = st.number_input(
                "Tornado ± (%/yr)", 0.0, 5.0, 1.0, 0.1,
                key="sens_rate_swing",
                help="Absolute swing in %/yr applied symmetrically around the base "
                     "rate escalator when running the tornado sweep.",
            )

        st.markdown("*Load growth escalator*")
        _ls_col1, _ls_col2 = st.columns(2)
        with _ls_col1:
            load_sigma = st.number_input(
                "σ (%/yr) ", 0.0, 5.0, 0.5, 0.1, key="sens_load_sigma",
            )
        with _ls_col2:
            load_swing = st.number_input(
                "Tornado ± (%/yr) ", 0.0, 5.0, 1.0, 0.1,
                key="sens_load_swing",
                help="Absolute swing in %/yr applied symmetrically around the "
                     "base load escalator when running the tornado sweep.",
            )

        st.markdown("*PV degradation (triangular MC; absolute swing for tornado)*")
        degrad_low, degrad_mode, degrad_high = st.columns(3)
        with degrad_low:
            d_low = st.number_input("MC low (%/yr)", 0.0, 2.0, 0.3, 0.05, key="sens_d_low")
        with degrad_mode:
            d_mode = st.number_input("MC mode (%/yr)", 0.0, 2.0, float(degradation_pct), 0.05, key="sens_d_mode")
        with degrad_high:
            d_high = st.number_input("MC high (%/yr)", 0.0, 2.0, 0.8, 0.05, key="sens_d_high")
        degrad_swing = st.number_input(
            "PV degradation tornado ± (%/yr)",
            0.0, 2.0, 0.5, 0.05,
            key="sens_degrad_swing",
            help="Absolute swing in %/yr applied symmetrically around the "
                 "base PV degradation (the MC mode) when running the tornado sweep.",
        )

        run_mc = st.button("Run Monte Carlo", type="primary", key="sens_run_mc")
        run_tornado = st.button("Run Tornado", key="sens_run_tornado")

    levers = [
        Lever(
            "rate_escalator", "normal",
            (float(rate_escalator), float(rate_sigma)),
            "Rate escalator", "%/yr",
            abs_swing=float(rate_swing),
        ),
        Lever(
            "load_escalator", "normal",
            (float(load_escalator), float(load_sigma)),
            "Load escalator", "%/yr",
            abs_swing=float(load_swing),
        ),
        Lever(
            "degradation", "triangular",
            (float(d_low), float(d_mode), float(d_high)),
            "PV degradation", "%/yr",
            abs_swing=float(degrad_swing),
        ),
    ]

    with out_col:
        # Inner tabs so the tornado chart is always discoverable even before the
        # user clicks Run Tornado — they can see the empty-state copy there.
        mc_sub, tornado_sub = st.tabs(["📊 Monte Carlo", "🌪 Tornado"])

    with mc_sub:
        placeholder = st.empty()

        # Counter for unique Plotly chart keys — Streamlit rejects duplicates,
        # and the final draw happens at the same len(npvs) as the last progress tick.
        draw_counter = {"n": 0}

        def _draw_mc(npvs: "np.ndarray", final: bool) -> None:
            draw_counter["n"] += 1
            pct = _sens_percentiles(npvs)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=npvs / 1_000_000, nbinsx=40, marker_color=NAVY, opacity=0.88,
                name="NPV of Customer Savings",
            ))
            for p, color in [(10, BLUE), (50, NAVY), (90, GREEN)]:
                fig.add_vline(
                    x=pct[p] / 1_000_000, line_dash="dash", line_color=color, line_width=2,
                    annotation_text=f"<b>P{p}</b>  ${pct[p]/1_000_000:,.2f}MM",
                    annotation_position="top",
                    annotation_font=dict(color=color, size=12),
                )
            fig.update_layout(
                title=dict(
                    text=f"NPV of Customer Savings — {len(npvs):,} sample"
                         f"{'s' if len(npvs)!=1 else ''}"
                         + (" (final)" if final else " (running…)"),
                    font=dict(size=15, color=NAVY),
                ),
                xaxis_title="NPV ($MM)",
                yaxis_title="Count",
                template="plotly_white",
                bargap=0.02,
                height=400,
                font=dict(family="Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif",
                          size=12, color="#1A1A1A"),
                margin=dict(l=50, r=30, t=70, b=50),
                showlegend=False,
            )
            placeholder.plotly_chart(
                fig, use_container_width=True,
                key=f"mc_{draw_counter['n']}_{'final' if final else 'live'}",
            )

        if run_mc:
            with st.status("Running Monte Carlo…", expanded=True) as status:
                def _cb(i: int, npvs_so_far):
                    status.write(f"{i:,} / {n_samples:,} samples")
                    _draw_mc(npvs_so_far, final=False)

                mc_df = _sens_monte_carlo(
                    result=result,
                    result_pv_only=result_pv_only,
                    system_cost=float(system_cost),
                    years=int(years),
                    discount_rate_pct=float(discount),
                    levers=levers,
                    n=int(n_samples),
                    seed=int(seed),
                    nem_regime_1=nem_regime_1,
                    progress_cb=_cb,
                    chunk=max(10, n_samples // 20),
                )
                status.update(label=f"Monte Carlo complete: {len(mc_df):,} samples", state="complete")

            _draw_mc(mc_df["npv"].to_numpy(), final=True)
            st.session_state["sensitivity_mc_df"] = mc_df

            pct = _sens_percentiles(mc_df["npv"].to_numpy())
            k1, k2, k3 = st.columns(3)
            k1.metric("P10 NPV (Customer Savings)", f"${pct[10]/1_000_000:,.2f}MM")
            k2.metric("P50 NPV (Customer Savings)", f"${pct[50]/1_000_000:,.2f}MM")
            k3.metric("P90 NPV (Customer Savings)", f"${pct[90]/1_000_000:,.2f}MM")
        elif st.session_state.get("sensitivity_mc_df") is None:
            st.caption(
                "Click **Run Monte Carlo** on the left to sample the NPV "
                "distribution across the selected levers."
            )

    with tornado_sub:
        if not run_tornado and st.session_state.get("sensitivity_tornado_df") is None:
            st.caption(
                "Click **Run Tornado** on the left for a ±10% one-at-a-time "
                "sensitivity sweep — shows which lever moves NPV the most."
            )

        if run_tornado:
            with st.spinner("Running tornado sweep…"):
                tdf = _sens_tornado(
                    result=result,
                    result_pv_only=result_pv_only,
                    system_cost=float(system_cost),
                    years=int(years),
                    discount_rate_pct=float(discount),
                    levers=levers,
                    pct_low=-0.10,
                    pct_high=0.10,
                    nem_regime_1=nem_regime_1,
                )
            base = tdf.attrs.get("base_npv", 0.0)

            fig = go.Figure()
            # Bars drawn as (base -> low) and (base -> high) segments around base NPV.
            for _, row in tdf[::-1].iterrows():
                fig.add_trace(go.Bar(
                    y=[row["lever"]], x=[row["low_npv"] - base],
                    base=base, orientation="h",
                    marker_color=BLUE, opacity=0.9, showlegend=False,
                    hovertemplate=(
                        f"{row['lever']}<br>low: ${row['low_npv']/1_000_000:,.2f}MM<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Bar(
                    y=[row["lever"]], x=[row["high_npv"] - base],
                    base=base, orientation="h",
                    marker_color=GREEN, opacity=0.9, showlegend=False,
                    hovertemplate=(
                        f"{row['lever']}<br>high: ${row['high_npv']/1_000_000:,.2f}MM<extra></extra>"
                    ),
                ))
            fig.add_vline(
                x=base, line_color=NAVY, line_width=2.5,
                annotation_text=f"<b>Base NPV</b>  ${base/1_000_000:,.2f}MM",
                annotation_position="top",
                annotation_font=dict(color=NAVY, size=12),
            )
            fig.update_layout(
                title=dict(
                    text="Tornado — impact on NPV of Customer Savings (±10% lever swing around base)",
                    font=dict(size=15, color=NAVY),
                ),
                xaxis_title="NPV ($)",
                yaxis_title="",
                barmode="overlay",
                template="plotly_white",
                height=max(260, 80 + 55 * len(tdf)),
                font=dict(family="Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif",
                          size=12, color="#1A1A1A"),
                margin=dict(l=120, r=30, t=70, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                tdf[["lever", "base", "low", "high", "low_npv", "high_npv", "swing"]]
                    .style.format({
                        "base": "{:.2f}", "low": "{:.2f}", "high": "{:.2f}",
                        "low_npv": "${:,.0f}", "high_npv": "${:,.0f}", "swing": "${:,.0f}",
                    }),
                use_container_width=True, hide_index=True,
            )
            st.session_state["sensitivity_tornado_df"] = tdf


# =============================================================================
# HELPER FUNCTIONS — Profiles (Load & Export)
# =============================================================================
def _save_profile_csv(directory, name, df):
    name = sanitize_filename(name)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    gcs_prefix = _DIR_TO_GCS_PREFIX.get(directory)
    if gcs_prefix:
        save_profile_bytes(directory, gcs_prefix, name, csv_bytes, ".csv")
    else:
        with open(os.path.join(directory, f"{name}.csv"), "wb") as f:
            f.write(csv_bytes)


@st.cache_data
def _load_profile_csv(directory, name) -> pd.DataFrame:
    local_path = os.path.join(directory, f"{name}.csv")
    if os.path.isfile(local_path):
        return pd.read_csv(local_path)
    # Try GCS fallback
    gcs_prefix = _DIR_TO_GCS_PREFIX.get(directory)
    if gcs_prefix:
        data = load_profile_bytes(directory, gcs_prefix, name, ".csv")
        if data is not None:
            return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(local_path)  # raise FileNotFoundError as before


def _save_system_profile(name: str) -> None:
    """Save current sidebar PV system settings + production data to a JSON file."""
    profile = {
        "location": st.session_state.get("sb_location", ""),
        "lat": st.session_state.get("_sp_lat"),
        "lon": st.session_state.get("_sp_lon"),
        "system_life": st.session_state.get("sb_system_life", 20),
        "system_size_kw": st.session_state.get("sb_system_size", 500.0),
        "dc_ac_ratio": st.session_state.get("sb_dc_ac_ratio", 1.2),
        "system_type": st.session_state.get("sb_system_type", "Fixed Tilt (Ground Mount)"),
        "module_type": st.session_state.get("sb_module_type", "Standard"),
        "system_losses": st.session_state.get("sb_system_losses", 14.08),
        "degradation": st.session_state.get("sb_degradation", 0.50),
        "cod_date": str(st.session_state.get("sb_cod_date", date(2026, 1, 1))),
    }
    prod = st.session_state.get("production_8760")
    if prod is not None:
        profile["production_8760"] = [float(v) for v in prod]
    summary = st.session_state.get("production_summary")
    if summary is not None:
        profile["production_summary"] = summary
    json_bytes = json.dumps(profile).encode("utf-8")
    save_profile_bytes(SYSTEM_PROFILES_DIR, _DIR_TO_GCS_PREFIX[SYSTEM_PROFILES_DIR], name, json_bytes, ".json")


@st.cache_data
def _load_system_profile(name: str) -> dict:
    """Load a system profile JSON and return the dict."""
    fp = os.path.join(SYSTEM_PROFILES_DIR, f"{name}.json")
    if os.path.isfile(fp):
        with open(fp, "r") as f:
            return json.load(f)
    # GCS fallback
    data = load_profile_bytes(SYSTEM_PROFILES_DIR, _DIR_TO_GCS_PREFIX[SYSTEM_PROFILES_DIR], name, ".json")
    if data is not None:
        return json.loads(data)
    # Raise FileNotFoundError as before
    with open(fp, "r") as f:
        return json.load(f)


def _load_nema_profile_into_session(profile_name: str):
    """Load a saved NEM-A profile bundle into session state.

    Restores: nema_utility, nema_meters, nema_meter_loads, nema_meter_tariffs,
    existing_solar_nema_meters, and sets load_8760 from the generating meter.
    """
    path = os.path.join(NEMA_PROFILES_DIR, f"{profile_name}.json")
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    else:
        raw = load_profile_bytes(NEMA_PROFILES_DIR, _DIR_TO_GCS_PREFIX[NEMA_PROFILES_DIR], profile_name, ".json")
        if raw is not None:
            data = json.loads(raw)
        else:
            with open(path) as f:  # raise FileNotFoundError
                data = json.load(f)
    st.session_state["nema_utility"] = data.get("utility", "PG&E")
    year = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
    dt = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    meters, loads, tariffs = [], {}, {}
    for i, m in enumerate(data.get("meters", [])):
        meters.append({
            "name": m["name"],
            "is_generating": m.get("is_generating", False),
            "use_gen_tariff": m.get("use_gen_tariff", not m.get("is_generating", False)),
            "load_key": f"nema_load_{i}",
            "tariff_key": f"nema_tariff_{i}",
        })
        if m.get("load_8760"):
            loads[i] = pd.Series(m["load_8760"], index=dt, name="load_kwh")
        if m.get("tariff"):
            tariffs[i] = TariffSchedule(**m["tariff"])
    st.session_state["nema_meters"] = meters
    st.session_state["nema_meter_loads"] = loads
    st.session_state["_raw_nema_meter_loads"] = {k: v.copy() for k, v in loads.items()}
    st.session_state["nema_meter_tariffs"] = tariffs
    st.session_state["existing_solar_nema_meters"] = data.get("existing_solar_meters", [])
    st.session_state["load_mode"] = "NEM-A Aggregation"
    st.session_state["load_mode_radio"] = "NEM-A Aggregation"
    # Set generating meter load as the main load_8760
    for i, m in enumerate(meters):
        if m["is_generating"] and i in loads:
            st.session_state["load_8760"] = loads[i]
            st.session_state["_raw_load_8760"] = loads[i].copy()
            break


def _parse_8760_csv(df: pd.DataFrame) -> np.ndarray:
    """Extract the load numeric column from a DataFrame, validate 8760 rows.

    If the first numeric column is an hour-year index (1-8760 sequential integers),
    skip it and use the next numeric column instead.

    Raises if the selected column has blank / non-numeric cells — those would
    become NaN and silently poison downstream billing + projection math.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV.")
    col = numeric_cols[0]
    if len(numeric_cols) > 1 and len(df) == 8760:
        first_vals = df[col].values
        if np.array_equal(first_vals, np.arange(1, 8761)):
            col = numeric_cols[1]
    values = np.asarray(df[col].values, dtype=float)
    if len(values) != 8760:
        raise ValueError(f"Expected 8760 rows, got {len(values)}.")

    bad = ~np.isfinite(values)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return values

    # Largest run of consecutive NaNs.
    run = max_run = 0
    for flag in bad:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)

    MAX_FILLABLE_HOURS = 10
    MAX_FILLABLE_RUN = 3
    if n_bad > MAX_FILLABLE_HOURS or max_run > MAX_FILLABLE_RUN:
        bad_idx = np.where(bad)[0]
        sample = ", ".join(str(int(i) + 2) for i in bad_idx[:5])
        more = f" (+{len(bad_idx) - 5} more)" if len(bad_idx) > 5 else ""
        raise ValueError(
            f"Column '{col}' has {n_bad:,} blank/non-numeric cells "
            f"(longest consecutive gap: {max_run} hour{'s' if max_run != 1 else ''}). "
            f"First at CSV row{'s' if n_bad > 1 else ''}: {sample}{more}. "
            f"Too many gaps to auto-fill — clean the CSV and re-upload."
        )

    # Small, isolated gaps — linear-interpolate with a visible warning.
    filled = pd.Series(values).interpolate(method="linear",
                                           limit=MAX_FILLABLE_RUN,
                                           limit_direction="both").to_numpy()
    bad_idx = np.where(bad)[0]
    rows_str = ", ".join(str(int(i) + 2) for i in bad_idx[:5])
    more = f" (+{len(bad_idx) - 5} more)" if len(bad_idx) > 5 else ""
    st.warning(
        f"Filled {n_bad} missing hour{'s' if n_bad != 1 else ''} in column '{col}' "
        f"via linear interpolation (CSV row{'s' if n_bad > 1 else ''}: {rows_str}{more}). "
        f"Review the source data if this was unexpected."
    )
    return filled


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="PV Solar Rate Simulator",
    page_icon="☀️",
    layout="wide",
)

# --- Logo in top-right corner ---
import base64

# --- Global font override — Aptos Narrow ---
# Exclude Material Symbols / icon spans so arrows and icons render correctly
st.markdown("""
<style>
html, body, [class*="css"], [data-testid="stAppViewContainer"],
.stMarkdown, .stDataFrame, .stMetric, .stTabs, .stButton,
input, select, textarea, button, p, div, h1, h2, h3, h4, h5, h6, label, li {
    font-family: "Aptos Narrow", "Aptos", "Calibri", "Arial Narrow", sans-serif !important;
}
span:not([class*="material"]):not([data-testid*="Icon"]):not([class*="icon"]) {
    font-family: "Aptos Narrow", "Aptos", "Calibri", "Arial Narrow", sans-serif !important;
}

/* Typography scale */
h1 { font-size: 22px !important; letter-spacing: 0.3px !important; font-weight: 600 !important; }
h2 { font-size: 18px !important; letter-spacing: 0.3px !important; font-weight: 600 !important; }
h3 { font-size: 16px !important; letter-spacing: 0.2px !important; font-weight: 600 !important; }
p, li, div, label, input, select, textarea {
    font-size: 13px !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 11px !important;
    color: #6b7280 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #0E2841 !important;
}

/* Tighter content spacing */
.block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
/* Subtle dividers */
hr {
    border: none !important;
    border-top: 1px solid #e5e7eb !important;
    margin: 12px 0 !important;
}
/* Consistent section gaps */
[data-testid="stVerticalBlock"] > div {
    margin-bottom: 4px !important;
}

/* ===== Fixed Navy Top Bar — full width, above sidebar ===== */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
.nav-bar-wrapper {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 54px !important;
    background-color: #0E2841 !important;
    z-index: 999999 !important;
    padding: 8px 16px 8px 330px !important;
    display: flex !important;
    align-items: center !important;
    overflow: visible !important;
    gap: 0 !important;
}
.nav-bar-wrapper > div[data-testid="stColumn"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
}
/* Push main content below the fixed bar */
.block-container {
    padding-top: 70px !important;
}
/* Sidebar below the bar */
section[data-testid="stSidebar"] {
    top: 54px !important;
    height: calc(100vh - 54px) !important;
}
/* When collapsed: keep a 28px visible strip so user can click to re-open */
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 28px !important;
    border-right: 1px solid #ddd !important;
    background-color: #f0f2f6 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    min-width: 28px !important;
    width: 28px !important;
}
section[data-testid="stSidebar"][aria-expanded="false"]:hover {
    min-width: 40px !important;
    cursor: pointer !important;
}
/* Sidebar refinements */
section[data-testid="stSidebar"] [data-testid="stHeader"] {
    font-size: 16px !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 15px !important;
    padding-top: 8px !important;
    border-top: 1px solid #ddd !important;
    margin-top: 8px !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 14px !important;
    color: #374151 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #4b5563 !important;
}
/* Fill the full (user-resized) sidebar width. Streamlit's default content
   wrappers and inner block-container can stay at a compressed/capped width when
   the panel is dragged wider, leaving the widgets squeezed in the middle. Force
   the content wrappers + block container to span the panel. Scoped to the
   expanded state so the 28px collapse-strip rule above still wins when
   collapsed; widgets themselves already stretch once their parents do. */
section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child,
section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarUserContent"] {
    width: 100% !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {
    max-width: 100% !important;
}
/* Freeze the input-loading tracker at the top of the sidebar while the rest of
   the configuration scrolls. position:sticky must live on Streamlit's element
   container (a direct child of the scrolling sidebar column), NOT on the inner
   markdown div — that div's containing block is only as tall as itself, so it
   has nothing to stick over. We key off the #sb-sticky-tracker marker the
   tracker HTML carries, via :has(), so the selector is robust to wrapper-class
   churn across Streamlit versions. */
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(#sb-sticky-tracker),
section[data-testid="stSidebar"] .element-container:has(#sb-sticky-tracker) {
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    background: #FFFFFF !important;
}
/* Popovers: ensure decent width */
[data-testid="stPopoverBody"] {
    min-width: 320px !important;
}
/* ALL nav-bar buttons: transparent, white bold text, no shape */
.nav-bar-wrapper button[data-testid="stPopoverButton"],
.nav-bar-wrapper button[data-testid="stPopoverButton"] p,
.nav-bar-wrapper button[data-testid="stPopoverButton"] span {
    background: transparent !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
.nav-bar-wrapper button[data-testid="stPopoverButton"]:hover {
    /* Subtle hairline white outline — no background fill flash. */
    background: transparent !important;
    color: #ffffff !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55) !important;
    border-radius: 3px !important;
}
.nav-bar-wrapper button[data-testid="stPopoverButton"]:focus,
.nav-bar-wrapper button[data-testid="stPopoverButton"]:focus-visible {
    outline: none !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.7) !important;
}
.nav-bar-wrapper button[data-testid="stPopoverButton"]:active {
    background: transparent !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.85) !important;
}
</style>
""", unsafe_allow_html=True)

# Phase 5: install the institutional theme (Inter + JetBrains Mono + tokens
# from assets/theme.css) AFTER the legacy inline CSS so the new tokens win
# on collisions. The legacy block is kept around for the navy top-bar rules
# (the theme file deliberately doesn't touch `.nav-bar-wrapper`).
from modules.ui import install_theme as _install_theme, set_dense_mode as _set_dense_mode
from modules.ui.components import section_header as _ui_section_header
_install_theme()
# Density preference reads from session_state; the toggle lives in the
# sidebar (see _render_sidebar). Set the attribute every rerun so the
# selected mode survives navigation and script reruns.
_set_dense_mode(bool(st.session_state.get("ui_dense_mode", False)))

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    # Embedded logo: sits in the document flow at the top-right of the
    # first content row, so it scrolls away with the page like any other
    # static asset. No longer fixed.
    st.markdown(
        f"""
        <style>
        .embedded-logo {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            margin: -6px 0 8px 0;
            padding: 0 4px;
        }}
        .embedded-logo img {{
            height: 40px;
            width: 40px;
            object-fit: contain;
            opacity: 0.92;
        }}
        </style>
        <div class="embedded-logo">
            <img src="data:image/png;base64,{logo_b64}" alt="38DN Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

def _init_session_state():
    """Initialize all session state defaults and handle pending load actions."""
    # =============================================================================
    # SESSION STATE INITIALIZATION
    # =============================================================================
    for key, default in {
        "production_8760": None,
        "production_summary": None,
        "load_8760": None,
        "available_rates": None,
        "tariff": None,
        "export_rates": None,
        "export_rates_multiyear": None,
        "billing_result": None,
        "billing_result_pv_only": None,
        "billing_result_batt": None,
        "saved_view": None,
        "battery_enabled": False,
        "battery_config": None,
        "battery_capacity_kwh": 0,
        "sizing_result": None,
        "active_mgmt_tab": None,
        "editing_saved_sim": False,
        "nem_regime_1": "NEM-3 / NVBT",
        "nem_switch": False,
        "nem_regime_2": "NEM-3 / NVBT",
        "nem_years_1": 5,
        "export_rates_2": None,
        "export_rates_multiyear_2": None,
        "billing_engine": "Custom",
        "ecc_cost_calculator": None,
        "ecc_tariff_metadata": None,
        "ecc_tariff_data": None,
        "nbc_rate": 0.0,
        "nsc_rate": NSC_DEFAULT_RATE,
        "billing_option": "ABO",
        "pending_sim_load": None,
        "pending_system_profile": None,
        "show_all_sims": False,
        "load_mode": "Single Meter",
        "nema_meters": [],
        "nema_meter_loads": {},
        "nema_meter_tariffs": {},
        "nema_utility": "PG&E",
        "existing_solar_enabled": False,
        "existing_solar_production_8760": None,
        "existing_solar_nema_meters": [],
        "custom_rate_extracted": None,
        "custom_rate_warnings": None,
        # Master Rate Switch toggle (UI grouping). When off, the projection
        # is a plain single-tariff run (rate_shift_enabled False, regime-2
        # keys None). rate_shift_enabled is derived from the "At Repower /
        # project start" trigger; regime_2_* from the "At NEM switch" trigger.
        "rate_switch_enabled": False,
        "rate_shift_enabled": False,
        "rate_shift_old_tariff": None,
        "nema_rate_shift_tariffs": {},
        # Post-transition (regime 2) tariff overrides. None ⇒ reuse the
        # regime-1 tariff (default, backwards compatible). When set AND
        # nem_switch is on, the projection re-bills the post-switch years
        # on this second tariff.
        "regime_2_tariff": None,
        "regime_2_ecc_calculator": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Handle pending simulation load ---
    if st.session_state.get("pending_sim_load"):
        _pending_name = st.session_state["pending_sim_load"]
        st.session_state["pending_sim_load"] = None
        _sim_data = _load_simulation(_pending_name)
        touch_simulation_mtime(_pending_name)
        populate_session_from_simulation(st.session_state, _sim_data)
        # Phase 4: keep the simulation name around so the Proposals selector
        # and GCS persistence can scope by it.
        st.session_state["_active_simulation_name"] = _pending_name
        st.session_state["_last_loaded_simulation_name"] = _pending_name
        # Hydrate saved proposals from GCS + local disk on load.
        try:
            _loaded_props = _load_proposals_gcs(_pending_name)
            if _loaded_props:
                st.session_state["proposals"] = {
                    p.id: _proposals.to_dict(p) for p in _loaded_props
                }
        except Exception:
            pass
        st.rerun()

    # --- Handle pending system profile load ---
    if st.session_state.get("pending_system_profile"):
        _sp_name = st.session_state["pending_system_profile"]
        st.session_state["pending_system_profile"] = None
        _sp_data = _load_system_profile(_sp_name)
        _sp_loc = _sp_data.get("location", "")
        st.session_state["sb_location"] = _sp_loc
        st.session_state["_sp_lat"] = _sp_data.get("lat")
        st.session_state["_sp_lon"] = _sp_data.get("lon")
        st.session_state["_sp_cached_location"] = _sp_loc
        st.session_state["sb_system_life"] = _sp_data.get("system_life", 20)
        st.session_state["sb_system_size"] = _sp_data.get("system_size_kw", 500.0)
        st.session_state["sb_dc_ac_ratio"] = _sp_data.get("dc_ac_ratio", 1.2)
        st.session_state["sb_system_type"] = _sp_data.get("system_type", "Fixed Tilt (Ground Mount)")
        st.session_state["sb_module_type"] = _sp_data.get("module_type", "Standard")
        st.session_state["sb_system_losses"] = _sp_data.get("system_losses", 14.08)
        st.session_state["sb_degradation"] = _sp_data.get("degradation", 0.50)
        cod_str = _sp_data.get("cod_date")
        if cod_str:
            st.session_state["sb_cod_date"] = date.fromisoformat(cod_str)
        if _sp_data.get("production_8760"):
            _cod_year = date.fromisoformat(cod_str).year if cod_str else 2024
            _dt_idx = pd.date_range(
                start=f"{_cod_year}-01-01 00:00", periods=8760, freq="h"
            )
            st.session_state["production_8760"] = pd.Series(
                _sp_data["production_8760"], index=_dt_idx, name="solar_kwh"
            )
        if _sp_data.get("production_summary"):
            st.session_state["production_summary"] = _sp_data["production_summary"]
        st.rerun()


_init_session_state()

# --- All Simulations view (inline) ---
if st.session_state.get("show_all_sims"):
    from datetime import datetime as _dt_cls

    st.title("All Simulations")
    st.caption("Click **Load** to open a simulation in the main app.")

    _all_sim_names = list_saved_simulations()

    if not _all_sim_names:
        st.info("No saved simulations found. Run and save a simulation from the main page.")
        if st.button("Back to Simulator", type="primary"):
            st.session_state["show_all_sims"] = False
            st.rerun()
        st.stop()

    _search = st.text_input("Search simulations...", key="sim_search", placeholder="Filter by name")
    if _search:
        _all_sim_names = [n for n in _all_sim_names if _search.lower() in n.lower()]

    if not _all_sim_names:
        st.warning("No simulations match the search.")
        if st.button("Back to Simulator", type="primary"):
            st.session_state["show_all_sims"] = False
            st.rerun()
        st.stop()

    # Table header
    _hdr = st.columns([2.5, 1.2, 1.2, 1, 1.5, 0.7, 0.7])
    _hdr[0].markdown("**Name**")
    _hdr[1].markdown("**PV System Size**")
    _hdr[2].markdown("**BESS Size**")
    _hdr[3].markdown("**Utility**")
    _hdr[4].markdown("**Last Run Date**")
    _hdr[5].markdown("**&nbsp;**")
    _hdr[6].markdown("**&nbsp;**")
    st.markdown(
        "<hr style='margin:4px 0; border:none; border-top:2px solid #2A7B7B;'>",
        unsafe_allow_html=True,
    )

    for _sname in _all_sim_names:
        _smeta = get_simulation_metadata(_sname)
        _sbatt = _smeta.get("battery_capacity_kwh", 0)
        _ssaved = _smeta.get("saved_at", "")
        try:
            _sdt = _dt_cls.fromisoformat(_ssaved)
            _sdisp_date = _sdt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            _sdisp_date = _ssaved[:16] if _ssaved else "N/A"

        _scols = st.columns([2.5, 1.2, 1.2, 1, 1.5, 0.7, 0.7])
        _scols[0].markdown(f"**{_sname}**")
        _scols[1].write(f"{_smeta['system_size_kw']:,.0f} kW")
        _scols[2].write(f"{_sbatt:,.0f} kWh" if _sbatt else "None")
        _scols[3].write(_smeta["utility"])
        _scols[4].write(_sdisp_date)

        with _scols[5]:
            if st.button("Load", key=f"load_{_sname}", width="stretch"):
                touch_simulation_mtime(_sname)
                populate_session_from_simulation(
                    st.session_state, _load_simulation(_sname)
                )
                st.session_state["show_all_sims"] = False
                st.rerun()

        with _scols[6]:
            if st.button("Del", key=f"del_{_sname}", width="stretch"):
                st.session_state[f"confirm_del_{_sname}"] = True

        if st.session_state.get(f"confirm_del_{_sname}"):
            _cc1, _cc2, _cc3 = st.columns([3, 1, 1])
            _cc1.warning(f"Delete **{_sname}**? This cannot be undone.")
            with _cc2:
                if st.button("Yes, delete", key=f"yes_del_{_sname}", type="primary"):
                    delete_simulation(_sname)
                    st.session_state.pop(f"confirm_del_{_sname}", None)
                    st.rerun()
            with _cc3:
                if st.button("Cancel", key=f"cancel_del_{_sname}"):
                    st.session_state.pop(f"confirm_del_{_sname}", None)
                    st.rerun()

        st.markdown(
            "<hr style='margin:2px 0; border:none; border-top:1px solid #e0e0e0;'>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("Back to Simulator", type="primary"):
        st.session_state["show_all_sims"] = False
        st.rerun()
    st.stop()


def _render_custom_tariff_loader(
    *,
    select_label: str,
    button_label: str,
    sel_key: str,
    button_key: str,
    pending_key: str,
    target_key: str,
    loaded_prefix: str,
    empty_warning: str,
    select_help: str | None = None,
    no_rates_caption: str = "Fetch rates above first to select a tariff.",
) -> None:
    """Render the Custom-engine "rate selectbox + Load button + status" loader.

    Used in several sidebar spots (original/pre-switch rate, post-NEM switch
    rate, and the Load-Profiles mirror) that were previously copy-pasted — the
    source of the label drift the reviewers flagged. On click it stashes the
    chosen rate label in ``st.session_state[pending_key]``; the shared
    pending-load handler does the actual fetch. Status reads from
    ``st.session_state[target_key]`` (the loaded TariffSchedule, or None).
    """
    rates = st.session_state.get("available_rates")
    if rates:
        options = {f"{r['name']}": r["label"] for r in rates}
        selected = st.selectbox(select_label, list(options.keys()), key=sel_key, help=select_help)
        if st.button(button_label, key=button_key):
            st.session_state[pending_key] = options[selected]
    else:
        st.caption(no_rates_caption)
    current = st.session_state.get(target_key)
    if current is not None:
        st.success(f"{loaded_prefix}: {current.name}")
    else:
        st.warning(empty_warning)


def _render_top_bar():
    """Render the top management bar (Simulations, Profiles, Save popovers) and tab sections."""
    # =============================================================================
    # TOP MANAGEMENT BUTTONS — Simulations / Load Profiles / Export Profiles
    # =============================================================================
    # JS to tag the button-row container as the fixed nav bar.
    # st.html (Streamlit 1.54+) injects directly into the DOM (no iframe).
    st.html(
        """
        <script>
        (function tagNavBar() {
            function apply() {
                const bc = document.querySelector('.block-container');
                if (!bc) return false;
                const hb = bc.querySelector('[data-testid="stHorizontalBlock"]');
                if (hb && !hb.classList.contains('nav-bar-wrapper')) {
                    hb.classList.add('nav-bar-wrapper');
                }
                return !!hb;
            }
            // Try immediately
            if (!apply()) {
                // Retry after DOM renders
                const iv = setInterval(function() { if (apply()) clearInterval(iv); }, 100);
                setTimeout(function() { clearInterval(iv); }, 5000);
            }
            // Observe for Streamlit re-renders
            const obs = new MutationObserver(function() { apply(); });
            const bc = document.querySelector('.block-container');
            if (bc) obs.observe(bc, {childList: true, subtree: true});
        })();

        /* --- Sidebar: expand if collapsed, and add click-to-expand on collapsed strip --- */
        (function sidebarHelper() {
            function expandSidebar() {
                var sb = document.querySelector('section[data-testid="stSidebar"]');
                if (!sb) return;
                if (sb.getAttribute('aria-expanded') === 'false') {
                    /* Find any expand button inside the sidebar or header and click it */
                    var btn = document.querySelector('[data-testid="collapsedControl"] button')
                           || document.querySelector('header button[kind="headerNoPadding"]')
                           || document.querySelector('header button');
                    if (btn) btn.click();
                }
            }
            /* Force sidebar open on load */
            setTimeout(expandSidebar, 500);
            setTimeout(expandSidebar, 1500);

            /* Click handler: clicking the collapsed strip re-opens the sidebar */
            document.addEventListener('click', function(e) {
                var sb = e.target.closest('section[data-testid="stSidebar"]');
                if (sb && sb.getAttribute('aria-expanded') === 'false') {
                    var btn = document.querySelector('[data-testid="collapsedControl"] button')
                           || document.querySelector('header button[kind="headerNoPadding"]')
                           || document.querySelector('header button');
                    if (btn) btn.click();
                }
            });
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    # Hide the st.html element container so it doesn't take vertical space
    st.markdown("""
    <style>
    /* Hide the st.html script container (it's an stHtml element before the button row) */
    [data-testid="stHtml"] {
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stHtml"]:has(+ [data-testid="stHorizontalBlock"]) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Proposals popover added to the right of Save to mirror the save flow —
    # users can pick a named bundle for the active simulation without leaving
    # the top bar. Column ratios rebalanced to keep the row symmetric.
    _mgmt_btn_cols = st.columns([0.15, 1, 1, 1, 1, 1, 1, 1, 1.5])

    # --- Simulations popover ---
    with _mgmt_btn_cols[1]:
        with st.popover("Simulations", width="stretch"):
            _all_sims = list_saved_simulations()
            _recent_3 = _all_sims[:3]

            if _recent_3:
                st.markdown("**Recent**")
                for _rsim in _recent_3:
                    _rmeta = get_simulation_metadata(_rsim)
                    _rsize = _rmeta.get("system_size_kw", 0)
                    _rutil = _rmeta.get("utility", "N/A")
                    _rbatt = _rmeta.get("battery_capacity_kwh", 0)
                    _rbatt_lbl = f" + {_rbatt:,.0f} kWh BESS" if _rbatt else ""
                    _rdate = _rmeta.get("saved_at", "")[:10]
                    if st.button(
                        _rsim,
                        key=f"popover_sim_{_rsim}",
                        width="stretch",
                        help=f"{_rsize:,.0f} kW | {_rutil}{_rbatt_lbl} | {_rdate}",
                    ):
                        touch_simulation_mtime(_rsim)
                        populate_session_from_simulation(
                            st.session_state, _load_simulation(_rsim)
                        )
                        st.rerun()
                st.divider()
            else:
                st.caption("No saved simulations yet.")

            if st.button("View All Simulations", width="stretch", type="primary"):
                st.session_state["show_all_sims"] = True
                st.rerun()

    # --- System Profiles popover ---
    with _mgmt_btn_cols[2]:
        with st.popover("System Profiles", width="stretch"):
            _sp_names = _list_saved(SYSTEM_PROFILES_DIR, ".json")
            _sp_recent_3 = _sp_names[:3]

            if _sp_recent_3:
                st.markdown("**Recent**")
                for _sp_r in _sp_recent_3:
                    _sp_r_data = _load_system_profile(_sp_r)
                    _sp_r_size = _sp_r_data.get("system_size_kw", 0)
                    _sp_r_loc = _sp_r_data.get("location", "N/A")
                    if st.button(
                        _sp_r,
                        key=f"popover_sp_{_sp_r}",
                        width="stretch",
                        help=f"{_sp_r_size:,.0f} kW | {_sp_r_loc}",
                    ):
                        st.session_state["pending_system_profile"] = _sp_r
                        st.rerun()
                st.divider()
            else:
                st.caption("No saved system profiles yet.")

            if st.button("View All System Profiles", width="stretch", type="primary"):
                st.session_state["active_mgmt_tab"] = "System Profiles"

    # --- Load Profiles popover ---
    with _mgmt_btn_cols[3]:
        with st.popover("Load Profiles", width="stretch"):
            _lp_all = _list_all_load_profiles()
            _lp_recent_3 = _lp_all[:3]

            if _lp_recent_3:
                st.markdown("**Recent**")
                for _lp_r_name, _lp_r_type in _lp_recent_3:
                    try:
                        if _lp_r_type == "csv":
                            _lp_df = _load_profile_csv(LOAD_PROFILES_DIR, _lp_r_name)
                            _lp_vals = _parse_8760_csv(_lp_df)
                            _lp_help = f"{_lp_vals.sum():,.0f} kWh/yr"
                        else:
                            with open(os.path.join(NEMA_PROFILES_DIR, f"{_lp_r_name}.json")) as _lp_f:
                                _lp_nd = json.load(_lp_f)
                            _lp_total = sum(sum(m.get("load_8760", [])) for m in _lp_nd.get("meters", []))
                            _lp_help = f"NEM-A · {_lp_total:,.0f} kWh/yr"
                    except Exception as e:
                        _lp_help = f"(load profile help unavailable: {e})"
                    if st.button(
                        _lp_r_name,
                        key=f"popover_lp_{_lp_r_name}",
                        width="stretch",
                        help=_lp_help,
                    ):
                        st.session_state["active_mgmt_tab"] = "Load Profiles"
                        st.session_state["lp_sel"] = _lp_r_name
                st.divider()
            else:
                st.caption("No saved load profiles yet.")

            if st.button("View All Load Profiles", width="stretch", type="primary"):
                st.session_state["active_mgmt_tab"] = "Load Profiles"

    # --- Export Profiles popover ---
    with _mgmt_btn_cols[4]:
        with st.popover("Export Profiles", width="stretch"):
            _ep_names = _list_saved(EXPORT_PROFILES_DIR, ".csv")
            _ep_recent_3 = _ep_names[:3]

            if _ep_recent_3:
                st.markdown("**Recent**")
                for _ep_r in _ep_recent_3:
                    try:
                        _ep_df = _load_profile_csv(EXPORT_PROFILES_DIR, _ep_r)
                        _ep_vals = _parse_8760_csv(_ep_df)
                        _ep_avg = _ep_vals.mean()
                        _ep_help = f"Avg ${_ep_avg:.4f}/kWh"
                    except Exception as e:
                        _ep_help = f"(export profile help unavailable: {e})"
                    if st.button(
                        _ep_r,
                        key=f"popover_ep_{_ep_r}",
                        width="stretch",
                        help=_ep_help,
                    ):
                        try:
                            _ep_load_df = _load_profile_csv(EXPORT_PROFILES_DIR, _ep_r)
                            _cod_yr = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
                            _ep_multiyear = parse_multiyear_export_rates(_ep_load_df, start_year=_cod_yr)
                            _ep_first_key = min(_ep_multiyear.keys())
                            st.session_state["export_rates"] = _ep_multiyear[_ep_first_key]
                            st.session_state["export_rates_multiyear"] = _ep_multiyear if len(_ep_multiyear) > 1 else None
                        except Exception as e:
                            st.warning(f"Failed to load export profile: {e}")
                st.divider()
            else:
                st.caption("No saved export profiles yet.")

            if st.button("View All Export Profiles", width="stretch", type="primary"):
                st.session_state["active_mgmt_tab"] = "Export Profiles"

    # --- Custom Rates popover ---
    with _mgmt_btn_cols[5]:
        with st.popover("Custom Rates", width="stretch"):
            _cr_saved = _list_saved(ECC_TARIFFS_DIR, ".json")
            _cr_recent_3 = _cr_saved[:3]

            if _cr_recent_3:
                st.markdown("**Recent Custom Rates**")
                for _cr_r in _cr_recent_3:
                    if st.button(
                        _cr_r,
                        key=f"popover_cr_{_cr_r}",
                        width="stretch",
                    ):
                        # Load the tariff JSON to preview on the Custom Rates tab
                        try:
                            _cr_r_path = os.path.join(ECC_TARIFFS_DIR, _cr_r + ".json")
                            with open(_cr_r_path, "r") as _cr_f:
                                _cr_r_data = json.load(_cr_f)
                            # Unwrap array to single dict for preview
                            if isinstance(_cr_r_data, list) and _cr_r_data:
                                _cr_r_data = _cr_r_data[0]
                            st.session_state["custom_rate_extracted"] = _cr_r_data
                            st.session_state["custom_rate_warnings"] = None
                            st.session_state["active_mgmt_tab"] = "Custom Rates"
                        except Exception:
                            st.session_state["active_mgmt_tab"] = "Custom Rates"
                st.divider()
            else:
                st.caption("No custom rates yet.")

            if st.button("Create Custom Rate", width="stretch", type="primary"):
                st.session_state["active_mgmt_tab"] = "Custom Rates"

    # --- Save popover ---
    save_btn = False
    sim_name = ""
    with _mgmt_btn_cols[6]:
        with st.popover("Save", width="stretch"):
            sim_name = st.text_input(
                "Simulation Name",
                placeholder="e.g., Ranch-500kW-AG1-SAT",
                key="sim_name_input",
            )
            save_btn = st.button(
                "Save Current Simulation",
                disabled=(not sim_name),
                width="stretch",
            )

    # --- Proposals popover ---
    # Mirrors the Save flow from the opposite direction: every saved Proposal
    # for the active simulation appears here, with a "+ New Proposal" that
    # primes the Proposals-tab builder. Streamlit doesn't expose a way to
    # programmatically switch tabs, so after clicking a button here we
    # surface a st.toast confirming the state change and nudging the user
    # to open the "PPA & Proposals → Proposals" tab.
    with _mgmt_btn_cols[7]:
        with st.popover("Proposals", width="stretch"):
            _top_sim_name = (
                st.session_state.get("_active_simulation_name")
                or st.session_state.get("_last_loaded_simulation_name")
            )
            _top_props = _list_proposals_session(
                st.session_state, simulation_name=_top_sim_name,
            )
            if _top_props:
                st.markdown(
                    f"**{len(_top_props)} Proposal{'s' if len(_top_props)!=1 else ''}"
                    + (f" · {_top_sim_name}" if _top_sim_name else "")
                    + "**"
                )
                for _p in _top_props[:4]:
                    _comp_n = len(_p.comparison_ppas)
                    _lbl = (
                        f"{_p.customer_name or 'Customer'}"
                        if _p.customer_name else _p.name
                    )
                    _help = (
                        f"{_p.name} · Primary: {_p.primary_ppa.name}"
                        + (f" + {_comp_n} alt{'s' if _comp_n != 1 else ''}"
                           if _comp_n else "")
                        + f" · Updated {_p.updated_at[:10]}"
                    )
                    if st.button(
                        _lbl,
                        key=f"popover_prop_{_p.id}",
                        width="stretch",
                        help=_help,
                    ):
                        st.session_state["active_proposal_id"] = _p.id
                        # Deep-link: auto-focus the Proposals sub-tab (where
                        # all editing + export lives after Tranche 1
                        # consolidation) so the user lands directly on the
                        # preview + export surface for the Proposal they
                        # just picked.
                        st.session_state["_focus_proposals_tab"] = True
                        st.session_state["_proposal_toast_pending"] = (
                            f"Activated: {_p.name} — loaded Proposals",
                            "📁",
                        )
                        st.rerun()
                if len(_top_props) > 4:
                    st.caption(f"+ {len(_top_props) - 4} more")
                st.divider()
            else:
                st.caption(
                    "No Proposals saved for this simulation yet. Open the "
                    "**PPA Rate** tab to build and save a PPA, then the "
                    "**Proposals** tab to bundle it into a Proposal."
                )

            _new_btn = st.button(
                "➕ New Proposal", width="stretch", type="primary",
                key="popover_new_proposal",
            )
            if _new_btn:
                st.session_state["_proposals_tab_new"] = True
                st.session_state["active_proposal_id"] = None
                st.session_state["_proposal_toast_pending"] = (
                    "New Proposal started — open **PPA & Proposals → Proposals** to continue",
                    "➕",
                )
                st.rerun()

    # Fire any pending toast from the popover buttons. Streamlit's
    # st.toast call has to run at the top level of the script (not inside
    # a popover context) to render reliably, so we stage it in
    # session_state and flush it here on the next rerun.
    _pending_toast = st.session_state.pop("_proposal_toast_pending", None)
    if _pending_toast:
        _msg, _icon = _pending_toast
        st.toast(_msg, icon=_icon)


    # ---- LOAD PROFILES SECTION ----
    if st.session_state["active_mgmt_tab"] == "Load Profiles":
        with st.expander("Load Profiles", expanded=True):
            # ================================================================
            # Per-NEM-period tariff (mirrors the sidebar Section-2 selector).
            # Surfaced here so the post-transition tariff can be set alongside
            # the load profile. Only relevant when the NEM switch is enabled.
            # Writes into the same regime_2_tariff / regime_2_ecc_calculator
            # session keys via the shared pending-load handlers.
            # ================================================================
            if (st.session_state.get("nem_switch")
                    and st.session_state.get("rate_switch_enabled")
                    and st.session_state.get("rate_switch_at_nem")):
                _lp_engine = st.session_state.get("billing_engine", "Custom")
                st.markdown("**Post-NEM switch rate**")
                st.caption(
                    "After the NEM regime switch, the projection re-bills on this tariff. "
                    "Leave on 'Same as regime 1' to reuse the regime-1 tariff. "
                    "Mirrors the sidebar Rate Switch → At NEM switch loader."
                )
                _lp_r2_mode = st.radio(
                    "Post-NEM switch tariff",
                    ["Same as regime 1", "Load a different tariff"],
                    key="lp_regime2_tariff_mode",
                    horizontal=True,
                )
                if _lp_r2_mode == "Same as regime 1":
                    if _lp_engine == "Custom":
                        st.session_state["regime_2_tariff"] = None
                    else:
                        st.session_state["regime_2_ecc_calculator"] = None
                else:
                    if _lp_engine == "Custom":
                        _render_custom_tariff_loader(
                            select_label="Select Post-NEM switch Rate Schedule",
                            button_label="Load Post-NEM switch Tariff",
                            sel_key="lp_regime2_tariff_sel",
                            button_key="lp_regime2_tariff_load_btn",
                            pending_key="_pending_regime2_tariff_load",
                            target_key="regime_2_tariff",
                            loaded_prefix="Post-NEM switch tariff loaded",
                            empty_warning="No post-NEM switch tariff loaded — will reuse the regime-1 tariff.",
                            no_rates_caption="Fetch rates in the sidebar (Section 4) first to select a tariff.",
                        )
                    else:
                        _lp_r2_saved = _list_saved(ECC_TARIFFS_DIR, ".json")
                        if _lp_r2_saved:
                            _lp_r2_ecc_sel = st.selectbox(
                                "Select Saved Post-NEM switch Tariff", _lp_r2_saved,
                                key="lp_regime2_ecc_saved_sel",
                            )
                            if st.button("Load Post-NEM switch Tariff", key="lp_regime2_ecc_load_btn") and _lp_r2_ecc_sel:
                                st.session_state["_pending_regime2_ecc_saved_path"] = os.path.join(
                                    ECC_TARIFFS_DIR, _lp_r2_ecc_sel + ".json"
                                )
                        else:
                            st.caption("No saved ECC tariffs — add one via the sidebar or Custom Rates.")
                        if st.session_state.get("regime_2_ecc_calculator") is not None:
                            st.success("Post-NEM switch ECC tariff loaded.")
                st.markdown("---")
            else:
                # Rate Switch → "At NEM switch" is NOT armed (master off, trigger
                # off, or NEM Switch off). The post-transition re-bill consumer
                # keys off regime_2_tariff being set, so a tariff loaded here
                # earlier must not linger — clear it, mirroring the sidebar's
                # clear so the two writers can't diverge and silently mis-price.
                st.session_state["regime_2_tariff"] = None
                st.session_state["regime_2_ecc_calculator"] = None

            # ================================================================
            # A. Saved Load Profiles — unified dropdown (CSV + NEM-A)
            # ================================================================
            _all_profiles = _list_all_load_profiles()

            _sel_name = None
            _sel_type = None

            if _all_profiles:
                st.markdown("**Saved Load Profiles**")
                _profile_names = [p[0] for p in _all_profiles]
                _sel_name = st.selectbox("Select profile", _profile_names, key="lp_sel", index=None, placeholder="Choose a profile to edit...")

                if _sel_name:
                    _sel_idx = _profile_names.index(_sel_name)
                    _sel_type = _all_profiles[_sel_idx][1]

                    # Show profile details
                    try:
                        if _sel_type == "csv":
                            _det_df = _load_profile_csv(LOAD_PROFILES_DIR, _sel_name)
                            _det_vals = _parse_8760_csv(_det_df)
                            st.caption(
                                f"**{_sel_name}** — Single Meter CSV · "
                                f"{_det_vals.sum():,.0f} kWh/yr · Peak: {_det_vals.max():,.1f} kW"
                            )
                        else:
                            with open(os.path.join(NEMA_PROFILES_DIR, f"{_sel_name}.json")) as _det_f:
                                _det_nd = json.load(_det_f)
                            _det_meters = _det_nd.get("meters", [])
                            _det_total = sum(sum(m.get("load_8760", [])) for m in _det_meters)
                            _det_gen = next((m["name"] for m in _det_meters if m.get("is_generating")), "—")
                            st.caption(
                                f"**{_sel_name}** — NEM-A · {_det_nd.get('utility', '')} · "
                                f"{len(_det_meters)} meters · {_det_total:,.0f} kWh/yr · Gen: {_det_gen}"
                            )
                    except Exception as e:
                        st.warning(f"Could not load profile details: {e}")

                    _btn_col1, _btn_col2 = st.columns(2)
                    with _btn_col1:
                        _lp_load_btn = st.button("Load into Session", key="lp_load_session", type="primary")
                    with _btn_col2:
                        _lp_del_btn = st.button("Delete", key="lp_del")

                    if _lp_load_btn:
                        try:
                            _cod_yr = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
                            if _sel_type == "csv":
                                _al_df = _load_profile_csv(LOAD_PROFILES_DIR, _sel_name)
                                _al_vals = _parse_8760_csv(_al_df)
                                _al_dt = pd.date_range(f"{_cod_yr}-01-01", periods=8760, freq="h")
                                st.session_state["load_8760"] = pd.Series(_al_vals, index=_al_dt, name="load_kwh")
                                st.session_state["_raw_load_8760"] = st.session_state["load_8760"].copy()
                                st.session_state["load_mode"] = "Single Meter"
                                st.session_state["load_mode_radio"] = "Single Meter"
                                st.success(f"Loaded '{_sel_name}': {_al_vals.sum():,.0f} kWh/yr")
                            else:
                                _load_nema_profile_into_session(_sel_name)
                                st.success(f"Loaded NEM-A '{_sel_name}' ({len(st.session_state.get('nema_meters', []))} meters)")
                        except Exception as e:
                            st.error(f"Error loading profile: {e}")

                    if _lp_del_btn:
                        if _sel_type == "csv":
                            _delete_file(LOAD_PROFILES_DIR, _sel_name, ".csv")
                        else:
                            _delete_file(NEMA_PROFILES_DIR, _sel_name, ".json")
                        st.success(f"Deleted '{_sel_name}'.")
            else:
                st.caption("No saved load profiles yet.")

            # ================================================================
            # B. Create New Profile (CSV upload or NEM-A builder)
            # ================================================================
            st.markdown("---")
            _new_profile_type = st.radio(
                "New Profile Type",
                ["Single Meter CSV", "NEM-A Multi-Meter"],
                horizontal=True,
                key="mgmt_new_profile_type",
            )

            if _new_profile_type == "Single Meter CSV":
                # --- CSV upload & save ---
                lp_name = st.text_input("Profile Name", placeholder="e.g., Dairy-Farm-2024", key="lp_name")
                lp_file = st.file_uploader("Upload 8760 Load CSV", type=["csv"], key="lp_upload")
                lp_save_btn = st.button("Save Load Profile", disabled=(not lp_name or lp_file is None))

                if lp_save_btn and lp_file is not None and lp_name:
                    try:
                        df_up = pd.read_csv(lp_file)
                        _parse_8760_csv(df_up)  # validate
                        _save_profile_csv(LOAD_PROFILES_DIR, lp_name, df_up)
                        st.success(f"Load profile '{lp_name}' saved!")
                    except Exception as e:
                        st.error(str(e))

            else:
                # --- NEM-A multi-meter profile builder ---
                _nema_profile_name = st.text_input(
                    "NEM-A Profile Name",
                    placeholder="e.g., Dairy-Farm-2024",
                    key="mgmt_nema_profile_name",
                )

                # --- Utility selector ---
                _mgmt_nema_utility = st.selectbox(
                    "NEM-A Utility (for fees)",
                    list(NEMA_FEES.keys()),
                    key="mgmt_nema_utility_sel",
                    index=list(NEMA_FEES.keys()).index(st.session_state.get("nema_utility", "PG&E")),
                )
                st.session_state["nema_utility"] = _mgmt_nema_utility

                _mgmt_fee_info = NEMA_FEES[_mgmt_nema_utility]
                st.caption(
                    f"Admin fees: ${_mgmt_fee_info['setup_per_meter']:.0f}/meter setup"
                    + (f" (cap ${_mgmt_fee_info['setup_cap']:.0f})" if _mgmt_fee_info['setup_cap'] else "")
                    + f", ${_mgmt_fee_info['monthly_per_meter']:.2f}/meter/month"
                )

                # --- Fetch Available Rates (shared across meters) ---
                _mgmt_fetch_col1, _mgmt_fetch_col2 = st.columns([1, 2])
                with _mgmt_fetch_col1:
                    if st.button(
                        "Fetch Available Rates", key="api_call_btn_mgmt_nema_fetch",
                    ):
                        st.session_state["_pending_mgmt_fetch_rates"] = _mgmt_nema_utility
                if st.session_state.get("available_rates"):
                    st.caption(f"{len(st.session_state['available_rates'])} rate schedules available for per-meter tariff selection.")

                # --- Initialize meter list if needed ---
                if "nema_meters" not in st.session_state or not st.session_state["nema_meters"]:
                    st.session_state["nema_meters"] = [
                        {"name": "Generating Meter", "is_generating": True, "load_key": "nema_load_0", "tariff_key": "nema_tariff_0"},
                        {"name": "Meter 2", "is_generating": False, "load_key": "nema_load_1", "tariff_key": "nema_tariff_1"},
                    ]

                # --- Add meter ---
                if st.button("+ Add Meter", key="mgmt_nema_add_meter"):
                    _mgmt_idx = len(st.session_state["nema_meters"])
                    st.session_state["nema_meters"].append({
                        "name": f"Meter {_mgmt_idx + 1}",
                        "is_generating": False,
                        "load_key": f"nema_load_{_mgmt_idx}",
                        "tariff_key": f"nema_tariff_{_mgmt_idx}",
                    })
                    st.rerun()

                st.markdown("---")
                st.caption("Upload an 8760 load CSV for each meter, then save the entire configuration as one profile.")

                # --- Per-meter expanders (config + upload only) ---
                _nema_staged_uploads: dict[int, pd.DataFrame] = {}
                for _lp_i, _lp_meter in enumerate(st.session_state["nema_meters"]):
                    with st.expander(f"{'*' if _lp_meter.get('is_generating') else ''} {_lp_meter['name']}", expanded=False):
                        _lp_meter["name"] = st.text_input(
                            "Meter Name", value=_lp_meter["name"], key=f"mgmt_nema_name_{_lp_i}",
                        )
                        _lp_meter["is_generating"] = st.checkbox(
                            "Generating meter (PV/ESS)",
                            value=_lp_meter["is_generating"],
                            key=f"mgmt_nema_gen_{_lp_i}",
                        )
                        _lp_m_file = st.file_uploader(
                            "Upload 8760 Load CSV", type=["csv"], key=f"mgmt_lp_upload_{_lp_i}",
                        )
                        if _lp_m_file is not None:
                            try:
                                _lp_m_df = pd.read_csv(_lp_m_file)
                                _lp_m_vals = _parse_8760_csv(_lp_m_df)
                                _nema_staged_uploads[_lp_i] = _lp_m_df
                                st.success(f"{len(_lp_m_vals):,} rows loaded ({_lp_m_vals.sum():,.0f} kWh)")
                            except Exception as e:
                                st.error(str(e))
                        else:
                            _existing_load = st.session_state.get("nema_meter_loads", {}).get(_lp_i)
                            if _existing_load is not None:
                                st.caption(f"Loaded: {len(_existing_load):,} rows ({_existing_load.sum():,.0f} kWh)")
                            elif _lp_meter.get("is_generating") and st.session_state.get("load_8760") is not None:
                                st.caption(f"Using generating meter load from sidebar ({st.session_state["load_8760"].sum():,.0f} kWh)")

                        # --- Per-meter tariff selection (non-generating meters) ---
                        if not _lp_meter.get("is_generating"):
                            st.markdown("**Tariff**")
                            _lp_use_gen = st.checkbox(
                                "Use generating meter's tariff",
                                value=_lp_meter.get("use_gen_tariff", True),
                                key=f"mgmt_nema_use_gen_tariff_{_lp_i}",
                            )
                            _lp_meter["use_gen_tariff"] = _lp_use_gen
                            if not _lp_use_gen and st.session_state.get("available_rates"):
                                _mgmt_rate_opts = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                                _mgmt_sel_rate = st.selectbox(
                                    "Select Rate Schedule", list(_mgmt_rate_opts.keys()),
                                    key=f"mgmt_nema_tariff_sel_{_lp_i}",
                                )
                                if st.button("Load Tariff", key=f"mgmt_nema_tariff_load_{_lp_i}", type="primary"):
                                    st.session_state[f"_pending_mgmt_nema_tariff_{_lp_i}"] = _mgmt_rate_opts[_mgmt_sel_rate]
                            _lp_cur_tariff = st.session_state.get("nema_meter_tariffs", {}).get(_lp_i)
                            if _lp_cur_tariff is not None:
                                st.success(f"Tariff loaded: {_lp_cur_tariff.name}")
                            elif not _lp_use_gen:
                                st.warning("No tariff loaded for this meter.")

                        # Remove meter
                        if len(st.session_state["nema_meters"]) > 2 and not _lp_meter["is_generating"]:
                            if st.button("Remove this meter", key=f"mgmt_nema_remove_{_lp_i}"):
                                st.session_state["nema_meters"].pop(_lp_i)
                                _old_tariffs = st.session_state.get("nema_meter_tariffs", {})
                                _new_tariffs = {}
                                for _tk, _tv in _old_tariffs.items():
                                    if _tk < _lp_i:
                                        _new_tariffs[_tk] = _tv
                                    elif _tk > _lp_i:
                                        _new_tariffs[_tk - 1] = _tv
                                st.session_state["nema_meter_tariffs"] = _new_tariffs
                                st.rerun()

                # --- Save NEM-A profile ---
                st.markdown("---")
                _nema_save_col1, _nema_save_col2 = st.columns([1, 1])
                with _nema_save_col1:
                    _nema_save_btn = st.button(
                        "Save NEM-A Profile",
                        type="primary",
                        disabled=(not _nema_profile_name),
                        key="mgmt_nema_save_profile",
                    )
                if _nema_save_btn and _nema_profile_name:
                    _nema_bundle_meters = []
                    _nema_save_ok = True
                    _nema_existing_loads = st.session_state.get("nema_meter_loads", {})
                    for _si, _sm in enumerate(st.session_state["nema_meters"]):
                        if _si in _nema_staged_uploads:
                            _s_vals = _parse_8760_csv(_nema_staged_uploads[_si])
                            _s_load_list = _s_vals.tolist()
                        elif _si in _nema_existing_loads:
                            _s_load_list = _nema_existing_loads[_si].tolist()
                        elif _sm.get("is_generating") and st.session_state.get("load_8760") is not None:
                            _s_load_list = st.session_state["load_8760"].tolist()
                        else:
                            _s_load_list = None

                        if _s_load_list is None:
                            st.error(f"No load data for meter '{_sm['name']}'. Upload a CSV or load a profile in the sidebar first.")
                            _nema_save_ok = False
                            break

                        _nema_save_tariffs = st.session_state.get("nema_meter_tariffs", {})
                        _nema_bundle_meters.append({
                            "name": _sm["name"],
                            "is_generating": _sm.get("is_generating", False),
                            "use_gen_tariff": _sm.get("use_gen_tariff", not _sm.get("is_generating", False)),
                            "load_8760": _s_load_list,
                            "tariff": asdict(_nema_save_tariffs[_si]) if _si in _nema_save_tariffs else None,
                        })

                    if _nema_save_ok:
                        import json as _json
                        _nema_bundle = {
                            "utility": _mgmt_nema_utility,
                            "meters": _nema_bundle_meters,
                            "existing_solar_meters": st.session_state.get("existing_solar_nema_meters", []),
                        }
                        _nema_safe_name = sanitize_filename(_nema_profile_name)
                        _nema_json_bytes = _json.dumps(_nema_bundle).encode("utf-8")
                        save_profile_bytes(NEMA_PROFILES_DIR, _DIR_TO_GCS_PREFIX[NEMA_PROFILES_DIR], _nema_safe_name, _nema_json_bytes, ".json")
                        st.success(f"NEM-A profile '{_nema_profile_name}' saved with {len(_nema_bundle_meters)} meters.")

            # ================================================================
            # C. Editing selected profile
            # ================================================================
            if _sel_name and _sel_type:
                st.markdown("---")
                # --- Auto-populate: CSV viewer/editor ---
                if _sel_type == "csv":
                    st.subheader(f"Editing: {_sel_name}")
                    edit_df = _load_profile_csv(LOAD_PROFILES_DIR, _sel_name)
                    try:
                        vals = _parse_8760_csv(edit_df)
                        st.write(f"**Rows:** {len(vals):,} | **Annual:** {vals.sum():,.0f} kWh | **Peak:** {vals.max():,.1f} kW")

                        # Chart selector
                        _preview_year = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
                        dt_idx = pd.date_range(f"{_preview_year}-01-01", periods=8760, freq="h")
                        _lp_chart_type = st.radio(
                            "Display",
                            ["Monthly Load", "Average Daily Profile", "Load Duration Curve"],
                            horizontal=True,
                            key="lp_chart_type",
                        )
                        import plotly.graph_objects as go
                        _chart_layout = dict(
                            height=380,
                            template="plotly_white",
                            font=dict(family="Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif", size=12),
                            title_font=dict(size=15, color="#0E2841"),
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                        if _lp_chart_type == "Monthly Load":
                            monthly_kwh = pd.Series(vals, index=dt_idx).resample("ME").sum()
                            fig = go.Figure(go.Bar(x=MONTH_NAMES, y=monthly_kwh.values, marker_color="#1D6FA9"))
                            fig.update_layout(title="Monthly Load (kWh)", yaxis_title="kWh", **_chart_layout)
                            st.plotly_chart(fig, use_container_width=True)
                        elif _lp_chart_type == "Average Daily Profile":
                            _lp_series = pd.Series(vals, index=dt_idx)
                            _lp_avg_hourly = _lp_series.groupby(_lp_series.index.hour).mean()
                            fig = go.Figure(go.Scatter(
                                x=list(range(24)), y=_lp_avg_hourly.values,
                                mode="lines+markers", line=dict(color="#1D6FA9", width=2.5),
                                marker=dict(size=5), fill="tozeroy", fillcolor="rgba(29,111,169,0.12)",
                            ))
                            fig.update_layout(
                                title="Average Daily Load Profile",
                                xaxis_title="Hour of Day", yaxis_title="Avg kW",
                                xaxis=dict(dtick=1, range=[-0.5, 23.5]),
                                **_chart_layout,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:  # Load Duration Curve
                            _lp_sorted = np.sort(vals)[::-1]
                            fig = go.Figure(go.Scatter(
                                x=list(range(1, 8761)), y=_lp_sorted,
                                mode="lines", line=dict(color="#1D6FA9", width=2),
                                fill="tozeroy", fillcolor="rgba(29,111,169,0.12)",
                            ))
                            fig.update_layout(
                                title="Load Duration Curve",
                                xaxis_title="Hours", yaxis_title="kW",
                                **_chart_layout,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(str(e))

                    st.caption("Edit the data below and click Save to update.")
                    edited_df = st.data_editor(edit_df, num_rows="fixed", width="stretch", height=400, key="lp_editor")

                    lp_save_edit = st.button("Save Changes", key="lp_save_edit")
                    if lp_save_edit:
                        try:
                            _parse_8760_csv(edited_df)  # validate
                            _save_profile_csv(LOAD_PROFILES_DIR, _sel_name, edited_df)
                            st.success(f"'{_sel_name}' updated!")
                        except Exception as e:
                            st.error(str(e))

                # --- Auto-populate: NEM-A inline editor ---
                if _sel_type == "nema":
                    import json as _json
                    _nema_edit_path = os.path.join(NEMA_PROFILES_DIR, f"{_sel_name}.json")
                    if os.path.exists(_nema_edit_path):
                        with open(_nema_edit_path, "r") as _f:
                            _ne_data = _json.load(_f)

                        st.subheader(f"Editing: {_sel_name}")

                        # Load into Session button
                        if st.button("Load into Session", key="mgmt_nema_load_profile", type="primary"):
                            _load_nema_profile_into_session(_sel_name)
                            st.success(f"Loaded NEM-A profile '{_sel_name}' ({len(st.session_state.get('nema_meters', []))} meters).")

                        # --- Utility selector ---
                        _ne_utility = st.selectbox(
                            "Utility",
                            list(NEMA_FEES.keys()),
                            key="edit_nema_utility",
                            index=list(NEMA_FEES.keys()).index(_ne_data.get("utility", "PG&E")),
                        )

                        _ne_fee_info = NEMA_FEES[_ne_utility]
                        st.caption(
                            f"Admin fees: ${_ne_fee_info['setup_per_meter']:.0f}/meter setup"
                            + (f" (cap ${_ne_fee_info['setup_cap']:.0f})" if _ne_fee_info['setup_cap'] else "")
                            + f", ${_ne_fee_info['monthly_per_meter']:.2f}/meter/month"
                        )

                        # Fetch rates for tariff selection
                        if st.button(
                            "Fetch Available Rates", key="api_call_btn_edit_nema_fetch",
                        ):
                            st.session_state["_pending_edit_nema_fetch_rates"] = _ne_utility
                        if st.session_state.get("available_rates"):
                            st.caption(f"{len(st.session_state['available_rates'])} rate schedules available.")

                        # Initialize edit-state meters from JSON (only on first load of this profile)
                        if st.session_state.get("_ne_edit_profile") != _sel_name:
                            st.session_state["_ne_edit_profile"] = _sel_name
                            _ne_edit_meters = []
                            _ne_edit_loads: dict[int, list] = {}
                            _ne_edit_tariffs: dict[int, dict | None] = {}
                            for _ei, _em in enumerate(_ne_data.get("meters", [])):
                                _ne_edit_meters.append({
                                    "name": _em["name"],
                                    "is_generating": _em.get("is_generating", False),
                                    "use_gen_tariff": _em.get("use_gen_tariff", not _em.get("is_generating", False)),
                                })
                                if _em.get("load_8760"):
                                    _ne_edit_loads[_ei] = _em["load_8760"]
                                _ne_edit_tariffs[_ei] = _em.get("tariff")
                            st.session_state["_ne_edit_meters"] = _ne_edit_meters
                            st.session_state["_ne_edit_loads"] = _ne_edit_loads
                            st.session_state["_ne_edit_tariffs"] = _ne_edit_tariffs

                        _ne_edit_meters = st.session_state.get("_ne_edit_meters", [])
                        _ne_edit_loads = st.session_state.get("_ne_edit_loads", {})
                        _ne_edit_tariffs = st.session_state.get("_ne_edit_tariffs", {})

                        # Add meter
                        if st.button("+ Add Meter", key="edit_nema_add_meter"):
                            _ne_new_idx = len(_ne_edit_meters)
                            _ne_edit_meters.append({
                                "name": f"Meter {_ne_new_idx + 1}",
                                "is_generating": False,
                                "use_gen_tariff": True,
                            })
                            st.session_state["_ne_edit_meters"] = _ne_edit_meters
                            st.rerun()

                        st.markdown("---")

                        # Per-meter expanders
                        _ne_staged_uploads: dict[int, pd.DataFrame] = {}
                        for _ei, _em in enumerate(_ne_edit_meters):
                            with st.expander(f"{'*' if _em.get('is_generating') else ''} {_em['name']}", expanded=False):
                                _em["name"] = st.text_input(
                                    "Meter Name", value=_em["name"], key=f"edit_nema_name_{_ei}",
                                )
                                _em["is_generating"] = st.checkbox(
                                    "Generating meter (PV/ESS)",
                                    value=_em.get("is_generating", False),
                                    key=f"edit_nema_gen_{_ei}",
                                )

                                # Load upload
                                _ne_m_file = st.file_uploader(
                                    "Upload 8760 Load CSV", type=["csv"], key=f"edit_nema_upload_{_ei}",
                                )
                                if _ne_m_file is not None:
                                    try:
                                        _ne_m_df = pd.read_csv(_ne_m_file)
                                        _ne_m_vals = _parse_8760_csv(_ne_m_df)
                                        _ne_staged_uploads[_ei] = _ne_m_df
                                        st.success(f"{len(_ne_m_vals):,} rows loaded ({_ne_m_vals.sum():,.0f} kWh)")
                                    except Exception as e:
                                        st.error(str(e))
                                else:
                                    _ne_cur_load = _ne_edit_loads.get(_ei)
                                    if _ne_cur_load is not None:
                                        _ne_load_sum = sum(_ne_cur_load)
                                        st.caption(f"Loaded: {len(_ne_cur_load):,} rows ({_ne_load_sum:,.0f} kWh)")

                                # Tariff selection (non-generating)
                                if not _em.get("is_generating"):
                                    st.markdown("**Tariff**")
                                    _ne_use_gen = st.checkbox(
                                        "Use generating meter's tariff",
                                        value=_em.get("use_gen_tariff", True),
                                        key=f"edit_nema_use_gen_tariff_{_ei}",
                                    )
                                    _em["use_gen_tariff"] = _ne_use_gen
                                    if not _ne_use_gen and st.session_state.get("available_rates"):
                                        _ne_rate_opts = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                                        _ne_sel_rate = st.selectbox(
                                            "Select Rate Schedule", list(_ne_rate_opts.keys()),
                                            key=f"edit_nema_tariff_sel_{_ei}",
                                        )
                                        if st.button("Load Tariff", key=f"edit_nema_tariff_load_{_ei}", type="primary"):
                                            st.session_state[f"_pending_edit_nema_tariff_{_ei}"] = _ne_rate_opts[_ne_sel_rate]
                                    _ne_cur_tariff = _ne_edit_tariffs.get(_ei)
                                    if _ne_cur_tariff is not None and isinstance(_ne_cur_tariff, dict) and _ne_cur_tariff.get("name"):
                                        st.success(f"Tariff: {_ne_cur_tariff['name']}")
                                    elif not _ne_use_gen:
                                        st.warning("No tariff loaded for this meter.")
                                else:
                                    _em["use_gen_tariff"] = False

                                # Remove meter
                                if len(_ne_edit_meters) > 2 and not _em.get("is_generating"):
                                    if st.button("Remove this meter", key=f"edit_nema_remove_{_ei}"):
                                        _ne_edit_meters.pop(_ei)
                                        # Re-index loads and tariffs
                                        _new_loads = {}
                                        for _k, _v in _ne_edit_loads.items():
                                            if _k < _ei:
                                                _new_loads[_k] = _v
                                            elif _k > _ei:
                                                _new_loads[_k - 1] = _v
                                        _new_tariffs = {}
                                        for _k, _v in _ne_edit_tariffs.items():
                                            if _k < _ei:
                                                _new_tariffs[_k] = _v
                                            elif _k > _ei:
                                                _new_tariffs[_k - 1] = _v
                                        st.session_state["_ne_edit_meters"] = _ne_edit_meters
                                        st.session_state["_ne_edit_loads"] = _new_loads
                                        st.session_state["_ne_edit_tariffs"] = _new_tariffs
                                        st.rerun()

                        # Save Changes button
                        st.markdown("---")
                        if st.button("Save Changes", key="edit_nema_save", type="primary"):
                            _ne_bundle_meters = []
                            _ne_save_ok = True
                            for _si, _sm in enumerate(_ne_edit_meters):
                                if _si in _ne_staged_uploads:
                                    _s_vals = _parse_8760_csv(_ne_staged_uploads[_si])
                                    _s_load_list = _s_vals.tolist()
                                elif _si in _ne_edit_loads:
                                    _s_load_list = _ne_edit_loads[_si]
                                    if not isinstance(_s_load_list, list):
                                        _s_load_list = list(_s_load_list)
                                else:
                                    _s_load_list = None

                                if _s_load_list is None:
                                    st.error(f"No load data for meter '{_sm['name']}'. Upload a CSV.")
                                    _ne_save_ok = False
                                    break

                                _ne_s_tariff = _ne_edit_tariffs.get(_si)
                                _ne_bundle_meters.append({
                                    "name": _sm["name"],
                                    "is_generating": _sm.get("is_generating", False),
                                    "use_gen_tariff": _sm.get("use_gen_tariff", not _sm.get("is_generating", False)),
                                    "load_8760": _s_load_list,
                                    "tariff": _ne_s_tariff,
                                })

                            if _ne_save_ok:
                                _ne_bundle = {
                                    "utility": _ne_utility,
                                    "meters": _ne_bundle_meters,
                                    "existing_solar_meters": _ne_data.get("existing_solar_meters", []),
                                }
                                with open(_nema_edit_path, "w") as _f:
                                    _json.dump(_ne_bundle, _f)
                                # Update edit state loads with staged uploads
                                for _ui, _udf in _ne_staged_uploads.items():
                                    _ne_edit_loads[_ui] = _parse_8760_csv(_udf).tolist()
                                st.session_state["_ne_edit_loads"] = _ne_edit_loads
                                st.success(f"'{_sel_name}' updated with {len(_ne_bundle_meters)} meters.")



    # ---- EXPORT PROFILES SECTION ----
    if st.session_state["active_mgmt_tab"] == "Export Profiles":
        with st.expander("Export Profiles", expanded=True):
            saved_exports = _list_saved(EXPORT_PROFILES_DIR, ".csv")
            ep_col1, ep_col2 = st.columns([2, 1])

            with ep_col1:
                st.markdown("**Upload & Save an Export Rate Profile**")
                ep_name = st.text_input("Profile Name", placeholder="e.g., PGE-ACC-2024", key="ep_name")
                ep_file = st.file_uploader("Upload 8760 Export Rate CSV", type=["csv"], key="ep_upload")
                ep_save_btn = st.button("Save Export Profile", disabled=(not ep_name or ep_file is None))

                if ep_save_btn and ep_file is not None and ep_name:
                    try:
                        df_up = pd.read_csv(ep_file)
                        _parse_8760_csv(df_up)  # validate
                        _save_profile_csv(EXPORT_PROFILES_DIR, ep_name, df_up)
                        st.success(f"Export profile '{ep_name}' saved!")
                    except Exception as e:
                        st.error(str(e))

            with ep_col2:
                if saved_exports:
                    st.markdown("**Saved Export Profiles**")
                    sel_ep = st.selectbox("Select profile", saved_exports, key="ep_sel")
                    ep_view_btn = st.button("View / Edit", key="ep_view")
                    ep_del_btn = st.button("Delete", key="ep_del")

                    if ep_del_btn and sel_ep:
                        _delete_file(EXPORT_PROFILES_DIR, sel_ep, ".csv")
                        st.success(f"Deleted '{sel_ep}'.")
                else:
                    st.caption("No saved export profiles yet.")
                    sel_ep = None
                    ep_view_btn = False

            # View / Edit section
            if saved_exports and ep_view_btn and sel_ep:
                st.session_state["ep_editing"] = sel_ep
            if st.session_state.get("ep_editing"):
                edit_name = st.session_state["ep_editing"]
                st.subheader(f"Editing: {edit_name}")
                edit_df = _load_profile_csv(EXPORT_PROFILES_DIR, edit_name)
                try:
                    vals = _parse_8760_csv(edit_df)
                    st.write(f"**Rows:** {len(vals):,} | **Avg Rate:** ${vals.mean():.4f}/kWh | **Range:** ${vals.min():.4f} - ${vals.max():.4f}")

                    _preview_year = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
                    dt_idx = pd.date_range(f"{_preview_year}-01-01", periods=8760, freq="h")
                    monthly_avg = pd.Series(vals, index=dt_idx).resample("ME").mean()
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(x=MONTH_NAMES, y=monthly_avg.values, marker_color="#45A750"))
                    fig.update_layout(title="Monthly Avg Export Rate ($/kWh)", yaxis_title="$/kWh", height=300, template="plotly_white")
                    st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.warning(str(e))

                st.caption("Edit the data below and click Save to update.")
                edited_df = st.data_editor(edit_df, num_rows="fixed", width="stretch", height=400, key="ep_editor")

                ep_save_edit = st.button("Save Changes", key="ep_save_edit")
                if ep_save_edit:
                    try:
                        _parse_8760_csv(edited_df)  # validate
                        _save_profile_csv(EXPORT_PROFILES_DIR, edit_name, edited_df)
                        st.success(f"'{edit_name}' updated!")
                    except Exception as e:
                        st.error(str(e))

                if st.button("Close Editor", key="ep_close_edit"):
                    del st.session_state["ep_editing"]


    # ---- SYSTEM PROFILES SECTION ----
    if st.session_state["active_mgmt_tab"] == "System Profiles":
        with st.expander("System Profiles", expanded=True):
            saved_sp = _list_saved(SYSTEM_PROFILES_DIR, ".json")
            sp_col1, sp_col2 = st.columns([2, 1])

            _sp_editing_name = st.session_state.get("sp_editing")

            with sp_col1:
                if _sp_editing_name:
                    st.markdown(f"**Editing: {_sp_editing_name}**")
                    st.caption("Modify the sidebar values, then click Update to overwrite this profile.")
                    _sp_edit_bcols = st.columns(2)
                    with _sp_edit_bcols[0]:
                        if st.button("Update Profile", key="sp_update_btn", type="primary", width="stretch"):
                            try:
                                _save_system_profile(_sp_editing_name)
                                st.session_state.pop("sp_editing", None)
                                st.success(f"Profile '{_sp_editing_name}' updated!")
                            except Exception as e:
                                st.error(str(e))
                    with _sp_edit_bcols[1]:
                        if st.button("Cancel Edit", key="sp_cancel_edit", width="stretch"):
                            st.session_state.pop("sp_editing", None)
                            st.rerun()
                else:
                    st.markdown("**Save Current System Profile**")
                    sp_name = st.text_input(
                        "Profile Name",
                        placeholder="e.g., Ranch-500kW-SAT",
                        key="sp_name",
                    )
                    sp_save_btn = st.button("Save System Profile", disabled=(not sp_name))

                    if sp_save_btn and sp_name:
                        try:
                            _save_system_profile(sp_name)
                            st.success(f"System profile '{sp_name}' saved!")
                        except Exception as e:
                            st.error(str(e))

            with sp_col2:
                if saved_sp:
                    st.markdown("**Saved System Profiles**")
                    sel_sp = st.selectbox("Select profile", saved_sp, key="sp_sel")
                    _sp_action_cols = st.columns(4)
                    with _sp_action_cols[0]:
                        sp_view_btn = st.button("View", key="sp_view", width="stretch")
                    with _sp_action_cols[1]:
                        sp_edit_btn = st.button("Edit", key="sp_edit", width="stretch")
                    with _sp_action_cols[2]:
                        sp_dup_btn = st.button("Duplicate", key="sp_dup", width="stretch")
                    with _sp_action_cols[3]:
                        sp_del_btn = st.button("Delete", key="sp_del", width="stretch")

                    if sp_del_btn and sel_sp:
                        _delete_file(SYSTEM_PROFILES_DIR, sel_sp, ".json")
                        st.success(f"Deleted '{sel_sp}'.")

                    if sp_edit_btn and sel_sp:
                        st.session_state["sp_editing"] = sel_sp
                        st.session_state["pending_system_profile"] = sel_sp
                        st.session_state.pop("sp_viewing", None)
                        st.rerun()

                    if sp_dup_btn and sel_sp:
                        _dup_base = f"{sel_sp} - Copy"
                        _dup_name = _dup_base
                        _dup_i = 2
                        while os.path.exists(os.path.join(SYSTEM_PROFILES_DIR, f"{_dup_name}.json")):
                            _dup_name = f"{_dup_base} {_dup_i}"
                            _dup_i += 1
                        _dup_data = _load_system_profile(sel_sp)
                        with open(os.path.join(SYSTEM_PROFILES_DIR, f"{_dup_name}.json"), "w") as _df:
                            json.dump(_dup_data, _df)
                        st.success(f"Duplicated as '{_dup_name}'.")
                else:
                    st.caption("No saved system profiles yet.")
                    sel_sp = None
                    sp_view_btn = False

            # View section
            if saved_sp and sp_view_btn and sel_sp:
                st.session_state["sp_viewing"] = sel_sp
            if st.session_state.get("sp_viewing"):
                view_name = st.session_state["sp_viewing"]
                st.subheader(f"Profile: {view_name}")
                try:
                    _sp_view_data = _load_system_profile(view_name)
                    _vinfo = [
                        f"**Location:** {_sp_view_data.get('location', 'N/A')}",
                        f"**Lat/Lon:** {_sp_view_data.get('lat', 'N/A')}, {_sp_view_data.get('lon', 'N/A')}",
                        f"**System Size:** {_sp_view_data.get('system_size_kw', 0):,.1f} kW-DC",
                        f"**DC/AC Ratio:** {_sp_view_data.get('dc_ac_ratio', 0):.2f}",
                        f"**System Type:** {_sp_view_data.get('system_type', 'N/A')}",
                        f"**Module Type:** {_sp_view_data.get('module_type', 'N/A')}",
                        f"**System Losses:** {_sp_view_data.get('system_losses', 0):.2f}%",
                        f"**Degradation:** {_sp_view_data.get('degradation', 0):.2f}%/yr",
                        f"**System Life:** {_sp_view_data.get('system_life', 0)} years",
                        f"**COD:** {_sp_view_data.get('cod_date', 'N/A')}",
                    ]
                    if _sp_view_data.get("production_summary"):
                        _ps = _sp_view_data["production_summary"]
                        _vinfo.append(f"**Annual Production:** {_ps.get('ac_annual_kwh', 0):,.0f} kWh")
                        _vinfo.append(f"**Capacity Factor:** {_ps.get('capacity_factor', 0):.1f}%")
                    else:
                        _vinfo.append("**Production:** Not saved (will need to re-run PVWatts)")
                    st.markdown("  \n".join(_vinfo))
                except Exception as e:
                    st.error(str(e))

                if st.button("Close", key="sp_close_view"):
                    del st.session_state["sp_viewing"]


    # ---- CUSTOM RATES SECTION ----
    if st.session_state["active_mgmt_tab"] == "Custom Rates":
        with st.expander("Custom Rates", expanded=True):

            # ================================================================
            # A. Saved Custom Rates
            # ================================================================
            _cr_all = _list_saved(ECC_TARIFFS_DIR, ".json")
            if _cr_all:
                st.markdown("**Saved Custom Rates**")
                _cr_sel = st.selectbox(
                    "Select a saved rate", _cr_all, key="cr_sel", index=None,
                    placeholder="Choose a custom rate...",
                )
                if _cr_sel:
                    _cr_btn_c1, _cr_btn_c2 = st.columns(2)
                    with _cr_btn_c1:
                        _cr_load_btn = st.button(
                            "Load into Simulator", key="cr_load", type="primary",
                        )
                    with _cr_btn_c2:
                        _cr_del_btn = st.button("Delete", key="cr_del")

                    if _cr_load_btn:
                        try:
                            _cr_path = os.path.join(ECC_TARIFFS_DIR, _cr_sel + ".json")
                            _cr_calc, _cr_data = load_ecc_tariff_from_json(_cr_path)
                            st.session_state["ecc_cost_calculator"] = _cr_calc
                            st.session_state["ecc_tariff_data"] = _cr_data
                            st.session_state["ecc_tariff_metadata"] = {
                                "source": f"Custom: {_cr_sel}",
                                "num_tariffs": len(_cr_data) if isinstance(_cr_data, list) else 1,
                                "tariff_names": [t.get("name", _cr_sel) for t in (_cr_data if isinstance(_cr_data, list) else [_cr_data])],
                            }
                            st.session_state["billing_engine_radio"] = "ECC"
                            st.session_state["billing_engine"] = "ECC"
                            st.session_state["active_mgmt_tab"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load tariff: {e}")

                    if _cr_del_btn:
                        _delete_file(ECC_TARIFFS_DIR, _cr_sel, ".json")
                        st.success(f"Deleted '{_cr_sel}'.")
            else:
                st.caption("No saved custom rates yet.")

            # ================================================================
            # B. Create New Custom Rate
            # ================================================================
            st.markdown("---")
            st.markdown("**Create New Custom Rate**")

            _cr_up_c1, _cr_up_c2, _cr_up_c3 = st.columns([2, 1, 1])
            with _cr_up_c1:
                _cr_pdf = st.file_uploader("Upload Tariff PDF", type=["pdf"], key="cr_pdf_upload")
            with _cr_up_c2:
                _cr_utility = st.text_input("Utility", placeholder="e.g., PG&E", key="cr_utility")
            with _cr_up_c3:
                _cr_rate_name = st.text_input("Rate Name", placeholder="e.g., AG-C", key="cr_rate_name")

            _cr_extract_btn = st.button(
                "Extract Rate Data",
                disabled=(_cr_pdf is None),
                type="primary",
            )

            if _cr_extract_btn and _cr_pdf is not None:
                with st.spinner("Extracting text from PDF..."):
                    try:
                        _cr_text = extract_text_from_pdf(_cr_pdf)
                    except Exception as e:
                        st.error(f"PDF extraction failed: {e}")
                        _cr_text = None

                if _cr_text:
                    with st.spinner("Analyzing tariff with Claude AI..."):
                        try:
                            _cr_result = extract_tariff_from_text(
                                _cr_text, utility=_cr_utility, rate_name=_cr_rate_name,
                            )
                            st.session_state["custom_rate_extracted"] = _cr_result
                            st.session_state["custom_rate_warnings"] = validate_tariff_structure(_cr_result)
                            st.success("Rate data extracted successfully!")
                        except Exception as e:
                            st.error(f"Claude API extraction failed: {e}")

            # ---- Preview extracted data ----
            _cr_extracted = st.session_state.get("custom_rate_extracted")
            if _cr_extracted:
                st.markdown("---")
                st.markdown("**Extracted Data Preview**")

                _cr_warnings = st.session_state.get("custom_rate_warnings", [])
                if _cr_warnings:
                    for _cw in _cr_warnings:
                        st.warning(_cw)

                # Tariff name / description
                st.caption(f"**{_cr_extracted.get('name', 'Unnamed')}** — {_cr_extracted.get('utility', 'N/A')}")
                if _cr_extracted.get("description"):
                    st.caption(_cr_extracted["description"])

                # Energy rates table
                _cr_energy = _cr_extracted.get("energyratestructure", [])
                if _cr_energy:
                    with st.expander("Energy Rates ($/kWh)", expanded=True):
                        # Parse period labels from energycomments
                        import re as _re
                        _cr_period_labels: dict[int, str] = {}
                        _cr_comments = _cr_extracted.get("energycomments", "")
                        if _cr_comments:
                            for _m in _re.finditer(r"Period\s+(\d+)\s*:\s*([^.]+)", _cr_comments):
                                _cr_period_labels[int(_m.group(1))] = _m.group(2).strip()
                        _cr_erows = []
                        for i, period in enumerate(_cr_energy):
                            rate = period[0].get("rate", 0) if period else 0
                            label = _cr_period_labels.get(i, "—")
                            _cr_erows.append({"Period": i, "Type": label, "Rate ($/kWh)": f"${rate:.5f}"})
                        st.table(pd.DataFrame(_cr_erows))

                # TOU schedule heatmap
                _cr_wk_sched = _cr_extracted.get("energyweekdayschedule")
                if _cr_wk_sched:
                    with st.expander("TOU Schedule (Weekday)", expanded=False):
                        import plotly.graph_objects as go
                        _cr_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        _cr_hours = [f"{h}:00" for h in range(24)]
                        fig = go.Figure(data=go.Heatmap(
                            z=_cr_wk_sched,
                            x=_cr_hours,
                            y=_cr_months,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Period"),
                        ))
                        fig.update_layout(
                            height=350,
                            margin=dict(l=60, r=20, t=30, b=40),
                            xaxis_title="Hour",
                            yaxis_title="Month",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Demand charges
                _cr_demand = _cr_extracted.get("demandratestructure", [])
                if _cr_demand:
                    with st.expander("Demand Charges ($/kW)", expanded=True):
                        _cr_demand_labels: dict[int, str] = {}
                        _cr_dcomments = _cr_extracted.get("demandcomments", "")
                        if _cr_dcomments:
                            for _dm in _re.finditer(r"Period\s+(\d+)\s*:\s*([^.]+)", _cr_dcomments):
                                _cr_demand_labels[int(_dm.group(1))] = _dm.group(2).strip()
                        _cr_drows = []
                        for i, period in enumerate(_cr_demand):
                            rate = period[0].get("rate", 0) if period else 0
                            label = _cr_demand_labels.get(i, "—")
                            _cr_drows.append({"Period": i, "Type": label, "Rate ($/kW)": f"${rate:.2f}"})
                        st.table(pd.DataFrame(_cr_drows))

                # Flat demand
                _cr_flat = _cr_extracted.get("flatdemandstructure")
                if _cr_flat:
                    _cr_flat_rate = _cr_flat[0][0].get("rate", 0) if _cr_flat and _cr_flat[0] else 0
                    st.caption(f"**Flat Demand Charge:** ${_cr_flat_rate:.2f}/kW")

                # Fixed charges
                _cr_fixed = _cr_extracted.get("fixedchargefirstmeter")
                if _cr_fixed:
                    st.caption(f"**Fixed Charge:** ${_cr_fixed:.5f}/{_cr_extracted.get('fixedchargeunits', '$/day')}")

                # Energy comments
                if _cr_extracted.get("energycomments"):
                    with st.expander("AI Period Descriptions"):
                        st.write(_cr_extracted["energycomments"])

                # ---- Save ----
                st.markdown("---")
                _cr_save_c1, _cr_save_c2 = st.columns([2, 1])
                with _cr_save_c1:
                    _cr_save_name = st.text_input(
                        "Save Name",
                        value=_cr_extracted.get("label", ""),
                        placeholder="e.g., PGE_AG-C_2026",
                        key="cr_save_name",
                    )
                with _cr_save_c2:
                    _cr_save_btn = st.button(
                        "Save Custom Rate",
                        disabled=(not _cr_save_name),
                        type="primary",
                    )

                if _cr_save_btn and _cr_save_name:
                    try:
                        _cr_saved_path = save_custom_tariff(
                            _cr_save_name, _cr_extracted, ECC_TARIFFS_DIR,
                        )
                        st.success(f"Saved as '{os.path.basename(_cr_saved_path)}'")
                        st.session_state["custom_rate_extracted"] = None
                        st.session_state["custom_rate_warnings"] = None
                    except Exception as e:
                        st.error(f"Save failed: {e}")


    # ---------------------------------------------------------------------------
    # Storage Diagnostics (below management tabs, before main content)
    # ---------------------------------------------------------------------------
    with st.expander("Storage Diagnostics"):
        _diag = gcs_diagnostic()
        if _diag["connected"]:
            st.success(f"GCS connected — bucket: {_diag['bucket_name']}, {_diag['blob_count']} files")
        else:
            st.warning(f"GCS not available: {_diag['error']}. Using local storage only.")

    st.title("PV Solar Rate Simulator")
    st.markdown(
        '<p style="font-size: 12px; color: rgba(150,150,150,0.9); margin-top: -10px;">'
        'California Net Value Billing Tariff (NVBT) — Hourly Import/Export Analysis</p>',
        unsafe_allow_html=True,
    )


    return save_btn, sim_name


save_btn, sim_name = _render_top_bar()

# --- Getting Started guidance (only shown when no simulation has run yet) ---
if st.session_state["billing_result"] is None and st.session_state["saved_view"] is None:
    st.info(
        "**Getting Started:** Use the sidebar to configure your simulation inputs, "
        "working through each numbered section (1-8). Once all checklist items below "
        "are complete, click **Run Simulation** to generate results.",
        icon="👋",
    )

st.divider()

def _render_sidebar():
    """Render the sidebar configuration panel and return computed context values."""
    # =============================================================================
    # SIDEBAR — INPUTS
    # =============================================================================
    with st.sidebar:
        # ── STICKY INPUT-LOAD TRACKER ───────────────────────────────────
        # Four oval pills showing which core inputs have been loaded. The
        # pills turn green as each input populates and the whole tracker
        # stays affixed to the top of the sidebar while the user scrolls
        # further configuration below.
        _sb_prod = st.session_state.get("production_8760") is not None
        _sb_load = st.session_state.get("load_8760") is not None
        _sb_tariff = (
            st.session_state.get("tariff") is not None
            or st.session_state.get("ecc_cost_calculator") is not None
        )
        # Export rates are only required under NEM-3 / NVBT single-meter.
        # NEM-1 / NEM-2 / NEM-A value exports at retail TOU so the check
        # is always satisfied there — mirror the Run-button's ready_checks
        # logic so the tracker can't disagree with the button.
        _sb_nem_regime = st.session_state.get("nem_regime_1", "NEM-3 / NVBT")
        _sb_nema_mode = st.session_state.get("load_mode") == "NEM-A Aggregation"
        _sb_export_required = (
            _sb_nem_regime == "NEM-3 / NVBT" and not _sb_nema_mode
        )
        _sb_export = (
            st.session_state.get("export_rates") is not None
            if _sb_export_required else True
        )
        _sb_checks = [
            ("Production", _sb_prod),
            ("Load",       _sb_load),
            ("Tariff",     _sb_tariff),
            ("Export",     _sb_export),
        ]
        # Rate-shift adds a 5th gate when the toggle is on — loading a
        # saved sim that had rate-shift enabled but no old-tariff reference
        # used to silently disable the Run button while the tracker showed
        # 4/4 ready. Surface it as its own pill instead.
        if st.session_state.get("rate_shift_enabled"):
            _sb_rs_ready = (
                st.session_state.get("rate_shift_old_tariff") is not None
                if st.session_state.get("billing_engine", "Custom") == "Custom"
                else st.session_state.get("rate_shift_old_ecc_calculator") is not None
            )
            _sb_checks.append(("Rate-shift", _sb_rs_ready))
        _sb_done = sum(1 for _, ok in _sb_checks if ok)
        _sb_total = len(_sb_checks)
        _sb_complete = _sb_done == _sb_total

        # Teal → green ombre palette. User prefers the darker end (teal) as
        # the dominant tone; green shows up lighter at the right of the
        # gradient. We ramp through 4 stops, one per pill, so the row reads
        # as a single teal-to-green sweep rather than 4 independent badges.
        _TEAL = "#518484"
        _GREEN = "#45A750"
        # Ombre stops — sized for up to 5 pills (4 core + optional rate-shift).
        # The ramp stays teal→green; an extra mid-stop keeps the sweep even
        # when the 5th pill shows up.
        _ombre_stops = ("#3E6F74", "#47767A", "#4F8369", "#59925E", _GREEN)

        # Status copy — subtle tone shift at completion; the color is
        # always a teal/green variant now (no amber/grey headline).
        if _sb_complete:
            _headline_text = "Inputs ready — run simulation"
            _headline_color = "#3E6F74"  # dark teal
        elif _sb_done >= _sb_total // 2:
            _headline_text = f"{_sb_done} of {_sb_total} inputs loaded"
            _headline_color = "#4B7E71"  # mid teal/green
        else:
            _headline_text = f"{_sb_done} of {_sb_total} inputs loaded"
            _headline_color = "#518484"  # straight teal

        # Oval pills — loaded pills take their color from the ombre stops
        # (positional, so the row visually sweeps dark-teal to green).
        # Pending pills stay ghost-outlined.
        _pill_pieces = []
        for idx, (name, ok) in enumerate(_sb_checks):
            if ok:
                _color = _ombre_stops[idx % len(_ombre_stops)]
                _pill_pieces.append(
                    f'<span style="display:inline-flex; align-items:center; gap:4px;'
                    f'padding:3px 10px; margin:0 4px 4px 0;'
                    f'font-size:11px; font-weight:600; letter-spacing:0.02em;'
                    f'border-radius:999px; background:{_color}; color:#ffffff;'
                    f'border:1px solid {_color};">'
                    f'<span style="font-size:9px; line-height:1;">✓</span> {name}'
                    f'</span>'
                )
            else:
                _pill_pieces.append(
                    f'<span style="display:inline-flex; align-items:center; gap:4px;'
                    f'padding:3px 10px; margin:0 4px 4px 0;'
                    f'font-size:11px; font-weight:500; letter-spacing:0.02em;'
                    f'border-radius:999px; background:#ffffff; color:#94A3B8;'
                    f'border:1px solid #E2E8F0;">'
                    f'<span style="font-size:9px; line-height:1;">○</span> {name}'
                    f'</span>'
                )
        _pills_html = "".join(_pill_pieces)

        # Progress bar — teal → green ombre gradient, width scales with
        # completion. At 0% the bar is invisible; at 100% it reads as the
        # full ombre sweep.
        _pct = int(round(_sb_done / _sb_total * 100))
        _bar = (
            f'<div style="height:3px; background:#F1F5F9; border-radius:2px; '
            f'margin-top:8px; overflow:hidden;">'
            f'<div style="width:{_pct}%; height:100%; '
            f'background:linear-gradient(90deg, {_TEAL} 0%, {_GREEN} 100%); '
            f'transition: width 240ms ease;"></div>'
            f'</div>'
        )

        # Sticky wrapper — the #sb-sticky-tracker marker lets the CSS above make
        # this block's Streamlit element container position:sticky, so the
        # tracker stays pinned to the top of the sidebar as the configuration
        # below scrolls. (Sticky can't live on this inner div: its containing
        # block is only as tall as itself.) Rendered as a single flush-left HTML
        # line because Streamlit's markdown parser treats indented HTML as a
        # code block even with unsafe_allow_html=True.
        _eyebrow_color = _TEAL  # always teal; matches the ombre identity
        _celebration = (
            '<div style="position:absolute;top:0;left:0;right:0;height:2px;'
            f'background:linear-gradient(90deg,{_TEAL} 0%,{_GREEN} 100%);"></div>'
            if _sb_complete else ""
        )
        _tracker_html = (
            '<div id="sb-sticky-tracker" style="'
            'background:#FFFFFF;padding:12px 4px 10px 4px;'
            'margin:-8px -4px 10px -4px;'
            'border-bottom:1px solid #E5E7EB;position:relative;">'
            f'{_celebration}'
            f'<div style="font-size:10px;font-weight:600;'
            f'letter-spacing:0.08em;text-transform:uppercase;'
            f'color:{_eyebrow_color};margin-bottom:6px;">Input loading</div>'
            f'<div style="font-size:13px;font-weight:600;'
            f'color:{_headline_color};margin-bottom:8px;'
            f'letter-spacing:-0.005em;">{_headline_text}</div>'
            f'<div style="line-height:1.8;">{_pills_html}</div>'
            f'{_bar}'
            '</div>'
        )
        st.markdown(_tracker_html, unsafe_allow_html=True)

        st.markdown(
            '<hr style="border:none;border-top:1px solid #E5E7EB;'
            'margin:4px 0 12px 0;">',
            unsafe_allow_html=True,
        )

        # Dense-mode toggle — lives just under the tracker so power users
        # get a single place to manage viewing preferences.
        _prev_dense = bool(st.session_state.get("ui_dense_mode", False))
        _dense_choice = st.toggle(
            "Dense view",
            value=_prev_dense,
            key="ui_dense_mode_toggle",
            help="Tightens padding, metric sizes, and row height. Useful on smaller screens or when comparing many scenarios.",
        )
        if _dense_choice != _prev_dense:
            st.session_state["ui_dense_mode"] = _dense_choice
            st.rerun()

        st.header("System & Site Configuration")

        # --- Load a System Profile ---
        _sp_names = _list_saved(SYSTEM_PROFILES_DIR, ".json")
        if _sp_names:
            _sp_options = ["(none)"] + _sp_names
            _sp_selected = st.selectbox(
                "Load a System Profile",
                _sp_options,
                key="sp_sidebar_sel",
                help="Select a saved system profile to auto-fill Location and PV System settings.",
            )
            if _sp_selected != "(none)":
                if st.button("Apply Profile", key="sp_apply_btn", type="primary", width="stretch"):
                    st.session_state["pending_system_profile"] = _sp_selected
                    st.rerun()
            st.divider()

        # --- 1. Location ---
        st.subheader("1. Location")
        location_input = st.text_input(
            "Address or City, CA",
            placeholder="e.g., Fresno, CA or 123 Main St, Bakersfield, CA",
            key="sb_location",
            help="Enter a California address or city to geocode. Used for PVWatts solar resource data.",
        )

        lat, lon = None, None
        # Invalidate cached lat/lon when the user changes location text
        if location_input != st.session_state.get("_sp_cached_location", ""):
            st.session_state["_sp_lat"] = None
            st.session_state["_sp_lon"] = None
            st.session_state["_sp_cached_location"] = location_input
        if location_input:
            # Use cached lat/lon from a loaded system profile if available
            _cached_lat = st.session_state.get("_sp_lat")
            _cached_lon = st.session_state.get("_sp_lon")
            if _cached_lat is not None and _cached_lon is not None:
                lat, lon = _cached_lat, _cached_lon
                st.success(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
            else:
                try:
                    lat, lon = geocode_address(location_input)
                    st.session_state["_sp_lat"] = lat
                    st.session_state["_sp_lon"] = lon
                    st.session_state["_sp_cached_location"] = location_input
                    st.success(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                except ValueError as e:
                    st.error(str(e))

        # --- 2. System Configuration ---
        st.subheader("2. PV System")
        # Seed session default if unset; drop `value=` to avoid Streamlit's
        # "default + session value" warning that fires when a saved sim
        # populated the key before widget instantiation.
        st.session_state.setdefault("sb_system_life", 20)
        system_life_years = st.number_input(
            "System Life (years)", min_value=1, max_value=50, step=1,
            key="sb_system_life",
            help="Duration used for annual projection and payback calculation",
        )
        system_size_kw = st.number_input(
            "System Size (kW-DC)", min_value=1.0, max_value=100000.0, value=500.0, step=10.0,
            key="sb_system_size",
            help="Nameplate DC capacity of the PV array",
        )
        dc_ac_ratio = st.number_input(
            "DC/AC Ratio", min_value=1.0, max_value=2.0, value=1.2, step=0.05,
            key="sb_dc_ac_ratio",
            help="Ratio of DC array capacity to AC inverter capacity. Typical range: 1.1-1.4",
        )
        system_type = st.radio(
            "System Type",
            ["Fixed Tilt (Ground Mount)", "Single Axis Tracker"],
            key="sb_system_type",
        )

        with st.expander("Advanced PV Options", expanded=False):
            module_type_label = st.selectbox(
                "Module Type",
                ["Standard", "Premium", "Thin Film"],
                key="sb_module_type",
                help="Standard: crystal silicon; Premium: higher efficiency; Thin Film: CdTe or a-Si",
            )
            _module_type_map = {"Standard": 0, "Premium": 1, "Thin Film": 2}
            module_type_code = _module_type_map[module_type_label]

            system_losses_pct = st.number_input(
                "System Losses (%)",
                min_value=0.0,
                max_value=50.0,
                value=14.08,
                step=0.5,
                format="%.2f",
                key="sb_system_losses",
                help="Total DC-to-AC derate losses (soiling, shading, wiring, mismatch, etc.). PVWatts default: 14.08%",
            )

            annual_degradation_pct = st.number_input(
                "Annual Degradation (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.50,
                step=0.05,
                format="%.2f",
                key="sb_degradation",
                help="Annual decline in solar output due to module aging. Industry standard: ~0.5%/yr",
            )

        cod_date = st.date_input(
            "Commercial Operation Date (COD)",
            value=date(2026, 1, 1),
            key="sb_cod_date",
            help="Start date for the simulation. The year determines TMY solar resource alignment.",
        )
        cod_year = cod_date.year

        generate_prod = st.button(
            "Generate Production Profile",
            type="primary",
            key="api_call_btn_generate_prod",  # picked up by .st-key-* CSS below
        )

        # --- 3. Load Profile ---
        st.subheader("3. Load Profile")

        # Initialize radio key from load_mode if not yet set
        if "load_mode_radio" not in st.session_state:
            st.session_state["load_mode_radio"] = st.session_state.get("load_mode", "Single Meter")
        # Apply pending load mode from saved profile restore
        if "_pending_load_mode_radio" in st.session_state:
            st.session_state["load_mode_radio"] = st.session_state.pop("_pending_load_mode_radio")
        load_mode = st.radio(
            "Configuration",
            ["Single Meter", "NEM-A Aggregation"],
            horizontal=True,
            key="load_mode_radio",
        )
        st.session_state["load_mode"] = load_mode

        load_file = None

        # --- A. Unified saved profiles dropdown (CSV + NEM-A) ---
        _sb_all_profiles = _list_all_load_profiles()

        if _sb_all_profiles:
            _sb_profile_names = [p[0] for p in _sb_all_profiles]
            _sb_sel_name = st.selectbox(
                "Saved Load Profile",
                _sb_profile_names,
                key="sidebar_profile_sel",
            )
            _sb_sel_idx = _sb_profile_names.index(_sb_sel_name)
            _sb_sel_type = _sb_all_profiles[_sb_sel_idx][1]

            # Auto-load on selection change
            _last_loaded = st.session_state.get("_last_loaded_sidebar_profile")
            if _sb_sel_name != _last_loaded:
                st.session_state["_last_loaded_sidebar_profile"] = _sb_sel_name
                try:
                    if _sb_sel_type == "csv":
                        _sb_df = _load_profile_csv(LOAD_PROFILES_DIR, _sb_sel_name)
                        _sb_vals = _parse_8760_csv(_sb_df)
                        _sb_dt = pd.date_range(f"{cod_year}-01-01", periods=8760, freq="h")
                        st.session_state["load_8760"] = pd.Series(_sb_vals, index=_sb_dt, name="load_kwh")
                        st.session_state["_raw_load_8760"] = st.session_state["load_8760"].copy()
                        st.session_state["load_mode"] = "Single Meter"
                        st.session_state["load_mode_radio"] = "Single Meter"
                        st.sidebar.success(
                            f"Loaded '{_sb_sel_name}': {_sb_vals.sum():,.0f} kWh/yr, "
                            f"Peak: {_sb_vals.max():,.1f} kW"
                        )
                    else:
                        _load_nema_profile_into_session(_sb_sel_name)
                        st.sidebar.success(
                            f"Loaded NEM-A '{_sb_sel_name}' "
                            f"({len(st.session_state.get('nema_meters', []))} meters)"
                        )
                except Exception as e:
                    st.sidebar.error(f"Error loading profile: {e}")

            # Show NEM-A meter breakdown when a NEM-A profile is active
            if _sb_sel_type == "nema":
                _sb_meters = st.session_state.get("nema_meters", [])
                _sb_meter_loads = st.session_state.get("nema_meter_loads", {})
                if _sb_meters:
                    for _mi, _m in enumerate(_sb_meters):
                        _m_label = _m["name"]
                        if _m.get("is_generating"):
                            _m_label += " *"
                        _m_kwh = ""
                        if _mi in _sb_meter_loads:
                            _m_kwh = f" — {_sb_meter_loads[_mi].sum():,.0f} kWh/yr"
                        st.caption(f"{_m_label}{_m_kwh}")
        else:
            st.caption("No saved profiles. Create one in the Load Profiles tab.")

        # --- B. Ad-hoc CSV upload ---
        st.caption("Or upload a CSV directly:")
        load_file = st.file_uploader("Upload 8760 Load CSV", type=["csv"], key="sidebar_load_upload")

        # --- Existing Solar (Repower) ---
        st.divider()
        existing_solar_enabled = st.toggle(
            "Existing Solar (Decommission)",
            key="existing_solar_toggle",
        )
        st.session_state["existing_solar_enabled"] = existing_solar_enabled

        generate_existing_solar = False
        existing_solar_size_kw = 100.0
        existing_solar_dc_ac = 1.2
        existing_solar_system_type = "Fixed Tilt (Ground Mount)"
        existing_solar_age = 10
        existing_solar_degradation = 0.50
        if existing_solar_enabled:
            st.caption(
                "If the site has an existing solar system being decommissioned, "
                "enter its specs below. The old system's estimated production will be added "
                "back to the interval data to recover the true gross load."
            )
            existing_solar_size_kw = st.number_input(
                "Existing System Size (kW-DC)",
                min_value=0.1,
                value=100.0,
                step=10.0,
                key="sb_existing_solar_size",
            )
            existing_solar_system_type = st.radio(
                "System Type",
                ["Fixed Tilt (Ground Mount)", "Single Axis Tracker"],
                key="sb_existing_solar_type",
                horizontal=True,
            )
            existing_solar_dc_ac = st.number_input(
                "DC/AC Ratio",
                min_value=0.5,
                max_value=3.0,
                value=1.2,
                step=0.05,
                key="sb_existing_solar_dc_ac",
            )
            existing_solar_age = st.number_input(
                "System Age (years)",
                min_value=0,
                max_value=50,
                value=10,
                step=1,
                key="sb_existing_solar_age",
            )
            existing_solar_degradation = st.number_input(
                "Annual Degradation (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.50,
                step=0.05,
                format="%.2f",
                key="sb_existing_solar_degradation",
            )

            # NEM-A mode: meter selection checkboxes
            if load_mode == "NEM-A Aggregation":
                st.caption("Select which meters the existing system was offsetting:")
                _nema_selected = []
                for _esi, _esm in enumerate(st.session_state.get("nema_meters", [])):
                    _es_default = _esm.get("is_generating", False)
                    _es_checked = st.checkbox(
                        _esm["name"],
                        value=_es_default,
                        key=f"existing_solar_meter_{_esi}",
                    )
                    if _es_checked:
                        _nema_selected.append(_esi)
                st.session_state["existing_solar_nema_meters"] = _nema_selected

            generate_existing_solar = st.button("Generate Existing Solar Profile", key="gen_existing_solar_btn")

            # Show status if profile already generated
            _es_prod = st.session_state.get("existing_solar_production_8760")
            if _es_prod is not None:
                st.success(f"Existing solar profile loaded: {_es_prod.sum():,.0f} kWh/yr (degraded)")

        # --- 4. Utility & Rate ---
        st.subheader("4. Utility & Rate")
        billing_engine = st.radio(
            "Billing Engine", ["Custom", "ECC"],
            key="billing_engine_radio",
            horizontal=True,
            help="Custom: uses OpenEI tariff data with built-in TOU billing. ECC: uses the Energy Cost Calculator engine.",
        )
        st.session_state["billing_engine"] = billing_engine

        utility_name = st.selectbox("Utility", list(UTILITY_EIA_IDS.keys()), key="sb_utility")

        # --- Default button/widget states for whichever branch is inactive ---
        fetch_rates_btn = False
        selected_rate_name = None
        selected_label = None
        load_tariff_btn = False
        ecc_fetch_btn = False
        ecc_load_json_btn = False

        if billing_engine == "Custom":
            # ---- Existing Custom engine UI ----
            fetch_rates_btn = st.button(
                "Fetch Available Rates", key="api_call_btn_fetch_rates",
            )

            # Rate selection (inline, right under Fetch button)
            if st.session_state["available_rates"]:
                rate_options = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                selected_rate_name = st.selectbox("Select Rate Schedule", list(rate_options.keys()))
                selected_label = rate_options[selected_rate_name]
                load_tariff_btn = st.button("Load Tariff Details")

            if st.session_state["tariff"]:
                with st.expander("View Tariff Details"):
                    st.markdown(format_tariff_summary(st.session_state["tariff"]))

        else:
            # ---- ECC engine UI ----
            _saved_ecc = _list_saved(ECC_TARIFFS_DIR, ".json")
            _ecc_source_options = ["Upload JSON", "OpenEI API"]
            if _saved_ecc:
                _ecc_source_options.insert(0, "Use Saved Tariff")

            ecc_tariff_source = st.radio(
                "Tariff Source", _ecc_source_options,
                key="ecc_tariff_source",
                horizontal=True,
            )

            if ecc_tariff_source == "Use Saved Tariff":
                _sel_ecc = st.selectbox("Select Saved Tariff", _saved_ecc, key="ecc_saved_sel")
                ecc_load_json_btn = st.button("Load Tariff", type="primary")
                if ecc_load_json_btn and _sel_ecc:
                    st.session_state["_ecc_saved_path"] = os.path.join(ECC_TARIFFS_DIR, _sel_ecc + ".json")

            elif ecc_tariff_source == "Upload JSON":
                st.file_uploader(
                    "Upload Tariff JSON", type=["json"], key="ecc_json_upload",
                )
                ecc_load_json_btn = st.button("Load from JSON", type="primary")

            elif ecc_tariff_source == "OpenEI API":
                st.selectbox(
                    "Sector", ["Commercial", "Residential", "Industrial"],
                    key="ecc_sector",
                )
                st.selectbox(
                    "Distribution Level", ["Secondary", "Primary"],
                    key="ecc_distrib",
                )
                st.selectbox(
                    "Phase Wiring", ["Three", "Single", "None"],
                    key="ecc_phase",
                )
                st.checkbox("Time-of-Use (TOU)", value=True, key="ecc_tou")
                st.checkbox("Peak Day Pricing (PDP)", value=False, key="ecc_pdp")
                st.text_input(
                    "Rate Schedule Filter",
                    placeholder="e.g., A-6, E-19, AG-4",
                    key="ecc_rate_filter",
                )
                ecc_fetch_btn = st.button("Fetch & Load ECC Tariff", type="primary")

            if st.session_state["ecc_tariff_metadata"]:
                with st.expander("View ECC Tariff Info"):
                    meta = st.session_state["ecc_tariff_metadata"]
                    st.write(f"**Source:** {meta.get('source', 'N/A')}")
                    st.write(f"**Utility ID:** {meta.get('utility_id', 'N/A')}")
                    st.write(f"**Sector:** {meta.get('sector', 'N/A')}")
                    st.write(f"**Rate Filter:** {meta.get('rate_filter', 'N/A')}")
                    n_tariffs = meta.get("num_tariffs", 0)
                    st.write(f"**Tariff blocks loaded:** {n_tariffs}")
                    if meta.get("tariff_names"):
                        for tname in meta["tariff_names"][:10]:
                            st.caption(f"  - {tname}")

        # --- Per-Meter Tariff Selection (NEM-A with Custom engine) ---
        if billing_engine == "Custom" and st.session_state.get("load_mode") == "NEM-A Aggregation":
            _pmt_meters = st.session_state.get("nema_meters", [])
            _pmt_needs_tariff = [
                (_pmt_i, _pmt_m) for _pmt_i, _pmt_m in enumerate(_pmt_meters)
                if not _pmt_m.get("is_generating") and not st.session_state.get(f"nema_use_gen_tariff_{_pmt_i}", True)
            ]
            if _pmt_needs_tariff:
                st.markdown("---")
                st.markdown("**Per-Meter Tariff Selection**")
                st.caption("Load a separate tariff for meters not using the generating meter's tariff.")
                _pmt_loaded_tariffs = st.session_state.get("nema_meter_tariffs", {})
                for _pmt_i, _pmt_m in _pmt_needs_tariff:
                    with st.expander(f"Tariff for: {_pmt_m['name']}", expanded=False):
                        if st.session_state["available_rates"]:
                            _pmt_rate_options = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                            _pmt_sel_name = st.selectbox(
                                "Select Rate Schedule", list(_pmt_rate_options.keys()),
                                key=f"nema_tariff_sel_{_pmt_i}",
                            )
                            _pmt_sel_label = _pmt_rate_options[_pmt_sel_name]
                            if st.button("Load Tariff", key=f"nema_tariff_load_{_pmt_i}", type="primary"):
                                st.session_state[f"_pending_nema_tariff_load_{_pmt_i}"] = _pmt_sel_label
                        else:
                            st.caption("Fetch rates above first to select a tariff.")

                        # Show current tariff status
                        _pmt_current = _pmt_loaded_tariffs.get(_pmt_i)
                        if _pmt_current is not None:
                            st.success(f"Loaded: {_pmt_current.name}")
                        else:
                            st.warning("No tariff loaded for this meter.")

        # ================================================================
        # --- Rate Switch ---
        # Single master toggle. When OFF, behavior is identical to a plain
        # single-tariff projection: rate_shift_enabled=False and the
        # regime-2 tariff keys are cleared (see end of the NEM-switch block).
        # When ON, two INDEPENDENT triggers control where the rate changes:
        #   (a) "At Repower / project start" → the existing rate-shift
        #       mechanism (rate_shift_enabled + rate_shift_old_tariff /
        #       rate_shift_old_ecc_calculator), and
        #   (b) "At NEM switch (year N)" → the regime-2 post-transition
        #       tariff (regime_2_tariff / regime_2_ecc_calculator). The
        #       regime-2 LOADER renders inline directly under its trigger
        #       checkbox below, gated on the rate_switch_at_nem checkbox.
        # ================================================================
        st.markdown("---")
        rate_switch_enabled = st.toggle(
            "Rate Switch",
            key="rate_switch_toggle",
            value=st.session_state.get("rate_switch_enabled", False),
            help="Model the customer changing tariffs over the project life. "
                 "Choose where the switch happens: at repower / project start, "
                 "at the NEM regime transition, or both.",
        )
        st.session_state["rate_switch_enabled"] = rate_switch_enabled

        # Defaults — overwritten only when Rate Switch is on.
        rate_switch_at_start = False
        rate_switch_at_nem = False

        if rate_switch_enabled:
            st.caption("Switch the rate at:")
            # Trigger (a): repower / project start — the rate-shift baseline.
            rate_switch_at_start = st.checkbox(
                "At Repower / project start",
                key="rate_switch_at_start",
                help="Bill an original (pre-switch) tariff as a standalone "
                     "baseline alongside the Section-4 tariff. Use for a "
                     "tariff change that takes effect at repower / project start.",
            )
            # Trigger (b): NEM transition — the regime-2 tariff. Requires the
            # NEM Switch (Section 5) to be configured so num_years_1 exists.
            _nem_switch_on = bool(st.session_state.get("nem_switch"))
            rate_switch_at_nem = st.checkbox(
                "At NEM switch year",
                key="rate_switch_at_nem",
                disabled=not _nem_switch_on,
                help="Re-bill the years after the NEM transition on a different "
                     "tariff. The post-NEM switch loader appears directly below "
                     "once NEM Switch is enabled.",
            )
            if not _nem_switch_on:
                # A disabled checkbox retains its stored value, so force the
                # local False when NEM Switch is off (regime-2 keys are also
                # backstop-cleared downstream, and the LP-tab gate checks
                # nem_switch too).
                rate_switch_at_nem = False
                st.caption("Disabled until NEM Switch is enabled below (Section 5).")

        # --- Trigger (a): original (pre-switch) rate loader ---
        # Drives the EXISTING rate-shift mechanism. rate_shift_enabled mirrors
        # the trigger so the engines read it unchanged.
        rate_shift_enabled = bool(rate_switch_enabled and rate_switch_at_start)
        st.session_state["rate_shift_enabled"] = rate_shift_enabled

        if rate_shift_enabled:
            st.markdown("**Original rate (pre-switch)**")

        if rate_shift_enabled and billing_engine == "Custom":
            _rs_is_nema = st.session_state.get("load_mode") == "NEM-A Aggregation"

            if not _rs_is_nema:
                # Single meter: one original-tariff selector
                st.caption(
                    "Select the customer's **original (pre-switch) tariff** below. "
                    "The analysis shows what the customer would pay on this rate as a "
                    "separate baseline, alongside the new tariff selected in Section 4."
                )
                _render_custom_tariff_loader(
                    select_label="Original tariff (pre-switch baseline)",
                    button_label="Load Original Tariff",
                    sel_key="rate_shift_old_rate_sel",
                    button_key="api_call_btn_rate_shift_load",
                    pending_key="_pending_rate_shift_load",
                    target_key="rate_shift_old_tariff",
                    loaded_prefix="Original tariff loaded",
                    empty_warning="No original tariff loaded — rate switch baseline will be unavailable.",
                    select_help="The tariff the customer is on today, before switching to the "
                                "Section-4 tariff. Billed as a standalone comparison baseline.",
                    no_rates_caption="Fetch rates above first to select an original tariff.",
                )
            else:
                # NEM-A: per-meter original tariff selectors only (no blanket option)
                _rs_meters = st.session_state.get("nema_meters", [])
                if _rs_meters and st.session_state["available_rates"]:
                    st.markdown("**Per-Meter Original Tariffs (NEM-A)**")
                    _rs_nema_tariffs = st.session_state.get("nema_rate_shift_tariffs", {})
                    _rs_all_loaded = True
                    for _rs_i, _rs_m in enumerate(_rs_meters):
                        with st.expander(f"Original tariff for: {_rs_m['name']}", expanded=(_rs_i not in _rs_nema_tariffs)):
                            _rs_pmt_options = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                            _rs_pmt_sel = st.selectbox(
                                "Original Rate", list(_rs_pmt_options.keys()),
                                key=f"nema_rs_tariff_sel_{_rs_i}",
                            )
                            _rs_pmt_label = _rs_pmt_options[_rs_pmt_sel]
                            if st.button("Load", key=f"nema_rs_tariff_load_{_rs_i}"):
                                st.session_state[f"_pending_nema_rs_tariff_{_rs_i}"] = _rs_pmt_label
                            _rs_pmt_current = _rs_nema_tariffs.get(_rs_i)
                            if _rs_pmt_current is not None:
                                st.success(f"Loaded: {_rs_pmt_current.name}")
                            else:
                                _rs_all_loaded = False
                    if not _rs_all_loaded:
                        st.warning("Load an original tariff for each meter to enable the rate switch baseline.")
                else:
                    st.caption("Fetch rates and configure NEM-A meters first.")

        elif rate_shift_enabled and billing_engine == "ECC":
            st.caption("Original rate with the ECC engine: upload a second ECC tariff JSON for the pre-switch rate.")
            ecc_rs_upload = st.file_uploader(
                "Original Rate Tariff JSON (ECC)", type=["json"], key="ecc_rs_json_upload",
            )
            if st.button("Load Original ECC Tariff", key="ecc_rs_load_btn") and ecc_rs_upload:
                st.session_state["_pending_ecc_rs_load"] = ecc_rs_upload

            _rs_ecc_current = st.session_state.get("rate_shift_old_ecc_calculator")
            if _rs_ecc_current is not None:
                st.success("Original ECC tariff loaded.")

        # --- Trigger (b): post-NEM switch rate loader ---
        # Co-located directly under the "At NEM switch year" trigger above (it
        # used to live down in Section 5). Gated on rate_switch_at_nem, which is
        # forced False whenever Rate Switch is off OR NEM Switch is off (see the
        # checkbox backstop above), so the else-clear is the single place that
        # zeroes the regime-2 keys on every not-armed path. The keys written
        # (regime_2_tariff / regime_2_ecc_calculator) are unchanged. Reads
        # available_rates (fetched in Section 4, above) and nem_switch from
        # session_state.
        if rate_switch_at_nem:
            st.markdown("**Post-NEM switch rate**")
            st.caption(
                "Re-bill the post-transition years (after the NEM switch year) "
                "on a different tariff. Leave unloaded to reuse the Section-4 tariff."
            )
            if billing_engine == "Custom":
                _render_custom_tariff_loader(
                    select_label="Select Post-NEM switch Rate Schedule",
                    button_label="Load Post-NEM switch Tariff",
                    sel_key="regime2_tariff_sel",
                    button_key="regime2_tariff_load_btn",
                    pending_key="_pending_regime2_tariff_load",
                    target_key="regime_2_tariff",
                    loaded_prefix="Post-NEM switch tariff loaded",
                    empty_warning="No post-NEM switch tariff loaded — will reuse the regime-1 tariff.",
                    select_help="The tariff the customer moves to after the NEM regime switch.",
                    no_rates_caption="Fetch rates above first to select a post-NEM switch tariff.",
                )
            else:
                _r2_saved_ecc = _list_saved(ECC_TARIFFS_DIR, ".json")
                _r2_ecc_src = st.radio(
                    "Post-NEM switch tariff source",
                    (["Use Saved Tariff"] if _r2_saved_ecc else []) + ["Upload JSON"],
                    key="regime2_ecc_source",
                    horizontal=True,
                )
                if _r2_ecc_src == "Use Saved Tariff":
                    _r2_sel_ecc = st.selectbox(
                        "Select Saved Post-NEM switch Tariff", _r2_saved_ecc,
                        key="regime2_ecc_saved_sel",
                    )
                    if st.button("Load Post-NEM switch Tariff", key="regime2_ecc_saved_load_btn") and _r2_sel_ecc:
                        st.session_state["_pending_regime2_ecc_saved_path"] = os.path.join(
                            ECC_TARIFFS_DIR, _r2_sel_ecc + ".json"
                        )
                else:
                    _r2_ecc_upload = st.file_uploader(
                        "Upload Post-NEM switch Tariff JSON", type=["json"],
                        key="regime2_ecc_json_upload",
                    )
                    if st.button("Load Post-NEM switch Tariff", key="regime2_ecc_upload_load_btn") and _r2_ecc_upload:
                        st.session_state["_pending_regime2_ecc_upload"] = _r2_ecc_upload
                if st.session_state.get("regime_2_ecc_calculator") is not None:
                    st.success("Post-NEM switch ECC tariff loaded.")
                else:
                    st.warning("No post-NEM switch ECC tariff loaded — will reuse the regime-1 tariff.")
        else:
            # Trigger off (Rate Switch off, "At NEM switch" off, or NEM Switch
            # off — which forces the trigger False above) — clear the regime-2
            # tariff so the post-switch years reuse the regime-1 tariff (default,
            # single-tariff path). This is the single clear that covers every
            # not-armed path, including NEM Switch being off.
            st.session_state["regime_2_tariff"] = None
            st.session_state["regime_2_ecc_calculator"] = None

        st.markdown("---")

        # --- 5. Export Compensation ---
        st.subheader("5. Export Compensation")
        nem_options = ["NEM-1", "NEM-2", "NEM-3 / NVBT"]

        def _render_export_rate_widgets(section_suffix: str, disabled: bool = False):
            """Render export rate source widgets. Returns (method, selected_profile, flat_rate_val)."""
            saved_names = _list_saved(EXPORT_PROFILES_DIR, ".csv")
            _export_options = ["Use saved profile", "Upload CSV", "Flat rate ($/kWh)"]
            method = st.radio(
                "Export rate source", _export_options,
                key=f"export_method_radio{section_suffix}",
                disabled=disabled,
            )
            sel_profile = None
            flat_val = None
            if method == "Use saved profile":
                if saved_names:
                    sel_profile = st.selectbox(
                        "Select Export Profile", saved_names,
                        key=f"sidebar_export_sel{section_suffix}",
                        disabled=disabled,
                    )
                else:
                    st.caption("No saved profiles. Upload via the Export Profiles tab above.")
            elif method == "Upload CSV":
                st.file_uploader(
                    "Upload ACC Export Rate CSV (8760 rows/yr, multi-year supported)",
                    type=["csv"],
                    key=f"sidebar_export_upload{section_suffix}",
                    disabled=disabled,
                )
            elif method == "Flat rate ($/kWh)":
                flat_val = st.number_input(
                    "Flat export rate ($/kWh)", min_value=0.0, max_value=1.0,
                    value=0.05, step=0.005, format="%.4f",
                    key=f"sb_flat_rate{section_suffix}",
                    disabled=disabled,
                )
            return method, sel_profile, flat_val

        nem_switch = st.toggle(
            "NEM Switch", value=False, key="nem_switch_toggle",
            help="Enable to model a mid-life NEM regime transition (e.g., NEM-1 for first 5 years, then NEM-3/NVBT)",
        )
        st.session_state["nem_switch"] = nem_switch

        def _render_nem12_widgets(suffix: str, regime: str):
            """Render NEM-1/NEM-2 specific widgets. Returns (nsc_rate, nbc_rate, billing_opt)."""
            st.caption("Exports valued at retail TOU energy rate (per NEM tariff)")
            _nsc = st.number_input(
                "NSC Rate ($/kWh)", min_value=0.0, max_value=1.0,
                value=st.session_state.get("nsc_rate", NSC_DEFAULT_RATE),
                step=0.005, format="%.4f",
                key=f"sb_nsc_rate{suffix}",
                help="Net Surplus Compensation rate for annual surplus export",
            )
            _nbc = 0.0
            if regime == "NEM-2":
                _nbc_default = NBC_DEFAULTS.get(utility_name, 0.025)
                _nbc = st.number_input(
                    "NBC Rate ($/kWh)", min_value=0.0, max_value=1.0,
                    value=st.session_state.get("nbc_rate", _nbc_default) or _nbc_default,
                    step=0.005, format="%.4f",
                    key=f"sb_nbc_rate{suffix}",
                    help="Non-Bypassable Charge applied to net consumption each interval",
                )
            _billing = st.radio(
                "Billing Option",
                ["Annual (ABO)", "Monthly (MBO)"],
                key=f"sb_billing_option{suffix}",
                horizontal=True,
            )
            _billing_opt = "ABO" if "Annual" in _billing else "MBO"
            return _nsc, _nbc, _billing_opt

        # Defaults for NEM-specific params
        nsc_rate = st.session_state.get("nsc_rate", NSC_DEFAULT_RATE)
        nbc_rate = st.session_state.get("nbc_rate", 0.0)
        billing_option = st.session_state.get("billing_option", "ABO")

        if not nem_switch:
            # Single export section
            nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=2,
                                        key="sb_nem_regime_1_single")
            if billing_engine == "ECC" and nem_regime_1 in ("NEM-1", "NEM-2"):
                st.warning(
                    "The ECC engine does not support TOU netting or credit carryover used by "
                    f"{nem_regime_1}. Annual projections may be inaccurate. "
                    "Use the Custom billing engine for full NEM-1/NEM-2 support."
                )
            if nem_regime_1 in ("NEM-1", "NEM-2"):
                nsc_rate, nbc_rate, billing_option = _render_nem12_widgets("", nem_regime_1)
                st.session_state["nsc_rate"] = nsc_rate
                st.session_state["nbc_rate"] = nbc_rate
                st.session_state["billing_option"] = billing_option
                # No export rate widgets needed — exports valued at retail TOU rate
                export_method = None
                selected_export_profile = None
                flat_rate = None
            else:
                export_method, selected_export_profile, flat_rate = _render_export_rate_widgets("")
            # Placeholders for section-2 variables (unused when switch is off)
            nem_regime_2 = None
            num_years_1 = None
            export_method_2 = None
            selected_export_profile_2 = None
            flat_rate_2 = None
        else:
            # --- Section 1 ---
            st.markdown("---")
            st.markdown("**Section 1 — Export Rates**")
            nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=0,
                                        key="sb_nem_regime_1_dual")
            if billing_engine == "ECC" and nem_regime_1 in ("NEM-1", "NEM-2"):
                st.warning(
                    "The ECC engine does not support TOU netting or credit carryover used by "
                    f"{nem_regime_1}. Annual projections may be inaccurate. "
                    "Use the Custom billing engine for full NEM-1/NEM-2 support."
                )
            num_years_1 = st.number_input(
                "Tenor (years)", min_value=1,
                max_value=max(1, system_life_years - 1),
                value=min(5, max(1, system_life_years - 1)),
                step=1, key="sb_nem_years_1",
            )
            if nem_regime_1 in ("NEM-1", "NEM-2"):
                nsc_rate, nbc_rate, billing_option = _render_nem12_widgets("", nem_regime_1)
                st.session_state["nsc_rate"] = nsc_rate
                st.session_state["nbc_rate"] = nbc_rate
                st.session_state["billing_option"] = billing_option
                export_method = None
                selected_export_profile = None
                flat_rate = None
            else:
                export_method, selected_export_profile, flat_rate = _render_export_rate_widgets("")

            # --- Section 2 ---
            st.markdown("---")
            st.markdown("**Section 2 — Export Rates**")
            nem_regime_2 = st.selectbox("NEM Regime", nem_options, index=2, key="sb_nem_regime_2")
            remaining_years = system_life_years - num_years_1
            st.number_input(
                "Tenor (years)", min_value=remaining_years, max_value=remaining_years,
                value=remaining_years, step=1, disabled=True, key="sb_nem_years_2",
            )
            if nem_regime_2 in ("NEM-1", "NEM-2"):
                # Section 2 NEM-1/NEM-2 widgets (separate keys)
                nsc_rate_2, nbc_rate_2, billing_option_2 = _render_nem12_widgets("_2", nem_regime_2)
                st.session_state["nsc_rate_2"] = nsc_rate_2
                st.session_state["nbc_rate_2"] = nbc_rate_2
                st.session_state["billing_option_2"] = billing_option_2
                export_method_2 = None
                selected_export_profile_2 = None
                flat_rate_2 = None
            else:
                export_method_2, selected_export_profile_2, flat_rate_2 = _render_export_rate_widgets("_2")

            # NOTE: the Post-NEM switch tariff loader (Rate Switch → "At NEM
            # switch") now renders up in the Rate Switch section, directly under
            # its trigger checkbox. Its else-clear there zeroes the regime-2 keys
            # on every not-armed path. The backstop below is retained as
            # defense-in-depth for the NEM-Switch-off case.

        # When the NEM switch is off, a regime-2 tariff is meaningless — clear it
        # so a stale selection can't leak into a single-regime projection.
        if not nem_switch:
            st.session_state["regime_2_tariff"] = None
            st.session_state["regime_2_ecc_calculator"] = None

        # --- 6. Battery (BESS) ---
        st.subheader("6. BESS")
        battery_enabled = st.toggle(
            "Enable Battery Storage", value=False, key="bess_toggle",
        )
        st.session_state["battery_enabled"] = battery_enabled

        battery_hours = st.number_input(
            "Battery Duration (hours)",
            min_value=0.5, max_value=12.0, value=4.0, step=0.5,
            disabled=not battery_enabled,
            help="Hours of storage at rated power",
            key="sb_batt_hours",
        )
        discharge_limit_pct = st.number_input(
            "Discharge Limit (%)",
            min_value=0.0, max_value=100.0, value=80.0, step=5.0,
            disabled=not battery_enabled,
            help="Max fraction of battery discharge that may be exported",
            key="sb_discharge_limit",
        )

        # --- Sizing: fixed kWh or optimize ---
        optimize_size = st.toggle(
            "Optimize Size", value=False, key="bess_optimize",
            disabled=not battery_enabled,
        )

        if not optimize_size:
            battery_capacity_kwh = st.number_input(
                "Battery Capacity (kWh)",
                min_value=1.0, max_value=500000.0, value=500.0, step=50.0,
                disabled=not battery_enabled,
                help="Nameplate energy capacity of the BESS",
                key="sb_batt_capacity",
            )
            bess_opt_min = bess_opt_max = bess_opt_step = 0.0
        else:
            opt_c1, opt_c2, opt_c3 = st.columns(3)
            with opt_c1:
                bess_opt_min = st.number_input(
                    "Min kWh", min_value=0.0, value=100.0, step=50.0,
                    disabled=not battery_enabled, key="bess_opt_min",
                )
            with opt_c2:
                bess_opt_max = st.number_input(
                    "Max kWh", min_value=0.0, value=2000.0, step=50.0,
                    disabled=not battery_enabled, key="bess_opt_max",
                )
            with opt_c3:
                bess_opt_step = st.number_input(
                    "Step kWh", min_value=1.0, value=100.0, step=50.0,
                    disabled=not battery_enabled, key="bess_opt_step",
                )
            battery_capacity_kwh = bess_opt_min  # placeholder; sweep happens at run time

        if battery_enabled:
            batt_power_kw = battery_capacity_kwh / battery_hours
            st.caption(f"Rated Power: {batt_power_kw:,.0f} kW"
                       + (" (per candidate)" if optimize_size else ""))

        # --- Charge / Discharge window presets ---
        WINDOW_PRESETS = {
            "Optimized (Best Export Hours)": "optimized",
            "Charge 9-15 / Discharge 16-21": (9, 15, 16, 21),
            "Charge 10-16 / Discharge 16-21": (10, 16, 16, 21),
            "Charge 8-14 / Discharge 17-22": (8, 14, 17, 22),
            "Charge 10-15 / Discharge 18-23": (10, 15, 18, 23),
            "Custom": None,
        }
        window_preset = st.selectbox(
            "Operating Windows",
            list(WINDOW_PRESETS.keys()),
            disabled=not battery_enabled,
            key="bess_window_preset",
        )
        preset_vals = WINDOW_PRESETS[window_preset]
        optimized_discharge = (preset_vals == "optimized")
        if optimized_discharge:
            charge_window_start, charge_window_end = 0, 23
            discharge_window_start, discharge_window_end = 0, 23
            if battery_enabled:
                st.caption(
                    f"Auto-selects best {int(battery_hours)}hr export block per day"
                )
        elif preset_vals is not None:
            charge_window_start, charge_window_end = preset_vals[0], preset_vals[1]
            discharge_window_start, discharge_window_end = preset_vals[2], preset_vals[3]
            if battery_enabled:
                st.caption(
                    f"Charge {charge_window_start}:00-{charge_window_end}:00 | "
                    f"Discharge {discharge_window_start}:00-{discharge_window_end}:00"
                )
        else:
            cw_col1, cw_col2 = st.columns(2)
            with cw_col1:
                charge_window_start = st.number_input(
                    "Charge Start Hr", min_value=0, max_value=23, value=10, step=1,
                    key="cw_start", disabled=not battery_enabled,
                )
            with cw_col2:
                charge_window_end = st.number_input(
                    "Charge End Hr", min_value=0, max_value=23, value=16, step=1,
                    key="cw_end", disabled=not battery_enabled,
                )
            dw_col1, dw_col2 = st.columns(2)
            with dw_col1:
                discharge_window_start = st.number_input(
                    "Discharge Start Hr", min_value=0, max_value=23, value=16, step=1,
                    key="dw_start", disabled=not battery_enabled,
                )
            with dw_col2:
                discharge_window_end = st.number_input(
                    "Discharge End Hr", min_value=0, max_value=23, value=21, step=1,
                    key="dw_end", disabled=not battery_enabled,
                )

        with st.expander("Advanced BESS Settings", expanded=False):
            bess_col1, bess_col2 = st.columns(2)
            with bess_col1:
                charge_eff = st.number_input(
                    "Charge Efficiency",
                    min_value=0.50, max_value=1.00, value=0.95, step=0.01,
                    format="%.2f", disabled=not battery_enabled,
                    key="sb_charge_eff",
                )
                discharge_eff = st.number_input(
                    "Discharge Efficiency",
                    min_value=0.50, max_value=1.00, value=0.95, step=0.01,
                    format="%.2f", disabled=not battery_enabled,
                    key="sb_discharge_eff",
                )
            with bess_col2:
                min_soc_pct = st.number_input(
                    "Min SoC (%)",
                    min_value=0.0, max_value=100.0, value=10.0, step=5.0,
                    disabled=not battery_enabled,
                    key="sb_min_soc",
                )
                max_soc_pct = st.number_input(
                    "Max SoC (%)",
                    min_value=0.0, max_value=100.0, value=100.0, step=5.0,
                    disabled=not battery_enabled,
                    key="sb_max_soc",
                )
            fast_dispatch = st.toggle(
                "Fast Dispatch (monthly LP)",
                value=True,
                disabled=not battery_enabled,
                help="Decompose the annual LP into 12 monthly sub-problems for faster solving",
                key="bess_fast_dispatch",
            )

        if battery_enabled:
            st.session_state["battery_capacity_kwh"] = battery_capacity_kwh
            st.session_state["battery_optimize"] = optimize_size
            st.session_state["battery_opt_range"] = (bess_opt_min, bess_opt_max, bess_opt_step)
            st.session_state["battery_fast_dispatch"] = fast_dispatch
            st.session_state["battery_config"] = BatteryConfig(
                battery_hours=battery_hours,
                discharge_limit_pct=discharge_limit_pct,
                charge_eff=charge_eff,
                discharge_eff=discharge_eff,
                min_soc_pct=min_soc_pct,
                max_soc_pct=max_soc_pct,
                charge_window_start=charge_window_start,
                charge_window_end=charge_window_end,
                discharge_window_start=discharge_window_start,
                discharge_window_end=discharge_window_end,
                optimized_discharge=optimized_discharge,
            )
        else:
            st.session_state["battery_config"] = None
            st.session_state["battery_capacity_kwh"] = 0
            st.session_state["battery_optimize"] = False
            st.session_state["battery_opt_range"] = (0, 0, 0)
            st.session_state["battery_fast_dispatch"] = False

        # --- 7. Escalators ---
        st.subheader("7. Escalators (Annual Projection)")
        # Seed session defaults before widget instantiation so a re-opened
        # saved simulation overrides them, but a fresh session lands on the
        # project-standard 3.0% / 0.0% pair.
        st.session_state.setdefault("sb_rate_escalator", 3.0)
        st.session_state.setdefault("sb_load_escalator", 0.0)
        rate_escalator = st.number_input(
            "Utility Rate Escalator (%/yr)", min_value=0.0, max_value=20.0, step=0.5,
            help="Applied annually to TOU energy rates",
            key="sb_rate_escalator",
        )
        load_escalator = st.number_input(
            "Demand Growth Escalator (%/yr)", min_value=0.0, max_value=20.0, step=0.5,
            help="Applied annually to load profile (increases consumption & peak demand)",
            key="sb_load_escalator",
        )
        compound_escalation = st.toggle(
            "Compound Escalation",
            value=True,
            key="sb_compound_escalation",
            help="Compound: (1 + rate%)^yr. Linear: 1 + rate% × yr. Compound is more realistic.",
        )

        # --- 8. System Cost ---
        st.subheader("8. System Cost (for Payback)")
        cost_input_method = st.radio(
            "Cost input", ["$/W-DC", "Total ($)"], key="sb_cost_method",
            help="Choose how to specify system cost. Used only for payback and ROI calculations.",
        )
        if cost_input_method == "$/W-DC":
            st.session_state.setdefault("sb_cost_per_watt", 2.10)
            cost_per_watt = st.number_input(
                "Installed Cost ($/W-DC)", min_value=0.0, step=0.05,
                key="sb_cost_per_watt",
            )
            system_cost = cost_per_watt * system_size_kw * 1000
            st.caption(f"Total: ${system_cost:,.0f}")
        else:
            system_cost = st.number_input(
                "Total Installed Cost ($)", min_value=0.0, value=750000.0, step=10000.0,
                key="sb_total_cost",
            )

    # Compute existing-solar offset for display columns
    _es_offset_monthly = None
    _es_offset_annual = 0.0
    if st.session_state.get("existing_solar_enabled") and st.session_state.get("existing_solar_production_8760") is not None:
        _es_prod = st.session_state["existing_solar_production_8760"]
        _dt = pd.date_range(f"{cod_year}-01-01", periods=8760, freq="h")
        _es_offset_monthly = [float(_es_prod[_dt.month == m].sum()) for m in range(1, 13)]
        _es_offset_annual = float(_es_prod.sum())

    return {
        "lat": lat,
        "lon": lon,
        "location_input": location_input,
        "system_life_years": system_life_years,
        "system_size_kw": system_size_kw,
        "dc_ac_ratio": dc_ac_ratio,
        "system_type": system_type,
        "module_type_code": module_type_code,
        "system_losses_pct": system_losses_pct,
        "annual_degradation_pct": annual_degradation_pct,
        "cod_date": cod_date,
        "cod_year": cod_year,
        "generate_prod": generate_prod,
        "load_mode": load_mode,
        "load_file": load_file,
        "existing_solar_enabled": existing_solar_enabled,
        "generate_existing_solar": generate_existing_solar,
        "existing_solar_size_kw": existing_solar_size_kw,
        "existing_solar_dc_ac": existing_solar_dc_ac,
        "existing_solar_system_type": existing_solar_system_type,
        "existing_solar_age": existing_solar_age,
        "existing_solar_degradation": existing_solar_degradation,
        "billing_engine": billing_engine,
        "utility_name": utility_name,
        "fetch_rates_btn": fetch_rates_btn,
        "selected_rate_name": selected_rate_name,
        "selected_label": selected_label,
        "load_tariff_btn": load_tariff_btn,
        "ecc_fetch_btn": ecc_fetch_btn,
        "ecc_load_json_btn": ecc_load_json_btn,
        "nem_switch": nem_switch,
        "nem_regime_1": nem_regime_1,
        "nem_regime_2": nem_regime_2,
        "num_years_1": num_years_1,
        "nsc_rate": nsc_rate,
        "nbc_rate": nbc_rate,
        "billing_option": billing_option,
        "export_method": export_method,
        "selected_export_profile": selected_export_profile,
        "flat_rate": flat_rate,
        "export_method_2": export_method_2,
        "selected_export_profile_2": selected_export_profile_2,
        "flat_rate_2": flat_rate_2,
        "battery_enabled": battery_enabled,
        "battery_capacity_kwh": battery_capacity_kwh,
        "rate_escalator": rate_escalator,
        "load_escalator": load_escalator,
        "compound_escalation": compound_escalation,
        "cost_input_method": cost_input_method,
        "system_cost": system_cost,
        "_es_offset_monthly": _es_offset_monthly,
        "_es_offset_annual": _es_offset_annual,
    }


_sidebar_ctx = _render_sidebar()
lat = _sidebar_ctx["lat"]
lon = _sidebar_ctx["lon"]
location_input = _sidebar_ctx["location_input"]
system_life_years = _sidebar_ctx["system_life_years"]
system_size_kw = _sidebar_ctx["system_size_kw"]
dc_ac_ratio = _sidebar_ctx["dc_ac_ratio"]
system_type = _sidebar_ctx["system_type"]
module_type_code = _sidebar_ctx["module_type_code"]
system_losses_pct = _sidebar_ctx["system_losses_pct"]
annual_degradation_pct = _sidebar_ctx["annual_degradation_pct"]
cod_date = _sidebar_ctx["cod_date"]
cod_year = _sidebar_ctx["cod_year"]
generate_prod = _sidebar_ctx["generate_prod"]
load_mode = _sidebar_ctx["load_mode"]
load_file = _sidebar_ctx["load_file"]
existing_solar_enabled = _sidebar_ctx["existing_solar_enabled"]
generate_existing_solar = _sidebar_ctx["generate_existing_solar"]
existing_solar_size_kw = _sidebar_ctx["existing_solar_size_kw"]
existing_solar_dc_ac = _sidebar_ctx["existing_solar_dc_ac"]
existing_solar_system_type = _sidebar_ctx["existing_solar_system_type"]
existing_solar_age = _sidebar_ctx["existing_solar_age"]
existing_solar_degradation = _sidebar_ctx["existing_solar_degradation"]
billing_engine = _sidebar_ctx["billing_engine"]
utility_name = _sidebar_ctx["utility_name"]
fetch_rates_btn = _sidebar_ctx["fetch_rates_btn"]
selected_rate_name = _sidebar_ctx["selected_rate_name"]
selected_label = _sidebar_ctx["selected_label"]
load_tariff_btn = _sidebar_ctx["load_tariff_btn"]
ecc_fetch_btn = _sidebar_ctx["ecc_fetch_btn"]
ecc_load_json_btn = _sidebar_ctx["ecc_load_json_btn"]
nem_switch = _sidebar_ctx["nem_switch"]
nem_regime_1 = _sidebar_ctx["nem_regime_1"]
nem_regime_2 = _sidebar_ctx["nem_regime_2"]
num_years_1 = _sidebar_ctx["num_years_1"]
nsc_rate = _sidebar_ctx["nsc_rate"]
nbc_rate = _sidebar_ctx["nbc_rate"]
billing_option = _sidebar_ctx["billing_option"]
export_method = _sidebar_ctx["export_method"]
selected_export_profile = _sidebar_ctx["selected_export_profile"]
flat_rate = _sidebar_ctx["flat_rate"]
export_method_2 = _sidebar_ctx["export_method_2"]
selected_export_profile_2 = _sidebar_ctx["selected_export_profile_2"]
flat_rate_2 = _sidebar_ctx["flat_rate_2"]
battery_enabled = _sidebar_ctx["battery_enabled"]
battery_capacity_kwh = _sidebar_ctx["battery_capacity_kwh"]
rate_escalator = _sidebar_ctx["rate_escalator"]
load_escalator = _sidebar_ctx["load_escalator"]
compound_escalation = _sidebar_ctx["compound_escalation"]
cost_input_method = _sidebar_ctx["cost_input_method"]
system_cost = _sidebar_ctx["system_cost"]
_es_offset_monthly = _sidebar_ctx["_es_offset_monthly"]
_es_offset_annual = _sidebar_ctx["_es_offset_annual"]


# =============================================================================
# SAVE SIMULATION HANDLER (after sidebar so variables are available)
# =============================================================================
# Handle main-area Save Simulation button (triggers via session state)
if st.session_state.get("_pending_save_name"):
    save_btn = True
    sim_name = st.session_state.pop("_pending_save_name")

if save_btn and sim_name and st.session_state.get("billing_result") is not None:
    result_to_save = st.session_state["billing_result"]
    summary_to_save = build_savings_summary(result_to_save, system_cost)
    _save_rs_old_baseline = result_to_save.old_rate_annual_baseline if result_to_save.old_rate_annual_baseline is not None else None
    proj_to_save = build_annual_projection(
        result=result_to_save,
        system_cost=system_cost,
        rate_escalator_pct=rate_escalator,
        load_escalator_pct=load_escalator,
        years=system_life_years,
        export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
        nem_regime_1=nem_regime_1,
        nem_regime_2=nem_regime_2 if nem_switch else None,
        num_years_1=num_years_1 if nem_switch else None,
        export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
        cod_year=cod_year,
        degradation_pct=annual_degradation_pct,
        nbc_rate_2=st.session_state.get("nbc_rate_2", 0.0) if nem_switch else 0.0,
        nsc_rate_2=st.session_state.get("nsc_rate_2", 0.0) if nem_switch else 0.0,
        compound_escalation=compound_escalation,
        rate_shift_old_baseline=_save_rs_old_baseline,
        existing_solar_offset_kwh=_es_offset_annual,
    )

    # Build extra battery data for saved view parity
    extra_save: dict[str, object] = {"has_battery": False}
    pv_only_res = st.session_state.get("billing_result_pv_only")
    batt_res = st.session_state.get("billing_result_batt")
    if batt_res is not None and pv_only_res is not None:
        extra_save["has_battery"] = True
        extra_save["monthly_summary_pv_only"] = pv_only_res.monthly_summary.to_dict(orient="records")
        extra_save["monthly_summary_batt"] = batt_res.monthly_summary.to_dict(orient="records")
        extra_save["summary_pv_only"] = build_savings_summary(pv_only_res, system_cost)
        extra_save["summary_batt"] = build_savings_summary(batt_res, system_cost)
        extra_save["projection_pv_only"] = build_annual_projection(
            result=pv_only_res, system_cost=system_cost,
            rate_escalator_pct=rate_escalator, load_escalator_pct=load_escalator,
            years=system_life_years,
            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
            cod_year=cod_year,
            degradation_pct=annual_degradation_pct,
            nbc_rate_2=st.session_state.get("nbc_rate_2", 0.0) if nem_switch else 0.0,
            nsc_rate_2=st.session_state.get("nsc_rate_2", 0.0) if nem_switch else 0.0,
            compound_escalation=compound_escalation,
            rate_shift_old_baseline=_save_rs_old_baseline,
            existing_solar_offset_kwh=_es_offset_annual,
        ).to_dict(orient="records")
        extra_save["projection_batt"] = build_annual_projection(
            result=batt_res, system_cost=system_cost,
            rate_escalator_pct=rate_escalator, load_escalator_pct=load_escalator,
            years=system_life_years,
            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
            result_pv_only=pv_only_res,
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
            cod_year=cod_year,
            degradation_pct=annual_degradation_pct,
            nbc_rate_2=st.session_state.get("nbc_rate_2", 0.0) if nem_switch else 0.0,
            nsc_rate_2=st.session_state.get("nsc_rate_2", 0.0) if nem_switch else 0.0,
            compound_escalation=compound_escalation,
            rate_shift_old_baseline=_save_rs_old_baseline,
            existing_solar_offset_kwh=_es_offset_annual,
        ).to_dict(orient="records")

        batt_cap = st.session_state.get("battery_capacity_kwh", 0)
        batt_cfg = st.session_state.get("battery_config")
        extra_save["battery_capacity_kwh"] = batt_cap
        extra_save["battery_hours"] = batt_cfg.battery_hours if batt_cfg else 4.0
        extra_save["battery_kpis"] = build_battery_kpi_summary(pv_only_res, batt_res, batt_cap)

        # Scenario comparison table
        extra_save["scenario_comparison"] = {
            "no_solar_bill": round(result_to_save.annual_bill_without_solar, 0),
            "pv_only_bill": round(pv_only_res.annual_bill_with_solar, 0),
            "pv_only_energy": round(pv_only_res.annual_energy_cost, 0),
            "pv_only_demand": round(pv_only_res.annual_demand_cost, 0),
            "pv_only_export": round(pv_only_res.annual_export_credit, 0),
            "pv_only_savings": round(pv_only_res.annual_savings, 0),
            "batt_bill": round(batt_res.annual_bill_with_solar, 0),
            "batt_energy": round(batt_res.annual_energy_cost, 0),
            "batt_demand": round(batt_res.annual_demand_cost, 0),
            "batt_export": round(batt_res.annual_export_credit, 0),
            "batt_savings": round(batt_res.annual_savings, 0),
            "battery_value": round(pv_only_res.annual_bill_with_solar - batt_res.annual_bill_with_solar, 0),
        }
        if st.session_state.get("rate_shift_enabled") and pv_only_res.rate_shift_annual_savings is not None:
            extra_save["scenario_comparison"]["pv_only_rate_shift_savings"] = round(pv_only_res.rate_shift_annual_savings, 0)
            extra_save["scenario_comparison"]["batt_rate_shift_savings"] = round(batt_res.rate_shift_annual_savings, 0)
            extra_save["scenario_comparison"]["pv_only_total_savings"] = round(pv_only_res.annual_savings + pv_only_res.rate_shift_annual_savings, 0)
            extra_save["scenario_comparison"]["batt_total_savings"] = round(batt_res.annual_savings + batt_res.rate_shift_annual_savings, 0)

        sizing_res = st.session_state.get("sizing_result")
        if sizing_res is not None:
            extra_save["sizing_table"] = sizing_res.table.to_dict(orient="records")
            extra_save["best_size_kwh"] = sizing_res.best_size_kwh

    # Grid exchange data — compute peak period from tariff
    _sv_tariff = st.session_state["tariff"]
    _sv_peak_idx = 0
    if _sv_tariff and _sv_tariff.energy_rate_structure:
        _sv_max_rate = 0.0
        for _i, _t in enumerate(_sv_tariff.energy_rate_structure):
            if _t and _t[0]["effective_rate"] > _sv_max_rate:
                _sv_max_rate = _t[0]["effective_rate"]
                _sv_peak_idx = _i
    _, ge_raw_save = build_grid_exchange_summary(result_to_save, _sv_peak_idx)
    extra_save["grid_exchange"] = ge_raw_save.to_dict(orient="records")
    if extra_save.get("has_battery") and pv_only_res is not None and batt_res is not None:
        _, ge_raw_pv = build_grid_exchange_summary(pv_only_res, _sv_peak_idx)
        _, ge_raw_bt = build_grid_exchange_summary(batt_res, _sv_peak_idx)
        extra_save["grid_exchange_pv_only"] = ge_raw_pv.to_dict(orient="records")
        extra_save["grid_exchange_batt"] = ge_raw_bt.to_dict(orient="records")

    # --- Serialize prerequisites for Edit Simulation ---
    _tariff_obj = st.session_state["tariff"]
    _tariff_dict = asdict(_tariff_obj) if _tariff_obj else None
    _prod_list = st.session_state["production_8760"].tolist() if st.session_state["production_8760"] is not None else None
    _load_list = st.session_state["load_8760"].tolist() if st.session_state["load_8760"] is not None else None
    _export_list = st.session_state["export_rates"].tolist() if st.session_state["export_rates"] is not None else None

    _batt_cfg = st.session_state.get("battery_config")

    # Save section 2 export rates when NEM switch is on
    if nem_switch:
        _export_rates_2 = st.session_state.get("export_rates_2")
        _export_multiyear_2 = st.session_state.get("export_rates_multiyear_2")
        extra_save["export_rates_2"] = _export_rates_2.tolist() if _export_rates_2 is not None else None
        extra_save["export_rates_multiyear_2"] = (
            {k: list(v.values) for k, v in _export_multiyear_2.items()}
            if _export_multiyear_2 else None
        )

    # Save NEM-A meter data when in aggregation mode
    _save_load_mode = st.session_state.get("load_mode", "Single Meter")
    if _save_load_mode == "NEM-A Aggregation":
        extra_save["nema_meters"] = st.session_state.get("nema_meters", [])
        _nema_meter_loads = st.session_state.get("nema_meter_loads", {})
        extra_save["nema_meter_loads"] = {
            str(k): list(v.values) for k, v in _nema_meter_loads.items()
        }
        _nema_meter_tariffs = st.session_state.get("nema_meter_tariffs", {})
        if _nema_meter_tariffs:
            extra_save["nema_meter_tariffs"] = {
                str(k): asdict(v) for k, v in _nema_meter_tariffs.items()
            }

    # Save existing solar production profile and raw (pre-adjustment) load
    _es_prod_save = st.session_state.get("existing_solar_production_8760")
    if _es_prod_save is not None:
        extra_save["existing_solar_production_8760"] = list(_es_prod_save.values)
    _raw_load_save = st.session_state.get("_raw_load_8760")
    if _raw_load_save is not None:
        extra_save["raw_load_8760"] = list(_raw_load_save.values)
    _raw_nema_save = st.session_state.get("_raw_nema_meter_loads")
    if _raw_nema_save:
        extra_save["raw_nema_meter_loads"] = {
            str(k): list(v.values) for k, v in _raw_nema_save.items()
        }

    # Phase 3+4: PPAs and Proposals travel with the simulation so reopening
    # a saved sim restores the full deal context (not just the billing run).
    _saved_ppas = st.session_state.get("saved_ppa_scenarios") or {}
    if _saved_ppas:
        extra_save["saved_ppa_scenarios"] = _saved_ppas
    _proposals_store = st.session_state.get("proposals") or {}
    if _proposals_store:
        extra_save["proposals"] = _proposals_store
    _active_proposal_save = st.session_state.get("active_proposal_id")
    if _active_proposal_save:
        extra_save["active_proposal_id"] = _active_proposal_save

    # Track the simulation name for Proposal scoping / GCS persistence.
    st.session_state["_active_simulation_name"] = sim_name
    st.session_state["_last_loaded_simulation_name"] = sim_name

    _save_simulation(
        name=sim_name,
        result=result_to_save,
        summary=summary_to_save,
        projection_df=proj_to_save,
        inputs={
            "cod_date": cod_date.isoformat(),
            "location": location_input,
            "system_size_kw": system_size_kw,
            "dc_ac_ratio": dc_ac_ratio,
            "system_type": system_type,
            "utility": utility_name,
            "rate_escalator": rate_escalator,
            "load_escalator": load_escalator,
            "cost_input_method": cost_input_method,
            "system_cost": system_cost,
            "battery_enabled": battery_enabled,
            "battery_capacity_kwh": st.session_state.get("battery_capacity_kwh", 0),
            "battery_hours": _batt_cfg.battery_hours if _batt_cfg else 4.0,
            "battery_config": asdict(_batt_cfg) if _batt_cfg else None,
            "system_life_years": system_life_years,
            "nem_regime_1": nem_regime_1,
            "nem_switch": nem_switch,
            "nem_regime_2": nem_regime_2 if nem_switch else None,
            "nem_years_1": num_years_1 if nem_switch else None,
            "billing_engine": billing_engine,
            "ecc_tariff_metadata": st.session_state.get("ecc_tariff_metadata") if billing_engine == "ECC" else None,
            "nbc_rate": nbc_rate,
            "nsc_rate": nsc_rate,
            "billing_option": billing_option,
            "load_mode": _save_load_mode,
            "nema_utility": st.session_state.get("nema_utility", "PG&E") if _save_load_mode == "NEM-A Aggregation" else None,
            "existing_solar_enabled": st.session_state.get("existing_solar_enabled", False),
            "existing_solar_size_kw": st.session_state.get("sb_existing_solar_size", 100.0),
            "existing_solar_system_type": st.session_state.get("sb_existing_solar_type", "Fixed Tilt (Ground Mount)"),
            "existing_solar_dc_ac_ratio": st.session_state.get("sb_existing_solar_dc_ac", 1.2),
            "existing_solar_age": st.session_state.get("sb_existing_solar_age", 10),
            "existing_solar_degradation_pct": st.session_state.get("sb_existing_solar_degradation", 0.5),
            "existing_solar_nema_meters": st.session_state.get("existing_solar_nema_meters", []),
            "rate_shift_enabled": st.session_state.get("rate_shift_enabled", False),
        },
        production_8760=_prod_list,
        load_8760=_load_list,
        export_rates=_export_list,
        tariff_data=_tariff_dict,
        **extra_save,
    )
    st.success(f"Simulation '{sim_name}' saved!")
    st.rerun()
elif save_btn and st.session_state.get("billing_result") is None:
    st.warning("No simulation results to save. Run a simulation first using the **Run Simulation** button below.")


# =============================================================================
# PRODUCTION PROFILE GENERATION
# =============================================================================
if generate_prod and lat is not None and lon is not None:
    api_key = _get_secret("NREL_API_KEY")
    if not api_key:
        st.error("NREL_API_KEY not found. Add `NREL_API_KEY` to your Streamlit secrets (Manage app → Settings → Secrets), or to a local `.env` for local runs. Get a free key at https://developer.nlr.gov/signup/")
    else:
        with st.spinner("Calling PVWatts API..."):
            try:
                config = PVSystemConfig(
                    system_capacity_kw_dc=system_size_kw,
                    dc_ac_ratio=dc_ac_ratio,
                    array_type=get_array_type_code(system_type),
                    losses=system_losses_pct,
                    module_type=module_type_code,
                )
                prod, summary = fetch_production_8760(api_key, lat, lon, config, start_year=cod_year)
                st.session_state["production_8760"] = prod
                st.session_state["production_summary"] = summary
                st.sidebar.success(
                    f"Production generated: {summary['ac_annual_kwh']:,.0f} kWh/yr "
                    f"(CF: {summary['capacity_factor']:.1f}%)"
                )
            except Exception as e:
                st.error(f"PVWatts error: {e}")


# =============================================================================
# EXISTING SOLAR PROFILE GENERATION
# =============================================================================
if generate_existing_solar and lat is not None and lon is not None:
    api_key = _get_secret("NREL_API_KEY")
    if not api_key:
        st.error("NREL_API_KEY not found.")
    else:
        with st.spinner("Generating existing solar profile via PVWatts..."):
            try:
                _es_config = PVSystemConfig(
                    system_capacity_kw_dc=existing_solar_size_kw,
                    dc_ac_ratio=existing_solar_dc_ac,
                    array_type=get_array_type_code(existing_solar_system_type),
                )
                _es_prod, _es_summary = fetch_production_8760(
                    api_key, lat, lon, _es_config, start_year=cod_year
                )
                # Apply compound degradation
                _es_degradation_factor = (1 - existing_solar_degradation / 100) ** existing_solar_age
                _es_prod = _es_prod * _es_degradation_factor
                st.session_state["existing_solar_production_8760"] = _es_prod
                st.sidebar.success(
                    f"Existing solar profile: {_es_prod.sum():,.0f} kWh/yr "
                    f"(degraded {existing_solar_age}yr @ {existing_solar_degradation}%/yr)"
                )
            except Exception as e:
                st.error(f"Existing solar PVWatts error: {e}")


# =============================================================================
# LOAD PROFILE PARSING
# =============================================================================
if load_mode == "Single Meter":
    # Only handle ad-hoc CSV upload; saved profiles already loaded by sidebar selection
    if load_file is not None:
        try:
            df_load = pd.read_csv(load_file)
            load_values = _parse_8760_csv(df_load)
            dt_index = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
            st.session_state["load_8760"] = pd.Series(load_values, index=dt_index, name="load_kwh")
            st.session_state["_raw_load_8760"] = st.session_state["load_8760"].copy()
            annual_load = load_values.sum()
            peak_load = load_values.max()
            load_factor = annual_load / (peak_load * 8760) * 100 if peak_load > 0 else 0
            st.sidebar.success(
                f"Load profile loaded: {annual_load:,.0f} kWh/yr, "
                f"Peak: {peak_load:,.1f} kW, LF: {load_factor:.1f}%"
            )
        except Exception as e:
            st.error(f"Error reading load file: {e}")
else:
    # NEM-A: preserve session data loaded by _load_nema_profile_into_session
    _prev_loads = st.session_state.get("nema_meter_loads", {})
    st.session_state["nema_meter_loads"] = _prev_loads
    _prev_raw = st.session_state.get("_raw_nema_meter_loads", {})
    st.session_state["_raw_nema_meter_loads"] = _prev_raw

# =============================================================================
# EXISTING SOLAR LOAD ADJUSTMENT
# =============================================================================
# Bootstrap raw loads from NEM-A profile file if not yet set (migration).
# This ensures sessions started before the raw-load fix get clean base data.
if (
    load_mode != "Single Meter"
    and not st.session_state.get("_raw_nema_meter_loads")
    and st.session_state.get("nema_meter_loads")
):
    _bootstrap_name = st.session_state.get("_last_loaded_sidebar_profile")
    if _bootstrap_name:
        try:
            _bp = os.path.join(NEMA_PROFILES_DIR, f"{_bootstrap_name}.json")
            if os.path.exists(_bp):
                with open(_bp) as _bf:
                    _bd = json.load(_bf)
                _by = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
                _bdt = pd.date_range(f"{_by}-01-01", periods=8760, freq="h")
                _raw_boots: dict[int, pd.Series] = {}
                for _bi, _bm in enumerate(_bd.get("meters", [])):
                    if _bm.get("load_8760"):
                        _raw_boots[_bi] = pd.Series(_bm["load_8760"], index=_bdt, name="load_kwh")
                if _raw_boots:
                    st.session_state["_raw_nema_meter_loads"] = _raw_boots
                    st.session_state["nema_meter_loads"] = {k: v.copy() for k, v in _raw_boots.items()}
                    for _bi2, _bm2 in enumerate(st.session_state.get("nema_meters", [])):
                        if _bm2.get("is_generating") and _bi2 in _raw_boots:
                            st.session_state["load_8760"] = _raw_boots[_bi2].copy()
                            st.session_state["_raw_load_8760"] = _raw_boots[_bi2].copy()
                            break
        except Exception as e:
            logger.warning("Failed to bootstrap raw NEM-A loads: %s", e)

# Bootstrap raw load for single meter if missing
if (
    load_mode == "Single Meter"
    and st.session_state.get("_raw_load_8760") is None
    and st.session_state.get("load_8760") is not None
):
    _sb_profile_name = st.session_state.get("_last_loaded_sidebar_profile")
    if _sb_profile_name:
        try:
            _sb_df = _load_profile_csv(LOAD_PROFILES_DIR, _sb_profile_name)
            _sb_raw_vals = _parse_8760_csv(_sb_df)
            _sb_yr = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
            _sb_dtr = pd.date_range(f"{_sb_yr}-01-01", periods=8760, freq="h")
            _sb_raw_series = pd.Series(_sb_raw_vals, index=_sb_dtr, name="load_kwh")
            st.session_state["_raw_load_8760"] = _sb_raw_series
            st.session_state["load_8760"] = _sb_raw_series.copy()
        except Exception as e:
            logger.warning("Failed to bootstrap raw load: %s", e)

_es_enabled = st.session_state.get("existing_solar_enabled", False)
_es_production = st.session_state.get("existing_solar_production_8760")
if _es_enabled and _es_production is not None:
    if load_mode == "Single Meter":
        _raw_load = st.session_state.get("_raw_load_8760")
        if _raw_load is not None:
            st.session_state["load_8760"] = adjust_load_single_meter(
                _raw_load, _es_production
            )
    else:
        _es_selected = st.session_state.get("existing_solar_nema_meters", [])
        _raw_nema = st.session_state.get("_raw_nema_meter_loads", {})
        if _es_selected and _raw_nema:
            _adjusted_nema = adjust_loads_nema(_raw_nema, _es_production, _es_selected)
            st.session_state["nema_meter_loads"] = _adjusted_nema
            # Update load_8760 if generating meter was adjusted
            for _ami, _aminfo in enumerate(st.session_state.get("nema_meters", [])):
                if _aminfo.get("is_generating") and _ami in _adjusted_nema:
                    st.session_state["load_8760"] = _adjusted_nema[_ami]
                    break


# =============================================================================
# RATE SCHEDULE FETCHING (handlers for sidebar buttons)
# =============================================================================
if fetch_rates_btn:
    with st.spinner(f"Fetching rates for {utility_name}..."):
        try:
            rates = fetch_available_rates(utility_name)
            st.session_state["available_rates"] = rates
            st.sidebar.success(f"Found {len(rates)} rate schedules.")
        except Exception as e:
            st.error(f"Error fetching rates: {e}")

if load_tariff_btn and selected_label:
    with st.spinner("Loading tariff details..."):
        try:
            tariff = fetch_tariff_detail(selected_label)
            st.session_state["tariff"] = tariff
            st.sidebar.success(f"Tariff loaded: {tariff.name}")
        except Exception as e:
            st.error(f"Error loading tariff: {e}")

# --- Management tab: Fetch rates handler ---
if st.session_state.get("_pending_mgmt_fetch_rates"):
    _mgmt_fetch_util = st.session_state.pop("_pending_mgmt_fetch_rates")
    with st.spinner(f"Fetching rates for {_mgmt_fetch_util}..."):
        try:
            rates = fetch_available_rates(_mgmt_fetch_util)
            st.session_state["available_rates"] = rates
            st.success(f"Found {len(rates)} rate schedules.")
        except Exception as e:
            st.error(f"Error fetching rates: {e}")

# --- Management tab: Per-meter tariff load handlers ---
for _mgmt_ti in range(len(st.session_state.get("nema_meters", []))):
    _mgmt_tariff_key = f"_pending_mgmt_nema_tariff_{_mgmt_ti}"
    _mgmt_tariff_label = st.session_state.get(_mgmt_tariff_key)
    if _mgmt_tariff_label:
        st.session_state.pop(_mgmt_tariff_key)
        with st.spinner(f"Loading tariff for meter {_mgmt_ti}..."):
            try:
                _mgmt_tariff = fetch_tariff_detail(_mgmt_tariff_label)
                if "nema_meter_tariffs" not in st.session_state:
                    st.session_state["nema_meter_tariffs"] = {}
                st.session_state["nema_meter_tariffs"][_mgmt_ti] = _mgmt_tariff
                _mgmt_meter_name = st.session_state.get("nema_meters", [])[_mgmt_ti].get("name", f"Meter {_mgmt_ti}")
                st.success(f"Tariff loaded for {_mgmt_meter_name}: {_mgmt_tariff.name}")
            except Exception as e:
                st.error(f"Error loading per-meter tariff: {e}")

# --- Per-meter tariff load handlers (NEM-A sidebar) ---
for _pmt_load_i in range(len(st.session_state.get("nema_meters", []))):
    _pmt_pending_key = f"_pending_nema_tariff_load_{_pmt_load_i}"
    _pmt_pending_label = st.session_state.get(_pmt_pending_key)
    if _pmt_pending_label:
        st.session_state.pop(_pmt_pending_key)
        with st.spinner(f"Loading tariff for meter {_pmt_load_i}..."):
            try:
                _pmt_tariff = fetch_tariff_detail(_pmt_pending_label)
                if "nema_meter_tariffs" not in st.session_state:
                    st.session_state["nema_meter_tariffs"] = {}
                st.session_state["nema_meter_tariffs"][_pmt_load_i] = _pmt_tariff
                _pmt_meter_name = st.session_state.get("nema_meters", [])[_pmt_load_i].get("name", f"Meter {_pmt_load_i}")
                st.sidebar.success(f"Tariff loaded for {_pmt_meter_name}: {_pmt_tariff.name}")
            except Exception as e:
                st.error(f"Error loading per-meter tariff: {e}")

# --- Rate Shift tariff load handlers ---
if st.session_state.get("_pending_rate_shift_load"):
    _rs_load_label = st.session_state.pop("_pending_rate_shift_load")
    with st.spinner("Loading old tariff for rate shift..."):
        try:
            _rs_tariff = fetch_tariff_detail(_rs_load_label)
            st.session_state["rate_shift_old_tariff"] = _rs_tariff
            st.sidebar.success(f"Old tariff loaded: {_rs_tariff.name}")
        except Exception as e:
            st.error(f"Error loading old tariff: {e}")

for _rs_nema_i in range(len(st.session_state.get("nema_meters", []))):
    _rs_nema_key = f"_pending_nema_rs_tariff_{_rs_nema_i}"
    _rs_nema_label = st.session_state.get(_rs_nema_key)
    if _rs_nema_label:
        st.session_state.pop(_rs_nema_key)
        with st.spinner(f"Loading old tariff for meter {_rs_nema_i}..."):
            try:
                _rs_nema_tariff = fetch_tariff_detail(_rs_nema_label)
                if "nema_rate_shift_tariffs" not in st.session_state:
                    st.session_state["nema_rate_shift_tariffs"] = {}
                st.session_state["nema_rate_shift_tariffs"][_rs_nema_i] = _rs_nema_tariff
                _rs_meter_name = st.session_state.get("nema_meters", [])[_rs_nema_i].get("name", f"Meter {_rs_nema_i}")
                st.sidebar.success(f"Old tariff loaded for {_rs_meter_name}: {_rs_nema_tariff.name}")
            except Exception as e:
                st.error(f"Error loading old tariff: {e}")

# ECC rate shift load handler
if st.session_state.get("_pending_ecc_rs_load"):
    _ecc_rs_file = st.session_state.pop("_pending_ecc_rs_load")
    with st.spinner("Loading old ECC tariff..."):
        try:
            import tempfile
            _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            _tmp.write(_ecc_rs_file.read())
            _tmp.close()
            _rs_calc, _rs_tdata = load_ecc_tariff_from_json(_tmp.name)
            os.remove(_tmp.name)
            st.session_state["rate_shift_old_ecc_calculator"] = _rs_calc
            st.sidebar.success("Old ECC tariff loaded for rate shift.")
        except Exception as e:
            st.error(f"Error loading old ECC tariff: {e}")

# --- Regime-2 (post-transition) tariff load handlers ---
if st.session_state.get("_pending_regime2_tariff_load"):
    _r2_load_label = st.session_state.pop("_pending_regime2_tariff_load")
    with st.spinner("Loading post-transition tariff..."):
        try:
            _r2_tariff = fetch_tariff_detail(_r2_load_label)
            st.session_state["regime_2_tariff"] = _r2_tariff
            st.sidebar.success(f"Post-transition tariff loaded: {_r2_tariff.name}")
        except Exception as e:
            st.error(f"Error loading post-transition tariff: {e}")

if st.session_state.get("_pending_regime2_ecc_saved_path"):
    _r2_ecc_path = st.session_state.pop("_pending_regime2_ecc_saved_path")
    if _r2_ecc_path and os.path.isfile(_r2_ecc_path):
        with st.spinner("Loading post-transition ECC tariff..."):
            try:
                _r2_calc, _r2_tdata = load_ecc_tariff_from_json(_r2_ecc_path)
                st.session_state["regime_2_ecc_calculator"] = _r2_calc
                st.session_state["regime_2_ecc_tariff_data"] = _r2_tdata
                st.sidebar.success("Post-transition ECC tariff loaded.")
            except Exception as e:
                st.error(f"Error loading post-transition ECC tariff: {e}")

if st.session_state.get("_pending_regime2_ecc_upload"):
    _r2_ecc_file = st.session_state.pop("_pending_regime2_ecc_upload")
    with st.spinner("Loading post-transition ECC tariff..."):
        try:
            import tempfile
            _r2_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            _r2_tmp.write(_r2_ecc_file.read())
            _r2_tmp.close()
            _r2_calc, _r2_tdata = load_ecc_tariff_from_json(_r2_tmp.name)
            os.remove(_r2_tmp.name)
            st.session_state["regime_2_ecc_calculator"] = _r2_calc
            st.session_state["regime_2_ecc_tariff_data"] = _r2_tdata
            st.sidebar.success("Post-transition ECC tariff loaded.")
        except Exception as e:
            st.error(f"Error loading post-transition ECC tariff: {e}")

# --- ECC tariff fetch/load handlers ---
if ecc_fetch_btn:
    _ecc_eia_id = UTILITY_EIA_IDS.get(utility_name, 0)
    with st.spinner(f"Fetching ECC tariff for {utility_name}..."):
        try:
            calc, tdata = fetch_and_populate_ecc_tariff(
                utility_id=_ecc_eia_id,
                sector=st.session_state.get("ecc_sector", "Commercial"),
                tariff_rate_filter=st.session_state.get("ecc_rate_filter", ""),
                distrib_level=st.session_state.get("ecc_distrib", "Secondary"),
                phase_wiring=st.session_state.get("ecc_phase", "Three"),
                tou=st.session_state.get("ecc_tou", True),
                pdp=st.session_state.get("ecc_pdp", False),
            )
            st.session_state["ecc_cost_calculator"] = calc
            st.session_state["ecc_tariff_data"] = tdata
            _tnames = []
            if isinstance(tdata, list):
                for td in tdata[:10]:
                    if isinstance(td, dict):
                        _tnames.append(td.get("name", td.get("label", "Unknown")))
            st.session_state["ecc_tariff_metadata"] = {
                "source": "OpenEI API",
                "utility_id": _ecc_eia_id,
                "utility": utility_name,
                "sector": st.session_state.get("ecc_sector", "Commercial"),
                "rate_filter": st.session_state.get("ecc_rate_filter", ""),
                "distrib_level": st.session_state.get("ecc_distrib", "Secondary"),
                "phase_wiring": st.session_state.get("ecc_phase", "Three"),
                "tou": st.session_state.get("ecc_tou", True),
                "pdp": st.session_state.get("ecc_pdp", False),
                "num_tariffs": len(tdata) if isinstance(tdata, list) else 0,
                "tariff_names": _tnames,
            }
            # Save a copy to ECC_TARIFFS_DIR for future "Use Saved Tariff"
            import json as _json_mod
            _rate_tag = st.session_state.get("ecc_rate_filter", "").strip()
            _save_label = _rate_tag if _rate_tag else utility_name
            _ecc_bytes = _json_mod.dumps(tdata).encode("utf-8")
            save_profile_bytes(ECC_TARIFFS_DIR, _DIR_TO_GCS_PREFIX[ECC_TARIFFS_DIR], _save_label, _ecc_bytes, ".json")
            st.sidebar.success(f"ECC tariff loaded ({len(tdata)} block(s)).")
        except Exception as e:
            st.error(f"ECC tariff fetch error: {e}")

if ecc_load_json_btn:
    # --- "Use Saved Tariff" path ---
    _ecc_saved = st.session_state.pop("_ecc_saved_path", None)
    if _ecc_saved and os.path.isfile(_ecc_saved):
        try:
            calc, tdata = load_ecc_tariff_from_json(_ecc_saved)
            st.session_state["ecc_cost_calculator"] = calc
            st.session_state["ecc_tariff_data"] = tdata
            _tnames = []
            if isinstance(tdata, list):
                for td in tdata[:10]:
                    if isinstance(td, dict):
                        _tnames.append(td.get("name", td.get("label", "Unknown")))
            _fname = os.path.splitext(os.path.basename(_ecc_saved))[0]
            st.session_state["ecc_tariff_metadata"] = {
                "source": f"Saved tariff: {_fname}",
                "utility_id": "N/A",
                "utility": utility_name,
                "sector": "N/A",
                "rate_filter": "N/A",
                "num_tariffs": len(tdata) if isinstance(tdata, list) else 0,
                "tariff_names": _tnames,
            }
            st.sidebar.success(f"ECC tariff loaded from saved file ({len(tdata)} block(s)).")
        except Exception as e:
            st.error(f"ECC saved tariff load error: {e}")
    else:
        # --- "Upload JSON" path ---
        _ecc_uploaded = st.session_state.get("ecc_json_upload")
        if _ecc_uploaded is not None:
            import tempfile as _tmpmod
            try:
                with _tmpmod.TemporaryDirectory() as _tmp_dir:
                    _safe_ecc_name = sanitize_filename(_ecc_uploaded.name)
                    _tmp_path = os.path.join(_tmp_dir, _safe_ecc_name)
                    with open(_tmp_path, "wb") as _f:
                        _f.write(_ecc_uploaded.getvalue())
                    calc, tdata = load_ecc_tariff_from_json(_tmp_path)
                    st.session_state["ecc_cost_calculator"] = calc
                    st.session_state["ecc_tariff_data"] = tdata
                    _tnames = []
                    if isinstance(tdata, list):
                        for td in tdata[:10]:
                            if isinstance(td, dict):
                                _tnames.append(td.get("name", td.get("label", "Unknown")))
                    st.session_state["ecc_tariff_metadata"] = {
                        "source": f"JSON upload: {_ecc_uploaded.name}",
                        "utility_id": "N/A",
                        "utility": utility_name,
                        "sector": "N/A",
                        "rate_filter": "N/A",
                        "num_tariffs": len(tdata) if isinstance(tdata, list) else 0,
                        "tariff_names": _tnames,
                    }
                    # Save a copy to ECC_TARIFFS_DIR for future "Use Saved Tariff"
                    _save_name = os.path.splitext(_safe_ecc_name)[0]
                    with open(_tmp_path, "rb") as _rb:
                        _ecc_upload_bytes = _rb.read()
                    save_profile_bytes(ECC_TARIFFS_DIR, _DIR_TO_GCS_PREFIX[ECC_TARIFFS_DIR], _save_name, _ecc_upload_bytes, ".json")
                st.sidebar.success(f"ECC tariff loaded from JSON ({len(tdata)} block(s)).")
            except Exception as e:
                st.error(f"ECC JSON load error: {e}")
        else:
            st.warning("Upload a tariff JSON file first.")


# =============================================================================
# EXPORT RATE LOADING (handlers for sidebar selections)
# =============================================================================
def _handle_export_rate_loading(
    method, selected_profile, flat_val,
    upload_key, rates_key, multiyear_key,
    label_suffix="",
    start_year: int = 2026,
):
    """Load export rates based on selected method and store in session state."""
    if method == "Use saved profile" and selected_profile:
        try:
            df_exp = _load_profile_csv(EXPORT_PROFILES_DIR, selected_profile)
            multiyear = parse_multiyear_export_rates(df_exp, start_year=start_year)
            first_year_key = min(multiyear.keys())
            st.session_state[rates_key] = multiyear[first_year_key]
            if len(multiyear) > 1:
                st.session_state[multiyear_key] = multiyear
                st.sidebar.success(
                    f"Export profile{label_suffix} loaded: '{selected_profile}' "
                    f"({len(multiyear)}-year forecast, {first_year_key}-{max(multiyear.keys())})"
                )
            else:
                st.session_state[multiyear_key] = None
                st.sidebar.success(f"Export profile{label_suffix} loaded: '{selected_profile}'")
        except Exception as e:
            st.sidebar.error(f"Error loading export profile: {e}")

    elif method == "Upload CSV":
        if upload_key in st.session_state and st.session_state[upload_key] is not None:
            try:
                year1_rates, multiyear = load_acc_from_upload(st.session_state[upload_key], start_year=start_year)
                st.session_state[rates_key] = year1_rates
                st.session_state[multiyear_key] = multiyear
                if multiyear is not None:
                    n_years = len(multiyear)
                    st.sidebar.success(f"Export rates{label_suffix} loaded: {n_years}-year multi-year CSV.")
                else:
                    st.sidebar.success(f"Export rates{label_suffix} loaded from uploaded CSV.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    elif method == "Flat rate ($/kWh)" and flat_val is not None:
        st.session_state[multiyear_key] = None
        st.session_state[rates_key] = create_flat_export_rates(flat_val, start_year=start_year)


# Section 1 (only when NEM-3/NVBT — NEM-1/NEM-2 don't need export rates)
if export_method is not None:
    _handle_export_rate_loading(
        method=export_method,
        selected_profile=selected_export_profile,
        flat_val=flat_rate,
        upload_key="sidebar_export_upload",
        rates_key="export_rates",
        multiyear_key="export_rates_multiyear",
        start_year=cod_year,
    )

# Section 2 (only when NEM switch is on and NEM-3)
if nem_switch and export_method_2 is not None:
    _handle_export_rate_loading(
        method=export_method_2,
        selected_profile=selected_export_profile_2,
        flat_val=flat_rate_2,
        upload_key="sidebar_export_upload_2",
        rates_key="export_rates_2",
        multiyear_key="export_rates_multiyear_2",
        label_suffix=" (Section 2)",
        start_year=cod_year,
    )


# =============================================================================
# RUN SIMULATION
# =============================================================================
_nema_mode = st.session_state.get("load_mode") == "NEM-A Aggregation"

if _nema_mode:
    # NEM-A: check that all meters have load profiles
    _nema_loads_ready = st.session_state["load_8760"] is not None  # generating meter
    _nema_all_loads = st.session_state.get("nema_meter_loads", {})
    _nema_meter_list = st.session_state.get("nema_meters", [])
    for _mi, _minfo in enumerate(_nema_meter_list):
        if not _minfo.get("is_generating") and _mi not in _nema_all_loads:
            _nema_loads_ready = False
            break
    _load_ready = _nema_loads_ready
else:
    _load_ready = st.session_state["load_8760"] is not None

# Check per-meter tariffs for NEM-A with Custom engine
_tariff_ready = (
    st.session_state["tariff"] is not None
    if billing_engine == "Custom"
    else st.session_state.get("ecc_cost_calculator") is not None
)
if _tariff_ready and billing_engine == "Custom" and _nema_mode:
    _nema_meter_tariffs = st.session_state.get("nema_meter_tariffs", {})
    for _rc_i, _rc_m in enumerate(st.session_state.get("nema_meters", [])):
        if not _rc_m.get("is_generating") and not _rc_m.get("use_gen_tariff", True):
            if _rc_i not in _nema_meter_tariffs:
                _tariff_ready = False
                break

ready_checks = {
    "Production profile": st.session_state["production_8760"] is not None,
    "Load profile": _load_ready,
    "Tariff schedule": _tariff_ready,
    "Export rates": (
        st.session_state["export_rates"] is not None
        if nem_regime_1 == "NEM-3 / NVBT" and not _nema_mode
        else True  # Not needed for NEM-1/NEM-2 or NEM-A (exports valued at retail rate)
    ),
}

# Rate-shift guard: when enabled, the user must have loaded an old-tariff
# reference so the shift baseline is computable. Silent None-fallbacks used
# to leave rate_shift_annual_savings empty without explanation.
if st.session_state.get("rate_shift_enabled"):
    if billing_engine == "Custom":
        ready_checks["Rate-shift old tariff"] = (
            st.session_state.get("rate_shift_old_tariff") is not None
        )
    else:
        ready_checks["Rate-shift old tariff"] = (
            st.session_state.get("rate_shift_old_ecc_calculator") is not None
        )
all_ready = all(ready_checks.values())

# Note (Phase 5): the Simulation Checklist UI that previously rendered here
# has been replaced by the sticky input-load tracker at the top of the
# sidebar. That gives the same information (which inputs are loaded) with
# less visual noise in the main pane.

_run_col, _save_col, _edit_col = st.columns(3)
with _run_col:
    run_sim = st.button("Run Simulation", type="primary", disabled=not all_ready, width="stretch")
with _save_col:
    _has_result = st.session_state.get("billing_result") is not None
    with st.popover("Save Simulation", use_container_width=True, disabled=not _has_result):
        _main_sim_name = st.text_input(
            "Simulation Name",
            placeholder="e.g., Ranch-500kW-AG1-SAT",
            key="main_sim_name_input",
        )
        if st.button("Save", disabled=not _main_sim_name, width="stretch", key="main_save_btn"):
            st.session_state["_pending_save_name"] = _main_sim_name
            st.rerun()
with _edit_col:
    _has_saved_view = st.session_state["saved_view"] is not None
    edit_sim = st.button(
        "Edit Simulation",
        disabled=not _has_saved_view,
        width="stretch",
        help="Populate sidebar with the saved simulation's inputs so you can tweak and re-run",
    )

# --- Edit Simulation handler ---
if edit_sim and _has_saved_view:
    populate_session_from_simulation(st.session_state, st.session_state["saved_view"])
    st.rerun()

def _run_simulation():
    """Execute the billing simulation based on current sidebar inputs."""
    st.session_state["active_mgmt_tab"] = None
    try:
      with st.spinner("Running simulation..."):
        if billing_engine == "ECC":
            # ============ ECC billing engine ============
            _ecc_export = st.session_state["export_rates"]
            if _ecc_export is None:
                _ecc_dt = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                _ecc_export = pd.Series(np.zeros(8760), index=_ecc_dt, name="export_rate_per_kwh")
                st.warning("No export rates loaded — export credits will be $0. Load ACC/avoided cost rates in Section 5 for accurate NEM-3/NVBT results.")
            result_pv_only = run_ecc_billing_simulation(
                load_8760=st.session_state["load_8760"],
                production_8760=st.session_state["production_8760"],
                cost_calculator=st.session_state["ecc_cost_calculator"],
                export_rates_8760=_ecc_export,
                tariff_data=st.session_state.get("ecc_tariff_data"),
                nsc_rate=nsc_rate,
                min_monthly_charge=getattr(st.session_state.get("tariff"), "min_monthly_charge", 0.0),
            )
            st.session_state["billing_result_pv_only"] = result_pv_only
            st.session_state["billing_result"] = result_pv_only
            st.session_state["billing_result_batt"] = None
            st.session_state["sizing_result"] = None

            # ECC battery dispatch
            if st.session_state.get("battery_enabled") and st.session_state.get("battery_config"):
                batt_cap = st.session_state.get("battery_capacity_kwh", 0)
                if batt_cap > 0:
                    result_batt = run_ecc_billing_simulation(
                        load_8760=st.session_state["load_8760"],
                        production_8760=st.session_state["production_8760"],
                        cost_calculator=st.session_state["ecc_cost_calculator"],
                        export_rates_8760=_ecc_export,
                        tariff_data=st.session_state.get("ecc_tariff_data"),
                        battery_config=st.session_state["battery_config"],
                        capacity_kwh=batt_cap,
                        monthly_dispatch=st.session_state.get("battery_fast_dispatch", True),
                        nsc_rate=nsc_rate,
                        min_monthly_charge=getattr(st.session_state.get("tariff"), "min_monthly_charge", 0.0),
                    )
                    st.session_state["billing_result"] = result_batt
                    st.session_state["billing_result_batt"] = result_batt
                    _check_battery_solver(result_batt)

            st.success("Simulation complete (ECC engine)!")

        else:
            # ============ Custom billing engine ============
            # For NEM-1/NEM-2, export rates are not used (valued at retail TOU),
            # but the function signature requires an array — provide zeros as placeholder.
            _export_rates_for_sim = st.session_state["export_rates"]
            if _export_rates_for_sim is None:
                # NEM-1/NEM-2 value exports at retail TOU (zeros placeholder).
                # NEM-A may also reach here if NEM-3 export rates weren't loaded.
                _dt_idx_placeholder = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                _export_rates_for_sim = pd.Series(
                    np.zeros(8760), index=_dt_idx_placeholder, name="export_rate_per_kwh",
                )

            # NEM params for the billing call. NSC rate flows for NEM-1/2 AND
            # NEM-3 / NVBT — under NBT, year-end NSC re-prices net surplus from
            # avg ACC down to wholesale NSC (per CPUC D.22-12-056).
            _nem_nbc = nbc_rate if nem_regime_1 == "NEM-2" else 0.0
            _nem_nsc = nsc_rate
            _nem_billing = billing_option if nem_regime_1 in ("NEM-1", "NEM-2") else "ABO"

            if st.session_state.get("load_mode") == "NEM-A Aggregation":
                # ============ NEM-A Aggregation path ============

                # Build MeterConfig list from session state
                _nema_meter_loads = st.session_state.get("nema_meter_loads", {})
                _nema_meters_info = st.session_state.get("nema_meters", [])
                _gen_tariff = st.session_state["tariff"]
                _meter_configs = []

                for _mi, _minfo in enumerate(_nema_meters_info):
                    if _minfo.get("is_generating"):
                        _m_load = st.session_state["load_8760"]
                        _m_tariff = _gen_tariff
                    else:
                        _m_load = _nema_meter_loads.get(_mi)
                        if _m_load is None:
                            raise ValueError(f"No load profile for meter '{_minfo['name']}'")
                        # Use generating meter's tariff if checkbox is set, else use per-meter tariff
                        if _minfo.get("use_gen_tariff", True):
                            _m_tariff = _gen_tariff
                        else:
                            _m_tariff = st.session_state.get("nema_meter_tariffs", {}).get(_mi)
                            if _m_tariff is None:
                                raise ValueError(
                                    f"No tariff loaded for meter '{_minfo['name']}'. "
                                    f"Load a tariff in Section 4 or check 'Use generating meter's tariff'."
                                )

                    _meter_configs.append(MeterConfig(
                        name=_minfo["name"],
                        load_8760=_m_load,
                        tariff=_m_tariff,
                        is_generating=_minfo.get("is_generating", False),
                    ))

                _nema_profile = NemAProfile(
                    utility=st.session_state.get("nema_utility", utility_name),
                    meters=_meter_configs,
                    nem_regime=nem_regime_1,
                    nbc_rate=_nem_nbc,
                    nsc_rate=_nem_nsc,
                    billing_option=_nem_billing,
                )

                # PV-only aggregation run (no battery)
                result_pv_only = run_aggregation_simulation(
                    profile=_nema_profile,
                    production_8760=st.session_state["production_8760"],
                    export_rates_8760=_export_rates_for_sim,
                )
                st.session_state["billing_result_pv_only"] = result_pv_only
                st.session_state["sizing_result"] = None

                # Battery dispatch (if enabled)
                if st.session_state["battery_enabled"] and st.session_state["battery_config"] is not None:
                    batt_cfg = st.session_state["battery_config"]
                    _use_monthly = st.session_state.get("battery_fast_dispatch", False)
                    batt_cap = st.session_state.get("battery_capacity_kwh", 0)

                    if batt_cap > 0:

                        # Use effective export price for battery dispatch
                        _dt_idx = cast(pd.DatetimeIndex, st.session_state["load_8760"].index)
                        _eff_export = compute_effective_export_price(_meter_configs, _dt_idx)
                        _eff_export_series = pd.Series(_eff_export, index=_dt_idx, name="export_rate_per_kwh")

                        result_batt = run_aggregation_simulation(
                            profile=_nema_profile,
                            production_8760=st.session_state["production_8760"],
                            export_rates_8760=_eff_export_series,
                            battery_config=batt_cfg,
                            capacity_kwh=batt_cap,
                            monthly_dispatch=_use_monthly,
                        )
                        st.session_state["billing_result"] = result_batt
                        st.session_state["billing_result_batt"] = result_batt
                        _check_battery_solver(result_batt)
                    else:
                        st.session_state["billing_result"] = result_pv_only
                        st.session_state["billing_result_batt"] = None
                else:
                    st.session_state["billing_result"] = result_pv_only
                    st.session_state["billing_result_batt"] = None

                # Show NEM-A fee summary
                _nema_agg_count = sum(1 for m in _meter_configs if not m.is_generating)
                _nema_fees = compute_nema_fees(
                    st.session_state.get("nema_utility", utility_name), _nema_agg_count
                )
                st.success(
                    f"NEM-A simulation complete! "
                    f"{len(_meter_configs)} meters, "
                    f"${_nema_fees['annual_admin']:,.0f}/yr admin fees"
                )

            else:
                # ============ Single-meter Custom billing path ============
                # Phase 1: pipeline runs through modules.simulation.run_simulation
                # so Monte Carlo / AI callers share one code path.
                _base_sim_inputs = inputs_from_session_state(
                    st.session_state,
                    nem_regime=nem_regime_1,
                    nbc_rate=_nem_nbc,
                    nsc_rate=_nem_nsc,
                    billing_option=_nem_billing,
                    export_rates_placeholder=_export_rates_for_sim,
                    include_battery=False,
                )

                # --- Step 1: PV-only billing ---
                result_pv_only = run_simulation(_base_sim_inputs).pv_only_result
                st.session_state["billing_result_pv_only"] = result_pv_only
                st.session_state["sizing_result"] = None


                # --- Step 2: Battery dispatch (if enabled) ---
                if st.session_state["battery_enabled"] and st.session_state["battery_config"] is not None:
                    batt_cfg = st.session_state["battery_config"]
                    _use_monthly = st.session_state.get("battery_fast_dispatch", False)

                    if st.session_state.get("battery_optimize", False):
                        # ---- Sizing sweep ----
                        opt_min, opt_max, opt_step = st.session_state["battery_opt_range"]
                        if opt_max > opt_min and opt_step > 0:
                            import numpy as _np
                            candidates = _np.arange(opt_min, opt_max + opt_step / 2, opt_step).tolist()


                            _tariff = st.session_state["tariff"]
                            _dt_idx = cast(pd.DatetimeIndex, st.session_state["load_8760"].index)
                            d_masks, d_prices = _build_demand_lp_inputs(_tariff, _dt_idx)
                            _energy_rates = _build_hourly_energy_rates(_tariff, _dt_idx)

                            _export_for_sizing = (
                                _export_rates_for_sim
                                if st.session_state["export_rates"] is None
                                else st.session_state["export_rates"]
                            )
                            sizing_res = optimize_capacity_kwh(
                                candidate_sizes_kwh=candidates,
                                pv_kwh=np.asarray(st.session_state["production_8760"]),
                                load_kwh=np.asarray(st.session_state["load_8760"].values),
                                import_price=_energy_rates,
                                export_price=np.asarray(_export_for_sizing.values),
                                demand_window_masks=d_masks,
                                demand_prices=d_prices,
                                battery_config=batt_cfg,
                                monthly=_use_monthly,
                                dt_index=_dt_idx,
                            )
                            st.session_state["sizing_result"] = sizing_res


                            # Run full billing with best size to get proper BillingResult
                            result_batt = run_simulation(
                                _dc_replace(
                                    _base_sim_inputs,
                                    battery_config=batt_cfg,
                                    battery_capacity_kwh=sizing_res.best_size_kwh,
                                    monthly_dispatch=_use_monthly,
                                )
                            ).billing_result
                            st.session_state["billing_result"] = result_batt
                            st.session_state["billing_result_batt"] = result_batt
                            _check_battery_solver(result_batt)
                            st.session_state["battery_capacity_kwh"] = sizing_res.best_size_kwh


                            st.success(
                                f"Optimization complete! Best size: "
                                f"{sizing_res.best_size_kwh:,.0f} kWh"
                            )
                        else:
                            st.session_state["billing_result"] = result_pv_only
                            st.session_state["billing_result_batt"] = None
                            st.warning("Invalid optimize range. Running PV-only.")
                    else:
                        # ---- Fixed-size dispatch ----
                        batt_cap = st.session_state.get("battery_capacity_kwh", 0)
                        if batt_cap > 0:
                            result_batt = run_simulation(
                                _dc_replace(
                                    _base_sim_inputs,
                                    battery_config=batt_cfg,
                                    battery_capacity_kwh=batt_cap,
                                    monthly_dispatch=_use_monthly,
                                )
                            ).billing_result
                            st.session_state["billing_result"] = result_batt
                            st.session_state["billing_result_batt"] = result_batt
                            _check_battery_solver(result_batt)


                            st.success("Simulation complete (PV + Battery)!")
                        else:
                            st.session_state["billing_result"] = result_pv_only
                            st.session_state["billing_result_batt"] = None
                            st.success("Simulation complete (PV only).")
                else:
                    st.session_state["billing_result"] = result_pv_only
                    st.session_state["billing_result_batt"] = None
                    st.success("Simulation complete!")

        # --- Post-simulation: Rate Shift Analysis ---
        if st.session_state.get("rate_shift_enabled"):
            _rs_result = st.session_state["billing_result"]
            _rs_pv_only = st.session_state.get("billing_result_pv_only")

            if billing_engine == "ECC" and st.session_state.get("rate_shift_old_ecc_calculator"):
                _rs_old_calc = st.session_state["rate_shift_old_ecc_calculator"]
                _rs_old = compute_old_rate_baseline_ecc(
                    st.session_state["load_8760"], _rs_old_calc,
                )
                # Apply to all result variants
                for _rs_r in [_rs_result, _rs_pv_only]:
                    if _rs_r is not None:
                        _rs_r.old_rate_annual_baseline = _rs_old["annual_cost"]
                        _rs_r.old_rate_monthly_baselines = _rs_old["monthly_costs"]
                        _rs_r.rate_shift_annual_savings = (
                            _rs_old["annual_cost"] - _rs_r.annual_bill_without_solar
                        )

            elif billing_engine == "Custom":
                if st.session_state.get("load_mode") == "NEM-A Aggregation":
                    # NEM-A: require per-meter old tariffs (no blanket fallback)
                    _rs_nema_tariffs = st.session_state.get("nema_rate_shift_tariffs", {})
                    _rs_nema_loads = st.session_state.get("nema_meter_loads", {})
                    _rs_nema_meters = st.session_state.get("nema_meters", [])
                    # Only compute if every meter has an old tariff assigned
                    if all(_rs_mi in _rs_nema_tariffs for _rs_mi in range(len(_rs_nema_meters))):
                        _rs_total_old = 0.0
                        _rs_monthly_old = [0.0] * 12
                        for _rs_mi, _rs_minfo in enumerate(_rs_nema_meters):
                            _rs_m_old_tariff = _rs_nema_tariffs[_rs_mi]
                            if _rs_minfo.get("is_generating"):
                                _rs_m_load = st.session_state["load_8760"]
                            else:
                                _rs_m_load = _rs_nema_loads.get(_rs_mi)
                            if _rs_m_load is not None:
                                _rs_m_old = compute_old_rate_baseline(_rs_m_load, _rs_m_old_tariff)
                                _rs_total_old += _rs_m_old["annual_cost"]
                                for _rs_j in range(12):
                                    _rs_monthly_old[_rs_j] += _rs_m_old["monthly_costs"][_rs_j]

                        for _rs_r in [_rs_result, _rs_pv_only]:
                            if _rs_r is not None:
                                _rs_r.old_rate_annual_baseline = _rs_total_old
                                _rs_r.old_rate_monthly_baselines = _rs_monthly_old
                                _rs_r.rate_shift_annual_savings = (
                                    _rs_total_old - _rs_r.annual_bill_without_solar
                                )
                else:
                    # Single meter
                    _rs_old_tariff = st.session_state.get("rate_shift_old_tariff")
                    if _rs_old_tariff is not None:
                        _rs_old = compute_old_rate_baseline(
                            st.session_state["load_8760"], _rs_old_tariff,
                        )
                        for _rs_r in [_rs_result, _rs_pv_only]:
                            if _rs_r is not None:
                                _rs_r.old_rate_annual_baseline = _rs_old["annual_cost"]
                                _rs_r.old_rate_monthly_baselines = _rs_old["monthly_costs"]
                                _rs_r.rate_shift_annual_savings = (
                                    _rs_old["annual_cost"] - _rs_r.annual_bill_without_solar
                                )

        # Clear editing flag when done
        st.session_state["editing_saved_sim"] = False
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.warning(
            "Check that all sidebar inputs are configured correctly. "
            "Common causes: mismatched profile lengths, missing tariff data, or invalid rate schedules."
        )
        with st.expander("Show error details"):
            import traceback
            st.code(traceback.format_exc())


if run_sim:
    _run_simulation()



def _render_results():
    """Display simulation results tabs (Monthly, Grid, Projection, PPA, Downloads)."""
    # =============================================================================
    # RESULTS DISPLAY
    # =============================================================================
    st.divider()
    st.subheader("Simulation Results")

    # CSS for white table backgrounds and bold totals row
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

    has_battery = st.session_state["billing_result_batt"] is not None

    # --- Scenario selector ---
    scenario: str | None = None
    if has_battery:
        scenario = st.radio(
            "View scenario",
            ["PV + Battery", "PV Only"],
            horizontal=True,
            key="scenario_selector",
        )
        if scenario == "PV Only":
            result = cast(BillingResult, st.session_state["billing_result_pv_only"])
        else:
            result = cast(BillingResult, st.session_state["billing_result_batt"])
    else:
        result = cast(BillingResult, st.session_state["billing_result"])

    # Persistent "Viewing" row — scenario badge + active-Proposal selector live
    # together on a single row so there's only one layer of chrome above the
    # results tabs. The Proposal selector reads/writes
    # ``st.session_state["active_proposal_id"]``; the Proposals tab uses it
    # to drive the comparison preview.
    _scenario_badge_label = (
        (scenario or ("PV + Battery" if has_battery else "PV Only"))
        if has_battery else "PV Only"
    )
    _badge_color = "#45A750" if "Battery" in _scenario_badge_label else "#1D6FA9"

    _sim_name_for_props = (
        st.session_state.get("_active_simulation_name")
        or st.session_state.get("_last_loaded_simulation_name")
    )
    _sim_proposals = _list_proposals_session(
        st.session_state, simulation_name=_sim_name_for_props,
    )
    _active_id = st.session_state.get("active_proposal_id")
    if _active_id and not any(p.id == _active_id for p in _sim_proposals):
        _active_id = None
        st.session_state["active_proposal_id"] = None

    _vrow_left, _vrow_sel, _vrow_new = st.columns([0.45, 0.40, 0.15], gap="small")
    with _vrow_left:
        st.markdown(
            f"""<div style="margin:6px 0 0 0; display:flex; align-items:center; gap:10px;">
                <span style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px;">Viewing</span>
                <span style="background:{_badge_color}; color:#ffffff; padding:3px 10px;
                             border-radius:12px; font-size:12px; font-weight:600;">
                    {_scenario_badge_label}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    with _vrow_sel:
        if _sim_proposals:
            _opt_ids = [p.id for p in _sim_proposals]
            _opt_labels = {p.id: p.name for p in _sim_proposals}
            _default_idx = _opt_ids.index(_active_id) if _active_id in _opt_ids else 0
            _picked = st.selectbox(
                "Active Proposal",
                options=_opt_ids,
                index=_default_idx,
                format_func=lambda pid: f"📁  {_opt_labels.get(pid, pid)}",
                key="top_proposal_selector",
                label_visibility="collapsed",
            )
            if _picked != st.session_state.get("active_proposal_id"):
                st.session_state["active_proposal_id"] = _picked
        else:
            st.markdown(
                '<div style="margin:8px 0 0 0; font-size:11px; color:#9ca3af;">'
                "No Proposals yet — save a PPA on the PPA Rate tab, then open the Proposals tab to bundle them."
                "</div>",
                unsafe_allow_html=True,
            )
    with _vrow_new:
        if st.button("➕ New Proposal", key="top_new_proposal_btn", width="stretch"):
            st.session_state["_proposals_tab_new"] = True
            st.session_state["active_proposal_id"] = None

    # Phase 6+: top-level navigation is now five institutional sections
    # (Overview / Bills & Projection / PPA & Proposals / Analysis /
    # Downloads). Downloads is promoted out of Analysis so the export
    # surface is a first-class destination, and so the top-bar Proposals
    # popover can deep-link into it. Each section renders a second tab
    # row for its sub-views; legacy tab variables (tab1, tab2, ...) are
    # re-pointed at the appropriate sub-tab containers so every existing
    # `with tab_X:` body below continues to work verbatim.
    section_tabs = st.tabs([
        "Overview",
        "Bills & Projection",
        "PPA & Proposals",
        "Analysis",
        "Downloads",
    ])

    # — Overview: headline KPIs + energy flow charts —
    with section_tabs[0]:
        _overview_sub = st.tabs(["Production vs Load", "Savings & Payback"])

    # — Bills & Projection: monthly detail + multi-year projection + battery —
    with section_tabs[1]:
        _bills_labels = ["Monthly Bills", "Annual Projection"]
        if has_battery:
            _bills_labels.append("Battery Analysis")
        _bills_sub = st.tabs(_bills_labels)

    # — PPA & Proposals: rate builder + named-Proposal workspace —
    with section_tabs[2]:
        _ppa_sub = st.tabs(["PPA Rate", "Proposals"])

    # — Analysis: sensitivity + AI assistant (Downloads moved out) —
    with section_tabs[3]:
        _analysis_sub = st.tabs(["Sensitivity", "AI Assistant"])

    # Re-point legacy tab variables into the new nested structure.
    tab3 = _overview_sub[0]            # Production vs Load
    tab4 = _overview_sub[1]            # Savings & Payback
    tab1 = _bills_sub[0]               # Monthly Bills (+ TOU expander)
    tab2 = _bills_sub[1]               # Annual Projection
    tab_batt = _bills_sub[2] if has_battery else None
    tab_indexed = _ppa_sub[0]          # PPA Rate
    tab_proposals = _ppa_sub[1]        # Proposals (Phase 4)
    tab_sensitivity = _analysis_sub[0] # Monte Carlo + tornado
    tab_ai = _analysis_sub[1]          # AI Assistant
    tab5 = section_tabs[4]             # Downloads (top-level now)

    # When the user activates a Proposal from the top-bar popover we do
    # two things on the next render:
    #   1. Click the "Proposals" sub-tab under PPA & Proposals so they
    #      land on the preview + export surface (Tranche 1 consolidation
    #      made this the single export home).
    #   2. Dismiss the popover, which Streamlit's st.popover doesn't do
    #      itself after a button inside it triggers a rerun. We simulate
    #      an Escape key press on the document (BaseWeb popovers listen
    #      for it) and click outside the popover as a fallback.
    _focus_ppa_tab = st.session_state.pop("_focus_ppa_rate_tab", False)
    _focus_proposals = st.session_state.pop("_focus_proposals_tab", False)
    if _focus_proposals or _focus_ppa_tab:
        _sub_label = "Proposals" if _focus_proposals else "PPA Rate"
        st.components.v1.html(
            f"""
            <script>
            (function focusProposalsAndClosePopover() {{
                const doc = window.parent.document;
                const subLabel = {json.dumps(_sub_label)};

                function clickTab(labels) {{
                    const tabs = doc.querySelectorAll(
                        '.stTabs [data-baseweb="tab-list"] > button'
                    );
                    if (!tabs || tabs.length === 0) return false;
                    for (const t of tabs) {{
                        const text = (t.innerText || "").trim();
                        if (labels.includes(text)) {{ t.click(); return true; }}
                    }}
                    return false;
                }}

                function focusProposals() {{
                    /* First click the top-level PPA & Proposals section,
                     * then the requested sub-tab. */
                    const topOk = clickTab(["PPA & Proposals"]);
                    if (!topOk) return false;
                    /* Sub-tab clicks must wait for DOM mount. */
                    setTimeout(function () {{ clickTab([subLabel]); }}, 150);
                    return true;
                }}

                function closeAnyOpenPopover() {{
                    /* BaseWeb popovers listen for Escape on the document. */
                    const esc = new KeyboardEvent("keydown", {{
                        key: "Escape", code: "Escape",
                        keyCode: 27, which: 27, bubbles: true,
                    }});
                    doc.dispatchEvent(esc);
                    doc.body.dispatchEvent(esc);
                    /* Fallback: click outside any visible popover layer. */
                    const layers = doc.querySelectorAll(
                        '[data-baseweb="popover"], [data-baseweb="layer"]'
                    );
                    if (layers.length === 0) return;
                    const outside = doc.querySelector('.block-container');
                    if (outside) outside.click();
                }}

                /* Try once immediately, then poll briefly for late mounts. */
                let done = focusProposals();
                closeAnyOpenPopover();
                if (!done) {{
                    const iv = setInterval(function () {{
                        if (focusProposals()) {{
                            closeAnyOpenPopover();
                            clearInterval(iv);
                        }}
                    }}, 100);
                    setTimeout(function () {{ clearInterval(iv); }}, 3000);
                }}
            }})();
            </script>
            """,
            height=0,
        )

    # Compute peak period index from tariff
    _tariff_for_peak = st.session_state["tariff"]
    _peak_period_idx = 0
    if _tariff_for_peak and _tariff_for_peak.energy_rate_structure:
        _max_rate = 0.0
        for _idx, _tiers in enumerate(_tariff_for_peak.energy_rate_structure):
            if _tiers and _tiers[0]["effective_rate"] > _max_rate:
                _max_rate = _tiers[0]["effective_rate"]
                _peak_period_idx = _idx
    elif billing_engine == "ECC" and st.session_state.get("ecc_tariff_data"):
        from modules.billing_ecc import _build_tou_arrays
        _dummy_idx = pd.date_range("2026-01-01", periods=1, freq="h")
        _, _, _peak_period_idx = _build_tou_arrays(_dummy_idx, st.session_state["ecc_tariff_data"])

    # Determine PV-only result for demand column display (BESS mode only)
    pv_only_for_display = st.session_state["billing_result_pv_only"] if (has_battery and scenario == "PV + Battery") else None

    # Pre-compute the main annual projection (reused across tabs)
    _common_nem_kw = {
        "nem_regime_1": nem_regime_1,
        "nem_regime_2": nem_regime_2 if nem_switch else None,
        "num_years_1": num_years_1 if nem_switch else None,
        "export_rates_multiyear_2": st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
        "cod_year": cod_year,
        "degradation_pct": annual_degradation_pct,
        "nbc_rate_2": st.session_state.get("nbc_rate_2", 0.0) if nem_switch else 0.0,
        "nsc_rate_2": st.session_state.get("nsc_rate_2", 0.0) if nem_switch else 0.0,
    }
    _rs_old_baseline_for_proj = result.old_rate_annual_baseline if result.old_rate_annual_baseline is not None else None

    # --- Regime-2 re-billing orchestration ---
    # When the NEM switch is on AND a distinct post-transition tariff is set,
    # run a SECOND year-1-style billing sim under tariff #2 + nem_regime_2 +
    # the regime-2 export rates. The resulting BillingResult is handed to the
    # engine as result_regime2 so the post-switch years re-bill on it. When no
    # distinct regime-2 tariff is selected, result_regime2 stays None (no
    # behavior change — the engine reuses the regime-1 result).
    _result_regime2 = None
    if nem_switch:
        try:
            # Regime-2 NEM params (mirror the year-1 NEM param derivation, but
            # for the second regime).
            _r2_regime = nem_regime_2
            _r2_nbc = st.session_state.get("nbc_rate_2", 0.0) if _r2_regime == "NEM-2" else 0.0
            _r2_nsc = st.session_state.get("nsc_rate_2", st.session_state.get("nsc_rate", NSC_DEFAULT_RATE))
            _r2_billing = st.session_state.get("billing_option_2", "ABO") if _r2_regime in ("NEM-1", "NEM-2") else "ABO"

            if billing_engine == "Custom" and st.session_state.get("regime_2_tariff") is not None:
                # Regime-2 export series: prefer the loaded Section-2 8760 series,
                # else a zeros placeholder (NEM-1/2 value exports at retail TOU).
                _r2_export = st.session_state.get("export_rates_2")
                if _r2_export is None:
                    _r2_dt = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                    _r2_export = pd.Series(np.zeros(8760), index=_r2_dt, name="export_rate_per_kwh")
                # Build base inputs from session, then swap in tariff #2,
                # regime-2 export rates, and regime-2 NEM params. Mirror the
                # ACTIVE scenario's battery treatment: if the viewed result is
                # the with-battery result, re-dispatch the battery under
                # tariff #2 and take .billing_result; otherwise PV-only. This
                # prevents the post-transition years from discontinuously losing
                # battery savings for a BESS project.
                _r2_use_battery = (
                    result is not None
                    and result is st.session_state.get("billing_result_batt")
                )
                _r2_inputs = inputs_from_session_state(
                    st.session_state,
                    nem_regime=_r2_regime,
                    nbc_rate=_r2_nbc,
                    nsc_rate=_r2_nsc,
                    billing_option=_r2_billing,
                    export_rates_placeholder=_r2_export,
                    include_battery=_r2_use_battery,
                )
                _r2_replace = {
                    "tariff": st.session_state["regime_2_tariff"],
                    "export_rates_8760": _r2_export,
                }
                if _r2_use_battery:
                    _r2_replace["battery_config"] = st.session_state.get("battery_config")
                    _r2_replace["battery_capacity_kwh"] = st.session_state.get("battery_capacity_kwh", 0)
                _r2_inputs = _dc_replace(_r2_inputs, **_r2_replace)
                _r2_sim = run_simulation(_r2_inputs)
                _result_regime2 = (
                    _r2_sim.billing_result if _r2_use_battery else _r2_sim.pv_only_result
                )

            elif billing_engine == "ECC" and st.session_state.get("regime_2_ecc_calculator") is not None:
                _r2_ecc_export = st.session_state.get("export_rates_2")
                if _r2_ecc_export is None:
                    _r2_ecc_dt = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                    _r2_ecc_export = pd.Series(np.zeros(8760), index=_r2_ecc_dt, name="export_rate_per_kwh")
                _result_regime2 = run_ecc_billing_simulation(
                    load_8760=st.session_state["load_8760"],
                    production_8760=st.session_state["production_8760"],
                    cost_calculator=st.session_state["regime_2_ecc_calculator"],
                    export_rates_8760=_r2_ecc_export,
                    tariff_data=st.session_state.get("regime_2_ecc_tariff_data"),
                    nsc_rate=_r2_nsc,
                    min_monthly_charge=getattr(st.session_state.get("tariff"), "min_monthly_charge", 0.0),
                )
        except Exception as _r2_err:
            st.warning(f"Post-transition re-billing failed; reusing regime-1 tariff. ({_r2_err})")
            _result_regime2 = None

    _main_projection = build_annual_projection(
        result=result,
        system_cost=system_cost,
        rate_escalator_pct=rate_escalator,
        load_escalator_pct=load_escalator,
        years=system_life_years,
        export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
        result_pv_only=pv_only_for_display,
        compound_escalation=compound_escalation,
        rate_shift_old_baseline=_rs_old_baseline_for_proj,
        existing_solar_offset_kwh=_es_offset_annual,
        result_regime2=_result_regime2,
        **_common_nem_kw,
    )

    # --- Tab 1: Monthly Bills ---
    with tab1:
        st.subheader("Monthly Bill Summary")
        display_df = build_monthly_summary_display(result, result_pv_only=pv_only_for_display, existing_solar_offset_kwh=_es_offset_monthly)
        raw = result.monthly_summary
        totals = {
            "Month": "TOTAL",
        }
        if _es_offset_monthly is not None:
            totals["Degraded System Load Offset (kWh)"] = fmt_num(sum(_es_offset_monthly))
        totals.update({
            "Solar (kWh)": fmt_num(raw['solar_kwh'].sum()),
            "Import (kWh)": fmt_num(raw['import_kwh'].sum()),
            "Export (kWh)": fmt_num(raw['export_kwh'].sum()),
            "↳ Peak (kWh)": fmt_num(raw['export_peak_kwh'].sum()),
            "↳ Off-Peak (kWh)": fmt_num(raw['export_offpeak_kwh'].sum()),
        })
        if pv_only_for_display is not None:
            totals["Demand kW (PV)"] = fmt_num(pv_only_for_display.monthly_summary['peak_demand_kw'].max())
            totals["Demand kW (PV+BESS)"] = fmt_num(raw['peak_demand_kw'].max())
        else:
            totals["Demand kW (PV)"] = fmt_num(raw['peak_demand_kw'].max())
        # Cost outflows negated so they render as red accounting negatives,
        # matching build_monthly_summary_display row formatting.
        totals.update({
            "Energy ($)": fmt_dollar(-raw['energy_cost'].sum()),
            "Demand ($)": fmt_dollar(-raw['total_demand_charge'].sum()),
            "Fixed ($)": fmt_dollar(-raw['fixed_charge'].sum()),
        })
        _has_nbc = "nbc_charge" in raw.columns and raw["nbc_charge"].sum() > 0
        if _has_nbc:
            totals["NBC ($)"] = fmt_dollar(-raw['nbc_charge'].sum())
        totals.update({
            "Export Credit ($)": fmt_dollar(raw['export_credit'].sum()),  # inflow, stays positive
            "Net Bill ($)": fmt_dollar(-raw['net_bill'].sum()),
        })
        if result.old_rate_monthly_baselines is not None and result.monthly_baseline_details is not None:
            _rs_old = result.old_rate_monthly_baselines
            _rs_new = [d["total"] for d in result.monthly_baseline_details]
            totals["Rate Shift Savings ($)"] = fmt_dollar(sum(_rs_old) - sum(_rs_new))
        totals_row = pd.DataFrame([totals])
        display_with_totals = pd.concat([display_df, totals_row], ignore_index=True)

        st.markdown(
            render_styled_table(
                display_with_totals,
                bold_last_row=True,
                bold_cols=["Month", "Export (kWh)", "Net Bill ($)"],
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "Bill components shown as accounting negatives; Export Credit is positive. "
            "Export Peak and Off-Peak are sub-components of the bolded Export (kWh) total."
        )

        # Show NSC adjustment info if applicable. Mechanism differs by regime:
        # NEM-1/2 reprices surplus from retail TOU rate down to NSC; NEM-3/NBT
        # reprices from rolling-12-mo avg ACC rate down to NSC (per AB 920 +
        # CPUC D.22-12-056).
        if hasattr(result, 'annual_nsc_adjustment') and result.annual_nsc_adjustment > 0:
            _nsc_basis = (
                "retail TOU rate"
                if getattr(result, "nem_regime", "").startswith(("NEM-1", "NEM-2", "NEM-A (NEM-1)", "NEM-A (NEM-2)"))
                else "avg ACC export rate"
            )
            st.info(
                f"Net Surplus Compensation adjustment applied in month 12: "
                f"${result.annual_nsc_adjustment:,.2f} (surplus repriced from "
                f"{_nsc_basis} to wholesale NSC rate)"
            )

        # ── Grid Exchange breakdown (inline, was its own tab) ─────────
        with st.expander("TOU breakdown — Grid Import & Export by TOU Period"):
            st.caption(
                "Monthly totals above, here decomposed into peak vs off-peak "
                "import / export and their associated cost / credit."
            )
            ge_display, ge_raw = build_grid_exchange_summary(result, _peak_period_idx)
            _ge_bold_cols = [c for c in ge_display.columns if "Total" in c]
            st.markdown(
                render_styled_table(ge_display, bold_last_row=True, bold_cols=_ge_bold_cols),
                unsafe_allow_html=True,
            )

    # --- Tab 2: Annual Summary ---
    with tab2:
        st.subheader(f"Annual Summary ({system_life_years}-Year)")
        projection_df = _main_projection

        # Pre-compute both Simple and Detailed display DataFrames once so the
        # radio toggle swaps views without re-running the format pipeline.
        # Keyed in session_state only to survive fragment reruns; cleared on
        # every outer render to avoid cross-scenario staleness.
        def _format_projection(pdf):
            display_proj = pdf.copy()
            outflow_dollar_cols = [
                "Bill w/o Solar ($)", "Energy ($)", "Demand ($)",
                "Fixed ($)", "NBC ($)", "Bill w/ Solar ($)",
            ]
            for col in outflow_dollar_cols:
                if col in display_proj.columns:
                    display_proj[col] = display_proj[col].apply(lambda x: -x)
            _rename = {}
            if "Export Peak (kWh)" in display_proj.columns:
                _rename["Export Peak (kWh)"] = "↳ Peak (kWh)"
            if "Export Off-Peak (kWh)" in display_proj.columns:
                _rename["Export Off-Peak (kWh)"] = "↳ Off-Peak (kWh)"
            if _rename:
                display_proj = display_proj.rename(columns=_rename)
            for col in [c for c in display_proj.columns if "(kWh)" in c]:
                display_proj[col] = display_proj[col].apply(fmt_num)
            for col in [c for c in display_proj.columns if "kW" in c and "(kWh)" not in c]:
                display_proj[col] = display_proj[col].apply(fmt_num)
            for col in [c for c in display_proj.columns if "($)" in c]:
                display_proj[col] = display_proj[col].apply(fmt_dollar)
            return display_proj

        _detailed_proj = _format_projection(projection_df)
        _simple_drop = [
            "Load (kWh)", "Customer Load (kWh)", "Solar (kWh)",
            "Solar Offset (kWh)", "Import (kWh)", "Export (kWh)",
            "↳ Peak (kWh)", "↳ Off-Peak (kWh)",
            "Demand kW (PV)", "Demand kW (PV+BESS)",
            "Energy ($)", "Demand ($)", "Fixed ($)",
            "NBC ($)", "NSC Adj ($)",
        ]
        _simple_proj = _detailed_proj.drop(
            columns=[c for c in _simple_drop if c in _detailed_proj.columns]
        )
        st.session_state["_proj_simple_df"] = _simple_proj
        st.session_state["_proj_detailed_df"] = _detailed_proj

        @st.fragment
        def _render_projection_table():
            _proj_detail = st.radio(
                "View", ["Simple", "Detailed"], horizontal=True,
                key="proj_view_toggle", label_visibility="collapsed",
            )
            display_proj = (st.session_state["_proj_detailed_df"]
                            if _proj_detail == "Detailed"
                            else st.session_state["_proj_simple_df"])

            _proj_bold = ["Calendar Year"] if "Calendar Year" in display_proj.columns else ["Year"]
            if "Export (kWh)" in display_proj.columns:
                _proj_bold.append("Export (kWh)")
            if "Bill w/ Solar ($)" in display_proj.columns:
                _proj_bold.append("Bill w/ Solar ($)")
            _proj_highlight = ["Cumulative Savings ($)"]
            if "Cumulative Total Savings ($)" in display_proj.columns:
                _proj_highlight.append("Cumulative Total Savings ($)")
            st.markdown(render_styled_table(
                display_proj,
                bold_cols=_proj_bold,
                highlight_cols=_proj_highlight,
            ), unsafe_allow_html=True)
            st.caption(
                "Cost components shown as accounting negatives; Export Credit "
                "and Savings columns are positive. Export Peak / Off-Peak are "
                "sub-components of the bolded Export (kWh) total."
            )

        _render_projection_table()

        # Cumulative savings chart — 38DN palette
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projection_df["Year"],
            y=projection_df["Cumulative Savings ($)"],
            name="Cumulative Solar Savings",
            mode="lines+markers",
            line=dict(color="#45A750", width=2.8),
            marker=dict(size=7, color="#45A750"),
        ))
        if "Cumulative Total Savings ($)" in projection_df.columns:
            fig.add_trace(go.Scatter(
                x=projection_df["Year"],
                y=projection_df["Cumulative Total Savings ($)"],
                name="Cumulative Total Savings (incl. Rate Shift)",
                mode="lines+markers",
                line=dict(color="#1D6FA9", width=2.8),
                marker=dict(size=7, color="#1D6FA9"),
            ))
        fig.add_hline(
            y=system_cost, line_dash="dash", line_color="#0E2841",
            line_width=2,
            annotation_text=f"<b>System Cost</b>: ${system_cost:,.0f}",
            annotation_font=dict(color="#0E2841", size=12),
        )
        fig.update_layout(
            title=dict(text="Cumulative Savings vs. System Cost",
                       font=dict(size=15, color="#0E2841")),
            xaxis_title="Year", yaxis_title="$",
            template="plotly_white", height=380,
            font=dict(family="Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif",
                      size=12, color="#1A1A1A"),
            margin=dict(l=60, r=30, t=70, b=55),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font=dict(color="#1A1A1A", size=11)),
            xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#1A1A1A")),
            yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#1A1A1A")),
        )
        st.plotly_chart(fig, width="stretch")

    # --- Tab 3: Charts ---
    with tab3:
        st.subheader("Production vs. Load")
        fig_prod = create_production_vs_load_chart(result)
        st.plotly_chart(fig_prod, width="stretch")
        st.subheader("Monthly Bill Breakdown")
        fig_bill = create_monthly_bill_chart(result)
        st.plotly_chart(fig_bill, width="stretch")

    # --- Tab 4: Savings & Payback ───────────────────────────────────────
    with tab4:
        _render_savings_dashboard(
            result=result,
            pv_only_result=st.session_state.get("billing_result_pv_only"),
            pv_batt_result=st.session_state.get("billing_result_batt"),
            system_cost=system_cost,
            system_life_years=system_life_years,
            has_battery=has_battery,
            main_projection=_main_projection,
        )

    # --- Battery Analysis tab (only when battery enabled) ---
    if tab_batt is not None:
        with tab_batt:
            st.subheader("Battery Analysis")

            # Show selected / optimized size
            batt_cap_display = st.session_state.get("battery_capacity_kwh", 0)
            batt_hrs_display = st.session_state["battery_config"].battery_hours if st.session_state["battery_config"] else 4
            batt_pw_display = batt_cap_display / batt_hrs_display if batt_hrs_display > 0 else 0
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Capacity", f"{batt_cap_display:,.0f} kWh")
            with bc2:
                st.metric("Power", f"{batt_pw_display:,.0f} kW")
            with bc3:
                st.metric("Duration", f"{batt_hrs_display:.1f} hrs")

            # --- Battery KPIs ---
            pv_only_res = st.session_state["billing_result_pv_only"]
            batt_res = st.session_state["billing_result_batt"]
            if pv_only_res is not None and batt_res is not None and batt_cap_display > 0:
                kpis = build_battery_kpi_summary(pv_only_res, batt_res, batt_cap_display)

                st.divider()
                st.subheader("Performance KPIs")

                kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
                with kpi_c1:
                    st.metric("Est. Annual Cycles", f"{kpis['cycles']:,.1f}")
                    st.metric("Throughput", f"{kpis['throughput_kwh']:,.0f} kWh")
                with kpi_c2:
                    st.metric("Total Charge", f"{kpis['total_charge_kwh']:,.0f} kWh")
                    st.metric("Total Discharge", f"{kpis['total_discharge_kwh']:,.0f} kWh")
                with kpi_c3:
                    st.metric("Discharge to Load", f"{kpis['discharge_to_load_kwh']:,.0f} kWh")
                    st.metric("Discharge to Grid", f"{kpis['discharge_to_grid_kwh']:,.0f} kWh")
                with kpi_c4:
                    st.metric("Import Change", f"{kpis['import_change_kwh']:,.0f} kWh")
                    export_delta_sign = "+" if kpis["export_change_kwh"] >= 0 else ""
                    st.metric(
                        "Export Change",
                        f"{export_delta_sign}{kpis['export_change_kwh']:,.0f} kWh",
                        delta=f"{kpis['export_change_pct']:+.1f}%",
                    )

                st.divider()
                st.subheader("Self-Consumption & Peak Demand")
                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    st.markdown("**PV Self-Consumption**")
                    st.metric(
                        "PV Only",
                        f"{kpis['pv_self_consumption_pv_only_pct']:.1f}%",
                    )
                    st.metric(
                        "PV + Battery",
                        f"{kpis['pv_self_consumption_batt_pct']:.1f}%",
                        delta=f"+{kpis['self_consumption_increase_pct']:.1f} pp",
                    )
                with sc_c2:
                    st.markdown("**Peak Demand (annual max)**")
                    st.metric("PV Only", f"{kpis['pv_only_peak_kw']:,.1f} kW")
                    st.metric(
                        "PV + Battery",
                        f"{kpis['batt_peak_kw']:,.1f} kW",
                        delta=f"-{kpis['peak_reduction_kw']:,.1f} kW ({kpis['peak_reduction_pct']:.1f}%)",
                    )

            # Sizing curve (if optimize was run)
            sizing_res = st.session_state.get("sizing_result")
            if sizing_res is not None:
                st.divider()
                st.subheader("Sizing Curve")

                import plotly.graph_objects as go
                sz = sizing_res.table
                fig_sz = go.Figure()
                fig_sz.add_trace(go.Scatter(
                    x=sz["size_kwh"], y=sz["net_bill"],
                    mode="lines+markers", name="Net Bill",
                    line=dict(color="#1D6FA9", width=2.5),
                    marker=dict(size=6),
                ))
                # Mark best point
                best_row = sz[sz["size_kwh"] == sizing_res.best_size_kwh]
                if not best_row.empty:
                    fig_sz.add_trace(go.Scatter(
                        x=best_row["size_kwh"], y=best_row["net_bill"],
                        mode="markers", name="Optimal",
                        marker=dict(color="#45A750", size=14, symbol="star"),
                    ))
                fig_sz.update_layout(
                    title=dict(text="Net Annual Bill vs. Battery Size",
                               font=dict(size=15, color="#0E2841")),
                    xaxis_title="Battery Capacity (kWh)",
                    yaxis_title="Net Annual Bill ($)",
                    template="plotly_white", height=400,
                    font=dict(family="Aptos Narrow, Aptos, Calibri, Arial Narrow, sans-serif",
                              size=12, color="#1A1A1A"),
                    margin=dict(l=60, r=30, t=70, b=55),
                    xaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#1A1A1A")),
                    yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(color="#1A1A1A")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(color="#1A1A1A", size=11)),
                )
                st.plotly_chart(fig_sz, width="stretch")

                # Show sizing table
                st.subheader("Sizing Detail")
                sz_display = sz.copy()
                for c in [col for col in sz_display.columns if col != "size_kwh" and col != "power_kw"]:
                    sz_display[c] = sz_display[c].apply(lambda x: fmt_dollar(x) if isinstance(x, (int, float)) else x)
                sz_display["size_kwh"] = sz_display["size_kwh"].apply(fmt_num)
                sz_display["power_kw"] = sz_display["power_kw"].apply(fmt_num)
                sz_display.columns = ["Size (kWh)", "Power (kW)", "Energy ($)", "Demand ($)", "Export Credit ($)", "Net Bill ($)"]
                st.markdown(render_styled_table(sz_display), unsafe_allow_html=True)

    # --- Indexed Tariff tab ---
    @st.fragment
    def _ppa_dashboard():
        st.subheader("Indicative PPA Rate")
        st.caption("Calculates the maximum PPA rate ($/kWh) the customer can pay while achieving a target savings percentage vs. their utility-only bill.")

        # Controls row
        it_col1, it_col2 = st.columns([1, 1])
        with it_col1:
            it_view = st.radio("View", ["Annual", "Monthly"], horizontal=True, key="indexed_tariff_view")
        with it_col2:
            it_savings_pct = st.number_input(
                "Customer Savings Target (%)", 0.0, 99.0, 10.0, 1.0,
                key="it_savings_pct",
            )

        # No PPA escalator input: the rate is solved *per year* so the customer
        # keeps the savings target as a flat % of each year's offset, held
        # across a NEM regime switch. A fixed escalator can't co-exist with flat
        # per-year savings, so the rate simply floats with the bill. The realized
        # effective escalator (CAGR of the solved rate) is computed below for
        # display/snapshot continuity.
        st.caption(
            "The PPA rate is solved each year to hold your savings target as a "
            "flat % of that year's utility savings — maintained across any NEM "
            "regime switch. The rate floats with the bill; there is no fixed "
            "escalator."
        )

        # Advanced Options expander
        it_savings_esc = 0.0
        it_regime_1_savings = None
        it_regime_2_savings = None
        with st.expander("Advanced Options"):
            it_savings_esc = st.number_input(
                "Savings Escalator (%/yr)", 0.0, 10.0, 0.0, 0.5,
                key="it_savings_esc",
                help="Savings target increases by this amount each year",
            )
            if nem_switch:
                st.markdown("**Per-Regime Savings Targets**")
                r1c, r2c = st.columns(2)
                with r1c:
                    it_regime_1_savings = st.number_input(
                        f"{nem_regime_1} Savings (%)", 0.0, 99.0, it_savings_pct, 1.0,
                        key="it_r1_sav",
                    )
                with r2c:
                    it_regime_2_savings = st.number_input(
                        f"{nem_regime_2} Savings (%)", 0.0, 99.0, it_savings_pct, 1.0,
                        key="it_r2_sav",
                    )

        # Shared kwargs for indexed tariff builders
        _it_kw = dict(
            base_savings_pct=it_savings_pct,
            savings_escalator_pct=it_savings_esc,
            regime_1_savings_pct=it_regime_1_savings,
            regime_2_savings_pct=it_regime_2_savings,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
        )

        # Always compute annual indexed tariff (drives dashboard + annual table)
        it_annual_df = build_indexed_tariff_annual(_main_projection, **_it_kw)

        # ── PPA Dashboard ──────────────────────────────────────────
        _cd = it_annual_df
        _ppa_cost_yr = _cd["PPA Rate ($/kWh)"] * _cd["Solar (kWh)"]
        _total_ppa = _cd["Bill w/ Solar ($)"] + _ppa_cost_yr
        _bill_no = _cd["Bill w/o Solar ($)"]
        _cust_sav = _cd["Customer Savings ($)"]
        _life_sav = _cust_sav.sum()
        _yr1_rate = float(_cd["PPA Rate ($/kWh)"].iloc[0]) if len(_cd) else 0.0
        _yr1_sav = float(_cust_sav.iloc[0]) if len(_cd) else 0.0
        _yr1_bno = float(_bill_no.iloc[0]) if len(_cd) else 1.0
        _yr1_pct = (_yr1_sav / _yr1_bno * 100) if _yr1_bno else 0.0

        # ── Per-regime Yr-1 PPA rate + summary KPIs ───────────────────
        # When a NEM switch is configured, each regime gets its own back-
        # solved Yr-1 rate. Expose both so the user can see the step-down.
        _yr1_rate_r1 = _yr1_rate
        _yr1_rate_r2 = None
        if nem_switch and num_years_1 and num_years_1 < len(_cd):
            _split = int(num_years_1)
            _yr1_rate_r2 = float(_cd["PPA Rate ($/kWh)"].iloc[_split])

        # Realized effective escalator: CAGR of the solved per-year rate within
        # each regime. There is no longer a fixed escalator input, so this is
        # what gets stored on the snapshot for display and proposal-term
        # extrapolation — an honest read of how the floating rate actually grows.
        _rate_list = _cd["PPA Rate ($/kWh)"].tolist()
        if nem_switch and num_years_1 and num_years_1 < len(_cd):
            _esc_real_r1 = _realized_cagr(_rate_list[: int(num_years_1)])
            _esc_real_r2 = _realized_cagr(_rate_list[int(num_years_1):])
        else:
            _esc_real_r1 = _realized_cagr(_rate_list)
            _esc_real_r2 = None

        if _yr1_rate_r2 is not None:
            _k1, _k2, _k3, _k4, _k5 = st.columns(5)
            _k1.metric(f"Yr-1 PPA · {nem_regime_1}", f"${_yr1_rate_r1:.4f}/kWh")

            # Streamlit's st.metric parses the delta's sign from the FIRST
            # non-whitespace character of the string. A leading "$" hides
            # the minus sign, so we build the string with the sign at
            # position 0. Default ("normal") delta_color is used so the
            # semantic maps to project economics: a HIGHER regime-2 PPA
            # rate is more revenue (green ▲), a LOWER regime-2 rate is
            # reduced revenue (red ▼).
            _ppa_diff = _yr1_rate_r2 - _yr1_rate_r1
            _delta_sign = "-" if _ppa_diff < 0 else "+"
            _delta_str = f"{_delta_sign}${abs(_ppa_diff):.4f}/kWh"
            _k2.metric(
                f"Yr-1 PPA · {nem_regime_2}", f"${_yr1_rate_r2:.4f}/kWh",
                delta=_delta_str,
                delta_color="normal",
            )
            _k3.metric("Year-1 Savings", f"${_yr1_sav:,.0f}")
            _k4.metric("Savings %", f"{_yr1_pct:.1f}%")
            _k5.metric("Lifetime Savings", f"${_life_sav:,.0f}")
        else:
            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric("Year-1 PPA Rate", f"${_yr1_rate_r1:.4f}/kWh")
            _k2.metric("Year-1 Savings", f"${_yr1_sav:,.0f}")
            _k3.metric("Savings %", f"{_yr1_pct:.1f}%")
            _k4.metric("Lifetime Savings", f"${_life_sav:,.0f}")

        # Pre-compute the x-axis series once — used by the Save PPA handler
        # below AND by the chart renderer further down. Previously this lived
        # inside the chart block, which meant clicking Save PPA raised a
        # NameError on _x because the handler ran before the chart body.
        _x = _cd["Calendar Year"].astype(int) if "Calendar Year" in _cd.columns else _cd["Year"]

        # ── Why does the NEM-3 bill often rise despite a lower PPA? ───
        # Custom callout styled with a light-teal background so it reads as
        # an explanatory panel tied to the 38DN palette rather than
        # Streamlit's default info-blue. Arrow direction in the headline
        # flips with the sign of the bill jump so a (rare) regime-switch
        # drop renders the correct semantic ↓ rather than the default ↑.
        if nem_switch and num_years_1 and num_years_1 < len(_cd):
            _jump = float(_total_ppa.iloc[num_years_1] - _total_ppa.iloc[num_years_1 - 1])
            if _jump != 0:
                _is_drop = _jump < 0
                _jump_arrow = "↓" if _is_drop else "↑"
                _jump_verb = "drops" if _is_drop else "steps up"
                _jump_rule_color = "#A8141A" if _is_drop else "#518484"
                _jump_bg = "#FDEDED" if _is_drop else "#E3EDED"
                _jump_border = "#F3C2C2" if _is_drop else "#C7DADA"
                _jump_eyebrow_color = "#A8141A" if _is_drop else "#518484"

                _jump_body = (
                    f"<strong>Why the bill {_jump_verb} in {nem_regime_2} even as the PPA steps "
                    f"down by ${abs(_yr1_rate_r2 - _yr1_rate_r1):.4f}/kWh:</strong> "
                    f"under {nem_regime_2}, exported solar is compensated at the ACC "
                    f"(avoided-cost) rate, which is typically <strong>5–10×</strong> "
                    f"lower than retail TOU. The lost export credit on the utility "
                    f"side outweighs the PPA reduction, so the customer's total bill "
                    f"{_jump_verb} by about <strong>{_jump_arrow} ${abs(_jump):,.0f}</strong> "
                    f"at the regime switch. The PPA steps down so the customer still "
                    f"keeps the savings target as a flat % of the (now smaller) "
                    f"{nem_regime_2} utility savings — the absolute dollar savings shrinks "
                    f"because the offset itself shrinks, not because the target slips. "
                    f"If the offset can't reach the target, the PPA floors at $0 and the "
                    f"customer keeps the full remaining savings."
                )
                st.markdown(
                    f'<div style="background:{_jump_bg};border:1px solid {_jump_border};'
                    f'border-left:3px solid {_jump_rule_color};border-radius:6px;'
                    'padding:12px 16px;margin:10px 0 18px 0;font-size:13px;'
                    'line-height:1.55;color:#0E2841;">'
                    f'<div style="font-size:10px;font-weight:600;color:{_jump_eyebrow_color};'
                    'text-transform:uppercase;letter-spacing:0.06em;'
                    f'margin-bottom:6px;">Regime-switch explainer</div>'
                    f'{_jump_body}</div>',
                    unsafe_allow_html=True,
                )

        # ── Save PPA → Proposal (nested structure) ───────────────────
        # Single action: saving a PPA simultaneously (a) appends it to the
        # session's saved_ppa_scenarios pool so the overlay chart can draw
        # it AND (b) attaches it to a Proposal for this simulation. If no
        # Proposal is active, one is auto-created with a sensible default
        # name, the PPA becomes its primary, and it is persisted to GCS.
        # Later saves append the new PPA as a comparison snapshot on the
        # active Proposal up to the 3-comparison cap, then wrap around to
        # replace the oldest comparison so the cap stays respected.
        _sp_save, _sp_tbl = st.container(), st.container()
        with _sp_save:
            # Current-Proposal indicator (teal card when active, ghost when not)
            _active_sim_name = (
                st.session_state.get("_active_simulation_name")
                or st.session_state.get("_last_loaded_simulation_name")
            )
            _active_prop = _get_active_proposal(st.session_state)
            if _active_prop and _active_prop.simulation_name == _active_sim_name:
                _n_snaps = 1 + len(_active_prop.comparison_ppas)
                _snap_limit_msg = (
                    f" (primary + {len(_active_prop.comparison_ppas)} of {_PROP_MAX_COMPARISONS} comparisons)"
                    if _active_prop.comparison_ppas else " (primary)"
                )
                # Styled via design-system tokens (see assets/theme.css).
                # Same visual language as the regime-switch explainer so
                # status callouts across the app stay coherent.
                st.markdown(
                    '<div style="background:var(--38dn-info-bg);'
                    'border:1px solid var(--38dn-border-1);'
                    'border-left:3px solid var(--38dn-teal);'
                    'border-radius:var(--38dn-radius-md);'
                    'padding:10px 14px;margin:6px 0 10px 0;'
                    'font-size:var(--38dn-fs-body);color:var(--38dn-ink);">'
                    '<div class="eyebrow-38dn" style="margin-bottom:4px;">'
                    'Active Proposal</div>'
                    f'<strong>{_active_prop.name}</strong> — {_n_snaps} PPA'
                    f'{"s" if _n_snaps != 1 else ""}{_snap_limit_msg}'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:var(--38dn-surface-1);'
                    'border:1px dashed var(--38dn-border-2);'
                    'border-radius:var(--38dn-radius-md);'
                    'padding:10px 14px;margin:6px 0 10px 0;'
                    'font-size:var(--38dn-fs-body);color:var(--38dn-slate-50);">'
                    '<strong>No Proposal active for this simulation.</strong> '
                    "Saving a PPA below will auto-create one you can edit "
                    "from the Proposals tab."
                    "</div>",
                    unsafe_allow_html=True,
                )

            _ui_section_header(
                "Save PPA to Proposal",
                caption=(
                    "Names and stores the current PPA configuration on the "
                    "active Proposal (auto-creates one if missing). Each save "
                    "appends a new PPA snapshot — build your primary first, "
                    f"then iterate up to {_PROP_MAX_COMPARISONS} comparison variants."
                ),
            )
            sv_name_col, sv_btn_col, sv_clear_col = st.columns([0.5, 0.25, 0.25])
            with sv_name_col:
                _sv_name = st.text_input(
                    "PPA scenario name", value="", key="ppa_save_name",
                    placeholder="e.g. 2.9% esc / 10% savings",
                    label_visibility="collapsed",
                )
            with sv_btn_col:
                _sv_btn_label = (
                    "💾 Add to Proposal" if _active_prop and _active_prop.simulation_name == _active_sim_name
                    else "💾 Save PPA + Start Proposal"
                )
                _save_clicked = st.button(
                    _sv_btn_label, key="ppa_save_btn", use_container_width=True,
                )
            if _save_clicked:
                # Spinner so the save-button click feels acknowledged instead
                # of dead during the session + GCS write. The blocking work
                # below is cheap (dict assignments, dataclass construction);
                # the slow step was GCS upload which is now asynchronous.
                with st.spinner("Saving PPA to Proposal…"):
                    name = (_sv_name or "").strip() or f"Scenario {len(st.session_state.get('saved_ppa_scenarios', {})) + 1}"

                    # --- 1. Persist to the session-wide scenarios pool so the
                    #        overlay chart and the comparison table can read it.
                    _proj_load_col = (
                        "Customer Load (kWh)"
                        if "Customer Load (kWh)" in _main_projection.columns
                        else "Load (kWh)"
                    )
                    _load_kwh = _main_projection[_proj_load_col].to_numpy() if _proj_load_col in _main_projection.columns else None
                    saved = st.session_state.setdefault("saved_ppa_scenarios", {})
                    saved[name] = {
                        "calendar_year": _x.tolist(),
                        "year_indices": _cd["Year"].astype(int).tolist()
                            if "Year" in _cd.columns else list(range(1, len(_cd) + 1)),
                        "ppa_rate_per_year": _cd["PPA Rate ($/kWh)"].round(5).tolist(),
                        "solar_kwh_per_year": _cd["Solar (kWh)"].round(0).tolist(),
                        "total_ppa_bill_k": (_total_ppa / 1000).round(2).tolist(),
                        "utility_only_bill_k": (_bill_no / 1000).round(2).tolist(),
                        "ppa_effective_rate": (
                            (_total_ppa.to_numpy() / _load_kwh).round(5).tolist()
                            if _load_kwh is not None and (_load_kwh > 0).all() else None
                        ),
                        "utility_effective_rate": (
                            (_bill_no.to_numpy() / _load_kwh).round(5).tolist()
                            if _load_kwh is not None and (_load_kwh > 0).all() else None
                        ),
                        "year1_rate_r1": _yr1_rate_r1,
                        "year1_rate_r2": _yr1_rate_r2,
                        "ppa_escalator_r1": float(_esc_real_r1),
                        "ppa_escalator_r2": float(_esc_real_r2) if (nem_switch and _esc_real_r2 is not None) else None,
                        "savings_pct": float(it_savings_pct),
                        "savings_pct_r2": float(st.session_state.get("it_r2_sav") or it_savings_pct)
                            if nem_switch else None,
                        "nem_regime_1": nem_regime_1,
                        "nem_regime_2": nem_regime_2 if nem_switch else None,
                        "num_years_1": int(num_years_1) if nem_switch and num_years_1 else None,
                        "term_years": int(len(_cd)),
                        "lifetime_savings": float(_life_sav),
                    }

                    # --- 2. Attach to a Proposal — create one if missing, or
                    #        append to the active Proposal for this simulation.
                    try:
                        _new_snap = _snapshot_from_saved(
                            name, saved[name], term_years=int(len(_cd)),
                            sim_system_size_kw=float(system_size_kw or 0.0),
                            sim_rate_escalator_pct=float(rate_escalator or 0.0),
                            sim_load_escalator_pct=float(load_escalator or 0.0),
                        )
                        _current = (
                            _active_prop
                            if (_active_prop and _active_prop.simulation_name == _active_sim_name)
                            else None
                        )
                        if _current is None:
                            # Auto-create a Proposal: first PPA is the primary.
                            # Name follows a consistent "Deal Folder · {sim}
                            # · {date}" pattern so proposals are easy to
                            # recognise in the top-bar popover and GCS.
                            _default_proposal_name = (
                                f"Deal Folder · {_active_sim_name or 'Untitled Sim'} "
                                f"· {date.today().strftime('%Y-%m-%d')}"
                            )
                            _proposal_obj = _create_proposal_obj(
                                name=_default_proposal_name,
                                simulation_name=_active_sim_name,
                                customer_name=st.session_state.get("customer_name", ""),
                                site_address=st.session_state.get("sb_location", ""),
                                utility_account="",
                                term_years=int(len(_cd)),
                                primary_ppa=_new_snap,
                                comparison_ppas=(),
                            )
                            _toast_msg = (
                                f"Created Proposal '{_proposal_obj.name}' with "
                                f"'{name}' as primary"
                            )
                        else:
                            # Append as comparison. If we're at the cap,
                            # replace the oldest comparison snapshot to keep
                            # the UX forgiving rather than silently blocking.
                            _existing_comps = list(_current.comparison_ppas)
                            # Don't duplicate — if a snap by this name exists, overwrite it.
                            _existing_comps = [s for s in _existing_comps if s.name != name]
                            _existing_comps.append(_new_snap)
                            if len(_existing_comps) > _PROP_MAX_COMPARISONS:
                                _existing_comps = _existing_comps[-_PROP_MAX_COMPARISONS:]
                            _proposal_obj = _update_proposal_obj(
                                _current,
                                comparison_ppas=tuple(_existing_comps),
                                term_years=int(len(_cd)),
                            )
                            _toast_msg = (
                                f"Added '{name}' to Proposal '{_proposal_obj.name}' "
                                f"({len(_existing_comps)}/{_PROP_MAX_COMPARISONS} comparisons)"
                            )
                        _save_proposal_session(st.session_state, _proposal_obj)
                        # Fire-and-forget GCS persist so the ~1–2s upload
                        # round-trip doesn't delay the fragment rerun. The
                        # session-state write above is the source of truth
                        # for the current render; GCS is the durable store
                        # that backs a subsequent simulation reload.
                        _persist_proposal_async(_proposal_obj)
                        st.session_state["_proposal_toast_pending"] = (
                            _toast_msg + " · open the Proposals tab to review",
                            "📁",
                        )
                    except Exception as exc:
                        st.warning(f"PPA saved to pool but could not attach to a Proposal: {exc}")
                        st.session_state["_proposal_toast_pending"] = (
                            f"Saved PPA '{name}' (no Proposal link)", "💾",
                        )

                    # Full rerun so the top-bar Proposals popover also
                    # picks up the newly created / updated Proposal — a
                    # fragment-scoped rerun would leave it stale since the
                    # popover renders outside this fragment. The JS focus
                    # handler below re-opens the PPA Rate sub-tab so the
                    # user stays where they were.
                    st.session_state["_focus_ppa_rate_tab"] = True
                    st.rerun()
            with sv_clear_col:
                if st.button("Clear PPA pool", key="ppa_clear_btn", use_container_width=True,
                             help="Clears the session's PPA scenarios pool (saved Proposals are untouched)."):
                    st.session_state["saved_ppa_scenarios"] = {}
                    st.session_state["_focus_ppa_rate_tab"] = True
                    st.rerun()

        # Saved scenarios table (above chart so the user sees their saved
        # trajectories before looking at the overlay).
        with _sp_tbl:
            if st.session_state.get("saved_ppa_scenarios"):
                sv_df = pd.DataFrame([
                    {
                        "Scenario": n,
                        f"Yr-1 PPA · {d.get('nem_regime_1', 'NEM-1')}":
                            f"${d.get('year1_rate_r1', d.get('year1_rate', 0)):.4f}",
                        f"Yr-1 PPA · {d['nem_regime_2']}" if d.get("nem_regime_2") else "":
                            f"${d['year1_rate_r2']:.4f}" if d.get("year1_rate_r2") is not None else "",
                        "Esc. R1 (%/yr)": f"{d['ppa_escalator_r1']:.1f}",
                        "Esc. R2 (%/yr)" if d.get("ppa_escalator_r2") is not None else "":
                            f"{d['ppa_escalator_r2']:.1f}" if d.get("ppa_escalator_r2") is not None else "",
                        "Savings target (%)": f"{d['savings_pct']:.1f}",
                        "Lifetime savings": fmt_dollar(d["lifetime_savings"]),
                    }
                    for n, d in st.session_state["saved_ppa_scenarios"].items()
                ])
                sv_df = sv_df.loc[:, [c for c in sv_df.columns if c != ""]]
                st.markdown(render_styled_table(sv_df, bold_cols=["Scenario"]),
                            unsafe_allow_html=True)

        st.divider()

        # ── Chart view toggle ─────────────────────────────────────────
        _chart_mode = st.radio(
            "Chart view",
            ["Annual Bill ($K)", "Effective Rate ($/kWh)"],
            horizontal=True,
            key="ppa_chart_mode",
            help=(
                "Annual Bill shows total dollars paid per year. Effective Rate "
                "divides that total by the customer's kWh consumed, giving an "
                "apples-to-apples $/kWh comparison against the utility-only path."
            ),
        )

        # (_x was pre-computed above alongside the Save PPA handler so both
        # callers reference the same instance.)
        import plotly.graph_objects as go

        _NAVY, _GREEN, _BLUE, _TEAL, _AMBER = "#0E2841", "#45A750", "#1D6FA9", "#518484", "#D48A1A"

        # Effective-rate series derived from the projection's per-year load.
        _proj_load_col = (
            "Customer Load (kWh)" if "Customer Load (kWh)" in _main_projection.columns
            else "Load (kWh)"
        )
        _load_kwh = (
            _main_projection[_proj_load_col].to_numpy()
            if _proj_load_col in _main_projection.columns else None
        )
        _bill_mode = _chart_mode.startswith("Annual Bill")
        if _bill_mode:
            _y_util = (_bill_no / 1000).round(1)
            _y_ppa = (_total_ppa / 1000).round(1)
            _y_axis_title = "Annual Cost ($K)"
            _hover_unit = "$%{y:.1f}K"
        else:
            if _load_kwh is None or not (_load_kwh > 0).all():
                st.warning(
                    "Effective $/kWh view requires per-year load from the projection; "
                    "falling back to Annual Bill view."
                )
                _bill_mode = True
                _y_util = (_bill_no / 1000).round(1)
                _y_ppa = (_total_ppa / 1000).round(1)
                _y_axis_title = "Annual Cost ($K)"
                _hover_unit = "$%{y:.1f}K"
            else:
                _y_util = pd.Series((_bill_no.to_numpy() / _load_kwh).round(5),
                                    index=_bill_no.index)
                _y_ppa = pd.Series((_total_ppa.to_numpy() / _load_kwh).round(5),
                                   index=_total_ppa.index)
                _y_axis_title = "Effective Rate ($/kWh)"
                _hover_unit = "$%{y:.4f}/kWh"

        _fig = go.Figure()

        _fig.add_trace(go.Scatter(
            x=_x, y=_y_util,
            name="Utility Only",
            mode="lines",
            line=dict(color=_NAVY, width=2.5, dash="dot"),
            hovertemplate=f"%{{x}}<br><b>{_hover_unit}</b><extra>Utility Only</extra>",
        ))

        if nem_switch and num_years_1 and num_years_1 < len(_cd):
            _split = int(num_years_1)
            _x_r1, _y_r1 = _x.iloc[:_split].tolist(), _y_ppa.iloc[:_split].tolist()
            _x_r2, _y_r2 = _x.iloc[_split - 1:].tolist(), _y_ppa.iloc[_split - 1:].tolist()
            _fig.add_trace(go.Scatter(
                x=_x_r1, y=_y_r1,
                name=f"Solar + PPA — {nem_regime_1}",
                mode="lines+markers",
                line=dict(color=_GREEN, width=3),
                marker=dict(size=5, color=_GREEN),
                fill="tonexty" if _bill_mode else None,
                fillcolor="rgba(69,167,80,0.15)" if _bill_mode else None,
                hovertemplate=f"%{{x}}<br><b>{_hover_unit}</b><extra>{nem_regime_1}</extra>",
            ))
            _fig.add_trace(go.Scatter(
                x=_x_r2, y=_y_r2,
                name=f"Solar + PPA — {nem_regime_2}",
                mode="lines+markers",
                line=dict(color=_BLUE, width=3),
                marker=dict(size=5, color=_BLUE),
                fill="tonexty" if _bill_mode else None,
                fillcolor="rgba(29,111,169,0.15)" if _bill_mode else None,
                hovertemplate=f"%{{x}}<br><b>{_hover_unit}</b><extra>{nem_regime_2}</extra>",
            ))
            _switch_x = _x.iloc[_split - 1] if _split - 1 < len(_x) else _x.iloc[-1]
            # Position the regime-switch label along the right-hand side of
            # the vertical rule at a mid y-value so it never collides with
            # the chart title or legend at the top.
            _fig.add_vline(
                x=_switch_x, line_color=_AMBER, line_width=2, line_dash="dash",
                annotation_text=f"→ {nem_regime_2}",
                annotation_position="top right",
                annotation_font=dict(color=_AMBER, size=11),
                annotation_bgcolor="rgba(255,255,255,0.9)",
                annotation_bordercolor=_AMBER,
                annotation_borderwidth=1,
                annotation_borderpad=3,
            )
        else:
            _fig.add_trace(go.Scatter(
                x=_x, y=_y_ppa,
                name="Solar + PPA",
                mode="lines+markers",
                line=dict(color=_GREEN, width=3),
                marker=dict(size=5, color=_GREEN),
                fill="tonexty" if _bill_mode else None,
                fillcolor="rgba(69,167,80,0.15)" if _bill_mode else None,
                hovertemplate=f"%{{x}}<br><b>{_hover_unit}</b><extra>Solar + PPA</extra>",
            ))

        # Overlay any saved PPA scenarios as thin ghost lines (same mode as the chart).
        _saved = st.session_state.get("saved_ppa_scenarios", {}) or {}
        _palette = [_TEAL, _AMBER, "#8E44AD", "#C0392B", "#117864"]
        for _si, (_sname, _sdata) in enumerate(_saved.items()):
            _syr = _sdata.get("calendar_year")
            if _bill_mode:
                _sy = _sdata.get("total_ppa_bill_k")
            else:
                _sy = _sdata.get("ppa_effective_rate")
            if not _syr or not _sy or len(_syr) != len(_sy):
                continue
            _fig.add_trace(go.Scatter(
                x=_syr, y=_sy,
                name=f"Saved · {_sname}",
                mode="lines",
                line=dict(color=_palette[_si % len(_palette)], width=1.8, dash="dashdot"),
                opacity=0.75,
                hovertemplate=f"%{{x}}<br>{_hover_unit}<extra>{_sname}</extra>",
            ))

        # Legend goes BELOW the plot so it never collides with the title
        # or regime-switch label. Top margin shrinks; bottom margin grows
        # to make room for a 2-row legend when many PPAs are overlaid.
        _num_traces = max(1, len(_fig.data))
        _legend_rows = 1 if _num_traces <= 3 else 2
        _bottom_margin = 70 + (_legend_rows - 1) * 22
        _fig.update_layout(
            title=dict(
                text=("Annual Bill: Utility Only vs Solar + PPA" if _bill_mode
                      else "Effective $/kWh Paid: Utility Only vs Solar + PPA"),
                font=dict(size=15, color=_NAVY),
                y=0.96, x=0.01, xanchor="left",
            ),
            xaxis_title="Year",
            yaxis_title=_y_axis_title,
            yaxis=dict(rangemode="tozero", gridcolor="#E5E7EB", automargin=True),
            xaxis=dict(gridcolor="#E5E7EB", automargin=True),
            template="plotly_white",
            height=500,
            margin=dict(l=70, r=40, t=60, b=_bottom_margin),
            font=dict(family="Inter, Aptos Narrow, sans-serif",
                      size=12, color="#1A1A1A"),
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.18,
                xanchor="center", x=0.5,
                font=dict(color="#1A1A1A", size=11),
                bgcolor="rgba(255,255,255,0)",
                borderwidth=0,
            ),
            hovermode="x unified",
            transition=dict(duration=350, easing="cubic-in-out"),
        )

        # Savings band midpoint annotations — bill-mode only. Place them
        # in the upper half of the savings band so the text doesn't sit on
        # top of either line or clip the fill patterning.
        if _bill_mode:
            def _annotate_savings(lo: int, hi: int, color: str) -> None:
                if hi <= lo:
                    return
                mid = (lo + hi) // 2
                if mid >= len(_cd):
                    return
                gap = float(_bill_no.iloc[mid] - _total_ppa.iloc[mid])
                # Anchor the label ~25% down from the utility line toward
                # the PPA line so it sits inside the savings band without
                # overlapping either line.
                band_y = float(
                    (0.75 * _bill_no.iloc[mid] + 0.25 * _total_ppa.iloc[mid]) / 1000
                )
                if gap > 0:
                    _fig.add_annotation(
                        x=_x.iloc[mid], y=band_y,
                        text=f"<b>${gap / 1000:.1f}K savings</b>",
                        showarrow=False,
                        font=dict(size=11, color=color),
                        bgcolor="rgba(255,255,255,0.94)",
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=4,
                    )

            if nem_switch and num_years_1 and num_years_1 < len(_cd):
                _annotate_savings(0, int(num_years_1), "#2D7A3C")
                _annotate_savings(int(num_years_1), len(_cd), "#13477A")
            else:
                _annotate_savings(0, len(_cd), "#2D7A3C")

        st.plotly_chart(_fig, use_container_width=True,
                        key=f"ppa_dashboard_chart_{_chart_mode}")

        # ── Data Table (Annual / Monthly) ──────────────────────────
        if it_view == "Annual":
            it_df = it_annual_df
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Bill w/ Solar ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            for _sav_col in ["Utility Savings ($)", "Customer Savings ($)", "NSC Adj ($)"]:
                if _sav_col in it_display.columns:
                    it_display[_sav_col] = it_display[_sav_col].apply(fmt_dollar)
            if "Solar (kWh)" in it_display.columns:
                it_display["Solar (kWh)"] = it_display["Solar (kWh)"].apply(fmt_num)
            if "Savings Target (%)" in it_display.columns:
                it_display["Savings Target (%)"] = it_display["Savings Target (%)"].apply(
                    lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x)
                )
            if "PPA Rate ($/kWh)" in it_display.columns:
                it_display["PPA Rate ($/kWh)"] = it_display["PPA Rate ($/kWh)"].apply(fmt_rate)
            st.markdown(
                render_styled_table(it_display, bold_cols=["PPA Rate ($/kWh)"]),
                unsafe_allow_html=True,
            )
        else:
            it_monthly = _build_multiyear_monthly_df(
                result=result,
                result_pv_only=pv_only_for_display,
                rate_escalator_pct=rate_escalator,
                load_escalator_pct=load_escalator,
                years=system_life_years,
                export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
                nem_regime_1=nem_regime_1,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
                export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
                cod_date=cod_date,
                degradation_pct=annual_degradation_pct,
                compound_escalation=compound_escalation,
                result_regime2=_result_regime2,
            )
            it_df = build_indexed_tariff_monthly(it_monthly, **_it_kw)
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Net Bill ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            for _sav_col in ["Utility Savings ($)", "Customer Savings ($)", "NSC Adj ($)"]:
                if _sav_col in it_display.columns:
                    it_display[_sav_col] = it_display[_sav_col].apply(fmt_dollar)
            if "Solar (kWh)" in it_display.columns:
                it_display["Solar (kWh)"] = it_display["Solar (kWh)"].apply(fmt_num)
            if "Savings Target (%)" in it_display.columns:
                it_display["Savings Target (%)"] = it_display["Savings Target (%)"].apply(
                    lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x)
                )
            if "PPA Rate ($/kWh)" in it_display.columns:
                it_display["PPA Rate ($/kWh)"] = it_display["PPA Rate ($/kWh)"].apply(fmt_rate)
            st.markdown(
                render_styled_table(it_display, bold_cols=["PPA Rate ($/kWh)"]),
                unsafe_allow_html=True,
            )

        # Formula explanation
        with st.expander("How is the PPA Rate calculated?"):
            st.markdown(
                "**PPA Rate** = [(1 − Savings%) × Bill w/o Solar − Bill w/ Solar] / Solar kWh\n\n"
                "This gives the maximum $/kWh a customer can pay for solar and still achieve "
                "their savings target relative to the utility-only bill."
            )

    with tab_indexed:
        _ppa_dashboard()

    with tab_proposals:
        _render_proposals_tab(
            simulation_name=_sim_name_for_props,
            result=result,
            pv_only_result=pv_only_for_display,
            main_projection=_main_projection,
            system_size_kw=system_size_kw,
            dc_ac_ratio=dc_ac_ratio,
            battery_cap_kwh=float(st.session_state.get("battery_capacity_kwh", 0.0) or 0.0),
            system_cost=system_cost,
            system_life_years=system_life_years,
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            utility_name=utility_name,
            selected_rate_name=selected_rate_name,
            rate_escalator=rate_escalator,
            load_escalator=load_escalator,
            compound_escalation=compound_escalation,
            cod_date=cod_date,
            annual_degradation_pct=annual_degradation_pct,
            common_nem_kw=_common_nem_kw,
            rs_old_baseline=_rs_old_baseline_for_proj,
            es_offset_annual=_es_offset_annual,
        )

    with tab_sensitivity:
        _render_sensitivity_tab(
            result=result,
            result_pv_only=pv_only_for_display,
            system_cost=system_cost,
            rate_escalator=rate_escalator,
            load_escalator=load_escalator,
            degradation_pct=st.session_state.get("pv_degradation_pct", 0.5),
            system_life_years=system_life_years,
            nem_regime_1=_common_nem_kw.get("nem_regime_1", "NEM-3 / NVBT"),
        )

    with tab_ai:
        _render_ai_assistant_tab(
            result=result,
            customer_name=st.session_state.get("customer_name", ""),
            address=st.session_state.get("sb_location", ""),
            system_size_kw=float(st.session_state.get("sb_system_size", 0.0) or 0.0),
            battery_capacity_kwh=float(st.session_state.get("battery_capacity_kwh", 0.0) or 0.0),
            nem_regime=_common_nem_kw.get("nem_regime_1", "NEM-3"),
            horizon_years=system_life_years,
            ppa_rate=st.session_state.get("ppa_rate_value"),
            tariff=st.session_state.get("tariff"),
        )

    # --- Downloads tab (always last) ---
    with tab5:
        st.subheader("Download Results")
        st.caption("Export simulation data as CSV or Excel files for further analysis.")

        # Monthly CSV — with year projection option
        dl_monthly_years = st.number_input(
            "Monthly CSV projection years",
            min_value=1, max_value=system_life_years, value=min(20, system_life_years), step=1,
            key="dl_monthly_years",
            help="1 = year-1 only (12 rows). >1 = multi-year monthly detail with escalation.",
        )

        # Build annual projection (shared by Annual CSV and Excel downloads)
        annual_proj_df = build_annual_projection(
            result=result,
            system_cost=system_cost,
            rate_escalator_pct=rate_escalator,
            load_escalator_pct=load_escalator,
            years=dl_monthly_years,
            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
            result_pv_only=pv_only_for_display,
            compound_escalation=compound_escalation,
            rate_shift_old_baseline=_rs_old_baseline_for_proj,
            existing_solar_offset_kwh=_es_offset_annual,
            **_common_nem_kw,
        )

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            monthly_csv = generate_monthly_csv(
                result,
                result_pv_only=pv_only_for_display,
                rate_escalator_pct=rate_escalator,
                load_escalator_pct=load_escalator,
                years=dl_monthly_years,
                export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
                nem_regime_1=nem_regime_1,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
                export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
                cod_date=cod_date,
                degradation_pct=annual_degradation_pct,
                compound_escalation=compound_escalation,
            )
            _monthly_label = (
                "Download Monthly Summary CSV"
                if dl_monthly_years <= 1
                else f"Download Monthly Summary CSV ({dl_monthly_years}yr)"
            )
            st.download_button(
                label=_monthly_label,
                data=monthly_csv,
                file_name="pv_sim_monthly_summary.csv",
                mime="text/csv",
            )
        with col_dl2:
            annual_csv = generate_annual_csv(_negate_outflow_columns(annual_proj_df))
            st.download_button(
                label=f"Download Annual Summary CSV ({dl_monthly_years}yr)",
                data=annual_csv,
                file_name="pv_sim_annual_summary.csv",
                mime="text/csv",
            )
        with col_dl3:
            hourly_csv = generate_hourly_csv(result, cod_date=cod_date)
            st.download_button(
                label="Download Hourly 8760 CSV",
                data=hourly_csv,
                file_name="pv_sim_hourly_8760.csv",
                mime="text/csv",
            )

        st.divider()
        excel_bytes = generate_simulation_excel(
            sim_name=sim_name,
            system_size_kw=system_size_kw,
            dc_ac_ratio=dc_ac_ratio,
            production_summary=st.session_state.get("production_summary"),
            location_input=location_input,
            lat=lat, lon=lon,
            system_life_years=system_life_years,
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            battery_capacity_kwh=st.session_state.get("battery_capacity_kwh", 0),
            discharge_limit_pct=(
                batt_cfg_.discharge_limit_pct
                if (batt_cfg_ := st.session_state.get("battery_config")) else 0.0
            ),
            utility_name=utility_name,
            selected_rate_name=selected_rate_name,
            rate_escalator_pct=rate_escalator,
            load_escalator_pct=load_escalator,
            annual_projection_df=annual_proj_df,
            result=result,
            result_pv_only=pv_only_for_display,
            export_rates_8760=st.session_state.get("export_rates"),
            export_rates_8760_2=st.session_state.get("export_rates_2") if nem_switch else None,
            nem_switch=nem_switch,
            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
            export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
            years=dl_monthly_years,
            cod_date=cod_date,
            degradation_pct=annual_degradation_pct,
        )
        st.download_button(
            label="Download Simulation Details (.xlsx)",
            data=excel_bytes,
            file_name="pv_sim_details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Monte Carlo sensitivity (if run) ---
        _mc_df = st.session_state.get("sensitivity_mc_df")
        if _mc_df is not None and len(_mc_df):
            st.divider()
            st.markdown("**Sensitivity — Monte Carlo samples**")
            st.caption(
                f"{len(_mc_df):,} samples from the last Monte Carlo run. Each row "
                "contains the sampled lever values and the resulting NPV of "
                "customer savings."
            )
            _mc_csv = _mc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Monte Carlo Samples (.csv)",
                data=_mc_csv,
                file_name="monte_carlo_samples.csv",
                mime="text/csv",
                key="dl_mc_csv",
            )
        _tornado_df = st.session_state.get("sensitivity_tornado_df")
        if _tornado_df is not None and len(_tornado_df):
            _t_csv = _tornado_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Tornado Results (.csv)",
                data=_t_csv,
                file_name="tornado_sensitivity.csv",
                mime="text/csv",
                key="dl_tornado_csv",
            )

        # Tranche 1 consolidation: Downloads no longer hosts a Customer
        # Proposal Deck block. The full deck build + export lives in the
        # PPA & Proposals → Proposals sub-tab so there's one surface that
        # owns Proposal state end-to-end. A thin pointer card stays here
        # for discoverability.
        st.divider()
        st.markdown(
            '<div style="background:var(--38dn-info-bg, #E8F1FA);'
            'border:1px solid #C9DBEC;border-left:3px solid #1D6FA9;'
            'border-radius:6px;padding:12px 16px;margin:8px 0;'
            'font-size:13px;color:#0E2841;">'
            '<div style="font-size:10px;font-weight:600;color:#1D6FA9;'
            'text-transform:uppercase;letter-spacing:0.06em;'
            'margin-bottom:6px;">Customer Proposal Deck</div>'
            "Deck (PPTX), Deck + Comparison Appendix, and Comparison XLSX "
            "exports live in <strong>PPA &amp; Proposals → Proposals</strong>. "
            "Pick a Proposal from the top-bar popover to jump there."
            "</div>",
            unsafe_allow_html=True,
        )


if st.session_state["billing_result"] is not None:
    _render_results()
