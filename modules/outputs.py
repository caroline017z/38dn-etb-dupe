"""
Output generation module — charts, CSV builders, and summary formatters.
"""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO, BytesIO
from .billing import BillingResult


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Negative-value formatting helpers (accounting style)
# ---------------------------------------------------------------------------
def fmt_num(x) -> str:
    """Format a number as XXX,XXX with parentheses for negatives."""
    if not isinstance(x, (int, float)):
        return str(x)
    if x < 0:
        return f"({abs(x):,.0f})"
    return f"{x:,.0f}"


def fmt_dollar(x) -> str:
    """Format a number as $XXX,XXX with parentheses for negatives."""
    if not isinstance(x, (int, float)):
        return str(x)
    if x < 0:
        return f"$({abs(x):,.0f})"
    return f"${x:,.0f}"


def fmt_rate(x) -> str:
    """Format a number as $0.XXXXX rate with parentheses for negatives."""
    if not isinstance(x, (int, float)):
        return str(x)
    if x < 0:
        return f"$({abs(x):.5f})"
    return f"${x:.5f}"


def style_negative_red(styler):
    """Apply red styling to any cell whose text contains '(' (accounting negative)."""
    def _color(val):
        if isinstance(val, str) and "(" in val:
            return "color: #cc0000; background-color: #ffe0e0"
        return ""
    return styler.map(_color)


def render_styled_table(
    df: pd.DataFrame,
    bold_last_row: bool = False,
    bold_cols: list[str] | None = None,
) -> str:
    """Render a DataFrame as an HTML table string with red-negative styling.

    This bypasses st.dataframe's unreliable Styler support by generating
    raw HTML that st.markdown can render directly.

    Args:
        df: Pre-formatted DataFrame (all values already strings).
        bold_last_row: If True, bold the last row (TOTAL).
        bold_cols: Column names whose headers and values should be bold.

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True).
    """
    bold_set = set(bold_cols) if bold_cols else set()
    col_list = list(df.columns)

    html = [
        '<div style="overflow-x:auto;">',
        '<table style="width:100%; border-collapse:collapse; font-size:13px;">',
        "<thead><tr>",
    ]
    for col in col_list:
        weight = "font-weight:700;" if col in bold_set else ""
        html.append(
            f'<th style="text-align:right; padding:6px 10px; border-bottom:2px solid #ccc;'
            f' background:#f8f8f8; white-space:nowrap; {weight}">{col}</th>'
        )
    html.append("</tr></thead><tbody>")

    for i, (_, row) in enumerate(df.iterrows()):
        is_last = i == len(df) - 1
        row_weight = "font-weight:700;" if (bold_last_row and is_last) else ""
        border = "border-top:2px solid #ccc;" if (bold_last_row and is_last) else ""
        html.append(f"<tr style='{row_weight}{border}'>")
        for j, val in enumerate(row):
            s = str(val)
            col_bold = "font-weight:700;" if col_list[j] in bold_set else ""
            cell_style = f"text-align:right; padding:5px 10px; white-space:nowrap; {col_bold}"
            if j == 0:
                cell_style = f"text-align:left; padding:5px 10px; white-space:nowrap; {col_bold}"
            if "(" in s:
                cell_style += " color:#cc0000; background-color:#ffe0e0;"
            html.append(f"<td style='{cell_style}'>{s}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "\n".join(html)


def _negate_outflow_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Negate cost outflow columns and Export (kWh) for display (accounting style)."""
    out = df.copy()
    for col in ["Bill w/o Solar ($)", "Energy ($)", "Demand ($)",
                 "Fixed ($)", "Bill w/ Solar ($)"]:
        if col in out.columns:
            out[col] = out[col] * -1
    if "Export (kWh)" in out.columns:
        out["Export (kWh)"] = out["Export (kWh)"] * -1
    return out


def build_monthly_summary_display(
    result: BillingResult,
    result_pv_only: BillingResult | None = None,
) -> pd.DataFrame:
    """
    Format the monthly summary for display in Streamlit.
    Adds month names and formats currency columns.

    If result_pv_only is provided, the table includes both
    "Demand kW (PV)" (from result_pv_only) and "Demand kW (PV+BESS)" (from result).
    Otherwise, only "Demand kW (PV)" is shown (from result itself).
    """
    df = result.monthly_summary.copy()
    df["month_name"] = [MONTH_NAMES[m - 1] for m in df["month"]]

    # Build display columns and rename map
    display_cols = [
        "month_name", "load_kwh", "solar_kwh", "import_kwh", "export_kwh",
        "export_peak_kwh", "export_offpeak_kwh",
    ]
    rename_map = {
        "month_name": "Month",
        "load_kwh": "Load (kWh)",
        "solar_kwh": "Solar (kWh)",
        "import_kwh": "Import (kWh)",
        "export_kwh": "Export (kWh)",
        "export_peak_kwh": "Export Peak (kWh)",
        "export_offpeak_kwh": "Export Off-Peak (kWh)",
    }

    if result_pv_only is not None:
        # BESS mode: show both PV-only and PV+BESS demand
        df["demand_kw_pv"] = result_pv_only.monthly_summary["peak_demand_kw"]
        df["demand_kw_bess"] = df["peak_demand_kw"]
        display_cols += ["demand_kw_pv", "demand_kw_bess"]
        rename_map["demand_kw_pv"] = "Demand kW (PV)"
        rename_map["demand_kw_bess"] = "Demand kW (PV+BESS)"
    else:
        # PV-only mode
        display_cols.append("peak_demand_kw")
        rename_map["peak_demand_kw"] = "Demand kW (PV)"

    display_cols += [
        "energy_cost", "total_demand_charge",
        "fixed_charge",
    ]

    # NBC column (NEM-2 only — include when any month has nbc_charge > 0)
    _has_nbc = "nbc_charge" in result.monthly_summary.columns and result.monthly_summary["nbc_charge"].sum() > 0
    if _has_nbc:
        display_cols.append("nbc_charge")

    display_cols += ["export_credit", "net_bill"]

    rename_map.update({
        "energy_cost": "Energy ($)",
        "total_demand_charge": "Demand ($)",
        "fixed_charge": "Fixed ($)",
        "nbc_charge": "NBC ($)",
        "export_credit": "Export Credit ($)",
        "net_bill": "Net Bill ($)",
    })

    # Negate export_credit so it displays as a negative value in the table,
    # making the equation visible: Energy + Demand + Fixed + Export Credit = Net Bill
    if "export_credit" in df.columns:
        df["export_credit"] = -df["export_credit"]

    # Only include columns that exist
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].rename(columns=rename_map)

    # Format kWh columns
    kwh_cols = [c for c in df.columns if "(kWh)" in c]
    for col in kwh_cols:
        df[col] = df[col].apply(fmt_num)

    # Format kW columns
    kw_cols = [c for c in df.columns if "kW" in c and "(kWh)" not in c]
    for col in kw_cols:
        df[col] = df[col].apply(fmt_num)

    # Format $ columns
    dollar_cols = [c for c in df.columns if "($)" in c]
    for col in dollar_cols:
        df[col] = df[col].apply(fmt_dollar)

    return df


def build_savings_summary(result: BillingResult, system_cost: float = 0.0) -> dict:
    """Build a savings summary dictionary."""
    simple_payback = None
    if result.annual_savings > 0 and system_cost > 0:
        simple_payback = system_cost / result.annual_savings

    return {
        "annual_load_kwh": round(result.annual_load_kwh, 0),
        "annual_solar_kwh": round(result.annual_solar_kwh, 0),
        "solar_offset_pct": round(
            result.annual_solar_kwh / result.annual_load_kwh * 100
            if result.annual_load_kwh > 0 else 0, 1
        ),
        "annual_import_kwh": round(result.annual_import_kwh, 0),
        "annual_export_kwh": round(result.annual_export_kwh, 0),
        "annual_bill_without_solar": round(result.annual_bill_without_solar, 2),
        "annual_bill_with_solar": round(result.annual_bill_with_solar, 2),
        "annual_savings": round(result.annual_savings, 2),
        "savings_pct": round(result.savings_pct, 1),
        "system_cost": round(system_cost, 2),
        "simple_payback_years": round(simple_payback, 1) if simple_payback else None,
    }


def _compute_tou_netted_monthly(hourly_detail: pd.DataFrame) -> tuple[
    float, float, dict[int, float], dict[int, float]
]:
    """Compute TOU-netted energy cost and export credit from hourly data.

    Returns:
        (annual_tou_energy, annual_tou_credit,
         per_month_tou_energy, per_month_tou_credit)

    Under TOU netting (NEM-1/2), each month×period's import and export are
    netted at the TOU retail rate. Months with net-positive charges produce
    energy cost; months with net-negative charges produce export credit.
    """
    annual_energy = 0.0
    annual_credit = 0.0
    month_energy: dict[int, float] = {}
    month_credit: dict[int, float] = {}

    has_rate = "energy_rate" in hourly_detail.columns
    for month in range(1, 13):
        mm = hourly_detail.index.month == month
        if not has_rate:
            # ECC fallback
            ec = float(hourly_detail.loc[mm, "export_credit"].sum()) if "export_credit" in hourly_detail.columns else 0.0
            month_energy[month] = 0.0
            month_credit[month] = ec
            annual_credit += ec
            continue

        monthly_charge = 0.0
        for pidx in hourly_detail.loc[mm, "energy_period"].unique():
            pm = mm & (hourly_detail["energy_period"] == pidx)
            rate = hourly_detail.loc[pm, "energy_rate"].iloc[0]
            net = hourly_detail.loc[pm, "import_kwh"].sum() - hourly_detail.loc[pm, "export_kwh"].sum()
            monthly_charge += net * rate

        if monthly_charge >= 0:
            month_energy[month] = monthly_charge
            month_credit[month] = 0.0
            annual_energy += monthly_charge
        else:
            month_energy[month] = 0.0
            month_credit[month] = abs(monthly_charge)
            annual_credit += abs(monthly_charge)

    return annual_energy, annual_credit, month_energy, month_credit


def _compute_tou_netted_credit(hourly_detail: pd.DataFrame) -> float:
    """Compute annual TOU-netted export credit from hourly data (NEM-1/2 logic)."""
    _, credit, _, _ = _compute_tou_netted_monthly(hourly_detail)
    return credit


def build_annual_projection(
    result: BillingResult,
    system_cost: float,
    rate_escalator_pct: float,
    load_escalator_pct: float,
    years: int = 10,
    export_rates_multiyear: dict[int, "pd.Series"] | None = None,
    result_pv_only: BillingResult | None = None,
    nem_regime_1: str = "NEM-3/NVBT",
    nem_regime_2: str | None = None,
    num_years_1: int | None = None,
    export_rates_multiyear_2: dict[int, "pd.Series"] | None = None,
    cod_year: int | None = None,
    degradation_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Build a multi-year annual projection table.

    Escalators:
      - rate_escalator_pct: applied linearly to TOU energy rates each year
      - load_escalator_pct: applied linearly to load profile each year
        (increases energy consumption AND peak demand)
      - degradation_pct: annual solar production decline (e.g. 0.5 for 0.5%/yr)

    Args:
        result: Year-1 BillingResult
        system_cost: Total installed cost ($)
        rate_escalator_pct: Annual TOU rate escalation (e.g., 3.0 for 3%)
        load_escalator_pct: Annual load/demand growth (e.g., 2.0 for 2%)
        years: Number of years to project
        result_pv_only: If provided, PV-only result for separate demand column
        nem_regime_1: NEM regime for the first period (default NEM-3/NVBT)
        nem_regime_2: NEM regime for the second period (None if no switch)
        num_years_1: Number of years under regime 1 (None if no switch)
        export_rates_multiyear_2: Multi-year export rates for regime 2
    """
    year1_energy = float(result.monthly_summary["energy_cost"].sum())
    year1_demand = float(result.monthly_summary["total_demand_charge"].sum())
    year1_fixed = float(result.monthly_summary["fixed_charge"].sum())
    year1_export = float(result.monthly_summary["export_credit"].sum())
    year1_nbc = float(result.monthly_summary["nbc_charge"].sum()) if "nbc_charge" in result.monthly_summary.columns else 0.0
    year1_load_kwh = result.annual_load_kwh
    year1_solar_kwh = result.annual_solar_kwh
    year1_import_kwh = result.annual_import_kwh
    year1_export_kwh = result.annual_export_kwh
    year1_export_peak_kwh = float(result.monthly_summary["export_peak_kwh"].sum())
    year1_export_offpeak_kwh = float(result.monthly_summary["export_offpeak_kwh"].sum())

    # Demand kW columns
    year1_demand_kw_bess = 0.0
    if result_pv_only is not None:
        year1_demand_kw_pv = float(result_pv_only.monthly_summary["peak_demand_kw"].max())
        year1_demand_kw_bess = float(result.monthly_summary["peak_demand_kw"].max())
    else:
        year1_demand_kw_pv = float(result.monthly_summary["peak_demand_kw"].max())

    year1_bill_no_solar = result.annual_bill_without_solar

    # Baseline breakdown (no solar)
    year1_baseline_demand = year1_bill_no_solar - (year1_bill_no_solar - year1_demand - year1_fixed) - year1_fixed
    # Simplify: baseline_demand ≈ year1_demand scaled by (baseline peak / solar peak)
    # For projection, use the no-solar bill components directly
    year1_baseline_energy = year1_bill_no_solar - year1_baseline_demand - year1_fixed

    rate_mult = rate_escalator_pct / 100.0
    load_mult = load_escalator_pct / 100.0
    degrad_rate = degradation_pct / 100.0

    # Precompute BOTH energy baselines from hourly data so regime-shift years
    # use the correct energy cost:
    #   raw_energy = sum(import_kwh * energy_rate) — NEM-3/NVBT (no netting)
    #   tou_energy = positive side of TOU per-period netting — NEM-1/2
    #   tou_credit = negative side of TOU per-period netting — NEM-1/2
    raw_year1_energy = float(result.hourly_detail["energy_cost"].sum())
    tou_year1_energy, year1_tou_credit, _, _ = _compute_tou_netted_monthly(
        result.hourly_detail,
    )

    rows = []
    cumulative_savings = 0.0
    # Multi-year export rates: keyed by calendar year (e.g. {2026: Series, 2027: ...})
    if export_rates_multiyear and len(export_rates_multiyear) > 1:
        _my_keys = sorted(export_rates_multiyear.keys())
        multiyear_start = _my_keys[0]   # first calendar year in CSV
        multiyear_max = _my_keys[-1]     # last calendar year in CSV
    else:
        multiyear_start = 0
        multiyear_max = 0

    # Multi-year export rates for regime 2
    if export_rates_multiyear_2 and len(export_rates_multiyear_2) > 1:
        _my2_keys = sorted(export_rates_multiyear_2.keys())
        multiyear_start_2 = _my2_keys[0]
        multiyear_max_2 = _my2_keys[-1]
    else:
        multiyear_start_2 = 0
        multiyear_max_2 = 0

    for yr in range(1, years + 1):
        rate_factor = 1.0 + rate_mult * (yr - 1)
        load_factor = 1.0 + load_mult * (yr - 1)

        # Solar degrades compounding each year (year 1 = full output)
        solar_factor = (1.0 - degrad_rate) ** (yr - 1)

        # Load grows → more import, higher peaks; solar declines → less export
        yr_load_kwh = year1_load_kwh * load_factor
        yr_solar_kwh = year1_solar_kwh * solar_factor
        # Net increase in demand from load growth + solar decline
        net_delta = year1_load_kwh * (load_factor - 1) + year1_solar_kwh * (1.0 - solar_factor)
        # Extra demand first absorbs exports, remainder increases import
        yr_export_kwh = max(0, year1_export_kwh - net_delta)
        absorbed = year1_export_kwh - yr_export_kwh
        yr_import_kwh = year1_import_kwh + (net_delta - absorbed)

        # Export TOU volumes scale proportionally to total export
        export_volume_ratio = yr_export_kwh / year1_export_kwh if year1_export_kwh > 0 else 1.0
        yr_export_peak_kwh = year1_export_peak_kwh * export_volume_ratio
        yr_export_offpeak_kwh = year1_export_offpeak_kwh * export_volume_ratio

        # Demand kW scales with load growth
        yr_demand_kw_pv = year1_demand_kw_pv * load_factor
        yr_demand_kw_bess = year1_demand_kw_bess * load_factor

        # Demand cost: higher peaks from load growth
        yr_demand = year1_demand * load_factor
        yr_fixed = year1_fixed
        # Export credit: rates escalate, but export volume may shrink
        volume_ratio = yr_export_kwh / year1_export_kwh if year1_export_kwh > 0 else 1.0
        # Import volume ratio (for scaling raw energy cost under NEM-3)
        import_ratio = yr_import_kwh / year1_import_kwh if year1_import_kwh > 0 else load_factor

        # Determine active regime for this year
        if nem_regime_2 and num_years_1 and yr > num_years_1:
            active_regime = nem_regime_2
            active_multiyear = export_rates_multiyear_2
            active_my_start = multiyear_start_2
            active_my_max = multiyear_max_2
        else:
            active_regime = nem_regime_1
            active_multiyear = export_rates_multiyear
            active_my_start = multiyear_start
            active_my_max = multiyear_max

        # Energy cost depends on active regime:
        #   NEM-1/2: TOU-netted energy (lower because exports offset imports within each period)
        #   NEM-3/NVBT: raw import energy cost (no netting, exports valued separately)
        if active_regime in ("NEM-1", "NEM-2"):
            yr_energy = tou_year1_energy * load_factor * rate_factor
        else:
            yr_energy = raw_year1_energy * import_ratio * rate_factor

        # Compute export credit based on active regime
        if active_regime in ("NEM-1", "NEM-2"):
            # TOU-netted credit scaled by rate escalation and volume change
            yr_export = year1_tou_credit * rate_factor * volume_ratio
        elif active_my_max > 0 and active_multiyear:
            # NEM-3/NVBT with multi-year CSV: look up by actual calendar year
            calendar_year = (cod_year if cod_year is not None else active_my_start) + (yr - 1)
            rate_year = max(active_my_start, min(calendar_year, active_my_max))
            yr_export_rates = active_multiyear[rate_year].values  # 8760 array
            hourly_export = result.hourly_detail["export_kwh"].values  # 8760 array
            base_credit = float(np.sum(hourly_export * yr_export_rates))
            # Apply rate escalation for projection years beyond the CSV range
            if calendar_year > active_my_max:
                overshoot = calendar_year - active_my_max
                export_esc = 1.0 + rate_mult * overshoot
            else:
                export_esc = 1.0
            yr_export = base_credit * volume_ratio * export_esc
        else:
            # Single-year / flat / auto: existing rate escalation
            yr_export = year1_export * rate_factor * volume_ratio

        # NBC: only applies during NEM-2 regime years
        if active_regime == "NEM-2":
            yr_nbc = year1_nbc * rate_factor if year1_nbc > 0 else 0.0
        else:
            yr_nbc = 0.0

        yr_bill_solar = yr_energy + yr_demand + yr_fixed + yr_nbc - yr_export
        yr_export_display = -yr_export  # negative for display

        # Baseline (no solar): load grows and rates increase
        yr_baseline_energy = year1_baseline_energy * load_factor * rate_factor
        yr_baseline_demand = year1_baseline_demand * load_factor
        yr_bill_no_solar = yr_baseline_energy + yr_baseline_demand + yr_fixed

        yr_savings = yr_bill_no_solar - yr_bill_solar
        cumulative_savings += yr_savings

        row = {
            "Year": yr,
        }
        if cod_year is not None:
            row["Calendar Year"] = cod_year + yr - 1
        row.update({
            "Load (kWh)": round(yr_load_kwh),
            "Solar (kWh)": round(yr_solar_kwh),
            "Import (kWh)": round(yr_import_kwh),
            "Export (kWh)": round(max(yr_export_kwh, 0)),
            "Export Peak (kWh)": round(yr_export_peak_kwh),
            "Export Off-Peak (kWh)": round(yr_export_offpeak_kwh),
            "Demand kW (PV)": round(yr_demand_kw_pv),
        })
        if result_pv_only is not None:
            row["Demand kW (PV+BESS)"] = round(yr_demand_kw_bess)
        row.update({
            "Bill w/o Solar ($)": round(yr_bill_no_solar),
            "Energy ($)": round(yr_energy),
            "Demand ($)": round(yr_demand),
            "Fixed ($)": round(yr_fixed),
        })
        _any_nem2 = (nem_regime_1 == "NEM-2") or (nem_regime_2 == "NEM-2")
        if _any_nem2 or year1_nbc > 0:
            row["NBC ($)"] = round(yr_nbc)
        row.update({
            "Export Credit ($)": round(yr_export_display),
            "Bill w/ Solar ($)": round(yr_bill_solar),
            "Annual Savings ($)": round(yr_savings),
            "Cumulative Savings ($)": round(cumulative_savings),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def create_production_vs_load_chart(result: BillingResult) -> go.Figure:
    """Create a monthly grouped bar chart: Load vs Solar Production."""
    df = result.monthly_summary
    fig = go.Figure()
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["load_kwh"], name="Load", marker_color="#EF553B", opacity=0.85))
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["solar_kwh"], name="Solar Production", marker_color="#FFB347", opacity=0.85))
    fig.add_trace(go.Scatter(x=MONTH_NAMES, y=df["import_kwh"], name="Net Import", mode="lines+markers", line=dict(color="#636EFA", width=2.5), marker=dict(size=7)))
    fig.update_layout(title="Monthly Production vs. Load", xaxis_title="Month", yaxis_title="Energy (kWh)", barmode="group", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=450)
    return fig


def create_monthly_bill_chart(result: BillingResult) -> go.Figure:
    """Create a stacked bar chart showing monthly bill components."""
    df = result.monthly_summary
    fig = go.Figure()
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["energy_cost"], name="Energy Charges", marker_color="#636EFA"))
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["total_demand_charge"], name="Demand Charges", marker_color="#EF553B"))
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["fixed_charge"], name="Fixed Charges", marker_color="#AB63FA"))
    # NBC bar (NEM-2 only)
    if "nbc_charge" in df.columns and df["nbc_charge"].sum() > 0:
        fig.add_trace(go.Bar(x=MONTH_NAMES, y=df["nbc_charge"], name="NBC Charges", marker_color="#FFA15A"))
    fig.add_trace(go.Bar(x=MONTH_NAMES, y=-df["export_credit"], name="Export Credit", marker_color="#00CC96"))
    fig.update_layout(title="Monthly Bill Breakdown (With Solar)", xaxis_title="Month", yaxis_title="Cost ($)", barmode="relative", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=450)
    return fig


def generate_hourly_csv(result: BillingResult, cod_date=None) -> str:
    """Generate CSV string of hourly detail data for download."""
    df = result.hourly_detail.copy()
    if cod_date is not None:
        df = df[df.index >= pd.Timestamp(cod_date)]
    if "export_kwh" in df.columns and "export_credit" in df.columns:
        exp = df["export_kwh"]
        df["value_of_energy_dollar_per_kwh"] = np.where(
            exp > 0, df["export_credit"] / exp, 0.0,
        )
    df.index.name = "datetime"
    buf = StringIO()
    df.to_csv(buf)
    return buf.getvalue()


def _build_multiyear_monthly_df(
    result: BillingResult,
    result_pv_only: BillingResult | None = None,
    rate_escalator_pct: float = 0.0,
    load_escalator_pct: float = 0.0,
    years: int = 1,
    export_rates_multiyear: dict[int, "pd.Series"] | None = None,
    nem_regime_1: str = "NEM-3/NVBT",
    nem_regime_2: str | None = None,
    num_years_1: int | None = None,
    export_rates_multiyear_2: dict[int, "pd.Series"] | None = None,
    cod_date=None,
    degradation_pct: float = 0.0,
) -> pd.DataFrame:
    """Build a multi-year monthly DataFrame (12 × years rows).

    Scales year-1 monthly values with escalators and regime-aware
    export credit / NBC logic matching build_annual_projection.
    """
    _cod_month = cod_date.month if cod_date else 1
    _cod_day = cod_date.day if cod_date else 1
    _cod_year = cod_date.year if cod_date else None

    ms = result.monthly_summary
    rate_mult = rate_escalator_pct / 100.0
    load_mult = load_escalator_pct / 100.0
    degrad_rate = degradation_pct / 100.0

    # Determine multi-year export rate calendar-year start (regime 1)
    if export_rates_multiyear and len(export_rates_multiyear) > 1:
        _my_keys = sorted(export_rates_multiyear.keys())
        my_start = _my_keys[0]
        my_max = _my_keys[-1]
    else:
        my_start = 0
        my_max = 0

    # Multi-year export rates for regime 2
    if export_rates_multiyear_2 and len(export_rates_multiyear_2) > 1:
        _my2_keys = sorted(export_rates_multiyear_2.keys())
        my_start_2 = _my2_keys[0]
        my_max_2 = _my2_keys[-1]
    else:
        my_start_2 = 0
        my_max_2 = 0

    year1_load = float(ms["load_kwh"].sum())
    year1_solar = float(ms["solar_kwh"].sum())
    year1_export = float(ms["export_kwh"].sum())

    # Precompute per-month TOU-netted energy AND credit (for NEM-1/2 regime years)
    # and per-month raw import energy cost (for NEM-3 regime years)
    hd = result.hourly_detail
    _, _, month_tou_energy, month_tou_credits = _compute_tou_netted_monthly(hd)
    raw_month_energy: dict[int, float] = {}
    month_import_kwh: dict[int, float] = {}
    for month in range(1, 13):
        mm = hd.index.month == month
        raw_month_energy[month] = float(hd.loc[mm, "energy_cost"].sum())
        month_import_kwh[month] = float(hd.loc[mm, "import_kwh"].sum())

    year1_import_total = sum(month_import_kwh.values())

    # Precompute per-month weighted average retail rate (import-weighted)
    month_wtd_rate: dict[int, float] = {}
    for month in range(1, 13):
        mm = hd.index.month == month
        imp = float(hd.loc[mm, "import_kwh"].sum())
        if imp > 0:
            month_wtd_rate[month] = float((hd.loc[mm, "import_kwh"] * hd.loc[mm, "energy_rate"]).sum()) / imp
        else:
            month_wtd_rate[month] = 0.0

    _any_nem2 = (nem_regime_1 == "NEM-2") or (nem_regime_2 == "NEM-2")

    rows = []
    for yr in range(1, years + 1):
        rate_factor = 1.0 + rate_mult * (yr - 1)
        load_factor = 1.0 + load_mult * (yr - 1)
        solar_factor = (1.0 - degrad_rate) ** (yr - 1)
        net_delta = year1_load * (load_factor - 1) + year1_solar * (1.0 - solar_factor)
        yr_export_total = max(0, year1_export - net_delta)
        volume_ratio = yr_export_total / year1_export if year1_export > 0 else 1.0

        # Import volume ratio (for scaling raw energy cost under NEM-3)
        absorbed = year1_export - yr_export_total
        yr_import_total = year1_import_total + (net_delta - absorbed)
        import_ratio = yr_import_total / year1_import_total if year1_import_total > 0 else load_factor

        # Determine active regime for this year
        if nem_regime_2 and num_years_1 and yr > num_years_1:
            active_regime = nem_regime_2
            active_multiyear = export_rates_multiyear_2
            active_my_start = my_start_2
            active_my_max = my_max_2
        else:
            active_regime = nem_regime_1
            active_multiyear = export_rates_multiyear
            active_my_start = my_start
            active_my_max = my_max

        # Per-month export credit recompute based on active regime
        month_export_credit_override = None
        if active_regime in ("NEM-1", "NEM-2"):
            # TOU-netted credits scaled by rate escalation and volume change
            month_export_credit_override = {}
            for m in range(1, 13):
                month_export_credit_override[m] = month_tou_credits[m] * rate_factor * volume_ratio
        elif active_my_max > 0 and active_multiyear:
            calendar_year = (_cod_year if _cod_year is not None else active_my_start) + (yr - 1)
            cal_yr = max(active_my_start, min(calendar_year, active_my_max))
            yr_rates = active_multiyear[cal_yr].values
            hourly_export = hd["export_kwh"].values
            dt_index = hd.index
            month_idx = dt_index.month
            # Apply rate escalation for projection years beyond the CSV range
            if calendar_year > active_my_max:
                overshoot = calendar_year - active_my_max
                export_esc = 1.0 + rate_mult * overshoot
            else:
                export_esc = 1.0
            month_export_credit_override = {}
            for m in range(1, 13):
                mask = month_idx == m
                base = float(np.sum(hourly_export[mask] * yr_rates[mask]))
                month_export_credit_override[m] = base * volume_ratio * export_esc

        for _, mrow in ms.iterrows():
            m = int(mrow["month"])

            # Skip pre-COD months for Year 1
            if yr == 1 and cod_date and m < _cod_month:
                continue

            r = {}
            r["Year"] = yr
            if _cod_year is not None:
                r["Calendar Year"] = _cod_year + (yr - 1)
            r["Month"] = MONTH_NAMES[m - 1]

            # Pro-rate COD month (Year 1 only)
            _prorate = 1.0
            if yr == 1 and cod_date and m == _cod_month and _cod_day > 1:
                _days = calendar.monthrange(_cod_year, m)[1]
                _prorate = (_days - _cod_day + 1) / _days
                r["Month"] = f"{MONTH_NAMES[m - 1]} (partial)"

            r["Load (kWh)"] = round(mrow["load_kwh"] * load_factor * _prorate, 1)
            r["Solar (kWh)"] = round(mrow["solar_kwh"] * solar_factor * _prorate, 1)
            r["Import (kWh)"] = round(mrow["import_kwh"] * load_factor * _prorate, 1)
            r["Export (kWh)"] = round(mrow["export_kwh"] * volume_ratio * _prorate, 1)
            r["Export Peak (kWh)"] = round(mrow["export_peak_kwh"] * volume_ratio * _prorate, 1)
            r["Export Off-Peak (kWh)"] = round(mrow["export_offpeak_kwh"] * volume_ratio * _prorate, 1)

            if result_pv_only is not None:
                pv_row = result_pv_only.monthly_summary[result_pv_only.monthly_summary["month"] == m].iloc[0]
                r["Demand kW (PV)"] = round(pv_row["peak_demand_kw"] * load_factor, 2)
                r["Demand kW (PV+BESS)"] = round(mrow["peak_demand_kw"] * load_factor, 2)
            else:
                r["Demand kW (PV)"] = round(mrow["peak_demand_kw"] * load_factor, 2)

            r["Wtd Avg Rate ($/kWh)"] = round(month_wtd_rate[m] * rate_factor, 5)

            # Energy cost depends on active regime:
            #   NEM-1/2: TOU-netted energy (exports offset imports within each TOU period)
            #   NEM-3/NVBT: raw import energy cost (no netting; exports valued separately)
            if active_regime in ("NEM-1", "NEM-2"):
                r["Energy ($)"] = round(month_tou_energy[m] * load_factor * rate_factor * _prorate, 2)
            else:
                r["Energy ($)"] = round(raw_month_energy[m] * import_ratio * rate_factor * _prorate, 2)
            r["Demand ($)"] = round(mrow["total_demand_charge"] * load_factor, 2)
            r["Fixed ($)"] = round(mrow["fixed_charge"] * _prorate, 2)

            # NBC: only applies during NEM-2 regime years
            _m_nbc = 0.0
            if active_regime == "NEM-2" and "nbc_charge" in ms.columns and mrow["nbc_charge"] > 0:
                _m_nbc = round(mrow["nbc_charge"] * rate_factor, 2)
            if _any_nem2:
                r["NBC ($)"] = _m_nbc

            if month_export_credit_override is not None:
                r["Export Credit ($)"] = -round(month_export_credit_override[m] * _prorate, 2)
            else:
                r["Export Credit ($)"] = -round(mrow["export_credit"] * rate_factor * volume_ratio * _prorate, 2)

            r["Net Bill ($)"] = round(
                r["Energy ($)"] + r["Demand ($)"] + r["Fixed ($)"] + _m_nbc + r["Export Credit ($)"], 2
            )

            # Baseline bill (no-solar) per month — for Indexed Tariff PPA rate calc
            if result.monthly_baseline_details is not None:
                _bd = result.monthly_baseline_details[m - 1]
                r["Baseline Bill ($)"] = round(
                    _bd["energy"] * load_factor * rate_factor * _prorate
                    + _bd["demand"] * load_factor
                    + _bd["fixed"] * _prorate, 2)

            rows.append(r)

    return pd.DataFrame(rows)


def generate_monthly_csv(
    result: BillingResult,
    result_pv_only: BillingResult | None = None,
    rate_escalator_pct: float = 0.0,
    load_escalator_pct: float = 0.0,
    years: int = 1,
    export_rates_multiyear: dict[int, "pd.Series"] | None = None,
    nem_regime_1: str = "NEM-3/NVBT",
    nem_regime_2: str | None = None,
    num_years_1: int | None = None,
    export_rates_multiyear_2: dict[int, "pd.Series"] | None = None,
    cod_date=None,
    degradation_pct: float = 0.0,
) -> str:
    """Generate CSV string of monthly summary data for download.

    Produces a multi-year monthly table with COD-aware partial months
    and Calendar Year columns.
    """
    df = _build_multiyear_monthly_df(
        result=result,
        result_pv_only=result_pv_only,
        rate_escalator_pct=rate_escalator_pct,
        load_escalator_pct=load_escalator_pct,
        years=max(years, 1),
        export_rates_multiyear=export_rates_multiyear,
        nem_regime_1=nem_regime_1,
        nem_regime_2=nem_regime_2,
        num_years_1=num_years_1,
        export_rates_multiyear_2=export_rates_multiyear_2,
        cod_date=cod_date,
        degradation_pct=degradation_pct,
    )
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def generate_annual_csv(projection_df: pd.DataFrame) -> str:
    """Generate CSV string of annual projection data for download."""
    buf = StringIO()
    projection_df.to_csv(buf, index=False)
    return buf.getvalue()


def _indexed_tariff_savings_target(
    yr: int,
    base_savings_pct: float,
    savings_escalator_pct: float,
    regime_1_savings_pct: float | None,
    regime_2_savings_pct: float | None,
    nem_regime_2: str | None,
    num_years_1: int | None,
) -> float:
    """Compute the savings target (%) for a given projection year."""
    if regime_1_savings_pct is not None and regime_2_savings_pct is not None:
        if nem_regime_2 and num_years_1 and yr > num_years_1:
            base = regime_2_savings_pct
            esc_yr = yr - num_years_1
        else:
            base = regime_1_savings_pct
            esc_yr = yr
        return base + savings_escalator_pct * (esc_yr - 1)
    return base_savings_pct + savings_escalator_pct * (yr - 1)


def build_indexed_tariff_annual(
    annual_proj_df: pd.DataFrame,
    base_savings_pct: float,
    savings_escalator_pct: float = 0.0,
    regime_1_savings_pct: float | None = None,
    regime_2_savings_pct: float | None = None,
    nem_regime_2: str | None = None,
    num_years_1: int | None = None,
) -> pd.DataFrame:
    """Build an annual Indexed Tariff table solving for PPA rate per year.

    PPA Rate = [(1 - savings_frac) × Bill w/o Solar - Bill w/ Solar] / Solar kWh
    """
    rows = []
    for _, row in annual_proj_df.iterrows():
        yr = int(row["Year"])
        savings_target = _indexed_tariff_savings_target(
            yr, base_savings_pct, savings_escalator_pct,
            regime_1_savings_pct, regime_2_savings_pct,
            nem_regime_2, num_years_1,
        )
        savings_frac = savings_target / 100.0
        bill_no_solar = row["Bill w/o Solar ($)"]
        bill_solar = row["Bill w/ Solar ($)"]
        solar_kwh = row["Solar (kWh)"]

        if solar_kwh > 0:
            ppa_rate = ((1.0 - savings_frac) * bill_no_solar - bill_solar) / solar_kwh
        else:
            ppa_rate = 0.0

        r = {"Year": yr}
        if "Calendar Year" in row.index:
            r["Calendar Year"] = int(row["Calendar Year"])
        customer_savings = bill_no_solar - bill_solar
        r.update({
            "Bill w/o Solar ($)": round(bill_no_solar, 2),
            "Bill w/ Solar ($)": round(bill_solar, 2),
            "Customer Savings ($)": round(customer_savings, 2),
            "Solar (kWh)": round(solar_kwh, 0),
            "Savings Target (%)": round(savings_target, 1),
            "PPA Rate ($/kWh)": round(ppa_rate, 5),
        })
        rows.append(r)

    return pd.DataFrame(rows)


def build_indexed_tariff_monthly(
    multiyear_monthly_df: pd.DataFrame,
    base_savings_pct: float,
    savings_escalator_pct: float = 0.0,
    regime_1_savings_pct: float | None = None,
    regime_2_savings_pct: float | None = None,
    nem_regime_2: str | None = None,
    num_years_1: int | None = None,
) -> pd.DataFrame:
    """Build a monthly Indexed Tariff table solving for PPA rate per month.

    PPA Rate = [(1 - savings_frac) × Baseline Bill - Net Bill] / Solar kWh
    """
    rows = []
    for _, row in multiyear_monthly_df.iterrows():
        yr = int(row["Year"])
        savings_target = _indexed_tariff_savings_target(
            yr, base_savings_pct, savings_escalator_pct,
            regime_1_savings_pct, regime_2_savings_pct,
            nem_regime_2, num_years_1,
        )
        savings_frac = savings_target / 100.0

        baseline_bill = row.get("Baseline Bill ($)", 0.0)
        if baseline_bill is None or (isinstance(baseline_bill, float) and np.isnan(baseline_bill)):
            baseline_bill = 0.0
        net_bill = row["Net Bill ($)"]
        solar_kwh = row["Solar (kWh)"]

        if solar_kwh > 0:
            ppa_rate = ((1.0 - savings_frac) * baseline_bill - net_bill) / solar_kwh
        else:
            ppa_rate = 0.0

        r = {"Year": yr}
        if "Calendar Year" in row.index:
            r["Calendar Year"] = int(row["Calendar Year"])
        customer_savings = baseline_bill - net_bill
        r.update({
            "Month": row["Month"],
            "Bill w/o Solar ($)": round(baseline_bill, 2),
            "Net Bill ($)": round(net_bill, 2),
            "Customer Savings ($)": round(customer_savings, 2),
            "Solar (kWh)": round(solar_kwh, 0),
            "Savings Target (%)": round(savings_target, 1),
            "PPA Rate ($/kWh)": round(ppa_rate, 5),
        })
        rows.append(r)

    return pd.DataFrame(rows)


def build_grid_exchange_summary(
    result: BillingResult, peak_period_idx: int | frozenset[int] = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly grid import/export summary broken out by peak vs off-peak.

    Args:
        result: BillingResult with hourly_detail containing energy_period column
        peak_period_idx: TOU period index(es) considered "peak" (highest rate).
            Can be a single int or a frozenset of ints.

    Returns:
        (display_df, raw_df) — formatted DataFrame with TOTAL row, and raw numeric DataFrame
    """
    hd = result.hourly_detail
    hd_month = hd.index.month

    ep = hd["energy_period"].values
    if isinstance(peak_period_idx, (set, frozenset)):
        is_peak = np.isin(ep, list(peak_period_idx))
    else:
        is_peak = ep == peak_period_idx

    rows = []
    for m in range(1, 13):
        mm = hd_month == m
        peak_m = mm & is_peak
        offpeak_m = mm & ~is_peak

        imp_peak = float(hd.loc[peak_m, "import_kwh"].sum())
        imp_offpeak = float(hd.loc[offpeak_m, "import_kwh"].sum())
        imp_total = imp_peak + imp_offpeak

        exp_peak = float(hd.loc[peak_m, "export_kwh"].sum())
        exp_offpeak = float(hd.loc[offpeak_m, "export_kwh"].sum())
        exp_total = exp_peak + exp_offpeak

        cost_peak = float(hd.loc[peak_m, "energy_cost"].sum())
        cost_offpeak = float(hd.loc[offpeak_m, "energy_cost"].sum())
        cost_total = cost_peak + cost_offpeak

        credit_peak = float(hd.loc[peak_m, "export_credit"].sum())
        credit_offpeak = float(hd.loc[offpeak_m, "export_credit"].sum())
        credit_total = credit_peak + credit_offpeak

        rows.append({
            "Month": MONTH_NAMES[m - 1],
            "Import Total (kWh)": round(imp_total, 0),
            "Import Peak (kWh)": round(imp_peak, 0),
            "Import Off-Peak (kWh)": round(imp_offpeak, 0),
            "Export Total (kWh)": round(exp_total, 0),
            "Export Peak (kWh)": round(exp_peak, 0),
            "Export Off-Peak (kWh)": round(exp_offpeak, 0),
            "Import Cost Total ($)": round(cost_total, 0),
            "Import Cost Peak ($)": round(cost_peak, 0),
            "Import Cost Off-Peak ($)": round(cost_offpeak, 0),
            "Export Credit Total ($)": -round(credit_total, 0),
            "Export Credit Peak ($)": -round(credit_peak, 0),
            "Export Credit Off-Peak ($)": -round(credit_offpeak, 0),
        })

    raw_df = pd.DataFrame(rows)

    # Format display copy
    df = raw_df.copy()
    for c in [col for col in df.columns if "(kWh)" in col]:
        df[c] = df[c].apply(fmt_num)
    for c in [col for col in df.columns if "($)" in col]:
        df[c] = df[c].apply(fmt_dollar)

    # TOTAL row
    totals = {"Month": "TOTAL"}
    for c in raw_df.columns:
        if c == "Month":
            continue
        if "(kWh)" in c:
            totals[c] = fmt_num(raw_df[c].sum())
        elif "($)" in c:
            totals[c] = fmt_dollar(raw_df[c].sum())
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df, raw_df


def build_battery_kpi_summary(
    result_pv_only: BillingResult,
    result_batt: BillingResult,
    capacity_kwh: float,
) -> dict:
    """Compute battery-specific KPIs comparing PV-only to PV+battery results.

    Args:
        result_pv_only: BillingResult from PV-only simulation
        result_batt: BillingResult from PV+battery simulation
        capacity_kwh: Battery nameplate capacity (kWh)

    Returns:
        dict of KPI name -> value
    """
    hd = result_batt.hourly_detail

    # --- Charge / discharge totals ---
    total_charge = float(hd["batt_charge_kwh"].sum()) if "batt_charge_kwh" in hd.columns else 0.0
    total_discharge_to_load = float(hd["batt_to_load_kwh"].sum()) if "batt_to_load_kwh" in hd.columns else 0.0
    total_discharge_to_grid = float(hd["batt_to_grid_kwh"].sum()) if "batt_to_grid_kwh" in hd.columns else 0.0
    total_curtailed = float(hd["batt_curtailed_kwh"].sum()) if "batt_curtailed_kwh" in hd.columns else 0.0
    total_discharge = total_discharge_to_load + total_discharge_to_grid + total_curtailed

    # --- Cycles estimate: throughput / (2 * capacity) ---
    throughput = total_charge + total_discharge
    cycles = throughput / (2.0 * capacity_kwh) if capacity_kwh > 0 else 0.0

    # --- PV self-consumption ---
    # Self-consumed PV = solar production - export
    pv_only_self_consumption = result_pv_only.annual_solar_kwh - result_pv_only.annual_export_kwh
    batt_self_consumption = result_batt.annual_solar_kwh - result_batt.annual_export_kwh

    pv_only_self_pct = (
        pv_only_self_consumption / result_pv_only.annual_solar_kwh * 100
        if result_pv_only.annual_solar_kwh > 0 else 0.0
    )
    batt_self_pct = (
        batt_self_consumption / result_batt.annual_solar_kwh * 100
        if result_batt.annual_solar_kwh > 0 else 0.0
    )
    self_consumption_increase_pct = batt_self_pct - pv_only_self_pct

    # --- Export change ---
    export_change_kwh = result_batt.annual_export_kwh - result_pv_only.annual_export_kwh
    export_change_pct = (
        export_change_kwh / result_pv_only.annual_export_kwh * 100
        if result_pv_only.annual_export_kwh > 0 else 0.0
    )

    # --- Peak demand before vs after ---
    pv_only_peak = float(result_pv_only.monthly_summary["peak_demand_kw"].max())
    batt_peak = float(result_batt.monthly_summary["peak_demand_kw"].max())
    peak_reduction_kw = pv_only_peak - batt_peak
    peak_reduction_pct = (
        peak_reduction_kw / pv_only_peak * 100 if pv_only_peak > 0 else 0.0
    )

    # --- Import change ---
    import_change_kwh = result_batt.annual_import_kwh - result_pv_only.annual_import_kwh

    return {
        "total_charge_kwh": round(total_charge, 1),
        "total_discharge_kwh": round(total_discharge, 1),
        "discharge_to_load_kwh": round(total_discharge_to_load, 1),
        "discharge_to_grid_kwh": round(total_discharge_to_grid, 1),
        "throughput_kwh": round(throughput, 1),
        "cycles": round(cycles, 1),
        "pv_self_consumption_pv_only_pct": round(pv_only_self_pct, 1),
        "pv_self_consumption_batt_pct": round(batt_self_pct, 1),
        "self_consumption_increase_pct": round(self_consumption_increase_pct, 1),
        "export_change_kwh": round(export_change_kwh, 1),
        "export_change_pct": round(export_change_pct, 1),
        "pv_only_peak_kw": round(pv_only_peak, 2),
        "batt_peak_kw": round(batt_peak, 2),
        "peak_reduction_kw": round(peak_reduction_kw, 2),
        "peak_reduction_pct": round(peak_reduction_pct, 1),
        "import_change_kwh": round(import_change_kwh, 1),
        "curtailed_kwh": round(total_curtailed, 1),
    }


def generate_simulation_excel(
    sim_name: str,
    system_size_kw: float,
    dc_ac_ratio: float,
    production_summary: dict | None,
    location_input: str | None,
    lat: float | None,
    lon: float | None,
    system_life_years: int,
    nem_regime_1: str,
    nem_regime_2: str | None,
    num_years_1: int | None,
    battery_capacity_kwh: float,
    discharge_limit_pct: float,
    utility_name: str | None,
    selected_rate_name: str | None,
    rate_escalator_pct: float,
    load_escalator_pct: float,
    annual_projection_df: pd.DataFrame,
    result: BillingResult,
    result_pv_only: BillingResult | None,
    export_rates_8760: "pd.Series | None",
    export_rates_8760_2: "pd.Series | None",
    nem_switch: bool,
    export_rates_multiyear: dict[int, "pd.Series"] | None,
    export_rates_multiyear_2: dict[int, "pd.Series"] | None,
    years: int,
    cod_date=None,
    degradation_pct: float = 0.0,
) -> bytes:
    """Generate a multi-sheet Excel workbook with full simulation details.

    Returns bytes of the .xlsx file content.
    """
    # ------------------------------------------------------------------
    # 1. Summary sheet
    # ------------------------------------------------------------------
    annual_solar = result.annual_solar_kwh
    if production_summary and "ac_annual" in production_summary:
        annual_production = production_summary["ac_annual"]
    else:
        annual_production = annual_solar

    yield_kwh_kw = (
        round(annual_production / system_size_kw, 1)
        if system_size_kw > 0 else 0.0
    )

    system_size_kwac = round(system_size_kw / dc_ac_ratio, 2) if dc_ac_ratio > 0 else system_size_kw

    self_consumed = annual_solar - result.annual_export_kwh
    self_consumption_frac = (
        self_consumed / annual_solar
        if annual_solar > 0 else 0.0
    )
    export_frac = 1.0 - self_consumption_frac

    regime_2_term = (
        system_life_years - num_years_1
        if num_years_1 is not None else None
    )

    summary_rows = [
        ("Simulation Name", sim_name or "N/A"),
        ("Commercial Operation Date", cod_date.strftime("%B %d, %Y") if cod_date else "N/A"),
        ("System Size (kW-DC)", round(system_size_kw, 2)),
        ("System Size (kW-AC)", system_size_kwac),
        ("Yield (kWh/kW)", yield_kwh_kw),
        ("Annual Production (kWh)", round(annual_production, 0)),
        ("Self-Consumption (%)", self_consumption_frac),
        ("Export (%)", export_frac),
        ("System Life (years)", system_life_years),
        ("Location", location_input or "N/A"),
        ("Latitude", round(lat, 4) if lat is not None else "N/A"),
        ("Longitude", round(lon, 4) if lon is not None else "N/A"),
        ("NEM Regime 1", nem_regime_1),
        ("NEM Regime 1 Term (years)", num_years_1 if num_years_1 is not None else system_life_years),
        ("NEM Regime 2", nem_regime_2 or "N/A"),
        ("NEM Regime 2 Term (years)", regime_2_term if regime_2_term is not None else "N/A"),
        ("BESS Size (kWh)", battery_capacity_kwh),
        ("BESS Export Limit (%)", round(discharge_limit_pct * 100, 1) if discharge_limit_pct else 0.0),
        ("Utility", utility_name or "N/A"),
        ("Rate Tariff", selected_rate_name or "N/A"),
        ("Utility Escalator (%/yr)", rate_escalator_pct),
        ("Demand Escalator (%/yr)", load_escalator_pct),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])

    # ------------------------------------------------------------------
    # 2. Export Rates (Hourly) — 8760 rows
    # ------------------------------------------------------------------
    hd = result.hourly_detail
    export_hourly_data = {
        "Datetime": hd.index,
        "Export (kWh)": hd["export_kwh"].values,
    }
    # Rate column: NEM-1/2 → retail TOU rate; NEM-3/NVBT → ACC export rate
    if nem_regime_1 in ("NEM-1", "NEM-2") or export_rates_8760 is None:
        export_hourly_data["Export Rate ($/kWh)"] = hd["energy_rate"].values
    else:
        export_hourly_data["Export Rate ($/kWh)"] = (
            export_rates_8760.values
            if hasattr(export_rates_8760, "values") else export_rates_8760
        )
    # Value of Energy: |export_credit| / export_kwh (0 when no export)
    exp_kwh = hd["export_kwh"].values
    exp_credit = np.abs(hd["export_credit"].values)
    export_hourly_data["Value of Energy ($/kWh)"] = np.where(
        exp_kwh > 0, exp_credit / exp_kwh, 0.0,
    )
    export_hourly_df = pd.DataFrame(export_hourly_data)

    # ------------------------------------------------------------------
    # 3. Retail Rates (Hourly) — 8760 rows
    # ------------------------------------------------------------------
    retail_hourly_df = pd.DataFrame({
        "Datetime": hd.index,
        "Retail Rate ($/kWh)": hd["energy_rate"].values,
        "Import (kWh)": hd["import_kwh"].values,
    })

    # ------------------------------------------------------------------
    # 4. Monthly sheets (export and retail)
    # ------------------------------------------------------------------
    monthly_df = _build_multiyear_monthly_df(
        result=result,
        result_pv_only=result_pv_only,
        rate_escalator_pct=rate_escalator_pct,
        load_escalator_pct=load_escalator_pct,
        years=years,
        export_rates_multiyear=export_rates_multiyear,
        nem_regime_1=nem_regime_1,
        nem_regime_2=nem_regime_2,
        num_years_1=num_years_1,
        export_rates_multiyear_2=export_rates_multiyear_2,
        cod_date=cod_date,
        degradation_pct=degradation_pct,
    )
    export_monthly_cols = ["Year", "Calendar Year", "Month", "Export (kWh)", "Export Credit ($)"]
    retail_monthly_cols = ["Year", "Calendar Year", "Month", "Import (kWh)", "Wtd Avg Rate ($/kWh)", "Energy ($)"]
    export_monthly_df = monthly_df[[c for c in export_monthly_cols if c in monthly_df.columns]].copy()
    # Weighted average Value of Energy: |Export Credit| / Export kWh per month
    if "Export (kWh)" in export_monthly_df.columns and "Export Credit ($)" in export_monthly_df.columns:
        m_exp = export_monthly_df["Export (kWh)"]
        m_credit = export_monthly_df["Export Credit ($)"].abs()
        export_monthly_df["Value of Energy ($/kWh)"] = np.where(
            m_exp > 0, m_credit / m_exp, 0.0,
        )
    retail_monthly_df = monthly_df[[c for c in retail_monthly_cols if c in monthly_df.columns]].copy()

    # ------------------------------------------------------------------
    # 5. Assemble workbook
    # ------------------------------------------------------------------
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        annual_display_df = _negate_outflow_columns(annual_projection_df)
        annual_display_df.to_excel(writer, sheet_name="Annual Savings", index=False)
        export_hourly_df.to_excel(writer, sheet_name="Export Rates (Hourly)", index=False)
        export_monthly_df.to_excel(writer, sheet_name="Export Rates (Monthly)", index=False)
        retail_hourly_df.to_excel(writer, sheet_name="Retail Rates (Hourly)", index=False)
        retail_monthly_df.to_excel(writer, sheet_name="Retail Rates (Monthly)", index=False)

        # Style header rows (row 1) across all sheets
        from openpyxl.styles import Alignment, Font, PatternFill
        _header_fill = PatternFill(start_color="212B48", end_color="212B48", fill_type="solid")
        _header_font = Font(color="FFFFFF", bold=True)
        for ws in writer.sheets.values():
            for cell in ws[1]:
                cell.fill = _header_fill
                cell.font = _header_font

        # Left-align the Value column on the Summary sheet
        ws_summary = writer.sheets["Summary"]
        _left = Alignment(horizontal="left")
        for row_idx in range(1, len(summary_df) + 2):  # header + data rows
            ws_summary.cell(row=row_idx, column=2).alignment = _left

        # Format percentage rows on the Summary sheet
        _pct_params = {"Self-Consumption (%)", "Export (%)"}
        for row_idx, (param, _) in enumerate(summary_rows, start=2):
            if param in _pct_params:
                ws_summary.cell(row=row_idx, column=2).number_format = '0.0%'

        # Apply number formats to data sheets
        _fmt_kwh = '#,##0'
        _fmt_dollar = '$#,##0'
        _fmt_dollar_acct = '$#,##0_);[Red]($#,##0)'
        _fmt_rate = '$0.00000'

        for sheet_name, df, dollar_fmt in [
            ("Export Rates (Hourly)", export_hourly_df, _fmt_dollar),
            ("Export Rates (Monthly)", export_monthly_df, _fmt_dollar),
            ("Retail Rates (Hourly)", retail_hourly_df, _fmt_dollar),
            ("Retail Rates (Monthly)", retail_monthly_df, _fmt_dollar),
            ("Annual Savings", annual_display_df, _fmt_dollar_acct),
        ]:
            ws = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(df.columns, start=1):
                if "(kWh)" in col_name:
                    fmt = _fmt_kwh
                elif "($/kWh)" in col_name:
                    fmt = _fmt_rate
                elif "($)" in col_name:
                    fmt = dollar_fmt
                elif "kW" in col_name and "(kWh)" not in col_name:
                    fmt = _fmt_kwh
                else:
                    continue
                for row_idx in range(2, len(df) + 2):  # skip header row
                    ws.cell(row=row_idx, column=col_idx).number_format = fmt

        # Auto-fit column widths across all sheets
        from openpyxl.utils import get_column_letter
        for ws in writer.sheets.values():
            for col_idx in range(1, ws.max_column + 1):
                max_len = 0
                col_letter = get_column_letter(col_idx)
                for row_idx in range(1, min(ws.max_row + 1, 1002)):  # sample header + up to 1000 rows
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if cell.value is not None:
                        cell_len = len(str(cell.value))
                        if cell_len > max_len:
                            max_len = cell_len
                ws.column_dimensions[col_letter].width = max_len + 3

    return buf.getvalue()
