"""
PV Solar Rate Simulator — Streamlit Application

Simulates annual electricity bills for California agricultural and commercial
customers with solar PV systems under Net Value Billing Tariff (NVBT).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
from dataclasses import asdict
from datetime import date, datetime
from typing import cast
from dotenv import load_dotenv

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
    get_export_rates,
    load_acc_from_upload,
    create_flat_export_rates,
    parse_multiyear_export_rates,
)
from modules.billing import run_billing_simulation, BillingResult
from modules.billing_ecc import (
    fetch_and_populate_ecc_tariff,
    load_ecc_tariff_from_json,
    run_ecc_billing_simulation,
)
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
    style_negative_red,
    render_styled_table,
)

load_dotenv()

# =============================================================================
# DIRECTORIES
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SIMULATIONS_DIR = os.path.join(DATA_DIR, "simulations")
LOAD_PROFILES_DIR = os.path.join(DATA_DIR, "load_profiles")
EXPORT_PROFILES_DIR = os.path.join(DATA_DIR, "export_profiles")
ECC_TARIFFS_DIR = os.path.join(DATA_DIR, "ecc_tariffs")
SYSTEM_PROFILES_DIR = os.path.join(DATA_DIR, "system_profiles")

for d in [SIMULATIONS_DIR, LOAD_PROFILES_DIR, EXPORT_PROFILES_DIR, ECC_TARIFFS_DIR, SYSTEM_PROFILES_DIR]:
    os.makedirs(d, exist_ok=True)


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
)


def _list_saved(directory: str, ext: str = ".json") -> list[str]:
    """Generic file lister — still used by Load Profiles / Export Profiles."""
    files = glob.glob(os.path.join(directory, f"*{ext}"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _delete_file(directory, name, ext):
    """Generic file deleter — still used by Load Profiles / Export Profiles."""
    fp = os.path.join(directory, f"{name}{ext}")
    if os.path.exists(fp):
        os.remove(fp)


# =============================================================================
# HELPER — Simulation Progress Overlay
# =============================================================================
def _progress_overlay_html(pct: int, step_text: str) -> str:
    """Return HTML for a translucent overlay with circular progress indicator."""
    deg = int(pct * 3.6)  # 0-360 degrees
    return f"""
    <div id="sim-overlay" style="
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0, 0, 0, 0.55); z-index: 99999;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        backdrop-filter: blur(2px);
    ">
        <p style="
            color: white; font-size: 22px; font-weight: 600;
            margin-bottom: 24px; letter-spacing: 0.5px;
        ">Simulation Progress</p>
        <div style="
            width: 130px; height: 130px; border-radius: 50%;
            background: conic-gradient(#636EFA {deg}deg, rgba(255,255,255,0.15) {deg}deg);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 30px rgba(99,110,250,0.4);
        ">
            <div style="
                width: 100px; height: 100px; border-radius: 50%;
                background: rgba(20, 20, 30, 0.92);
                display: flex; align-items: center; justify-content: center;
            ">
                <span style="color: white; font-size: 26px; font-weight: 700;">{pct}%</span>
            </div>
        </div>
        <p style="
            color: rgba(255,255,255,0.8); font-size: 14px;
            margin-top: 16px;
        ">{step_text}</p>
    </div>
    """


# =============================================================================
# HELPER FUNCTIONS — Profiles (Load & Export)
# =============================================================================
def _save_profile_csv(directory, name, df):
    df.to_csv(os.path.join(directory, f"{name}.csv"), index=False)


def _load_profile_csv(directory, name) -> pd.DataFrame:
    return pd.read_csv(os.path.join(directory, f"{name}.csv"))


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
    fp = os.path.join(SYSTEM_PROFILES_DIR, f"{name}.json")
    with open(fp, "w") as f:
        json.dump(profile, f)


def _load_system_profile(name: str) -> dict:
    """Load a system profile JSON and return the dict."""
    fp = os.path.join(SYSTEM_PROFILES_DIR, f"{name}.json")
    with open(fp, "r") as f:
        return json.load(f)


def _parse_8760_csv(df: pd.DataFrame) -> np.ndarray:
    """Extract the first numeric column from a DataFrame, validate 8760 rows."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV.")
    values = np.asarray(df[numeric_cols[0]].values)
    if len(values) != 8760:
        raise ValueError(f"Expected 8760 rows, got {len(values)}.")
    return values


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
</style>
""", unsafe_allow_html=True)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: absolute;
            top: -60px;
            right: 20px;
            z-index: 999999;
            pointer-events: none;
        }}
        .top-right-logo img {{
            height: 72px;
            width: 72px;
            object-fit: contain;
        }}
        </style>
        <div class="top-right-logo">
            <img src="data:image/png;base64,{logo_b64}" alt="38DN Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.session_state["production_8760"] = np.array(_sp_data["production_8760"])
    if _sp_data.get("production_summary"):
        st.session_state["production_summary"] = _sp_data["production_summary"]
    st.rerun()

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

# =============================================================================
# TOP MANAGEMENT BUTTONS — Simulations / Load Profiles / Export Profiles
# =============================================================================
# Toggle-button CSS: active button gets a blue highlight, others grey
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .mgmt-btn button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Teal styling for the top management button row */
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"],
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stPopoverButton"] {
    border-color: #2A7B7B !important;
    color: #2A7B7B !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"]:hover,
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="stPopoverButton"]:hover {
    border-color: #1E5C5C !important;
    color: #1E5C5C !important;
    background-color: rgba(42, 123, 123, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

_mgmt_btn_cols = st.columns([1.2, 0.05, 0.8, 0.05, 1, 0.05, 1, 0.05, 1.2])

# --- Simulations popover ---
with _mgmt_btn_cols[0]:
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

# --- System Profiles toggle ---
with _mgmt_btn_cols[2]:
    if st.button(
        "System Profiles",
        key="mgmt_btn_system",
        width="stretch",
        type="primary" if st.session_state.active_mgmt_tab == "System Profiles" else "secondary",
    ):
        st.session_state.active_mgmt_tab = (
            None if st.session_state.active_mgmt_tab == "System Profiles" else "System Profiles"
        )
        st.rerun()

# --- Load Profiles toggle ---
with _mgmt_btn_cols[4]:
    if st.button(
        "Load Profiles",
        key="mgmt_btn_load",
        width="stretch",
        type="primary" if st.session_state.active_mgmt_tab == "Load Profiles" else "secondary",
    ):
        st.session_state.active_mgmt_tab = (
            None if st.session_state.active_mgmt_tab == "Load Profiles" else "Load Profiles"
        )
        st.rerun()

# --- Export Profiles toggle ---
with _mgmt_btn_cols[6]:
    if st.button(
        "Export Profiles",
        key="mgmt_btn_export",
        width="stretch",
        type="primary" if st.session_state.active_mgmt_tab == "Export Profiles" else "secondary",
    ):
        st.session_state.active_mgmt_tab = (
            None if st.session_state.active_mgmt_tab == "Export Profiles" else "Export Profiles"
        )
        st.rerun()

# --- Save popover ---
save_btn = False
sim_name = ""
with _mgmt_btn_cols[8]:
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


# ---- LOAD PROFILES SECTION ----
if st.session_state.active_mgmt_tab == "Load Profiles":
    saved_loads = _list_saved(LOAD_PROFILES_DIR, ".csv")
    lp_col1, lp_col2 = st.columns([2, 1])

    with lp_col1:
        st.markdown("**Upload & Save a Load Profile**")
        lp_name = st.text_input("Profile Name", placeholder="e.g., Dairy-Farm-2024", key="lp_name")
        lp_file = st.file_uploader("Upload 8760 Load CSV", type=["csv"], key="lp_upload")
        lp_save_btn = st.button("Save Load Profile", disabled=(not lp_name or lp_file is None))

        if lp_save_btn and lp_file is not None and lp_name:
            try:
                df_up = pd.read_csv(lp_file)
                _parse_8760_csv(df_up)  # validate
                _save_profile_csv(LOAD_PROFILES_DIR, lp_name, df_up)
                st.success(f"Load profile '{lp_name}' saved!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with lp_col2:
        if saved_loads:
            st.markdown("**Saved Load Profiles**")
            sel_lp = st.selectbox("Select profile", saved_loads, key="lp_sel")
            lp_view_btn = st.button("View / Edit", key="lp_view")
            lp_del_btn = st.button("Delete", key="lp_del")

            if lp_del_btn and sel_lp:
                _delete_file(LOAD_PROFILES_DIR, sel_lp, ".csv")
                st.success(f"Deleted '{sel_lp}'.")
                st.rerun()
        else:
            st.caption("No saved load profiles yet.")
            sel_lp = None
            lp_view_btn = False

    # View / Edit section
    if saved_loads and lp_view_btn and sel_lp:
        st.session_state["lp_editing"] = sel_lp
    if st.session_state.get("lp_editing"):
        edit_name = st.session_state["lp_editing"]
        st.subheader(f"Editing: {edit_name}")
        edit_df = _load_profile_csv(LOAD_PROFILES_DIR, edit_name)
        try:
            vals = _parse_8760_csv(edit_df)
            st.write(f"**Rows:** {len(vals):,} | **Annual:** {vals.sum():,.0f} kWh | **Peak:** {vals.max():,.1f} kW")

            # Monthly summary chart
            _preview_year = st.session_state.get("sb_cod_date", date(2026, 1, 1)).year
            dt_idx = pd.date_range(f"{_preview_year}-01-01", periods=8760, freq="h")
            monthly_kwh = pd.Series(vals, index=dt_idx).resample("ME").sum()
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(x=MONTH_NAMES, y=monthly_kwh.values, marker_color="#636EFA"))
            fig.update_layout(title="Monthly Load (kWh)", yaxis_title="kWh", height=300, template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(str(e))

        st.caption("Edit the data below and click Save to update.")
        edited_df = st.data_editor(edit_df, num_rows="fixed", width="stretch", height=400, key="lp_editor")

        lp_save_edit = st.button("Save Changes", key="lp_save_edit")
        if lp_save_edit:
            try:
                _parse_8760_csv(edited_df)  # validate
                _save_profile_csv(LOAD_PROFILES_DIR, edit_name, edited_df)
                st.success(f"'{edit_name}' updated!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if st.button("Close Editor", key="lp_close_edit"):
            del st.session_state["lp_editing"]
            st.rerun()


# ---- EXPORT PROFILES SECTION ----
if st.session_state.active_mgmt_tab == "Export Profiles":
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
                st.rerun()
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
                st.rerun()
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
            fig = go.Figure(go.Bar(x=MONTH_NAMES, y=monthly_avg.values, marker_color="#00CC96"))
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
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if st.button("Close Editor", key="ep_close_edit"):
            del st.session_state["ep_editing"]
            st.rerun()


# ---- SYSTEM PROFILES SECTION ----
if st.session_state.active_mgmt_tab == "System Profiles":
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
                        st.rerun()
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
                    st.rerun()
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
                st.rerun()

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
                st.rerun()
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
            st.rerun()


st.title("PV Solar Rate Simulator")
st.markdown(
    '<p style="font-size: 12px; color: rgba(150,150,150,0.9); margin-top: -10px;">'
    'California Net Value Billing Tariff (NVBT) — Hourly Import/Export Analysis</p>',
    unsafe_allow_html=True,
)

# --- Getting Started guidance (only shown when no simulation has run yet) ---
if st.session_state.billing_result is None and st.session_state.saved_view is None:
    st.info(
        "**Getting Started:** Use the sidebar to configure your simulation inputs, "
        "working through each numbered section (1-8). Once all checklist items below "
        "are complete, click **Run Simulation** to generate results.",
        icon="👋",
    )

st.divider()

# =============================================================================
# SIDEBAR — INPUTS
# =============================================================================
with st.sidebar:
    st.header("System & Site Configuration")
    st.caption("Complete each section below, then click **Run Simulation** in the main panel.")

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
    system_life_years = st.number_input(
        "System Life (years)", min_value=1, max_value=50, value=20, step=1,
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

    with st.expander("Advanced PV Options"):
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

    generate_prod = st.button("Generate Production Profile", type="primary")

    # --- 3. Load Profile ---
    st.subheader("3. Load Profile")
    saved_load_names = _list_saved(LOAD_PROFILES_DIR, ".csv")
    load_source = st.radio(
        "Load profile source",
        ["Upload new CSV", "Use saved profile"],
        key="load_source_radio",
    )

    load_file = None
    selected_load_profile = None
    if load_source == "Upload new CSV":
        st.caption("Upload a CSV with 8760 rows of hourly kWh values.")
        load_file = st.file_uploader("Upload 8760 Load CSV", type=["csv"], key="sidebar_load_upload")
    else:
        if saved_load_names:
            selected_load_profile = st.selectbox("Select Load Profile", saved_load_names, key="sidebar_load_sel")
        else:
            st.caption("No saved profiles. Upload via the Load Profiles tab above.")

    # --- 4. Utility & Rate ---
    st.subheader("4. Utility & Rate")
    billing_engine = st.radio(
        "Billing Engine", ["Custom", "ECC"],
        key="billing_engine_radio",
        horizontal=True,
        help="Custom: uses OpenEI tariff data with built-in TOU billing. ECC: uses the Energy Cost Calculator engine.",
    )
    st.session_state.billing_engine = billing_engine

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
        fetch_rates_btn = st.button("Fetch Available Rates")

        # Rate selection (inline, right under Fetch button)
        if st.session_state.available_rates:
            rate_options = {f"{r['name']}": r["label"] for r in st.session_state.available_rates}
            selected_rate_name = st.selectbox("Select Rate Schedule", list(rate_options.keys()))
            selected_label = rate_options[selected_rate_name]
            load_tariff_btn = st.button("Load Tariff Details")

        if st.session_state.tariff:
            with st.expander("View Tariff Details"):
                st.markdown(format_tariff_summary(st.session_state.tariff))

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
                st.session_state._ecc_saved_path = os.path.join(ECC_TARIFFS_DIR, _sel_ecc + ".json")

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

        if st.session_state.ecc_tariff_metadata:
            with st.expander("View ECC Tariff Info"):
                meta = st.session_state.ecc_tariff_metadata
                st.write(f"**Source:** {meta.get('source', 'N/A')}")
                st.write(f"**Utility ID:** {meta.get('utility_id', 'N/A')}")
                st.write(f"**Sector:** {meta.get('sector', 'N/A')}")
                st.write(f"**Rate Filter:** {meta.get('rate_filter', 'N/A')}")
                n_tariffs = meta.get("num_tariffs", 0)
                st.write(f"**Tariff blocks loaded:** {n_tariffs}")
                if meta.get("tariff_names"):
                    for tname in meta["tariff_names"][:10]:
                        st.caption(f"  - {tname}")

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
    st.session_state.nem_switch = nem_switch

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
        nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=2, key="sb_nem_regime_1")
        if nem_regime_1 in ("NEM-1", "NEM-2"):
            nsc_rate, nbc_rate, billing_option = _render_nem12_widgets("", nem_regime_1)
            st.session_state.nsc_rate = nsc_rate
            st.session_state.nbc_rate = nbc_rate
            st.session_state.billing_option = billing_option
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
        nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=0, key="sb_nem_regime_1")
        num_years_1 = st.number_input(
            "Tenor (years)", min_value=1,
            max_value=max(1, system_life_years - 1),
            value=min(5, max(1, system_life_years - 1)),
            step=1, key="sb_nem_years_1",
        )
        if nem_regime_1 in ("NEM-1", "NEM-2"):
            nsc_rate, nbc_rate, billing_option = _render_nem12_widgets("", nem_regime_1)
            st.session_state.nsc_rate = nsc_rate
            st.session_state.nbc_rate = nbc_rate
            st.session_state.billing_option = billing_option
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
            st.caption("Exports valued at retail TOU energy rate (per NEM tariff)")
            export_method_2 = None
            selected_export_profile_2 = None
            flat_rate_2 = None
        else:
            export_method_2, selected_export_profile_2, flat_rate_2 = _render_export_rate_widgets("_2")

    # --- 6. Battery (BESS) ---
    st.subheader("6. BESS")
    if billing_engine == "ECC":
        st.info("Battery optimization is not yet available with the ECC billing engine.")
    battery_enabled = st.toggle(
        "Enable Battery Storage", value=False, key="bess_toggle",
        disabled=(billing_engine == "ECC"),
    )
    st.session_state.battery_enabled = battery_enabled if billing_engine != "ECC" else False

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
        st.session_state.battery_capacity_kwh = battery_capacity_kwh
        st.session_state.battery_optimize = optimize_size
        st.session_state.battery_opt_range = (bess_opt_min, bess_opt_max, bess_opt_step)
        st.session_state.battery_fast_dispatch = fast_dispatch
        st.session_state.battery_config = BatteryConfig(
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
        st.session_state.battery_config = None
        st.session_state.battery_capacity_kwh = 0
        st.session_state.battery_optimize = False
        st.session_state.battery_opt_range = (0, 0, 0)
        st.session_state.battery_fast_dispatch = False

    # --- 7. Escalators ---
    st.subheader("7. Escalators (Annual Projection)")
    rate_escalator = st.number_input(
        "Utility Rate Escalator (%/yr)", min_value=0.0, max_value=20.0, value=3.0, step=0.5,
        help="Applied annually to TOU energy rates",
        key="sb_rate_escalator",
    )
    load_escalator = st.number_input(
        "Demand Growth Escalator (%/yr)", min_value=0.0, max_value=20.0, value=2.0, step=0.5,
        help="Applied annually to load profile (increases consumption & peak demand)",
        key="sb_load_escalator",
    )

    # --- 8. System Cost ---
    st.subheader("8. System Cost (for Payback)")
    cost_input_method = st.radio(
        "Cost input", ["$/W-DC", "Total ($)"], key="sb_cost_method",
        help="Choose how to specify system cost. Used only for payback and ROI calculations.",
    )
    if cost_input_method == "$/W-DC":
        cost_per_watt = st.number_input(
            "Installed Cost ($/W-DC)", min_value=0.0, value=1.50, step=0.05,
            key="sb_cost_per_watt",
        )
        system_cost = cost_per_watt * system_size_kw * 1000
        st.caption(f"Total: ${system_cost:,.0f}")
    else:
        system_cost = st.number_input(
            "Total Installed Cost ($)", min_value=0.0, value=750000.0, step=10000.0,
            key="sb_total_cost",
        )


# =============================================================================
# SAVE SIMULATION HANDLER (after sidebar so variables are available)
# =============================================================================
if save_btn and sim_name and st.session_state.get("billing_result") is not None:
    result_to_save = st.session_state.billing_result
    summary_to_save = build_savings_summary(result_to_save, system_cost)
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

        sizing_res = st.session_state.get("sizing_result")
        if sizing_res is not None:
            extra_save["sizing_table"] = sizing_res.table.to_dict(orient="records")
            extra_save["best_size_kwh"] = sizing_res.best_size_kwh

    # Grid exchange data — compute peak period from tariff
    _sv_tariff = st.session_state.tariff
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
    _tariff_obj = st.session_state.tariff
    _tariff_dict = asdict(_tariff_obj) if _tariff_obj else None
    _prod_list = st.session_state.production_8760.tolist() if st.session_state.production_8760 is not None else None
    _load_list = st.session_state.load_8760.tolist() if st.session_state.load_8760 is not None else None
    _export_list = st.session_state.export_rates.tolist() if st.session_state.export_rates is not None else None

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
    api_key = os.getenv("NREL_API_KEY", "")
    if not api_key:
        st.error("NREL_API_KEY not found. Add `NREL_API_KEY=your_key` to the `.env` file in the project root. Get a free key at https://developer.nrel.gov/signup/")
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
                st.session_state.production_8760 = prod
                st.session_state.production_summary = summary
                st.sidebar.success(
                    f"Production generated: {summary['ac_annual_kwh']:,.0f} kWh/yr "
                    f"(CF: {summary['capacity_factor']:.1f}%)"
                )
            except Exception as e:
                st.error(f"PVWatts error: {e}")


# =============================================================================
# LOAD PROFILE PARSING
# =============================================================================
if load_source == "Upload new CSV" and load_file is not None and st.session_state.load_8760 is None:
    try:
        df_load = pd.read_csv(load_file)
        load_values = _parse_8760_csv(df_load)
        dt_index = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
        st.session_state.load_8760 = pd.Series(load_values, index=dt_index, name="load_kwh")
        annual_load = load_values.sum()
        peak_load = load_values.max()
        load_factor = annual_load / (peak_load * 8760) * 100 if peak_load > 0 else 0
        st.sidebar.success(
            f"Load profile loaded: {annual_load:,.0f} kWh/yr, "
            f"Peak: {peak_load:,.1f} kW, LF: {load_factor:.1f}%"
        )
    except Exception as e:
        st.error(f"Error reading load file: {e}")

elif load_source == "Use saved profile" and selected_load_profile:
    try:
        df_load = _load_profile_csv(LOAD_PROFILES_DIR, selected_load_profile)
        load_values = _parse_8760_csv(df_load)
        dt_index = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
        st.session_state.load_8760 = pd.Series(load_values, index=dt_index, name="load_kwh")
        annual_load = load_values.sum()
        peak_load = load_values.max()
        load_factor = annual_load / (peak_load * 8760) * 100 if peak_load > 0 else 0
        st.sidebar.success(
            f"Loaded '{selected_load_profile}': {annual_load:,.0f} kWh/yr, "
            f"Peak: {peak_load:,.1f} kW, LF: {load_factor:.1f}%"
        )
    except Exception as e:
        st.error(f"Error loading profile: {e}")


# =============================================================================
# RATE SCHEDULE FETCHING (handlers for sidebar buttons)
# =============================================================================
if fetch_rates_btn:
    with st.spinner(f"Fetching rates for {utility_name}..."):
        try:
            rates = fetch_available_rates(utility_name)
            st.session_state.available_rates = rates
            st.sidebar.success(f"Found {len(rates)} rate schedules.")
        except Exception as e:
            st.error(f"Error fetching rates: {e}")

if load_tariff_btn and selected_label:
    with st.spinner("Loading tariff details..."):
        try:
            tariff = fetch_tariff_detail(selected_label)
            st.session_state.tariff = tariff
            st.sidebar.success(f"Tariff loaded: {tariff.name}")
        except Exception as e:
            st.error(f"Error loading tariff: {e}")

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
            st.session_state.ecc_cost_calculator = calc
            st.session_state.ecc_tariff_data = tdata
            _tnames = []
            if isinstance(tdata, list):
                for td in tdata[:10]:
                    if isinstance(td, dict):
                        _tnames.append(td.get("name", td.get("label", "Unknown")))
            st.session_state.ecc_tariff_metadata = {
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
            _save_dest = os.path.join(ECC_TARIFFS_DIR, f"{_save_label}.json")
            with open(_save_dest, "w") as _sf:
                _json_mod.dump(tdata, _sf)
            st.sidebar.success(f"ECC tariff loaded ({len(tdata)} block(s)).")
        except Exception as e:
            st.error(f"ECC tariff fetch error: {e}")

if ecc_load_json_btn:
    # --- "Use Saved Tariff" path ---
    _ecc_saved = st.session_state.pop("_ecc_saved_path", None)
    if _ecc_saved and os.path.isfile(_ecc_saved):
        try:
            calc, tdata = load_ecc_tariff_from_json(_ecc_saved)
            st.session_state.ecc_cost_calculator = calc
            st.session_state.ecc_tariff_data = tdata
            _tnames = []
            if isinstance(tdata, list):
                for td in tdata[:10]:
                    if isinstance(td, dict):
                        _tnames.append(td.get("name", td.get("label", "Unknown")))
            _fname = os.path.splitext(os.path.basename(_ecc_saved))[0]
            st.session_state.ecc_tariff_metadata = {
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
                _tmp_dir = _tmpmod.mkdtemp()
                _tmp_path = os.path.join(_tmp_dir, _ecc_uploaded.name)
                with open(_tmp_path, "wb") as _f:
                    _f.write(_ecc_uploaded.getvalue())
                calc, tdata = load_ecc_tariff_from_json(_tmp_path)
                st.session_state.ecc_cost_calculator = calc
                st.session_state.ecc_tariff_data = tdata
                _tnames = []
                if isinstance(tdata, list):
                    for td in tdata[:10]:
                        if isinstance(td, dict):
                            _tnames.append(td.get("name", td.get("label", "Unknown")))
                st.session_state.ecc_tariff_metadata = {
                    "source": f"JSON upload: {_ecc_uploaded.name}",
                    "utility_id": "N/A",
                    "utility": utility_name,
                    "sector": "N/A",
                    "rate_filter": "N/A",
                    "num_tariffs": len(tdata) if isinstance(tdata, list) else 0,
                    "tariff_names": _tnames,
                }
                # Save a copy to ECC_TARIFFS_DIR for future "Use Saved Tariff"
                _save_name = os.path.splitext(_ecc_uploaded.name)[0]
                _save_dest = os.path.join(ECC_TARIFFS_DIR, f"{_save_name}.json")
                import shutil
                shutil.copy2(_tmp_path, _save_dest)
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
ready_checks = {
    "Production profile": st.session_state.production_8760 is not None,
    "Load profile": st.session_state.load_8760 is not None,
    "Tariff schedule": (
        st.session_state.tariff is not None
        if billing_engine == "Custom"
        else st.session_state.get("ecc_cost_calculator") is not None
    ),
    "Export rates": (
        st.session_state.export_rates is not None
        if nem_regime_1 == "NEM-3 / NVBT"
        else True  # Not needed for NEM-1/NEM-2 (exports valued at retail rate)
    ),
}
all_ready = all(ready_checks.values())

if not all_ready:
    st.subheader("Simulation Checklist")
    _checklist_hints = {
        "Production profile": "Sidebar Section 1-2: enter a location, configure PV system, and click **Generate Production Profile**",
        "Load profile": "Sidebar Section 3: upload an 8760 CSV or select a saved load profile",
        "Tariff schedule": "Sidebar Section 4: fetch and load a rate schedule from your selected utility",
        "Export rates": "Sidebar Section 5: choose an export compensation method (saved profile, CSV upload, or flat rate)",
    }
    for check, status in ready_checks.items():
        if status:
            st.write(f"✅ {check}")
        else:
            st.write(f"⬜ {check}")
            st.caption(f"  ↳ {_checklist_hints.get(check, '')}")
    st.info("Complete all inputs in the sidebar, then click **Run Simulation** below.")

_run_col, _edit_col = st.columns([1, 1])
with _run_col:
    run_sim = st.button("Run Simulation", type="primary", disabled=not all_ready, width="stretch")
with _edit_col:
    _has_saved_view = st.session_state.saved_view is not None
    edit_sim = st.button(
        "Edit Simulation",
        disabled=not _has_saved_view,
        width="stretch",
        help="Populate sidebar with the saved simulation's inputs so you can tweak and re-run",
    )

# --- Edit Simulation handler ---
if edit_sim and _has_saved_view:
    populate_session_from_simulation(st.session_state, st.session_state.saved_view)
    st.rerun()

if run_sim:
    _overlay = st.empty()
    _overlay.markdown(
        _progress_overlay_html(0, "Initializing..."),
        unsafe_allow_html=True,
    )
    try:
        if billing_engine == "ECC":
            # ============ ECC billing engine ============
            _overlay.markdown(
                _progress_overlay_html(25, "Running ECC billing simulation..."),
                unsafe_allow_html=True,
            )
            result_pv_only = run_ecc_billing_simulation(
                load_8760=st.session_state.load_8760,
                production_8760=st.session_state.production_8760,
                cost_calculator=st.session_state.ecc_cost_calculator,
                export_rates_8760=st.session_state.export_rates,
                tariff_data=st.session_state.get("ecc_tariff_data"),
            )
            st.session_state.billing_result_pv_only = result_pv_only
            st.session_state.billing_result = result_pv_only
            st.session_state.billing_result_batt = None
            st.session_state.sizing_result = None

            _overlay.markdown(
                _progress_overlay_html(50, "ECC simulation complete."),
                unsafe_allow_html=True,
            )
            _overlay.markdown(
                _progress_overlay_html(75, "Building results..."),
                unsafe_allow_html=True,
            )
            _overlay.markdown(
                _progress_overlay_html(100, "Done!"),
                unsafe_allow_html=True,
            )
            st.success("Simulation complete (ECC engine)!")

        else:
            # ============ Custom billing engine ============
            # For NEM-1/NEM-2, export rates are not used (valued at retail TOU),
            # but the function signature requires an array — provide zeros as placeholder.
            _export_rates_for_sim = st.session_state.export_rates
            if _export_rates_for_sim is None and nem_regime_1 in ("NEM-1", "NEM-2"):
                _dt_idx_placeholder = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                _export_rates_for_sim = pd.Series(
                    np.zeros(8760), index=_dt_idx_placeholder, name="export_rate_per_kwh",
                )

            # NEM params for the billing call
            _nem_nbc = nbc_rate if nem_regime_1 == "NEM-2" else 0.0
            _nem_nsc = nsc_rate if nem_regime_1 in ("NEM-1", "NEM-2") else 0.0
            _nem_billing = billing_option if nem_regime_1 in ("NEM-1", "NEM-2") else "ABO"

            # --- Step 1: PV-only billing ---
            _overlay.markdown(
                _progress_overlay_html(5, "Running PV-only billing simulation..."),
                unsafe_allow_html=True,
            )
            result_pv_only = run_billing_simulation(
                load_8760=st.session_state.load_8760,
                production_8760=st.session_state.production_8760,
                tariff=st.session_state.tariff,
                export_rates_8760=_export_rates_for_sim,
                nem_regime=nem_regime_1,
                nbc_rate=_nem_nbc,
                nsc_rate=_nem_nsc,
                billing_option=_nem_billing,
            )
            st.session_state.billing_result_pv_only = result_pv_only
            st.session_state.sizing_result = None

            _overlay.markdown(
                _progress_overlay_html(25, "PV-only simulation complete."),
                unsafe_allow_html=True,
            )

            # --- Step 2: Battery dispatch (if enabled) ---
            if st.session_state.battery_enabled and st.session_state.battery_config is not None:
                batt_cfg = st.session_state.battery_config
                _use_monthly = st.session_state.get("battery_fast_dispatch", False)

                if st.session_state.get("battery_optimize", False):
                    # ---- Sizing sweep ----
                    opt_min, opt_max, opt_step = st.session_state.battery_opt_range
                    if opt_max > opt_min and opt_step > 0:
                        import numpy as _np
                        candidates = _np.arange(opt_min, opt_max + opt_step / 2, opt_step).tolist()

                        _overlay.markdown(
                            _progress_overlay_html(30, f"Running sizing sweep ({len(candidates)} candidates)..."),
                            unsafe_allow_html=True,
                        )

                        _tariff = st.session_state.tariff
                        _dt_idx = cast(pd.DatetimeIndex, st.session_state.load_8760.index)
                        d_masks, d_prices = _build_demand_lp_inputs(_tariff, _dt_idx)
                        _energy_rates = _build_hourly_energy_rates(_tariff, _dt_idx)

                        _export_for_sizing = (
                            _export_rates_for_sim
                            if st.session_state.export_rates is None
                            else st.session_state.export_rates
                        )
                        sizing_res = optimize_capacity_kwh(
                            candidate_sizes_kwh=candidates,
                            pv_kwh=np.asarray(st.session_state.production_8760),
                            load_kwh=np.asarray(st.session_state.load_8760.values),
                            import_price=_energy_rates,
                            export_price=np.asarray(_export_for_sizing.values),
                            demand_window_masks=d_masks,
                            demand_prices=d_prices,
                            battery_config=batt_cfg,
                            monthly=_use_monthly,
                        )
                        st.session_state.sizing_result = sizing_res

                        _overlay.markdown(
                            _progress_overlay_html(50, "Sizing complete. Running final billing..."),
                            unsafe_allow_html=True,
                        )

                        # Run full billing with best size to get proper BillingResult
                        result_batt = run_billing_simulation(
                            load_8760=st.session_state.load_8760,
                            production_8760=st.session_state.production_8760,
                            tariff=st.session_state.tariff,
                            export_rates_8760=_export_rates_for_sim,
                            battery_config=batt_cfg,
                            capacity_kwh=sizing_res.best_size_kwh,
                            monthly_dispatch=_use_monthly,
                            nem_regime=nem_regime_1,
                            nbc_rate=_nem_nbc,
                            nsc_rate=_nem_nsc,
                            billing_option=_nem_billing,
                        )
                        st.session_state.billing_result = result_batt
                        st.session_state.billing_result_batt = result_batt
                        st.session_state.battery_capacity_kwh = sizing_res.best_size_kwh

                        _overlay.markdown(
                            _progress_overlay_html(75, "Building results..."),
                            unsafe_allow_html=True,
                        )

                        _overlay.markdown(
                            _progress_overlay_html(100, "Done!"),
                            unsafe_allow_html=True,
                        )
                        st.success(
                            f"Optimization complete! Best size: "
                            f"{sizing_res.best_size_kwh:,.0f} kWh"
                        )
                    else:
                        st.session_state.billing_result = result_pv_only
                        st.session_state.billing_result_batt = None
                        st.warning("Invalid optimize range. Running PV-only.")
                else:
                    # ---- Fixed-size dispatch ----
                    batt_cap = st.session_state.get("battery_capacity_kwh", 0)
                    if batt_cap > 0:
                        _overlay.markdown(
                            _progress_overlay_html(30, "Running PV + Battery dispatch..."),
                            unsafe_allow_html=True,
                        )
                        result_batt = run_billing_simulation(
                            load_8760=st.session_state.load_8760,
                            production_8760=st.session_state.production_8760,
                            tariff=st.session_state.tariff,
                            export_rates_8760=_export_rates_for_sim,
                            battery_config=batt_cfg,
                            capacity_kwh=batt_cap,
                            monthly_dispatch=_use_monthly,
                            nem_regime=nem_regime_1,
                            nbc_rate=_nem_nbc,
                            nsc_rate=_nem_nsc,
                            billing_option=_nem_billing,
                        )
                        st.session_state.billing_result = result_batt
                        st.session_state.billing_result_batt = result_batt

                        _overlay.markdown(
                            _progress_overlay_html(75, "Building results..."),
                            unsafe_allow_html=True,
                        )

                        _overlay.markdown(
                            _progress_overlay_html(100, "Done!"),
                            unsafe_allow_html=True,
                        )
                        st.success("Simulation complete (PV + Battery)!")
                    else:
                        st.session_state.billing_result = result_pv_only
                        st.session_state.billing_result_batt = None
                        _overlay.markdown(
                            _progress_overlay_html(100, "Done!"),
                            unsafe_allow_html=True,
                        )
                        st.success("Simulation complete (PV only).")
            else:
                st.session_state.billing_result = result_pv_only
                st.session_state.billing_result_batt = None
                _overlay.markdown(
                    _progress_overlay_html(50, "Building results..."),
                    unsafe_allow_html=True,
                )
                _overlay.markdown(
                    _progress_overlay_html(100, "Done!"),
                    unsafe_allow_html=True,
                )
                st.success("Simulation complete!")

        # Clear overlay and editing flag when done
        _overlay.empty()
        st.session_state.editing_saved_sim = False
    except Exception as e:
        _overlay.empty()
        st.error(f"Simulation failed: {e}")
        st.warning(
            "Check that all sidebar inputs are configured correctly. "
            "Common causes: mismatched profile lengths, missing tariff data, or invalid rate schedules."
        )
        with st.expander("Show error details"):
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# RESULTS DISPLAY
# =============================================================================
if st.session_state.billing_result is not None:
    st.divider()
    st.subheader("Simulation Results")

    # CSS for white table backgrounds and bold totals row
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] { background-color: #FFFFFF; padding: 4px; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    has_battery = st.session_state.billing_result_batt is not None

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
            result = cast(BillingResult, st.session_state.billing_result_pv_only)
        else:
            result = cast(BillingResult, st.session_state.billing_result_batt)
    else:
        result = cast(BillingResult, st.session_state.billing_result)

    tab_labels = ["Monthly Bills", "Grid Exchange", "Annual Projection", "Production vs Load", "Savings & Payback"]
    if has_battery:
        tab_labels.append("Battery Analysis")
    tab_labels.append("PPA Rate")
    tab_labels.append("Downloads")
    result_tabs = st.tabs(tab_labels)

    # Assign tab variables
    tab1 = result_tabs[0]
    tab_grid = result_tabs[1]
    tab2 = result_tabs[2]
    tab3 = result_tabs[3]
    tab4 = result_tabs[4]
    tab_batt = result_tabs[5] if has_battery else None
    tab_indexed = result_tabs[-2]  # Indexed Tariff (always second-to-last)
    tab5 = result_tabs[-1]         # Export / Download (always last)

    # Compute peak period index from tariff
    _tariff_for_peak = st.session_state.tariff
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
        _, _, _peak_period_idx = _build_tou_arrays(_dummy_idx, st.session_state.ecc_tariff_data)

    # Determine PV-only result for demand column display (BESS mode only)
    pv_only_for_display = st.session_state.billing_result_pv_only if (has_battery and scenario == "PV + Battery") else None

    # --- Tab 1: Monthly Bills ---
    with tab1:
        st.subheader("Monthly Bill Summary")
        display_df = build_monthly_summary_display(result, result_pv_only=pv_only_for_display)
        raw = result.monthly_summary
        totals = {
            "Month": "TOTAL",
            "Load (kWh)": fmt_num(raw['load_kwh'].sum()),
            "Solar (kWh)": fmt_num(raw['solar_kwh'].sum()),
            "Import (kWh)": fmt_num(raw['import_kwh'].sum()),
            "Export (kWh)": fmt_num(raw['export_kwh'].sum()),
            "Export Peak (kWh)": fmt_num(raw['export_peak_kwh'].sum()),
            "Export Off-Peak (kWh)": fmt_num(raw['export_offpeak_kwh'].sum()),
        }
        if pv_only_for_display is not None:
            totals["Demand kW (PV)"] = fmt_num(pv_only_for_display.monthly_summary['peak_demand_kw'].max())
            totals["Demand kW (PV+BESS)"] = fmt_num(raw['peak_demand_kw'].max())
        else:
            totals["Demand kW (PV)"] = fmt_num(raw['peak_demand_kw'].max())
        totals.update({
            "Energy ($)": fmt_dollar(raw['energy_cost'].sum()),
            "Demand ($)": fmt_dollar(raw['total_demand_charge'].sum()),
            "Fixed ($)": fmt_dollar(raw['fixed_charge'].sum()),
        })
        # NBC column (NEM-2 only)
        _has_nbc = "nbc_charge" in raw.columns and raw["nbc_charge"].sum() > 0
        if _has_nbc:
            totals["NBC ($)"] = fmt_dollar(raw['nbc_charge'].sum())
        totals.update({
            "Export Credit ($)": fmt_dollar(raw['export_credit'].sum()),
            "Net Bill ($)": fmt_dollar(raw['net_bill'].sum()),
        })
        totals_row = pd.DataFrame([totals])
        display_with_totals = pd.concat([display_df, totals_row], ignore_index=True)

        st.markdown(render_styled_table(display_with_totals, bold_last_row=True), unsafe_allow_html=True)

        # Show NSC adjustment info if applicable
        if hasattr(result, 'annual_nsc_adjustment') and result.annual_nsc_adjustment > 0:
            st.info(
                f"Net Surplus Compensation adjustment applied in month 12: "
                f"${result.annual_nsc_adjustment:,.2f} (surplus valued at NSC rate instead of retail)"
            )

    # --- Grid Exchange tab ---
    with tab_grid:
        st.subheader("Grid Import & Export by TOU Period")
        ge_display, ge_raw = build_grid_exchange_summary(result, _peak_period_idx)

        _ge_bold_cols = [c for c in ge_display.columns if "Total" in c]
        st.markdown(render_styled_table(ge_display, bold_last_row=True, bold_cols=_ge_bold_cols), unsafe_allow_html=True)

    # --- Tab 2: Annual Summary ---
    with tab2:
        st.subheader(f"Annual Summary ({system_life_years}-Year)")
        projection_df = build_annual_projection(
            result=result,
            system_cost=system_cost,
            rate_escalator_pct=rate_escalator,
            load_escalator_pct=load_escalator,
            years=system_life_years,
            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
            result_pv_only=pv_only_for_display,
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
            cod_year=cod_year,
            degradation_pct=annual_degradation_pct,
        )

        # Format for display — negate cost outflow columns so they show as (red)
        display_proj = projection_df.copy()
        outflow_dollar_cols = [
            "Bill w/o Solar ($)", "Energy ($)", "Demand ($)",
            "Fixed ($)", "Bill w/ Solar ($)",
        ]
        for col in outflow_dollar_cols:
            if col in display_proj.columns:
                display_proj[col] = display_proj[col].apply(lambda x: -x)
        if "Export (kWh)" in display_proj.columns:
            display_proj["Export (kWh)"] = display_proj["Export (kWh)"].apply(lambda x: -x)

        kwh_proj_cols = [c for c in display_proj.columns if "(kWh)" in c]
        for col in kwh_proj_cols:
            display_proj[col] = display_proj[col].apply(fmt_num)
        kw_proj_cols = [c for c in display_proj.columns if "kW" in c and "(kWh)" not in c]
        for col in kw_proj_cols:
            display_proj[col] = display_proj[col].apply(fmt_num)
        dollar_proj_cols = [c for c in display_proj.columns if "($)" in c]
        for col in dollar_proj_cols:
            display_proj[col] = display_proj[col].apply(fmt_dollar)

        st.markdown(render_styled_table(display_proj), unsafe_allow_html=True)

        # Cumulative savings chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projection_df["Year"],
            y=projection_df["Cumulative Savings ($)"],
            name="Cumulative Savings",
            mode="lines+markers",
            line=dict(color="#00CC96", width=2.5),
        ))
        fig.add_hline(
            y=system_cost, line_dash="dash", line_color="#EF553B",
            annotation_text=f"System Cost: ${system_cost:,.0f}",
        )
        fig.update_layout(
            title="Cumulative Savings vs. System Cost",
            xaxis_title="Year", yaxis_title="$",
            template="plotly_white", height=350,
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

    # --- Tab 4: Savings & Payback ---
    with tab4:
        st.subheader("Annual Savings & Payback")
        summary = build_savings_summary(result, system_cost)

        # --- Scenario comparison when battery is active ---
        if has_battery and st.session_state.billing_result_pv_only is not None:
            pv_only = cast(BillingResult, st.session_state.billing_result_pv_only)
            pv_batt = cast(BillingResult, st.session_state.billing_result_batt)

            st.markdown("**Scenario Comparison**")
            cmp_data = {
                "Metric": [
                    "Annual Bill",
                    "Energy Charges",
                    "Demand Charges",
                    "Export Credit",
                    "Savings vs No-Solar",
                ],
                "No Solar": [
                    fmt_dollar(result.annual_bill_without_solar),
                    "—", "—", "—",
                    "—",
                ],
                "PV Only": [
                    fmt_dollar(pv_only.annual_bill_with_solar),
                    fmt_dollar(pv_only.annual_energy_cost),
                    fmt_dollar(pv_only.annual_demand_cost),
                    fmt_dollar(pv_only.annual_export_credit),
                    fmt_dollar(pv_only.annual_savings),
                ],
                "PV + Battery": [
                    fmt_dollar(pv_batt.annual_bill_with_solar),
                    fmt_dollar(pv_batt.annual_energy_cost),
                    fmt_dollar(pv_batt.annual_demand_cost),
                    fmt_dollar(pv_batt.annual_export_credit),
                    fmt_dollar(pv_batt.annual_savings),
                ],
            }
            battery_value = pv_only.annual_bill_with_solar - pv_batt.annual_bill_with_solar
            cmp_data["Metric"].append("Battery Value")
            cmp_data["No Solar"].append("—")
            cmp_data["PV Only"].append("—")
            cmp_data["PV + Battery"].append(fmt_dollar(battery_value))

            cmp_df = pd.DataFrame(cmp_data)
            st.markdown(render_styled_table(cmp_df), unsafe_allow_html=True)
            st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Load", f"{summary['annual_load_kwh']:,.0f} kWh")
            st.metric("Annual Solar", f"{summary['annual_solar_kwh']:,.0f} kWh")
            st.metric("Solar Offset", f"{summary['solar_offset_pct']:.1f}%")
        with col2:
            label_bill = "Bill WITH Solar + Battery" if has_battery else "Bill WITH Solar"
            st.metric("Bill WITHOUT Solar", f"${summary['annual_bill_without_solar']:,.0f}")
            st.metric(label_bill, f"${summary['annual_bill_with_solar']:,.0f}")
            st.metric("Annual Savings", f"${summary['annual_savings']:,.0f}", delta=f"{summary['savings_pct']:.1f}%")
        with col3:
            st.metric("System Cost", f"${summary['system_cost']:,.0f}")
            if summary["simple_payback_years"] is not None:
                st.metric("Simple Payback", f"{summary['simple_payback_years']:.1f} years")
            else:
                st.metric("Simple Payback", "N/A")
        st.divider()
        st.subheader("Energy Balance")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Grid Import", f"{summary['annual_import_kwh']:,.0f} kWh")
        with col_b:
            st.metric("Grid Export", f"{summary['annual_export_kwh']:,.0f} kWh")

    # --- Battery Analysis tab (only when battery enabled) ---
    if tab_batt is not None:
        with tab_batt:
            st.subheader("Battery Analysis")

            # Show selected / optimized size
            batt_cap_display = st.session_state.get("battery_capacity_kwh", 0)
            batt_hrs_display = st.session_state.battery_config.battery_hours if st.session_state.battery_config else 4
            batt_pw_display = batt_cap_display / batt_hrs_display if batt_hrs_display > 0 else 0
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Capacity", f"{batt_cap_display:,.0f} kWh")
            with bc2:
                st.metric("Power", f"{batt_pw_display:,.0f} kW")
            with bc3:
                st.metric("Duration", f"{batt_hrs_display:.1f} hrs")

            # --- Battery KPIs ---
            pv_only_res = st.session_state.billing_result_pv_only
            batt_res = st.session_state.billing_result_batt
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
                    line=dict(color="#636EFA", width=2.5),
                    marker=dict(size=6),
                ))
                # Mark best point
                best_row = sz[sz["size_kwh"] == sizing_res.best_size_kwh]
                if not best_row.empty:
                    fig_sz.add_trace(go.Scatter(
                        x=best_row["size_kwh"], y=best_row["net_bill"],
                        mode="markers", name="Optimal",
                        marker=dict(color="#EF553B", size=14, symbol="star"),
                    ))
                fig_sz.update_layout(
                    title="Net Annual Bill vs. Battery Size",
                    xaxis_title="Battery Capacity (kWh)",
                    yaxis_title="Net Annual Bill ($)",
                    template="plotly_white", height=400,
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
    with tab_indexed:
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

        # Build projection & render table
        if it_view == "Annual":
            it_proj = build_annual_projection(
                result=result,
                system_cost=system_cost,
                rate_escalator_pct=rate_escalator,
                load_escalator_pct=load_escalator,
                years=system_life_years,
                export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
                result_pv_only=pv_only_for_display,
                nem_regime_1=nem_regime_1,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
                export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
                cod_year=cod_year,
                degradation_pct=annual_degradation_pct,
            )
            it_df = build_indexed_tariff_annual(
                it_proj,
                base_savings_pct=it_savings_pct,
                savings_escalator_pct=it_savings_esc,
                regime_1_savings_pct=it_regime_1_savings,
                regime_2_savings_pct=it_regime_2_savings,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
            )
            # Format for display
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Bill w/ Solar ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            if "Customer Savings ($)" in it_display.columns:
                it_display["Customer Savings ($)"] = it_display["Customer Savings ($)"].apply(fmt_dollar)
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
            )
            it_df = build_indexed_tariff_monthly(
                it_monthly,
                base_savings_pct=it_savings_pct,
                savings_escalator_pct=it_savings_esc,
                regime_1_savings_pct=it_regime_1_savings,
                regime_2_savings_pct=it_regime_2_savings,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
            )
            # Format for display
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Net Bill ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            if "Customer Savings ($)" in it_display.columns:
                it_display["Customer Savings ($)"] = it_display["Customer Savings ($)"].apply(fmt_dollar)
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
            nem_regime_1=nem_regime_1,
            nem_regime_2=nem_regime_2 if nem_switch else None,
            num_years_1=num_years_1 if nem_switch else None,
            export_rates_multiyear_2=st.session_state.get("export_rates_multiyear_2") if nem_switch else None,
            cod_year=cod_year,
            degradation_pct=annual_degradation_pct,
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
