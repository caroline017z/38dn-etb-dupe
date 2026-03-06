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
import logging
from dataclasses import asdict
from datetime import date, datetime
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
    get_export_rates,
    load_acc_from_upload,
    create_flat_export_rates,
    parse_multiyear_export_rates,
)
from modules.billing import run_billing_simulation, BillingResult, compute_old_rate_baseline
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
    style_negative_red,
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
)


def _list_saved(directory: str, ext: str = ".json") -> list[str]:
    """Generic file lister — still used by Load Profiles / Export Profiles."""
    files = glob.glob(os.path.join(directory, f"*{ext}"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _list_all_load_profiles() -> list[tuple[str, str]]:
    """Return unified (name, type) list of CSV + NEM-A profiles sorted by mtime (newest first)."""
    entries: list[tuple[float, str, str]] = []  # (mtime, name, type)
    for f in glob.glob(os.path.join(LOAD_PROFILES_DIR, "*.csv")):
        entries.append((os.path.getmtime(f), os.path.splitext(os.path.basename(f))[0], "csv"))
    for f in glob.glob(os.path.join(NEMA_PROFILES_DIR, "*.json")):
        entries.append((os.path.getmtime(f), os.path.splitext(os.path.basename(f))[0], "nema"))
    entries.sort(key=lambda x: x[0], reverse=True)
    return [(name, typ) for _, name, typ in entries]


def _delete_file(directory, name, ext):
    """Generic file deleter — still used by Load Profiles / Export Profiles."""
    fp = os.path.join(directory, f"{name}{ext}")
    if os.path.exists(fp):
        os.remove(fp)


# =============================================================================
# HELPER — Simulation Progress Overlay
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
    name = sanitize_filename(name)
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


def _load_nema_profile_into_session(profile_name: str):
    """Load a saved NEM-A profile bundle into session state.

    Restores: nema_utility, nema_meters, nema_meter_loads, nema_meter_tariffs,
    existing_solar_nema_meters, and sets load_8760 from the generating meter.
    """
    path = os.path.join(NEMA_PROFILES_DIR, f"{profile_name}.json")
    with open(path) as f:
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
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV.")
    col = numeric_cols[0]
    if len(numeric_cols) > 1 and len(df) == 8760:
        first_vals = df[col].values
        if np.array_equal(first_vals, np.arange(1, 8761)):
            col = numeric_cols[1]
    values = np.asarray(df[col].values)
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
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
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
            position: fixed;
            top: 64px;
            right: 20px;
            z-index: 999998;
            pointer-events: none;
        }}
        .top-right-logo img {{
            height: 48px;
            width: 48px;
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
    "rate_shift_enabled": False,
    "rate_shift_old_tariff": None,
    "nema_rate_shift_tariffs": {},
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

_mgmt_btn_cols = st.columns([0.15, 1, 1, 1, 1, 1, 1, 1.5])

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
            st.rerun()

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
                    st.rerun()
            st.divider()
        else:
            st.caption("No saved load profiles yet.")

        if st.button("View All Load Profiles", width="stretch", type="primary"):
            st.session_state["active_mgmt_tab"] = "Load Profiles"
            st.rerun()

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
                    st.rerun()
            st.divider()
        else:
            st.caption("No saved export profiles yet.")

        if st.button("View All Export Profiles", width="stretch", type="primary"):
            st.session_state["active_mgmt_tab"] = "Export Profiles"
            st.rerun()

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
                    st.rerun()
            st.divider()
        else:
            st.caption("No custom rates yet.")

        if st.button("Create Custom Rate", width="stretch", type="primary"):
            st.session_state["active_mgmt_tab"] = "Custom Rates"
            st.rerun()

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


# ---- LOAD PROFILES SECTION ----
if st.session_state["active_mgmt_tab"] == "Load Profiles":
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
                st.rerun()
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
                st.rerun()
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
            if st.button("Fetch Available Rates", key="mgmt_nema_fetch_rates"):
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
            with st.expander(f"{'*' if _lp_meter.get('is_generating') else ''} {_lp_meter['name']}", expanded=(_lp_i < 2)):
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
                _nema_save_path = os.path.join(NEMA_PROFILES_DIR, f"{sanitize_filename(_nema_profile_name)}.json")
                with open(_nema_save_path, "w") as _f:
                    _json.dump(_nema_bundle, _f)
                st.success(f"NEM-A profile '{_nema_profile_name}' saved with {len(_nema_bundle_meters)} meters.")
                st.rerun()

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
                    fig = go.Figure(go.Bar(x=MONTH_NAMES, y=monthly_kwh.values, marker_color="#636EFA"))
                    fig.update_layout(title="Monthly Load (kWh)", yaxis_title="kWh", **_chart_layout)
                    st.plotly_chart(fig, use_container_width=True)
                elif _lp_chart_type == "Average Daily Profile":
                    _lp_series = pd.Series(vals, index=dt_idx)
                    _lp_avg_hourly = _lp_series.groupby(_lp_series.index.hour).mean()
                    fig = go.Figure(go.Scatter(
                        x=list(range(24)), y=_lp_avg_hourly.values,
                        mode="lines+markers", line=dict(color="#636EFA", width=2.5),
                        marker=dict(size=5), fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
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
                        mode="lines", line=dict(color="#636EFA", width=2),
                        fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
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
                    st.rerun()
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
                    st.rerun()

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
                if st.button("Fetch Available Rates", key="edit_nema_fetch_rates"):
                    st.session_state["_pending_edit_nema_fetch_rates"] = _ne_utility
                if st.session_state.get("available_rates"):
                    st.caption(f"{len(st.session_state['available_rates'])} rate schedules available.")

                # Initialize edit-state meters from JSON (only on first load of this profile)
                _ne_edit_key = f"_ne_editing_{_sel_name}"
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
                    with st.expander(f"{'*' if _em.get('is_generating') else ''} {_em['name']}", expanded=(_ei < 2)):
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
                        st.rerun()



# ---- EXPORT PROFILES SECTION ----
if st.session_state["active_mgmt_tab"] == "Export Profiles":
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
if st.session_state["active_mgmt_tab"] == "System Profiles":
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


# ---- CUSTOM RATES SECTION ----
if st.session_state["active_mgmt_tab"] == "Custom Rates":
    st.subheader("Custom Rates")

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
                st.rerun()
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
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")


st.title("PV Solar Rate Simulator")
st.markdown(
    '<p style="font-size: 12px; color: rgba(150,150,150,0.9); margin-top: -10px;">'
    'California Net Value Billing Tariff (NVBT) — Hourly Import/Export Analysis</p>',
    unsafe_allow_html=True,
)

# --- Getting Started guidance (only shown when no simulation has run yet) ---
if st.session_state["billing_result"] is None and st.session_state["saved_view"] is None:
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
        fetch_rates_btn = st.button("Fetch Available Rates")

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
                with st.expander(f"Tariff for: {_pmt_m['name']}", expanded=True):
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

    # --- Rate Shift Analysis ---
    st.markdown("---")
    rate_shift_enabled = st.toggle(
        "Rate Shift Analysis",
        key="rate_shift_toggle",
        value=st.session_state.get("rate_shift_enabled", False),
        help="Compare savings from switching tariffs (e.g., TOU-C to TOU-D). "
             "Shows what you would pay on the old rate as a separate baseline.",
    )
    st.session_state["rate_shift_enabled"] = rate_shift_enabled

    if rate_shift_enabled and billing_engine == "Custom":
        _rs_is_nema = st.session_state.get("load_mode") == "NEM-A Aggregation"

        if not _rs_is_nema:
            # Single meter: one old tariff selector
            if st.session_state["available_rates"]:
                _rs_rate_options = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                _rs_selected = st.selectbox(
                    "Old Rate (pre-switch)",
                    list(_rs_rate_options.keys()),
                    key="rate_shift_old_rate_sel",
                    help="Select the tariff the customer was on before switching.",
                )
                _rs_label = _rs_rate_options[_rs_selected]
                if st.button("Load Old Tariff", key="rate_shift_load_btn"):
                    st.session_state["_pending_rate_shift_load"] = _rs_label
            else:
                st.caption("Fetch rates above first to select an old tariff.")

            _rs_current = st.session_state.get("rate_shift_old_tariff")
            if _rs_current is not None:
                st.success(f"Old tariff loaded: {_rs_current.name}")
        else:
            # NEM-A: per-meter old tariff selectors only (no blanket option)
            _rs_meters = st.session_state.get("nema_meters", [])
            if _rs_meters and st.session_state["available_rates"]:
                st.markdown("**Per-Meter Old Tariffs (NEM-A)**")
                _rs_nema_tariffs = st.session_state.get("nema_rate_shift_tariffs", {})
                _rs_all_loaded = True
                for _rs_i, _rs_m in enumerate(_rs_meters):
                    with st.expander(f"Old tariff for: {_rs_m['name']}", expanded=(_rs_i not in _rs_nema_tariffs)):
                        _rs_pmt_options = {f"{r['name']}": r["label"] for r in st.session_state["available_rates"]}
                        _rs_pmt_sel = st.selectbox(
                            "Old Rate", list(_rs_pmt_options.keys()),
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
                    st.warning("Load an old tariff for each meter to enable rate shift analysis.")
            else:
                st.caption("Fetch rates and configure NEM-A meters first.")

    elif rate_shift_enabled and billing_engine == "ECC":
        st.caption("Rate shift with ECC engine: upload a second ECC tariff JSON for the old rate.")
        ecc_rs_upload = st.file_uploader(
            "Old Rate Tariff JSON (ECC)", type=["json"], key="ecc_rs_json_upload",
        )
        if st.button("Load Old ECC Tariff", key="ecc_rs_load_btn") and ecc_rs_upload:
            st.session_state["_pending_ecc_rs_load"] = ecc_rs_upload

        _rs_ecc_current = st.session_state.get("rate_shift_old_ecc_calculator")
        if _rs_ecc_current is not None:
            st.success("Old ECC tariff loaded.")

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
        nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=2, key="sb_nem_regime_1")
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
        nem_regime_1 = st.selectbox("NEM Regime", nem_options, index=0, key="sb_nem_regime_1")
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
            import shutil as _shutil
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
                    _save_dest = os.path.join(ECC_TARIFFS_DIR, f"{_save_name}.json")
                    _shutil.copy2(_tmp_path, _save_dest)
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
all_ready = all(ready_checks.values())

if not all_ready:
    st.subheader("Simulation Checklist")
    _checklist_hints = {
        "Production profile": "Sidebar Section 1-2: enter a location, configure PV system, and click **Generate Production Profile**",
        "Load profile": "Sidebar Section 3: upload an 8760 CSV or select a saved load profile",
        "Tariff schedule": "Sidebar Section 4: fetch and load a rate schedule. For NEM-A meters not using the generating meter's tariff, load per-meter tariffs below the rate selector.",
        "Export rates": "Sidebar Section 5: choose an export compensation method (saved profile, CSV upload, or flat rate)",
    }
    for check, status in ready_checks.items():
        if status:
            st.write(f"✅ {check}")
        else:
            st.write(f"⬜ {check}")
            st.caption(f"  ↳ {_checklist_hints.get(check, '')}")
    st.info("Complete all inputs in the sidebar, then click **Run Simulation** below.")

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

if run_sim:
    st.session_state["active_mgmt_tab"] = None
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
            )
            st.session_state["billing_result_pv_only"] = result_pv_only
            st.session_state["billing_result"] = result_pv_only
            st.session_state["billing_result_batt"] = None
            st.session_state["sizing_result"] = None

            # ECC battery dispatch
            if st.session_state.get("battery_enabled") and st.session_state.get("battery_config"):
                batt_cap = st.session_state.get("battery_capacity_kwh", 0)
                if batt_cap > 0:
                    _overlay.markdown(
                        _progress_overlay_html(50, "Running ECC + Battery dispatch..."),
                        unsafe_allow_html=True,
                    )
                    result_batt = run_ecc_billing_simulation(
                        load_8760=st.session_state["load_8760"],
                        production_8760=st.session_state["production_8760"],
                        cost_calculator=st.session_state["ecc_cost_calculator"],
                        export_rates_8760=_ecc_export,
                        tariff_data=st.session_state.get("ecc_tariff_data"),
                        battery_config=st.session_state["battery_config"],
                        capacity_kwh=batt_cap,
                        monthly_dispatch=st.session_state.get("battery_fast_dispatch", True),
                    )
                    st.session_state["billing_result"] = result_batt
                    st.session_state["billing_result_batt"] = result_batt
                    _check_battery_solver(result_batt)

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
            _export_rates_for_sim = st.session_state["export_rates"]
            if _export_rates_for_sim is None:
                # NEM-1/NEM-2 value exports at retail TOU (zeros placeholder).
                # NEM-A may also reach here if NEM-3 export rates weren't loaded.
                _dt_idx_placeholder = pd.date_range(start=f"{cod_year}-01-01 00:00", periods=8760, freq="h")
                _export_rates_for_sim = pd.Series(
                    np.zeros(8760), index=_dt_idx_placeholder, name="export_rate_per_kwh",
                )

            # NEM params for the billing call
            _nem_nbc = nbc_rate if nem_regime_1 == "NEM-2" else 0.0
            _nem_nsc = nsc_rate if nem_regime_1 in ("NEM-1", "NEM-2") else 0.0
            _nem_billing = billing_option if nem_regime_1 in ("NEM-1", "NEM-2") else "ABO"

            if st.session_state.get("load_mode") == "NEM-A Aggregation":
                # ============ NEM-A Aggregation path ============
                _overlay.markdown(
                    _progress_overlay_html(10, "Building NEM-A meter profiles..."),
                    unsafe_allow_html=True,
                )

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
                    nsc_rate=_nem_nsc if nem_regime_1 in ("NEM-1", "NEM-2") else 0.04,
                    billing_option=_nem_billing,
                )

                # PV-only aggregation run (no battery)
                _overlay.markdown(
                    _progress_overlay_html(25, "Running NEM-A PV-only simulation..."),
                    unsafe_allow_html=True,
                )
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
                        _overlay.markdown(
                            _progress_overlay_html(50, "Running NEM-A PV + Battery simulation..."),
                            unsafe_allow_html=True,
                        )

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
                _overlay.markdown(
                    _progress_overlay_html(100, "Done!"),
                    unsafe_allow_html=True,
                )
                st.success(
                    f"NEM-A simulation complete! "
                    f"{len(_meter_configs)} meters, "
                    f"${_nema_fees['annual_admin']:,.0f}/yr admin fees"
                )

            else:
                # ============ Single-meter Custom billing path (original) ============
                # --- Step 1: PV-only billing ---
                _overlay.markdown(
                    _progress_overlay_html(5, "Running PV-only billing simulation..."),
                    unsafe_allow_html=True,
                )
                result_pv_only = run_billing_simulation(
                    load_8760=st.session_state["load_8760"],
                    production_8760=st.session_state["production_8760"],
                    tariff=st.session_state["tariff"],
                    export_rates_8760=_export_rates_for_sim,
                    nem_regime=nem_regime_1,
                    nbc_rate=_nem_nbc,
                    nsc_rate=_nem_nsc,
                    billing_option=_nem_billing,
                )
                st.session_state["billing_result_pv_only"] = result_pv_only
                st.session_state["sizing_result"] = None

                _overlay.markdown(
                    _progress_overlay_html(25, "PV-only simulation complete."),
                    unsafe_allow_html=True,
                )

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

                            _overlay.markdown(
                                _progress_overlay_html(30, f"Running sizing sweep ({len(candidates)} candidates)..."),
                                unsafe_allow_html=True,
                            )

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

                            _overlay.markdown(
                                _progress_overlay_html(50, "Sizing complete. Running final billing..."),
                                unsafe_allow_html=True,
                            )

                            # Run full billing with best size to get proper BillingResult
                            result_batt = run_billing_simulation(
                                load_8760=st.session_state["load_8760"],
                                production_8760=st.session_state["production_8760"],
                                tariff=st.session_state["tariff"],
                                export_rates_8760=_export_rates_for_sim,
                                battery_config=batt_cfg,
                                capacity_kwh=sizing_res.best_size_kwh,
                                monthly_dispatch=_use_monthly,
                                nem_regime=nem_regime_1,
                                nbc_rate=_nem_nbc,
                                nsc_rate=_nem_nsc,
                                billing_option=_nem_billing,
                            )
                            st.session_state["billing_result"] = result_batt
                            st.session_state["billing_result_batt"] = result_batt
                            _check_battery_solver(result_batt)
                            st.session_state["battery_capacity_kwh"] = sizing_res.best_size_kwh

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
                            st.session_state["billing_result"] = result_pv_only
                            st.session_state["billing_result_batt"] = None
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
                                load_8760=st.session_state["load_8760"],
                                production_8760=st.session_state["production_8760"],
                                tariff=st.session_state["tariff"],
                                export_rates_8760=_export_rates_for_sim,
                                battery_config=batt_cfg,
                                capacity_kwh=batt_cap,
                                monthly_dispatch=_use_monthly,
                                nem_regime=nem_regime_1,
                                nbc_rate=_nem_nbc,
                                nsc_rate=_nem_nsc,
                                billing_option=_nem_billing,
                            )
                            st.session_state["billing_result"] = result_batt
                            st.session_state["billing_result_batt"] = result_batt
                            _check_battery_solver(result_batt)

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
                            st.session_state["billing_result"] = result_pv_only
                            st.session_state["billing_result_batt"] = None
                            _overlay.markdown(
                                _progress_overlay_html(100, "Done!"),
                                unsafe_allow_html=True,
                            )
                            st.success("Simulation complete (PV only).")
                else:
                    st.session_state["billing_result"] = result_pv_only
                    st.session_state["billing_result_batt"] = None
                    _overlay.markdown(
                        _progress_overlay_html(50, "Building results..."),
                        unsafe_allow_html=True,
                    )
                    _overlay.markdown(
                        _progress_overlay_html(100, "Done!"),
                        unsafe_allow_html=True,
                    )
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

        # Clear overlay and editing flag when done
        _overlay.empty()
        st.session_state["editing_saved_sim"] = False
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
if st.session_state["billing_result"] is not None:
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
        **_common_nem_kw,
    )

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
        # Rate shift savings total
        if result.old_rate_monthly_baselines is not None and result.monthly_baseline_details is not None:
            _rs_old = result.old_rate_monthly_baselines
            _rs_new = [d["total"] for d in result.monthly_baseline_details]
            totals["Rate Shift Savings ($)"] = fmt_dollar(sum(_rs_old) - sum(_rs_new))
        totals_row = pd.DataFrame([totals])
        display_with_totals = pd.concat([display_df, totals_row], ignore_index=True)

        st.markdown(render_styled_table(display_with_totals, bold_last_row=True, bold_cols=["Month"]), unsafe_allow_html=True)

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
        projection_df = _main_projection

        # Store projection for fragment access
        st.session_state["_proj_display_df"] = projection_df.copy()

        @st.fragment
        def _render_projection_table():
            _proj_detail = st.radio(
                "View", ["Simple", "Detailed"], horizontal=True,
                key="proj_view_toggle", label_visibility="collapsed",
            )

            display_proj = st.session_state["_proj_display_df"].copy()
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

            if _proj_detail == "Simple":
                _drop_cols = [
                    "Load (kWh)", "Customer Load (kWh)", "Solar (kWh)",
                    "Solar Offset (kWh)", "Import (kWh)", "Export (kWh)",
                    "Export Peak (kWh)", "Export Off-Peak (kWh)",
                    "Demand kW (PV)", "Demand kW (PV+BESS)",
                    "Energy ($)", "Demand ($)", "Fixed ($)",
                    "NBC ($)", "NSC Adj ($)",
                ]
                display_proj = display_proj.drop(
                    columns=[c for c in _drop_cols if c in display_proj.columns]
                )

            _proj_bold = ["Calendar Year"] if "Calendar Year" in display_proj.columns else ["Year"]
            _proj_highlight = ["Cumulative Savings ($)"]
            if "Cumulative Total Savings ($)" in display_proj.columns:
                _proj_highlight.append("Cumulative Total Savings ($)")
            st.markdown(render_styled_table(
                display_proj,
                bold_cols=_proj_bold,
                highlight_cols=_proj_highlight,
            ), unsafe_allow_html=True)

        _render_projection_table()

        # Cumulative savings chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=projection_df["Year"],
            y=projection_df["Cumulative Savings ($)"],
            name="Cumulative Solar Savings",
            mode="lines+markers",
            line=dict(color="#00CC96", width=2.5),
        ))
        if "Cumulative Total Savings ($)" in projection_df.columns:
            fig.add_trace(go.Scatter(
                x=projection_df["Year"],
                y=projection_df["Cumulative Total Savings ($)"],
                name="Cumulative Total Savings (incl. Rate Shift)",
                mode="lines+markers",
                line=dict(color="#636EFA", width=2.5),
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
        if has_battery and st.session_state["billing_result_pv_only"] is not None:
            pv_only = cast(BillingResult, st.session_state["billing_result_pv_only"])
            pv_batt = cast(BillingResult, st.session_state["billing_result_batt"])

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

            # Rate shift rows
            if st.session_state.get("rate_shift_enabled") and pv_only.rate_shift_annual_savings is not None:
                cmp_data["Metric"].append("Rate Shift Savings")
                cmp_data["No Solar"].append("—")
                cmp_data["PV Only"].append(fmt_dollar(pv_only.rate_shift_annual_savings))
                cmp_data["PV + Battery"].append(fmt_dollar(pv_batt.rate_shift_annual_savings))

                pv_total = pv_only.annual_savings + pv_only.rate_shift_annual_savings
                batt_total = pv_batt.annual_savings + pv_batt.rate_shift_annual_savings
                cmp_data["Metric"].append("Total Savings")
                cmp_data["No Solar"].append("—")
                cmp_data["PV Only"].append(fmt_dollar(pv_total))
                cmp_data["PV + Battery"].append(fmt_dollar(batt_total))

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
        # Rate shift savings display
        if summary.get("rate_shift_annual_savings") is not None:
            st.divider()
            st.subheader("Rate Shift Analysis")
            rs_c1, rs_c2, rs_c3 = st.columns(3)
            with rs_c1:
                st.metric("Old Rate Baseline", f"${result.old_rate_annual_baseline:,.0f}")
            with rs_c2:
                st.metric("Rate Shift Savings", f"${summary['rate_shift_annual_savings']:,.0f}/yr")
            with rs_c3:
                st.metric("Total Combined Savings", f"${summary['total_annual_savings']:,.0f}/yr")

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

        # PPA Rate Escalator
        if nem_switch:
            esc_c1, esc_c2 = st.columns(2)
            with esc_c1:
                it_ppa_esc_1 = st.number_input(
                    f"PPA Escalator — {nem_regime_1} (%/yr)",
                    min_value=0.0, max_value=10.0, value=2.9, step=0.1,
                    format="%.1f", key="it_ppa_esc_1",
                )
            with esc_c2:
                it_ppa_esc_2 = st.number_input(
                    f"PPA Escalator — {nem_regime_2} (%/yr)",
                    min_value=0.0, max_value=10.0, value=2.9, step=0.1,
                    format="%.1f", key="it_ppa_esc_2",
                )
        else:
            it_ppa_esc_1 = st.number_input(
                "PPA Rate Escalator (%/yr)",
                min_value=0.0, max_value=10.0, value=2.9, step=0.1,
                format="%.1f", key="it_ppa_esc_1",
            )
            it_ppa_esc_2 = it_ppa_esc_1

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
            it_proj = _main_projection
            it_df = build_indexed_tariff_annual(
                it_proj,
                base_savings_pct=it_savings_pct,
                savings_escalator_pct=it_savings_esc,
                regime_1_savings_pct=it_regime_1_savings,
                regime_2_savings_pct=it_regime_2_savings,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
                ppa_escalator_pct=it_ppa_esc_1,
                ppa_escalator_pct_2=it_ppa_esc_2 if nem_switch else None,
            )
            # Format for display
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Bill w/ Solar ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            for _sav_col in ["Utility Savings ($)", "Customer Savings ($)"]:
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
            )
            it_df = build_indexed_tariff_monthly(
                it_monthly,
                base_savings_pct=it_savings_pct,
                savings_escalator_pct=it_savings_esc,
                regime_1_savings_pct=it_regime_1_savings,
                regime_2_savings_pct=it_regime_2_savings,
                nem_regime_2=nem_regime_2 if nem_switch else None,
                num_years_1=num_years_1 if nem_switch else None,
                ppa_escalator_pct=it_ppa_esc_1,
                ppa_escalator_pct_2=it_ppa_esc_2 if nem_switch else None,
            )
            # Format for display
            it_display = it_df.copy()
            for col in ["Bill w/o Solar ($)", "Net Bill ($)"]:
                if col in it_display.columns:
                    it_display[col] = (it_display[col] * -1).apply(fmt_dollar)
            for _sav_col in ["Utility Savings ($)", "Customer Savings ($)"]:
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

        # --- Customer Proposal (PPTX) ---
        st.divider()
        st.subheader("Customer Proposal (PPTX)")

        @st.fragment
        def _proposal_fragment():
            st.caption(
                "Generate a branded 38DN customer proposal deck from the current simulation. "
                "Fill in the fields below and click Generate."
            )

            # Pull indicative PPA rate from PPA Rate tab computation
            _prop_ppa_rate = 0.0
            _it_savings = st.session_state.get("it_savings_pct", 10.0)
            _it_sav_esc = st.session_state.get("it_savings_esc", 0.0)
            try:
                _it_proj = _main_projection
                _it_df = build_indexed_tariff_annual(
                    _it_proj,
                    base_savings_pct=_it_savings,
                    savings_escalator_pct=_it_sav_esc,
                    ppa_escalator_pct=st.session_state.get("it_ppa_esc_1", 2.9),
                    ppa_escalator_pct_2=st.session_state.get("it_ppa_esc_2", 2.9) if nem_switch else None,
                    nem_regime_2=nem_regime_2 if nem_switch else None,
                    num_years_1=num_years_1 if nem_switch else None,
                )
                if len(_it_df) >= 1 and "PPA Rate ($/kWh)" in _it_df.columns:
                    _yr1_rate = _it_df["PPA Rate ($/kWh)"].iloc[0]
                    if _yr1_rate > 0:
                        _prop_ppa_rate = round(_yr1_rate, 4)
            except Exception as e:
                logger.warning("Failed to compute indicative PPA rate: %s", e)

            # PPA escalator comes from PPA Rate tab inputs
            _prop_ppa_esc = st.session_state.get("it_ppa_esc_1", 2.9)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                prop_customer = st.text_input("Customer / Facility Name", key="prop_customer")
                prop_address = st.text_input("Site Address", key="prop_address")
                prop_account = st.text_input(
                    "Utility Account ID (optional)", key="prop_account",
                )
            with col_p2:
                prop_term = st.number_input(
                    "Term (years)", min_value=1, max_value=40,
                    value=25, step=1, key="prop_term",
                )
                prop_new_tariff = st.text_input(
                    "Proposed Tariff (if switching)", key="prop_new_tariff",
                    help="Leave blank to keep current tariff.",
                )

            # --- Custom per-regime savings toggle ---
            _prop_custom_savings = st.toggle(
                "Customize Customer Savings", key="prop_custom_savings",
                help="Override the PPA Rate tab savings target with per-regime values.",
            )
            _prop_sav_1 = _it_savings
            _prop_sav_2 = _it_savings
            if _prop_custom_savings:
                _cs_c1, _cs_c2 = st.columns(2)
                with _cs_c1:
                    _prop_sav_1 = st.number_input(
                        f"{nem_regime_1} Savings (%)",
                        min_value=0.0, max_value=99.0,
                        value=float(_it_savings), step=0.5,
                        key="prop_savings_regime_1",
                    )
                with _cs_c2:
                    if nem_switch:
                        _prop_sav_2 = st.number_input(
                            f"{nem_regime_2} Savings (%)",
                            min_value=0.0, max_value=99.0,
                            value=float(_it_savings), step=0.5,
                            key="prop_savings_regime_2",
                        )
                # Compute and display per-regime PPA rates
                try:
                    _cs_it_df = build_indexed_tariff_annual(
                        _main_projection,
                        base_savings_pct=_prop_sav_1,
                        savings_escalator_pct=_it_sav_esc,
                        regime_1_savings_pct=_prop_sav_1,
                        regime_2_savings_pct=_prop_sav_2 if nem_switch else None,
                        nem_regime_2=nem_regime_2 if nem_switch else None,
                        num_years_1=num_years_1 if nem_switch else None,
                        ppa_escalator_pct=_prop_ppa_esc,
                        ppa_escalator_pct_2=st.session_state.get("it_ppa_esc_2", 2.9) if nem_switch else None,
                    )
                    if len(_cs_it_df) >= 1 and "PPA Rate ($/kWh)" in _cs_it_df.columns:
                        _cs_r1 = _cs_it_df["PPA Rate ($/kWh)"].iloc[0]
                        _prop_ppa_rate = round(_cs_r1, 4) if _cs_r1 > 0 else _prop_ppa_rate
                        _mc1, _mc2 = st.columns(2)
                        with _mc1:
                            st.metric(f"{nem_regime_1} PPA Rate (Yr 1)", f"${_cs_r1:.4f}/kWh")
                        if nem_switch and num_years_1 and len(_cs_it_df) > num_years_1:
                            _cs_r2 = _cs_it_df["PPA Rate ($/kWh)"].iloc[num_years_1]
                            with _mc2:
                                st.metric(f"{nem_regime_2} PPA Rate (Yr {num_years_1 + 1})", f"${_cs_r2:.4f}/kWh")
                except Exception as e:
                    logger.warning("Failed to compute per-regime PPA rates: %s", e)

            _prop_date = date.today().strftime("%B %Y")
            _batt_cap = st.session_state.get("battery_capacity_kwh", 0) or 0
            _batt_cfg = st.session_state.get("battery_config")
            _batt_kw = _batt_cap / (_batt_cfg.battery_hours if _batt_cfg else 4.0) if _batt_cap > 0 else 0

            if st.button("Generate Customer Proposal", type="primary", key="btn_gen_proposal"):
                if not prop_customer:
                    st.warning("Please enter a customer name.")
                else:
                    with st.spinner("Building proposal deck..."):
                        # Build a term-length projection for the proposal
                        _prop_proj_df = build_annual_projection(
                            result=result,
                            system_cost=system_cost,
                            rate_escalator_pct=rate_escalator,
                            load_escalator_pct=load_escalator,
                            years=prop_term,
                            export_rates_multiyear=st.session_state.get("export_rates_multiyear"),
                            result_pv_only=pv_only_for_display,
                            compound_escalation=compound_escalation,
                            rate_shift_old_baseline=_rs_old_baseline_for_proj,
                            **_common_nem_kw,
                        )

                        # Keep the original (utility-only) projection for PPA
                        # backsolve — the savings matrix and regime-2 rate need
                        # Bill w/ Solar BEFORE any PPA overlay.
                        _prop_proj_original = _prop_proj_df.copy()

                        # When custom savings toggle is ON, compute per-year PPA
                        # cost from the backsolve and layer it onto the utility
                        # residual so the PPTX shows true customer economics:
                        #   Total Cost = Bill w/ Solar (utility) + PPA Cost
                        #   Customer Savings = Bill w/o Solar - Total Cost
                        if _prop_custom_savings:
                            _cs_tariff = build_indexed_tariff_annual(
                                _prop_proj_original,
                                base_savings_pct=_prop_sav_1,
                                savings_escalator_pct=_it_sav_esc,
                                regime_1_savings_pct=_prop_sav_1,
                                regime_2_savings_pct=_prop_sav_2 if nem_switch else None,
                                nem_regime_2=nem_regime_2 if nem_switch else None,
                                num_years_1=num_years_1 if nem_switch else None,
                                ppa_escalator_pct=_prop_ppa_esc,
                                ppa_escalator_pct_2=st.session_state.get("it_ppa_esc_2", 2.9) if nem_switch else None,
                            )
                            _prop_proj_df = _prop_proj_df.copy()
                            for idx, row in _prop_proj_df.iterrows():
                                yr = int(row["Year"])
                                # Look up PPA rate for this year from the tariff table
                                _tariff_row = _cs_tariff[_cs_tariff["Year"] == yr]
                                ppa_rate_yr = float(_tariff_row["PPA Rate ($/kWh)"].iloc[0]) if len(_tariff_row) else 0.0
                                solar_kwh_yr = row["Solar (kWh)"]
                                ppa_cost = max(ppa_rate_yr, 0.0) * solar_kwh_yr
                                # Total cost = utility residual + PPA payment
                                utility_residual = row["Bill w/ Solar ($)"]
                                total_cost = utility_residual + ppa_cost
                                bill_no = row["Bill w/o Solar ($)"]
                                _prop_proj_df.at[idx, "PPA Cost ($)"] = round(ppa_cost, 2)
                                _prop_proj_df.at[idx, "Bill w/ Solar ($)"] = round(total_cost, 2)
                                _prop_proj_df.at[idx, "Annual Savings ($)"] = round(bill_no - total_cost, 2)
                            _prop_proj_df["Cumulative Savings ($)"] = _prop_proj_df["Annual Savings ($)"].cumsum().round(2)

                        # Compute regime 2 PPA rate for exec summary using the
                        # ORIGINAL projection (before PPA overlay)
                        _prop_ppa_rate_r2 = None
                        if nem_switch and num_years_1 and _prop_ppa_rate > 0:
                            try:
                                _r2_tariff = build_indexed_tariff_annual(
                                    _prop_proj_original,
                                    base_savings_pct=_prop_sav_1 if _prop_custom_savings else _it_savings,
                                    savings_escalator_pct=_it_sav_esc,
                                    regime_1_savings_pct=_prop_sav_1 if _prop_custom_savings else None,
                                    regime_2_savings_pct=_prop_sav_2 if _prop_custom_savings else None,
                                    nem_regime_2=nem_regime_2,
                                    num_years_1=num_years_1,
                                    ppa_escalator_pct=_prop_ppa_esc,
                                    ppa_escalator_pct_2=st.session_state.get("it_ppa_esc_2", 2.9) if nem_switch else None,
                                )
                                if len(_r2_tariff) > num_years_1:
                                    _r2_rate = _r2_tariff["PPA Rate ($/kWh)"].iloc[num_years_1]
                                    if _r2_rate > 0:
                                        _prop_ppa_rate_r2 = round(float(_r2_rate), 4)
                            except Exception as e:
                                logger.warning("Failed to compute regime 2 PPA rate: %s", e)

                        proposal_bytes = generate_proposal_pptx(
                            customer_name=prop_customer,
                            address=prop_address,
                            utility_account=prop_account,
                            utility_name=utility_name,
                            tariff_name=selected_rate_name or "",
                            new_tariff_name=prop_new_tariff or None,
                            date_str=_prop_date,
                            system_size_kw=system_size_kw,
                            dc_ac_ratio=dc_ac_ratio,
                            battery_kwh=_batt_cap,
                            battery_kw=_batt_kw,
                            ppa_rate=_prop_ppa_rate if _prop_ppa_rate > 0 else None,
                            ppa_escalator_pct=_prop_ppa_esc if _prop_ppa_rate > 0 else None,
                            term_years=prop_term,
                            rate_escalator_pct=rate_escalator,
                            result=result,
                            annual_proj_df=_prop_proj_df,
                            nem_regime_1=nem_regime_1,
                            nem_regime_2=nem_regime_2 if nem_switch else None,
                            num_years_1=num_years_1 if nem_switch else None,
                            customer_savings_pct=_prop_sav_1 if _prop_custom_savings else st.session_state.get("it_savings_pct", 10.0),
                            customer_savings_pct_2=_prop_sav_2 if _prop_custom_savings and nem_switch else None,
                            ppa_rate_regime_2=_prop_ppa_rate_r2,
                            annual_proj_df_original=_prop_proj_original,
                        )
                    _safe_name = prop_customer.replace(" ", "_")[:30]
                    st.download_button(
                        label="Download Customer Proposal (.pptx)",
                        data=proposal_bytes,
                        file_name=f"{_safe_name}_Proposal_{_prop_date.replace(' ', '_')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )

        _proposal_fragment()
