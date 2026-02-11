"""
Shared simulation management helpers — used by both app.py and pages/All_Simulations.py.
"""
import os
import json
import glob
import pandas as pd
from datetime import date, datetime

from modules.tariff import TariffSchedule, NSC_DEFAULT_RATE, UTILITY_EIA_IDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SIMULATIONS_DIR = os.path.join(DATA_DIR, "simulations")
os.makedirs(SIMULATIONS_DIR, exist_ok=True)


def list_saved_simulations() -> list[str]:
    """Return simulation names sorted by file mtime (most recent first)."""
    files = glob.glob(os.path.join(SIMULATIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def load_simulation(name: str) -> dict:
    """Load a simulation JSON by name."""
    with open(os.path.join(SIMULATIONS_DIR, f"{name}.json"), "r") as f:
        return json.load(f)


def save_simulation(name, result, summary, projection_df, inputs, **extra):
    """Save a simulation to JSON."""
    data = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "inputs": inputs,
        "summary": summary,
        "monthly_summary": result.monthly_summary.to_dict(orient="records"),
        "projection": projection_df.to_dict(orient="records") if projection_df is not None else [],
    }
    data.update(extra)
    with open(os.path.join(SIMULATIONS_DIR, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=2)


def delete_simulation(name: str):
    """Delete a simulation JSON file."""
    fp = os.path.join(SIMULATIONS_DIR, f"{name}.json")
    if os.path.exists(fp):
        os.remove(fp)


def touch_simulation_mtime(name: str):
    """Update file mtime to now (marks simulation as recently accessed)."""
    fp = os.path.join(SIMULATIONS_DIR, f"{name}.json")
    if os.path.exists(fp):
        os.utime(fp, None)


def get_simulation_metadata(name: str) -> dict:
    """Return lightweight metadata for a saved simulation."""
    data = load_simulation(name)
    inp = data.get("inputs", {})
    return {
        "name": data.get("name", name),
        "saved_at": data.get("saved_at", ""),
        "system_size_kw": inp.get("system_size_kw", 0),
        "battery_capacity_kwh": inp.get(
            "battery_capacity_kwh", data.get("battery_capacity_kwh", 0)
        ),
        "utility": inp.get("utility", "N/A"),
        "has_battery": data.get("has_battery", False),
    }


def populate_session_from_simulation(st_session_state, sim_data: dict):
    """Populate all sidebar widget keys and prerequisites from a saved simulation dict.

    This is the extracted logic from the Edit Simulation handler.
    Does NOT call st.rerun() — the caller must do that.
    """
    inp = sim_data.get("inputs", {})

    # 1) Billing engine
    _saved_engine = inp.get("billing_engine", "Custom")
    st_session_state["billing_engine_radio"] = _saved_engine
    st_session_state["billing_engine"] = _saved_engine
    if _saved_engine == "ECC" and inp.get("ecc_tariff_metadata"):
        st_session_state["ecc_tariff_metadata"] = inp["ecc_tariff_metadata"]

    # 2) Location & system config
    st_session_state["sb_location"] = inp.get("location", "")
    st_session_state["sb_system_size"] = float(inp.get("system_size_kw", 500.0))
    st_session_state["sb_dc_ac_ratio"] = float(inp.get("dc_ac_ratio", 1.2))
    _sys_type = inp.get("system_type", "Fixed Tilt (Ground Mount)")
    _sys_type_options = ["Fixed Tilt (Ground Mount)", "Single Axis Tracker"]
    st_session_state["sb_system_type"] = (
        _sys_type if _sys_type in _sys_type_options else _sys_type_options[0]
    )
    _cod_str = inp.get("cod_date")
    if _cod_str:
        st_session_state["sb_cod_date"] = date.fromisoformat(_cod_str)
    _restore_cod_year = date.fromisoformat(_cod_str).year if _cod_str else 2026
    _util = inp.get("utility", "PG&E")
    _util_options = list(UTILITY_EIA_IDS.keys())
    st_session_state["sb_utility"] = (
        _util if _util in _util_options else _util_options[0]
    )
    st_session_state["sb_rate_escalator"] = float(inp.get("rate_escalator", 3.0))
    st_session_state["sb_load_escalator"] = float(inp.get("load_escalator", 2.0))

    # 3) Cost
    _cost_method = inp.get("cost_input_method", "$/W-DC")
    st_session_state["sb_cost_method"] = _cost_method
    if _cost_method == "$/W-DC":
        _size = float(inp.get("system_size_kw", 500.0))
        _total = float(inp.get("system_cost", 750000.0))
        st_session_state["sb_cost_per_watt"] = (
            round(_total / (_size * 1000), 2) if _size > 0 else 1.50
        )
    else:
        st_session_state["sb_total_cost"] = float(inp.get("system_cost", 750000.0))

    # 4) Battery
    _batt_enabled = inp.get("battery_enabled", sim_data.get("has_battery", False))
    st_session_state["bess_toggle"] = _batt_enabled
    if _batt_enabled:
        _batt_cfg = inp.get("battery_config", {})
        st_session_state["sb_batt_hours"] = float(
            _batt_cfg.get("battery_hours", sim_data.get("battery_hours", 4.0))
        )
        st_session_state["sb_discharge_limit"] = float(
            _batt_cfg.get("discharge_limit_pct", 80.0)
        )
        st_session_state["sb_batt_capacity"] = float(
            inp.get(
                "battery_capacity_kwh",
                sim_data.get("battery_capacity_kwh", 500.0),
            )
        )
        st_session_state["sb_charge_eff"] = float(
            _batt_cfg.get("charge_eff", 0.95)
        )
        st_session_state["sb_discharge_eff"] = float(
            _batt_cfg.get("discharge_eff", 0.95)
        )
        st_session_state["sb_min_soc"] = float(
            _batt_cfg.get("min_soc_pct", 10.0)
        )
        st_session_state["sb_max_soc"] = float(
            _batt_cfg.get("max_soc_pct", 100.0)
        )

    # 5) System Life & NEM regime
    st_session_state["sb_system_life"] = inp.get("system_life_years", 25)
    st_session_state["sb_nem_regime_1"] = inp.get("nem_regime_1", "NEM-3 / NVBT")
    st_session_state["nem_switch_toggle"] = inp.get("nem_switch", False)
    st_session_state["sb_nem_regime_2"] = inp.get("nem_regime_2", "NEM-3 / NVBT")
    st_session_state["sb_nem_years_1"] = inp.get("nem_years_1", 5)
    st_session_state["nbc_rate"] = inp.get("nbc_rate", 0.0)
    st_session_state["nsc_rate"] = inp.get("nsc_rate", NSC_DEFAULT_RATE)
    st_session_state["billing_option"] = inp.get("billing_option", "ABO")

    # 6) Restore section 2 export rates if present
    if sim_data.get("export_rates_2"):
        dt_idx = pd.date_range(
            start=f"{_restore_cod_year}-01-01 00:00", periods=8760, freq="h"
        )
        st_session_state["export_rates_2"] = pd.Series(
            sim_data["export_rates_2"], index=dt_idx, name="export_rate_per_kwh"
        )
    if sim_data.get("export_rates_multiyear_2"):
        _raw_my2 = sim_data["export_rates_multiyear_2"]
        dt_idx = pd.date_range(
            start=f"{_restore_cod_year}-01-01 00:00", periods=8760, freq="h"
        )
        st_session_state["export_rates_multiyear_2"] = {
            int(k): pd.Series(v, index=dt_idx, name="export_rate_per_kwh")
            for k, v in _raw_my2.items()
        }

    # 7) Restore the 4 prerequisites from saved data
    if sim_data.get("production_8760"):
        dt_idx = pd.date_range(
            start=f"{_restore_cod_year}-01-01 00:00", periods=8760, freq="h"
        )
        st_session_state["production_8760"] = pd.Series(
            sim_data["production_8760"], index=dt_idx, name="ac_watts"
        )
    if sim_data.get("load_8760"):
        dt_idx = pd.date_range(
            start=f"{_restore_cod_year}-01-01 00:00", periods=8760, freq="h"
        )
        st_session_state["load_8760"] = pd.Series(
            sim_data["load_8760"], index=dt_idx, name="load_kwh"
        )
    if sim_data.get("export_rates"):
        dt_idx = pd.date_range(
            start=f"{_restore_cod_year}-01-01 00:00", periods=8760, freq="h"
        )
        st_session_state["export_rates"] = pd.Series(
            sim_data["export_rates"], index=dt_idx, name="export_rate_per_kwh"
        )
    if sim_data.get("tariff_data"):
        td = sim_data["tariff_data"]
        st_session_state["tariff"] = TariffSchedule(**td)

    # 8) Close saved view and flag editing mode
    st_session_state["saved_view"] = None
    st_session_state["editing_saved_sim"] = True
