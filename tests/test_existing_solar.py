"""Tests for existing solar (repower) load adjustment functions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from modules.load_adjustment import adjust_load_single_meter, adjust_loads_nema


@pytest.fixture
def dt_index():
    """Standard 8760 DatetimeIndex for testing."""
    return pd.date_range("2026-01-01", periods=8760, freq="h")


@pytest.fixture
def make_series(dt_index):
    """Factory to create a pd.Series from a repeated hourly value."""
    def _make(value, name="load_kwh"):
        return pd.Series(np.full(8760, value), index=dt_index, name=name)
    return _make


# ---------- Single Meter Tests ----------

def test_single_meter_basic(dt_index):
    """Load + solar = correct gross load."""
    raw_load = pd.Series(np.full(8760, 50.0), index=dt_index, name="load_kwh")
    existing_solar = pd.Series(np.full(8760, 20.0), index=dt_index, name="solar_kwh")

    result = adjust_load_single_meter(raw_load, existing_solar)

    np.testing.assert_array_almost_equal(result.values, np.full(8760, 70.0))


def test_single_meter_floor_at_zero(dt_index):
    """Clips to zero when solar estimate > implied consumption."""
    raw_load = pd.Series(np.full(8760, 10.0), index=dt_index, name="load_kwh")
    existing_solar = pd.Series(np.full(8760, 50.0), index=dt_index, name="solar_kwh")

    result = adjust_load_single_meter(raw_load, existing_solar)

    # 10 + 50 = 60, no clipping needed (positive result)
    np.testing.assert_array_almost_equal(result.values, np.full(8760, 60.0))

    # Test with negative raw_load to trigger clipping
    raw_load_neg = pd.Series(np.full(8760, -30.0), index=dt_index, name="load_kwh")
    result_neg = adjust_load_single_meter(raw_load_neg, existing_solar)

    # -30 + 50 = 20, still positive
    np.testing.assert_array_almost_equal(result_neg.values, np.full(8760, 20.0))

    # Test where sum would be negative
    raw_load_very_neg = pd.Series(np.full(8760, -60.0), index=dt_index, name="load_kwh")
    result_floor = adjust_load_single_meter(raw_load_very_neg, existing_solar)

    # -60 + 50 = -10 → clipped to 0
    np.testing.assert_array_almost_equal(result_floor.values, np.full(8760, 0.0))


def test_single_meter_negative_interval(dt_index):
    """Handles negative interval data (net export hours) correctly."""
    # Mix of positive and negative values (net export hours)
    vals = np.zeros(8760)
    vals[:4380] = 100.0   # daytime consumption
    vals[4380:] = -20.0   # net export during high-solar hours
    raw_load = pd.Series(vals, index=dt_index, name="load_kwh")
    existing_solar = pd.Series(np.full(8760, 15.0), index=dt_index, name="solar_kwh")

    result = adjust_load_single_meter(raw_load, existing_solar)

    # Positive hours: 100 + 15 = 115
    np.testing.assert_array_almost_equal(result.values[:4380], np.full(4380, 115.0))
    # Negative hours: -20 + 15 = -5 → clipped to 0
    np.testing.assert_array_almost_equal(result.values[4380:], np.full(4380, 0.0))


# ---------- NEM-A Tests ----------

def test_nema_proportional_two_meters(dt_index):
    """60/40 load split → 60%/40% solar distribution."""
    meter_loads = {
        0: pd.Series(np.full(8760, 60.0), index=dt_index, name="load_kwh"),
        1: pd.Series(np.full(8760, 40.0), index=dt_index, name="load_kwh"),
    }
    existing_solar = pd.Series(np.full(8760, 100.0), index=dt_index, name="solar_kwh")

    result = adjust_loads_nema(meter_loads, existing_solar, [0, 1])

    # Meter 0: 60 + 100*(60/100) = 60 + 60 = 120
    np.testing.assert_array_almost_equal(result[0].values, np.full(8760, 120.0))
    # Meter 1: 40 + 100*(40/100) = 40 + 40 = 80
    np.testing.assert_array_almost_equal(result[1].values, np.full(8760, 80.0))


def test_nema_three_meters_one_unselected(dt_index):
    """Unselected meter unchanged."""
    meter_loads = {
        0: pd.Series(np.full(8760, 60.0), index=dt_index, name="load_kwh"),
        1: pd.Series(np.full(8760, 40.0), index=dt_index, name="load_kwh"),
        2: pd.Series(np.full(8760, 25.0), index=dt_index, name="load_kwh"),
    }
    existing_solar = pd.Series(np.full(8760, 50.0), index=dt_index, name="solar_kwh")

    result = adjust_loads_nema(meter_loads, existing_solar, [0, 1])

    # Meter 2 should be unchanged
    np.testing.assert_array_almost_equal(result[2].values, np.full(8760, 25.0))

    # Meter 0: 60 + 50*(60/100) = 60 + 30 = 90
    np.testing.assert_array_almost_equal(result[0].values, np.full(8760, 90.0))
    # Meter 1: 40 + 50*(40/100) = 40 + 20 = 60
    np.testing.assert_array_almost_equal(result[1].values, np.full(8760, 60.0))


def test_nema_only_generating_selected(dt_index):
    """Behaves like single-meter for that meter."""
    meter_loads = {
        0: pd.Series(np.full(8760, 50.0), index=dt_index, name="load_kwh"),
        1: pd.Series(np.full(8760, 30.0), index=dt_index, name="load_kwh"),
    }
    existing_solar = pd.Series(np.full(8760, 20.0), index=dt_index, name="solar_kwh")

    result = adjust_loads_nema(meter_loads, existing_solar, [0])

    # Meter 0: 50 + 20*(50/50) = 50 + 20 = 70 (100% share since it's the only selected)
    np.testing.assert_array_almost_equal(result[0].values, np.full(8760, 70.0))
    # Meter 1 unchanged
    np.testing.assert_array_almost_equal(result[1].values, np.full(8760, 30.0))


def test_nema_zero_load_hour_fallback(dt_index):
    """Equal share when all selected loads = 0."""
    vals_0 = np.full(8760, 50.0)
    vals_0[0] = 0.0  # hour 0 is zero for meter 0
    vals_1 = np.full(8760, 50.0)
    vals_1[0] = 0.0  # hour 0 is zero for meter 1

    meter_loads = {
        0: pd.Series(vals_0, index=dt_index, name="load_kwh"),
        1: pd.Series(vals_1, index=dt_index, name="load_kwh"),
    }
    existing_solar = pd.Series(np.full(8760, 40.0), index=dt_index, name="solar_kwh")

    result = adjust_loads_nema(meter_loads, existing_solar, [0, 1])

    # Hour 0: both zero → equal share → 0 + 40*(1/2) = 20 each
    assert result[0].iloc[0] == pytest.approx(20.0)
    assert result[1].iloc[0] == pytest.approx(20.0)

    # Normal hours: 50 + 40*(50/100) = 50 + 20 = 70
    assert result[0].iloc[1] == pytest.approx(70.0)


def test_nema_empty_selection(dt_index):
    """No adjustment if no meters selected."""
    meter_loads = {
        0: pd.Series(np.full(8760, 50.0), index=dt_index, name="load_kwh"),
        1: pd.Series(np.full(8760, 30.0), index=dt_index, name="load_kwh"),
    }
    existing_solar = pd.Series(np.full(8760, 20.0), index=dt_index, name="solar_kwh")

    result = adjust_loads_nema(meter_loads, existing_solar, [])

    np.testing.assert_array_almost_equal(result[0].values, np.full(8760, 50.0))
    np.testing.assert_array_almost_equal(result[1].values, np.full(8760, 30.0))


def test_adjustment_preserves_index(dt_index):
    """DatetimeIndex preserved after adjustment."""
    raw_load = pd.Series(np.full(8760, 50.0), index=dt_index, name="load_kwh")
    existing_solar = pd.Series(np.full(8760, 20.0), index=dt_index, name="solar_kwh")

    result = adjust_load_single_meter(raw_load, existing_solar)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result.index) == 8760
    pd.testing.assert_index_equal(result.index, dt_index)


def test_degradation_factor():
    """Compound degradation math: (1 - 0.005)^10 ≈ 0.951."""
    degradation_rate = 0.5  # percent
    age_years = 10
    factor = (1 - degradation_rate / 100) ** age_years

    assert factor == pytest.approx(0.95111, rel=1e-4)

    # Verify the factor applied to production
    original_production = 1000.0  # kWh
    degraded = original_production * factor
    assert degraded == pytest.approx(951.1, rel=1e-2)
