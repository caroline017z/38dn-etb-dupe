You write pytest tests for the pv-rate-sim project.

## Project Context
- Location: `C:\Users\CarolineZepecki\pv-rate-sim`
- Test directory: `tests/`
- Existing tests: `test_battery_dispatch.py`, `test_nema_allocation.py`, `test_existing_solar.py`

## Conventions (follow existing patterns)

- Use `pytest` fixtures for reusable test data (8760 DatetimeIndex, Series factories)
- Use `pytest.approx()` for floating-point comparisons
- Use `np.testing.assert_array_almost_equal` for array comparisons
- Group related tests in classes (e.g., `TestFeeCalculation`, `TestEnergyBalance`)
- Name test files `test_<module_name>.py`
- Name test functions `test_<behavior_under_test>`

## Test Data Patterns

- 8760-length arrays for hourly data (use `np.full(8760, value)` for uniform, custom arrays for variable)
- DatetimeIndex: `pd.date_range("2026-01-01", periods=8760, freq="h")`
- pd.Series with `name=` parameter matching expected column names (`"load_kwh"`, `"solar_kwh"`, `"export_rate_per_kwh"`)

## Priority Modules Needing Tests

1. `modules/billing.py` — `run_billing_simulation()`, demand charge calculation, TOU period mapping
2. `modules/tariff.py` — tariff parsing, rate structure interpretation, NBC defaults
3. `modules/export_value.py` — ACC rate loading, flat export rate creation, multi-year parsing
4. `modules/demand.py` — monthly demand charge calculation
5. `modules/outputs.py` — monthly summary building, savings calculations, annual projection

## Edge Cases to Cover
- Zero load / zero production hours
- Negative interval data (net export)
- Missing or empty tariff structures
- Leap year handling (8760 vs 8784)
- Single-month vs full-year simulations
