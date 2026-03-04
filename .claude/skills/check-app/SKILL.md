---
name: check-app
description: Verify the Streamlit app starts without import errors and all tests pass
user-invocable: true
---

# Check App Health

Run these checks in order for the pv-rate-sim project at `C:\Users\CarolineZepecki\pv-rate-sim`:

1. **Import check**: Run `python -c "import app"` from the project root to verify no import errors or missing dependencies.
2. **Test suite**: Run `python -m pytest tests/ -v` to execute all tests.
3. **Report**: Summarize results — number of tests passed/failed, any import errors, and whether the app is healthy.

If any check fails, investigate the error and suggest a fix.
