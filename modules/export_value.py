"""
NVBT / Avoided Cost Calculator (ACC) export compensation module.

Loads hourly export compensation rates ($/kWh) for California utilities.
Primary source: pre-downloaded ACC 8760 CSVs stored in data/acc_export_rates/
Fallback: user-uploaded CSV or flat rate input.
"""

import os
import pandas as pd
import numpy as np


# Expected CSV filename patterns per utility
ACC_FILE_PATTERNS = {
    "PG&E": "pge_acc",
    "SCE": "sce_acc",
    "SDG&E": "sdge_acc",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "acc_export_rates")


def _find_rate_column(df: pd.DataFrame) -> str:
    """Find the export rate column in a DataFrame.

    Searches for columns with keywords: rate, export, value, price, avoided.
    Falls back to the first numeric column.

    Raises:
        ValueError: If no numeric column is found.
    """
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["rate", "export", "value", "price", "avoided"]):
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return str(numeric_cols[0])

    raise ValueError(
        "Could not find a numeric column with export rates in the CSV. "
        "Expected a column with 'rate', 'export', 'value', or 'price' in the name."
    )


def _is_hour_index_column(col_name: str, values: np.ndarray) -> bool:
    """Return True if a column appears to be an hour index (1-8760) rather than rates."""
    if any(kw in col_name.lower() for kw in ["hour", "index", "row"]):
        return True
    # Values monotonically 1..8760 are clearly an index, not rates
    if len(values) == 8760 and values[0] == 1 and values[-1] == 8760:
        return True
    return False


def parse_multiyear_export_rates(df: pd.DataFrame, start_year: int = 2026) -> dict[int, pd.Series]:
    """Parse a multi-year export rate CSV into per-year 8760 Series.

    Supports wide-format CSVs where each numeric column is one year of
    hourly rates (8760 rows). Columns that look like hour indices
    (name contains "hour"/"index", or values are 1-8760) are skipped.

    Rate columns whose header is a 4-digit year (e.g. "2026") are keyed
    by that calendar year.  Other rate columns are keyed sequentially
    starting from the current calendar year.

    Args:
        df: DataFrame with 8760 rows and one or more numeric year columns.

    Returns:
        dict mapping **calendar year** (e.g. 2026) to 8760-value Series.

    Raises:
        ValueError: If the DataFrame does not have exactly 8760 rows or
            has no rate columns.
    """
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise ValueError("No numeric columns found in the CSV.")

    # Filter out hour-index columns
    rate_cols = [
        c for c in numeric_cols
        if not _is_hour_index_column(c, df[c].values)
    ]
    if not rate_cols:
        raise ValueError("No rate columns found — all numeric columns appear to be hour indices.")

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows, got {len(df)}. "
            "The file should contain one row per hour of the year."
        )

    dt_index = pd.date_range(start=f"{start_year}-01-01 00:00", periods=8760, freq="h")

    # Try to parse column headers as calendar years
    result = {}
    fallback_start = None
    for col in rate_cols:
        col_str = str(col).strip()
        try:
            year_int = int(float(col_str))
            if 2000 <= year_int <= 2100:
                result[year_int] = pd.Series(
                    df[col].values, index=dt_index, name="export_rate_per_kwh"
                )
                continue
        except (ValueError, OverflowError):
            pass
        # Non-year header — will assign sequentially below
        if fallback_start is None:
            from datetime import datetime as _dt
            fallback_start = _dt.now().year

    # Handle columns without parseable year headers
    if fallback_start is not None:
        yr = fallback_start
        for col in rate_cols:
            col_str = str(col).strip()
            try:
                year_int = int(float(col_str))
                if 2000 <= year_int <= 2100:
                    continue  # already added above
            except (ValueError, OverflowError):
                pass
            # Assign sequential calendar year
            while yr in result:
                yr += 1
            result[yr] = pd.Series(
                df[col].values, index=dt_index, name="export_rate_per_kwh"
            )
            yr += 1

    return result


def find_acc_file(utility_name: str) -> str | None:
    """
    Look for an ACC export rate CSV file for the given utility.
    Returns the file path if found, None otherwise.
    """
    prefix = ACC_FILE_PATTERNS.get(utility_name)
    if prefix is None:
        return None

    if not os.path.isdir(DATA_DIR):
        return None

    # Find the most recent file matching the pattern
    candidates = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().startswith(prefix.lower()) and fname.endswith(".csv"):
            candidates.append(os.path.join(DATA_DIR, fname))

    if not candidates:
        return None

    # Return the most recently modified file
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_acc_from_file(filepath: str, start_year: int = 2026) -> pd.Series:
    """
    Load ACC export rates from a CSV file.

    Expected format:
    - 8760 rows
    - At least one column containing $/kWh export rates
    - Column name should contain 'rate', 'export', 'value', or 'price' (case-insensitive)
    - Or just a single numeric column

    Returns:
        pd.Series of length 8760 with export rates in $/kWh
    """
    df = pd.read_csv(filepath)
    rate_col = _find_rate_column(df)
    rates = df[rate_col].values

    if len(rates) != 8760:
        raise ValueError(
            f"Expected 8760 rows in ACC CSV, got {len(rates)}. "
            "The file should contain one row per hour of the year."
        )

    # Build datetime index matching production 8760
    dt_index = pd.date_range(start=f"{start_year}-01-01 00:00", periods=8760, freq="h")
    return pd.Series(rates, index=dt_index, name="export_rate_per_kwh")


def load_acc_from_upload(uploaded_file, start_year: int = 2026) -> tuple[pd.Series, dict[int, pd.Series] | None]:
    """
    Load ACC export rates from a Streamlit uploaded file object.

    Supports:
    - Single-year CSV: 8760 rows, one numeric column → (Series, None)
    - Multi-year CSV: 8760 rows, multiple numeric columns (one per year)
      → (year1_series, {1: Series, 2: Series, ...})

    Returns:
        (year1_series, multiyear_dict_or_None)
    """
    df = pd.read_csv(uploaded_file)

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows, got {len(df)}. "
            "The file should contain one row per hour of the year."
        )

    multiyear = parse_multiyear_export_rates(df, start_year=start_year)
    first_year_key = min(multiyear.keys())
    year1_series = multiyear[first_year_key]

    if len(multiyear) == 1:
        return year1_series, None

    return year1_series, multiyear


def create_flat_export_rates(flat_rate: float, start_year: int = 2026) -> pd.Series:
    """
    Create an 8760 export rate series using a single flat $/kWh value.

    Args:
        flat_rate: $/kWh flat export compensation rate

    Returns:
        pd.Series of length 8760 with constant export rate
    """
    dt_index = pd.date_range(start=f"{start_year}-01-01 00:00", periods=8760, freq="h")
    return pd.Series(
        np.full(8760, flat_rate),
        index=dt_index,
        name="export_rate_per_kwh",
    )


def get_export_rates(utility_name: str, start_year: int = 2026) -> tuple[pd.Series | None, str]:
    """
    Attempt to load ACC export rates for a utility.

    Returns:
        (rates_series_or_None, status_message)
        - If found: (Series, "Loaded from {filepath}")
        - If not found: (None, "No ACC data found for {utility}")
    """
    filepath = find_acc_file(utility_name)

    if filepath is not None:
        try:
            rates = load_acc_from_file(filepath, start_year=start_year)
            return rates, f"Loaded from {os.path.basename(filepath)}"
        except Exception as e:
            return None, f"Error reading ACC file: {e}"

    return None, f"No ACC export rate data found for {utility_name}. Please upload a CSV or enter a flat rate."
