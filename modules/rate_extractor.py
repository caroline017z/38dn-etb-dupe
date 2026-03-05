"""
Custom Rate Extractor — PDF upload + Claude API structured tariff extraction.

Extracts OpenEI-format tariff JSON from utility rate tariff PDFs using
pdfplumber for text extraction and Claude API for structured parsing.
"""

import json
import os
import logging
from typing import Any

import pdfplumber
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from a PDF uploaded via Streamlit file_uploader."""
    text_pages: list[str] = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
            # Also try extracting tables as text
            for table in page.extract_tables():
                rows = ["\t".join(str(cell or "") for cell in row) for row in table]
                text_pages.append("\n".join(rows))
    return "\n\n".join(text_pages)


# ---------------------------------------------------------------------------
# Claude API tariff extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert utility rate tariff analyst. Given the text of a utility rate tariff PDF, \
extract the structured rate data into OpenEI JSON format.

Return ONLY a single JSON object (not an array) with these fields:

Required fields:
- "name": string — full tariff schedule name including effective date
- "label": string — short kebab-case label (e.g. "pge-ag-c-secondary-2026")
- "description": string — 1-2 sentence description of the tariff
- "utility": string — full utility name
- "eiaid": integer — EIA utility ID (PG&E=14328, SCE=17609, SDG&E=16609)
- "sector": string — "Commercial", "Residential", or "Industrial"
- "startdate": string — ISO date "YYYY-01-01T00:00:00.000Z"
- "enddate": string — ISO date "YYYY-12-31T23:59:59.000Z"
- "source": string — source document reference

Energy rates:
- "energyratestructure": array of arrays, each inner array has one object {"rate": float, "unit": "kWh"}
  Each element corresponds to a TOU period (index 0, 1, 2, ...).
- "energyweekdayschedule": 12×24 matrix (12 months × 24 hours). Each cell is the period index into energyratestructure.
- "energyweekendschedule": 12×24 matrix, same format.
- "energycomments": string describing each energy period (e.g., "Period 0: Off-Peak Winter. Period 1: Peak Summer.")
- "demandcomments": string describing each demand period (e.g., "Period 0: No Demand Charge. Period 1: Peak Summer Demand.")

Demand charges (if present):
- "demandunits": "kW"
- "demandrateunit": "kW"
- "demandratestructure": array of arrays, each inner array has one object {"rate": float}
  Index corresponds to demand TOU period.
- "demandweekdayschedule": 12×24 matrix of demand period indices.
- "demandweekendschedule": 12×24 matrix of demand period indices.

Flat demand (if present):
- "flatdemandstructure": [[{"rate": float}]]
- "flatdemandmonths": array of 12 integers (0 for all months typically)
- "flatdemandunit": "kW"

Fixed charges:
- "fixedchargefirstmeter": float — daily fixed charge in $/day
- "fixedchargeunits": "$/day"

Optional:
- "voltagecategory": "Secondary" or "Primary"
- "phasewiring": "Three Phase" or "Single Phase"
- "peakkwcapacitymin": integer — minimum kW for eligibility
- "minmonthlycharge": float — minimum monthly charge if specified
- "minmonthlychargeunits": "$/month"

Important rules:
- CRITICAL: Utility tariff PDFs often contain MULTIPLE effective dates and rate revisions \
on the same sheet (e.g., columns for different advice letters or effective dates). \
You MUST extract ONLY the MOST RECENT / LATEST effective rates — the ones with the newest \
effective date or the highest-numbered advice letter. Ignore all older/superseded columns. \
If rates are shown side-by-side for different effective dates, use ONLY the rightmost / newest column.
- TOU schedules must be exactly 12 rows (Jan-Dec) × 24 columns (hours 0-23).
- Period indices must be 0-based and match the corresponding ratestructure arrays.
- All rates should be in $/kWh for energy and $/kW for demand.
- If a charge type is not present in the tariff, omit those fields entirely.
- Return valid JSON only, no markdown fences or extra text.
"""


def _get_api_key() -> str:
    """Retrieve Anthropic API key from environment or Streamlit secrets."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it in your .env file or Streamlit secrets."
        )
    return key


def _load_example_tariff() -> str:
    """Load the PGE AG-C example tariff JSON as a string for the prompt."""
    example_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "ecc_tariffs", "PGE_AG-C_2026.json"
    )
    try:
        with open(example_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def extract_tariff_from_text(
    pdf_text: str, utility: str = "", rate_name: str = ""
) -> dict:
    """Call Claude API to extract structured OpenEI tariff JSON from PDF text."""
    import anthropic

    api_key = _get_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    example_json = _load_example_tariff()
    example_section = ""
    if example_json:
        example_section = (
            "\n\nHere is a complete example of the target JSON format "
            "(PG&E AG-C 2026):\n" + example_json
        )

    user_msg = (
        "Extract the tariff data from the following utility rate PDF text. "
        "IMPORTANT: This PDF may contain multiple rate revisions or effective dates. "
        "Extract ONLY the most recent/latest effective rates (newest date, highest advice letter number). "
        "Ignore all older superseded rates."
    )
    if utility:
        user_msg += f"\nUtility: {utility}"
    if rate_name:
        user_msg += f"\nRate schedule name: {rate_name}"
    user_msg += f"{example_section}\n\n--- PDF TEXT ---\n{pdf_text}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8192,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tariff_structure(tariff_data: dict) -> list[str]:
    """Validate required fields and schedule dimensions. Returns list of warnings."""
    warnings: list[str] = []

    required = ["name", "energyratestructure", "energyweekdayschedule", "energyweekendschedule"]
    for field in required:
        if field not in tariff_data:
            warnings.append(f"Missing required field: {field}")

    # Validate startdate / enddate (required by ECC loader)
    for date_field in ["startdate", "enddate"]:
        val = tariff_data.get(date_field)
        if not val:
            warnings.append(f"Missing {date_field} — will default to 2026 calendar year")
        elif not isinstance(val, str) or "T" not in val:
            warnings.append(f"{date_field} has unexpected format: {val}")

    # Validate 12×24 schedules
    for sched_key in [
        "energyweekdayschedule", "energyweekendschedule",
        "demandweekdayschedule", "demandweekendschedule",
    ]:
        sched = tariff_data.get(sched_key)
        if sched is None:
            continue
        if not isinstance(sched, list) or len(sched) != 12:
            warnings.append(f"{sched_key}: expected 12 rows (months), got {len(sched) if isinstance(sched, list) else 'non-list'}")
            continue
        for i, row in enumerate(sched):
            if not isinstance(row, list) or len(row) != 24:
                warnings.append(f"{sched_key}[{i}]: expected 24 columns (hours), got {len(row) if isinstance(row, list) else 'non-list'}")

    # Validate period indices match ratestructure length
    for prefix in ["energy", "demand"]:
        struct = tariff_data.get(f"{prefix}ratestructure")
        if struct is None:
            continue
        n_periods = len(struct)
        for sched_key in [f"{prefix}weekdayschedule", f"{prefix}weekendschedule"]:
            sched = tariff_data.get(sched_key)
            if sched is None:
                continue
            max_idx = max(max(row) for row in sched if row)
            if max_idx >= n_periods:
                warnings.append(
                    f"{sched_key} references period {max_idx} but {prefix}ratestructure "
                    f"only has {n_periods} periods (indices 0-{n_periods - 1})"
                )

    # Validate rate values are reasonable
    for entry in tariff_data.get("energyratestructure", []):
        if entry and isinstance(entry, list):
            rate = entry[0].get("rate", 0)
            if rate < 0 or rate > 2.0:
                warnings.append(f"Unusual energy rate: ${rate:.4f}/kWh")

    return warnings


# ---------------------------------------------------------------------------
# Save / list
# ---------------------------------------------------------------------------

def save_custom_tariff(name: str, tariff_data: dict, tariffs_dir: str) -> str:
    """Save tariff as OpenEI JSON array to the ECC tariffs directory. Returns file path."""
    from sim_helpers import sanitize_filename

    safe_name = sanitize_filename(name)
    path = os.path.join(tariffs_dir, f"{safe_name}.json")

    # Ensure startdate/enddate are present (required by ECC loader)
    td = dict(tariff_data)
    td.setdefault("startdate", "2026-01-01T00:00:00.000Z")
    td.setdefault("enddate", "2026-12-31T23:59:59.000Z")
    td.setdefault("approved", True)

    # OpenEI format expects an array
    data = [td]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
