"""Utility-bill PDF ingestion via pdfplumber + Anthropic tool use.

Two-step pipeline:

1. ``pdfplumber`` extracts raw text + tables from the PDF. This keeps
   token usage predictable — the model sees a compact text rendering, not
   the full binary.
2. The model is prompted with a single ``record_bill_data`` tool whose
   input schema captures the fields the app actually uses. The model
   invokes the tool; we parse its arguments into :class:`BillExtraction`.

Deterministic validation happens outside the model so downstream code
never sees half-populated data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import call_with_cache, tool_use_input, CachedBlock


TOOL_NAME = "record_bill_data"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Record structured data extracted from a utility bill. "
        "Use null for any field that is not present on the bill."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "utility": {
                "type": "string",
                "description": "Utility name (e.g. 'PG&E', 'SCE', 'SDG&E'). Null if unclear.",
            },
            "rate_schedule": {
                "type": "string",
                "description": "Tariff / rate schedule code (e.g. 'E-19', 'AG-C', 'B-19', 'TOU-GS-3').",
            },
            "billing_period_start": {"type": "string", "description": "YYYY-MM-DD."},
            "billing_period_end": {"type": "string", "description": "YYYY-MM-DD."},
            "total_kwh": {"type": "number", "description": "Total metered kWh for the period."},
            "peak_kwh": {"type": "number"},
            "mid_peak_kwh": {"type": "number"},
            "off_peak_kwh": {"type": "number"},
            "max_demand_kw": {"type": "number"},
            "peak_demand_kw": {"type": "number"},
            "total_charges_usd": {"type": "number"},
            "nem_true_up": {
                "type": "boolean",
                "description": "True if this bill is a NEM annual true-up statement.",
            },
            "notes": {
                "type": "string",
                "description": "Anything relevant not captured above, one sentence max.",
            },
        },
        "required": ["utility", "rate_schedule", "total_kwh", "total_charges_usd"],
    },
}


_SYSTEM = (
    "You extract structured data from utility bills. You always call the "
    f"`{TOOL_NAME}` tool exactly once. Never guess — use null when the bill "
    "does not clearly show a field. Round money to 2 decimals, energy to 0 "
    "decimals, demand to 1 decimal."
)


@dataclass(frozen=True)
class BillExtraction:
    utility: str | None
    rate_schedule: str | None
    billing_period_start: str | None
    billing_period_end: str | None
    total_kwh: float | None
    peak_kwh: float | None
    mid_peak_kwh: float | None
    off_peak_kwh: float | None
    max_demand_kw: float | None
    peak_demand_kw: float | None
    total_charges_usd: float | None
    nem_true_up: bool
    notes: str | None
    raw_text: str = field(repr=False, default="")


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    import io
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f"\n--- Page {i} ---\n")
            text = page.extract_text() or ""
            parts.append(text)
            for table in page.extract_tables() or []:
                parts.append("\n[table]")
                for row in table:
                    parts.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(parts)


def extract_bill(pdf_bytes: bytes) -> BillExtraction:
    """Extract :class:`BillExtraction` from a utility-bill PDF."""
    raw = _extract_pdf_text(pdf_bytes)

    message = call_with_cache(
        system=_SYSTEM,
        user="Extract this utility bill by calling the tool.",
        cached_context=[CachedBlock(kind="text", content=raw)],
        tools=[TOOL_SCHEMA],
        max_tokens=1024,
    )
    data = tool_use_input(message, TOOL_NAME) or {}
    return _from_tool_input(data, raw)


def _from_tool_input(data: dict[str, Any], raw: str) -> BillExtraction:
    def _f(key: str) -> float | None:
        v = data.get(key)
        return float(v) if v is not None else None

    def _s(key: str) -> str | None:
        v = data.get(key)
        return str(v) if v not in (None, "") else None

    return BillExtraction(
        utility=_s("utility"),
        rate_schedule=_s("rate_schedule"),
        billing_period_start=_s("billing_period_start"),
        billing_period_end=_s("billing_period_end"),
        total_kwh=_f("total_kwh"),
        peak_kwh=_f("peak_kwh"),
        mid_peak_kwh=_f("mid_peak_kwh"),
        off_peak_kwh=_f("off_peak_kwh"),
        max_demand_kw=_f("max_demand_kw"),
        peak_demand_kw=_f("peak_demand_kw"),
        total_charges_usd=_f("total_charges_usd"),
        nem_true_up=bool(data.get("nem_true_up", False)),
        notes=_s("notes"),
        raw_text=raw,
    )


__all__ = ["BillExtraction", "TOOL_NAME", "TOOL_SCHEMA", "extract_bill"]
