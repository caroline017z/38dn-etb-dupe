"""AI-generated executive-summary bullets for the proposal PPTX.

The model receives a compact JSON summary of the simulation outcome and
returns 3–5 short factual bullets. Voice is locked via the system prompt —
concise, factual, no marketing hype.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .client import call_with_cache, text_from


_SYSTEM = """You are drafting the executive summary bullets for a solar + battery proposal.

Rules:
- Exactly 3 to 5 bullets.
- Each bullet: one sentence, under 25 words, factual, no hype.
- Lead with the number where possible ($, %, years).
- No marketing adjectives (e.g. "amazing", "cutting-edge", "best-in-class").
- No greetings, no closings, no headers. Just the bullets.

Output format:
- Start each bullet with "- " on its own line.
- Nothing else."""


@dataclass(frozen=True)
class ProposalContext:
    customer_name: str
    site_address: str
    system_size_kw: float
    battery_capacity_kwh: float
    nem_regime: str
    year1_savings_usd: float
    year1_bill_without_solar_usd: float
    year1_bill_with_solar_usd: float
    savings_pct: float
    horizon_years: int
    total_projected_savings_usd: float
    ppa_rate_usd_per_kwh: float | None = None


def generate_executive_summary(ctx: ProposalContext) -> list[str]:
    """Returns 3–5 bullet strings. Does not raise on malformed responses —
    falls back to a deterministic summary if parsing produces fewer than
    3 usable bullets.
    """
    payload = json.dumps(asdict(ctx), indent=2, sort_keys=True)
    user = (
        "Write the executive-summary bullets for this scenario:\n\n```json\n"
        + payload
        + "\n```"
    )
    message = call_with_cache(system=_SYSTEM, user=user, max_tokens=512)
    bullets = _extract_bullets(text_from(message))
    if len(bullets) < 3:
        return _fallback(ctx)
    return bullets[:5]


_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")


def _extract_bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def _fallback(ctx: ProposalContext) -> list[str]:
    bullets = [
        f"Year-1 savings: ${ctx.year1_savings_usd:,.0f} "
        f"({ctx.savings_pct:.1f}% vs utility-only bill of "
        f"${ctx.year1_bill_without_solar_usd:,.0f}).",
        f"{ctx.horizon_years}-year projected savings: "
        f"${ctx.total_projected_savings_usd:,.0f}.",
        f"System size: {ctx.system_size_kw:,.0f} kW PV"
        + (f" with {ctx.battery_capacity_kwh:,.0f} kWh battery"
           if ctx.battery_capacity_kwh > 0 else "")
        + f" under {ctx.nem_regime}.",
    ]
    if ctx.ppa_rate_usd_per_kwh is not None:
        bullets.append(f"Indicative PPA rate: ${ctx.ppa_rate_usd_per_kwh:.3f}/kWh.")
    return bullets


__all__ = ["ProposalContext", "generate_executive_summary"]
