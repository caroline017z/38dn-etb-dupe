"""Grounded Q&A assistant for the selected system, tariff, NEM regime, and
billing structure.

No tool use, no browsing — the model answers strictly from the context
attached as a cached block. Callers pass either the URDB tariff JSON alone
or a composite dict with `urdb_tariff` + `system_context` keys.
"""

from __future__ import annotations

import json

from .client import CachedBlock, call_with_cache, text_from


_SYSTEM = (
    "You answer short factual questions about a California PV+BESS project's "
    "selected system, tariff, NEM regime, and billing structure. The user turn "
    "attaches a JSON context with two keys: `urdb_tariff` (the tariff rate "
    "structure from the URDB) and `system_context` (the selected PV/BESS size, "
    "NEM regime, utility). "
    "Ground every answer in the attached context. If a question cannot be "
    "answered from the context, say so explicitly. Keep answers under 100 "
    "words. Cite dollar amounts, kWh/kW thresholds, and TOU windows verbatim "
    "from the tariff where relevant. For NEM regime questions, explain how the "
    "regime treats exports (retail TOU netting for NEM-1/2, hourly ACC "
    "settlement for NEM-3) and point to the specific tariff fields that drive "
    "the math (energy rate structure, export compensation mechanism, NBC/NSC)."
)


def ask(question: str, *, tariff_label: str, urdb_json: dict) -> str:
    """Return a short grounded answer. The URDB JSON is attached as a
    cached block so follow-up questions within the same session reuse the
    cache and cost ~10% of the uncached input price.
    """
    context = json.dumps(urdb_json, indent=2, sort_keys=True)
    message = call_with_cache(
        system=_SYSTEM,
        user=f"Tariff: {tariff_label}\n\nQuestion: {question}",
        cached_context=[CachedBlock(kind="text", content=context)],
        max_tokens=512,
    )
    return text_from(message).strip()


__all__ = ["ask"]
