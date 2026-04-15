"""Grounded Q&A assistant for the currently-selected URDB tariff.

Intentionally minimal: no tool use, no browsing — the model answers
strictly from the URDB JSON we attach as a cached context block. Short
answers, direct quotes where appropriate.
"""

from __future__ import annotations

import json

from .client import CachedBlock, call_with_cache, text_from


_SYSTEM = (
    "You answer short factual questions about a California utility tariff. "
    "Answer only from the URDB JSON in the user turn — if the answer is not "
    "there, say 'Not stated in the tariff JSON.' Keep answers under 80 words. "
    "Cite dollar amounts, kWh/kW thresholds, and TOU windows verbatim."
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
