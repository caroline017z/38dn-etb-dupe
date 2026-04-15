"""Shared Anthropic client with prompt caching.

Every call site in this codebase routes through :func:`get_client` and
:func:`call_with_cache` so that:

* API key resolution (Streamlit secrets → env var) happens once.
* The default model is set in one place (see :data:`DEFAULT_MODEL`).
* System prompts and any large grounding blocks (URDB JSON, bill text)
  automatically get ``cache_control={"type": "ephemeral"}`` attached,
  which is a substantial cost win when users iterate in the UI.

See docs/anthropic prompt caching: same prefix reused within ~5 min TTL
reads at 10% of input price. For this app that's rates data, system
prompts, and uploaded bill text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


class AnthropicCreditError(RuntimeError):
    """Raised when the Anthropic API returns a 400 with "credit balance
    too low". Callers catch this separately to surface a friendlier
    message than dumping the raw 400 body in the UI."""


def _resolve_api_key() -> str | None:
    try:
        import streamlit as st

        try:
            return st.secrets["ANTHROPIC_API_KEY"]  # type: ignore[index]
        except (KeyError, FileNotFoundError):
            pass
    except ImportError:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def get_client() -> anthropic.Anthropic:
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .streamlit/secrets.toml "
            "or the environment before invoking AI features."
        )
    return anthropic.Anthropic(api_key=api_key)


@dataclass(frozen=True)
class CachedBlock:
    """A message-content block that should be cached.

    ``kind`` ∈ {"text"}; extended to "document"/"image" when needed.
    """

    kind: str
    content: str


def _to_cached_content(blocks: list[CachedBlock] | None) -> list[dict[str, Any]]:
    if not blocks:
        return []
    out: list[dict[str, Any]] = []
    for b in blocks:
        out.append({
            "type": b.kind,
            "text": b.content,
            "cache_control": {"type": "ephemeral"},
        })
    return out


def call_with_cache(
    *,
    system: str,
    user: str,
    cached_context: list[CachedBlock] | None = None,
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: anthropic.Anthropic | None = None,
) -> anthropic.types.Message:
    """Single-turn request with the system prompt + any grounding blocks
    marked for caching. Returns the raw ``Message`` so callers can pull
    tool calls or text as needed.
    """
    client = client or get_client()

    system_blocks = [{
        "type": "text",
        "text": system,
        "cache_control": {"type": "ephemeral"},
    }]

    user_content: list[dict[str, Any]] = []
    user_content.extend(_to_cached_content(cached_context))
    user_content.append({"type": "text", "text": user})

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user_content}],
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as exc:
        # 400 Bad Request from Anthropic covers a few cases, but the most
        # common runtime surprise for a deployed app is a depleted workspace
        # credit balance. Re-raise with a tighter error type so the UI
        # layer can show a friendlier message.
        msg = str(getattr(exc, "message", exc)) or ""
        body = getattr(exc, "body", None)
        body_msg = ""
        if isinstance(body, dict):
            body_msg = (body.get("error", {}) or {}).get("message", "")
        combined = f"{msg} {body_msg}".lower()
        if "credit balance" in combined or "credits" in combined:
            raise AnthropicCreditError(
                body_msg or msg
                or "Anthropic API credit balance is too low. Top up credits "
                   "in the Anthropic console to re-enable AI features."
            ) from exc
        raise


def text_from(message: anthropic.types.Message) -> str:
    """Concatenate text blocks from the response (ignoring tool uses)."""
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)  # type: ignore[attr-defined]
    return "".join(parts)


def tool_use_input(message: anthropic.types.Message, tool_name: str) -> dict[str, Any] | None:
    """Return the ``input`` of the first tool_use block matching ``tool_name``
    or ``None`` if the model did not invoke it.
    """
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return dict(getattr(block, "input", {}))
    return None


__all__ = [
    "DEFAULT_MODEL",
    "CachedBlock",
    "AnthropicCreditError",
    "get_client",
    "call_with_cache",
    "text_from",
    "tool_use_input",
]
