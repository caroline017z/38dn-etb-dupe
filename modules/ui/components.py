"""Reusable UI primitives for the 38DN finance-grade design system.

These are intentionally thin wrappers over ``st.markdown`` + HTML so the
theme CSS in ``assets/theme.css`` can own all visual detail. Every helper
renders to the current Streamlit container.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

from .tokens import PALETTE


_PILL_VARIANTS = {"primary", "green", "blue", "amber", "muted"}


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ─────────────────────────────────────────────────────────────────────
# Section header — a horizontal rule with title + optional caption
# ─────────────────────────────────────────────────────────────────────
def section_header(title: str, *, caption: str | None = None,
                   eyebrow: str | None = None) -> None:
    """Render a hairline-underlined section header (H2) with optional
    caption on the right.

    The eyebrow renders above the title as a short uppercase label — use
    it for the "Section 03" / "Scenario" kind of orientation cue common
    in finance decks.
    """
    bits: list[str] = []
    if eyebrow:
        bits.append(
            f'<div class="eyebrow-38dn" style="margin-bottom:2px;">{_esc(eyebrow)}</div>'
        )
    bits.append(f'<h3 style="margin:0;">{_esc(title)}</h3>')
    right = (
        f'<div class="caption-38dn" style="text-align:right;">{_esc(caption)}</div>'
        if caption else ""
    )
    st.markdown(
        f"""<div class="section-38dn">
            <div>{"".join(bits)}</div>
            {right}
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Card primitives
# ─────────────────────────────────────────────────────────────────────
def card(body_html: str, *, title: str | None = None, eyebrow: str | None = None,
         accent: bool = False) -> None:
    """Render an institutional card with optional eyebrow + title.

    ``body_html`` is raw HTML — callers that want to embed Streamlit
    widgets inside a card should use ``with st.container(border=True)``
    instead. This helper is the right choice for statically-formatted
    content (metric grids, notes, empty states).
    """
    eyebrow_html = (
        f'<div class="card-38dn__eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    )
    title_html = (
        f'<div class="card-38dn__title">{_esc(title)}</div>' if title else ""
    )
    klass = "card-38dn card-accent-left" if accent else "card-38dn"
    st.markdown(
        f'<div class="{klass}">{eyebrow_html}{title_html}{body_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Metric cards + rows
# ─────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, *, delta: str | None = None,
                delta_tone: Literal["positive", "negative", "neutral"] = "neutral",
                caption: str | None = None) -> None:
    """Single metric card — replaces ``st.metric`` when we want control
    over the delta tone and caption, matching the institutional CSS."""
    tone_color = {
        "positive": PALETTE["green"],
        "negative": PALETTE["red"],
        "neutral":  PALETTE["slate_50"],
    }[delta_tone]
    delta_html = (
        f'<div style="font-size:11px;color:{tone_color};'
        f'font-weight:500;margin-top:2px;">{_esc(delta)}</div>'
        if delta else ""
    )
    caption_html = (
        f'<div class="caption-38dn" style="margin-top:6px;">{_esc(caption)}</div>'
        if caption else ""
    )
    st.markdown(
        f"""<div class="card-38dn" style="padding:12px 16px;">
            <div class="card-38dn__eyebrow">{_esc(label)}</div>
            <div class="card-38dn__value">{_esc(value)}</div>
            {delta_html}
            {caption_html}
        </div>""",
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict]) -> None:
    """Render a grid of metric cards in a single equal-width row.

    Each dict may contain: ``label``, ``value``, ``delta``, ``delta_tone``,
    ``caption``.
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_tone=m.get("delta_tone", "neutral"),
                caption=m.get("caption"),
            )


# ─────────────────────────────────────────────────────────────────────
# Pills
# ─────────────────────────────────────────────────────────────────────
def pill(text: str, *, variant: Literal["primary", "green", "blue", "amber", "muted"] = "primary") -> str:
    """Return HTML for a status pill. Callers embed it inside their own
    markdown so the pill doesn't force a block-level line break."""
    v = variant if variant in _PILL_VARIANTS else "primary"
    klass = "pill-38dn" if v == "primary" else f"pill-38dn pill-{v}"
    return f'<span class="{klass}">{_esc(text)}</span>'


# ─────────────────────────────────────────────────────────────────────
# Segmented control
# ─────────────────────────────────────────────────────────────────────
def segmented_control(label: str, options: list[str], *, key: str,
                      default: int = 0,
                      label_hidden: bool = True) -> str:
    """Horizontal segmented control backed by ``st.radio``. The CSS theme
    styles the underlying radio buttons to look like an institutional
    segmented control (pill group) — use this instead of bare radios for
    primary view switches.
    """
    vis: Literal["visible", "hidden", "collapsed"] = (
        "collapsed" if label_hidden else "visible"
    )
    return st.radio(
        label=label,
        options=options,
        horizontal=True,
        index=default,
        key=key,
        label_visibility=vis,
    )


# ─────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────
def empty_state(title: str, body: str, *, icon: str = "") -> None:
    """Centered empty-state card used when a tab has nothing to show yet."""
    st.markdown(
        f"""<div class="card-38dn" style="
            margin: 32px auto; max-width: 520px; text-align: center; padding: 28px 32px;">
            <div style="font-size:28px; margin-bottom:8px;">{_esc(icon)}</div>
            <div class="card-38dn__title" style="margin-bottom:6px;">{_esc(title)}</div>
            <div class="caption-38dn" style="font-size:13px; color: var(--38dn-slate-70);">
                {_esc(body)}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
