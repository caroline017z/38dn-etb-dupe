"""Reusable UI primitives for the 38DN finance-grade design system.

These are intentionally thin wrappers over ``st.markdown`` + HTML so the
theme CSS in ``assets/theme.css`` can own all visual detail. Every helper
renders to the current Streamlit container.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

from .tokens import PALETTE


# ─────────────────────────────────────────────────────────────────────
# Inline sparkline — SVG, embeds directly in HTML tables
# ─────────────────────────────────────────────────────────────────────
def sparkline_svg(
    values: list[float],
    *,
    width: int = 80,
    height: int = 20,
    color: str | None = None,
    fill: bool = True,
) -> str:
    """Return an inline SVG sparkline for ``values``.

    Designed to embed inside HTML table cells (via ``render_styled_table``)
    so finance analysts get a trend line adjacent to the numeric figure.
    Values are auto-scaled to the SVG viewport; ``None`` / non-finite
    values render as gaps.

    Args:
        values: sequence of y-values (x is implicitly the index).
        width / height: SVG pixel dimensions.
        color: stroke + fill color; defaults to 38DN green for positive
            trends, red for negative.
        fill: when True, the area under the line gets a subtle fill
            matching the stroke color at 12% opacity.
    """
    if not values:
        return f'<svg width="{width}" height="{height}"></svg>'

    clean = [float(v) if (v is not None and v == v) else None for v in values]
    finite = [v for v in clean if v is not None]
    if not finite:
        return f'<svg width="{width}" height="{height}"></svg>'

    lo, hi = min(finite), max(finite)
    span = hi - lo or 1.0
    n = len(clean)

    # Auto-color: green if net positive slope, red if negative, navy if flat.
    if color is None:
        if finite[0] < finite[-1]:
            color = PALETTE["green"]
        elif finite[0] > finite[-1]:
            color = PALETTE["red"]
        else:
            color = PALETTE["navy"]

    pad = 1.5
    def _x(i: int) -> float:
        return pad + i * (width - 2 * pad) / max(n - 1, 1)
    def _y(v: float) -> float:
        return height - pad - (v - lo) * (height - 2 * pad) / span

    # Build the polyline — skip over gaps by ending the current segment
    # and starting a new one when a None is encountered.
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for i, v in enumerate(clean):
        if v is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append((_x(i), _y(v)))
    if current:
        segments.append(current)

    polylines = "".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in seg)}" />'
        for seg in segments if len(seg) >= 2
    )

    # Fill path — one continuous area under all segments joined to baseline.
    fill_path = ""
    if fill and segments:
        fill_segments = []
        for seg in segments:
            if len(seg) < 2:
                continue
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in seg)
            first_x, _ = seg[0]
            last_x, _ = seg[-1]
            fill_segments.append(
                f'<polygon fill="{color}" fill-opacity="0.12" '
                f'stroke="none" points="{first_x:.2f},{height - pad:.2f} '
                f'{pts} {last_x:.2f},{height - pad:.2f}" />'
            )
        fill_path = "".join(fill_segments)

    return (
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:inline-block;vertical-align:middle;">'
        f'{fill_path}{polylines}</svg>'
    )



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
