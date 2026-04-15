"""Python-side mirrors of the CSS custom properties in ``assets/theme.css``.

Any chart / Python-side component referencing a brand color should import
from here rather than inlining a hex literal, so a single edit to
``theme.css`` + this file re-skins the app.
"""

from __future__ import annotations

from typing import Final


# ─── Brand + accents ──────────────────────────────────────────────
PALETTE: Final[dict[str, str]] = {
    "navy":      "#0E2841",
    "green":     "#45A750",
    "blue":      "#1D6FA9",
    "teal":      "#518484",
    "amber":     "#D48A1A",
    "red":       "#A8141A",

    # neutrals
    "ink":       "#0E2841",
    "slate_70":  "#475569",
    "slate_50":  "#64748B",
    "slate_30":  "#94A3B8",

    # surfaces
    "surface_0": "#FFFFFF",
    "surface_1": "#F8FAFC",
    "surface_2": "#F1F5F9",
    "surface_3": "#E2E8F0",

    "border_1":  "#E5E7EB",
    "border_2":  "#CBD5E1",

    # state tints
    "info_bg":    "#E8F1FA",
    "success_bg": "#EDF7EE",
    "warning_bg": "#FDF3E2",
    "danger_bg":  "#FDEDED",
}

# Ordered categorical palette for PPA overlays, lever ranks, etc.
# Reserves navy for primary/utility baseline lines.
CATEGORICAL_ORDER: Final[tuple[str, ...]] = (
    PALETTE["green"],
    PALETTE["blue"],
    PALETTE["teal"],
    PALETTE["amber"],
    "#8E44AD",
    "#C0392B",
    "#117864",
)


# ─── Typography ────────────────────────────────────────────────────
TYPO: Final[dict[str, str | int]] = {
    "font_sans": (
        '"Inter", "Aptos Narrow", "Segoe UI", system-ui, '
        '-apple-system, BlinkMacSystemFont, sans-serif'
    ),
    "font_mono": (
        '"JetBrains Mono", "SF Mono", Menlo, Consolas, '
        '"Liberation Mono", monospace'
    ),
    "fs_h1":       22,
    "fs_h2":       16,
    "fs_h3":       14,
    "fs_body":     13,
    "fs_caption":  11,
    "fw_regular":  400,
    "fw_medium":   500,
    "fw_semibold": 600,
    "fw_bold":     700,
}


# ─── Spacing (4px grid) ─────────────────────────────────────────────
SP: Final[dict[str, int]] = {
    "sp_1": 4,  "sp_2": 8,  "sp_3": 12, "sp_4": 16,
    "sp_5": 24, "sp_6": 32, "sp_7": 48,
}


# ─── Radius ─────────────────────────────────────────────────────────
RADIUS: Final[dict[str, int]] = {
    "sm": 3,   # tags
    "md": 6,   # cards, buttons, inputs
    "pill": 999,
}


# ─── Reusable Plotly layout ─────────────────────────────────────────
# Apply via ``fig.update_layout(**PLOTLY_LAYOUT)`` then override title,
# axis_title, height as needed. Keeps every chart visually aligned
# without re-declaring the font stack each time.
PLOTLY_LAYOUT: Final[dict] = {
    "template": "plotly_white",
    "font": dict(
        family=TYPO["font_sans"],
        size=TYPO["fs_body"],
        color=PALETTE["ink"],
    ),
    "title_font": dict(
        size=TYPO["fs_h2"],
        color=PALETTE["navy"],
    ),
    "paper_bgcolor": PALETTE["surface_0"],
    "plot_bgcolor":  PALETTE["surface_0"],
    "margin": dict(l=60, r=30, t=60, b=50),
    "legend": dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1,
        font=dict(color=PALETTE["ink"], size=TYPO["fs_caption"]),
        bgcolor="rgba(0,0,0,0)",
    ),
    "xaxis": dict(
        gridcolor=PALETTE["border_1"],
        linecolor=PALETTE["border_2"],
        tickfont=dict(color=PALETTE["slate_70"], size=TYPO["fs_caption"]),
        title_font=dict(color=PALETTE["slate_70"], size=TYPO["fs_caption"]),
    ),
    "yaxis": dict(
        gridcolor=PALETTE["border_1"],
        linecolor=PALETTE["border_2"],
        tickfont=dict(color=PALETTE["slate_70"], size=TYPO["fs_caption"]),
        title_font=dict(color=PALETTE["slate_70"], size=TYPO["fs_caption"]),
        zeroline=True,
        zerolinecolor=PALETTE["border_2"],
    ),
    "hoverlabel": dict(
        bgcolor=PALETTE["surface_0"],
        bordercolor=PALETTE["border_2"],
        font=dict(family=TYPO["font_sans"], size=TYPO["fs_caption"],
                  color=PALETTE["ink"]),
    ),
}
