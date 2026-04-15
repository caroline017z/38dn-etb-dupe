"""38DN UI primitives — institutional finance-grade design system.

Public surface:
- :func:`install_theme` — inject `assets/theme.css` + Google Fonts into the
  current Streamlit page. Call once at top of ``app.py``.
- :mod:`modules.ui.tokens` — Python mirrors of the CSS custom properties
  (colors, typography, spacing) so charts and downstream helpers don't
  duplicate hex literals.
- :mod:`modules.ui.components` — cards, metrics, section headers,
  segmented controls, pills, and the underlying ``render_styled_table``
  replacement.
"""

from .theme import install_theme, set_dense_mode
from .tokens import PALETTE, TYPO, SP, RADIUS, PLOTLY_LAYOUT
from .components import (
    section_header,
    metric_card,
    metric_row,
    card,
    pill,
    segmented_control,
    empty_state,
    sparkline_svg,
)

__all__ = [
    "install_theme",
    "set_dense_mode",
    "PALETTE",
    "TYPO",
    "SP",
    "RADIUS",
    "PLOTLY_LAYOUT",
    "section_header",
    "metric_card",
    "metric_row",
    "card",
    "pill",
    "segmented_control",
    "empty_state",
    "sparkline_svg",
]
