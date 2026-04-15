"""Theme installer: inject Google Fonts + ``assets/theme.css`` into the page.

Call :func:`install_theme` once near the top of ``app.py``, ideally right
after ``st.set_page_config``. The function is idempotent and caches the
CSS text so successive Streamlit reruns don't re-read the file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st


_THEME_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "theme.css"

# Web-font preconnect + import. Inter for UI, JetBrains Mono for tabular
# numerals, Source Serif 4 as an optional accent (used sparingly).
_FONTS_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?
  family=Inter:wght@400;500;600;700&
  family=JetBrains+Mono:wght@400;500;600&
  family=Source+Serif+4:wght@400;600&display=swap"
  rel="stylesheet">
""".replace("\n  ", "")


@lru_cache(maxsize=1)
def _theme_css_text() -> str:
    try:
        return _THEME_CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def install_theme() -> None:
    """Inject fonts + theme CSS. Safe to call every rerun."""
    css = _theme_css_text()
    st.markdown(_FONTS_HTML, unsafe_allow_html=True)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def set_dense_mode(enabled: bool) -> None:
    """Toggle the ``data-dense`` attribute on the Streamlit app root.

    The theme stylesheet responds to ``.stApp[data-dense='true']`` by
    tightening padding, metric value sizes, table row height, and tab
    spacing. Works by injecting a tiny inline script that walks up to
    the root and flips the attribute on every rerun.
    """
    flag = "true" if enabled else "false"
    st.markdown(
        f"""<script>
        (function() {{
            const doc = window.parent && window.parent.document
                ? window.parent.document : document;
            const root = doc.querySelector('.stApp')
                      || doc.querySelector('[data-testid=\"stAppViewContainer\"]');
            if (root) {{
                root.setAttribute('data-dense', '{flag}');
            }}
        }})();
        </script>""",
        unsafe_allow_html=True,
    )
