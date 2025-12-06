"""Explainable NLP-based summary generator for daily picks.

Uses a small Jinja2 template when available; falls back to a simple string
builder to avoid hard dependency issues.
"""

from __future__ import annotations

from typing import Iterable, Optional

try:
    from jinja2 import Template  # type: ignore

    HAVE_JINJA = True
except ImportError:  # pragma: no cover
    HAVE_JINJA = False


DEFAULT_TEMPLATE = """
Today's picks are:
{% for stock, reason in pairs %}
- {{ stock }}: {{ reason }}
{% endfor %}
""".strip()


def generate_summary(
    top_stocks: Iterable[str],
    reasons: Iterable[str],
    template_str: Optional[str] = None,
) -> str:
    """Render a short natural-language summary of top picks and drivers.

    Args:
        top_stocks: Iterable of stock tickers in order of importance.
        reasons: Iterable of short strings explaining each pick.
        template_str: Optional custom template; defaults to a simple list format.

    Returns:
        Rendered summary string.
    """
    pairs = list(zip(top_stocks, reasons))
    tmpl = template_str or DEFAULT_TEMPLATE

    if HAVE_JINJA:
        return Template(tmpl).render(pairs=pairs)

    # Fallback: simple formatting without Jinja2.
    lines = ["Today's picks are:"]
    for stock, reason in pairs:
        lines.append(f"- {stock}: {reason}")
    return "\n".join(lines)


__all__ = ["generate_summary"]
