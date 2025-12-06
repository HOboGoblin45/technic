"""Live UI metrics overlay (placeholder for Sharpe, hit rate, volatility)."""

from __future__ import annotations

try:
    import streamlit as st

    HAVE_ST = True
except ImportError:  # pragma: no cover
    HAVE_ST = False


def render_metrics_overlay(sharpe: float, hit_rate: float, vol: float):
    if not HAVE_ST:
        return
    with st.sidebar:
        st.markdown("### Live Metrics")
        st.metric("Sharpe", f"{sharpe:.2f}")
        st.metric("Hit rate", f"{hit_rate:.2%}")
        st.metric("Volatility", f"{vol:.2f}")


__all__ = ["render_metrics_overlay"]
