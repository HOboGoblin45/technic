"""Meta-metric evaluation dashboard for Streamlit."""

from __future__ import annotations

try:
    import streamlit as st

    HAVE_ST = True
except ImportError:  # pragma: no cover
    HAVE_ST = False


def show_metrics_panel(metrics) -> None:
    """Render predictive MSE, realized alpha, and false positive rate."""
    if not HAVE_ST:
        raise ImportError("streamlit is required to show the meta-metrics panel.")
    st.title("Model Meta-Metrics")
    for k, v in metrics.items():
        st.metric(k, f"{v:.3f}")


__all__ = ["show_metrics_panel"]
