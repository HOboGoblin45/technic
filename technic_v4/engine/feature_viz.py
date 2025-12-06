"""Feature importance visualization via SHAP bar plot.

This is an optional utility; it requires `shap` and `matplotlib`.
"""

from __future__ import annotations

from typing import Optional

try:
    import shap  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_SHAP = True
except ImportError:  # pragma: no cover
    HAVE_SHAP = False


def visualize_feature_importance(model, X, show: bool = True):
    """Render a SHAP bar plot for feature importance.

    Args:
        model: Trained model compatible with SHAP Explainer.
        X: Feature matrix (DataFrame or ndarray).
        show: If True, call plt.show(); if False, return the figure/axes.

    Returns:
        The SHAP plot object or None if dependencies are missing.
    """
    if not HAVE_SHAP:
        raise ImportError("shap/matplotlib not installed; cannot visualize feature importance.")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    fig = shap.plots.bar(shap_values, show=False)
    if show:
        plt.show()
    return fig


__all__ = ["visualize_feature_importance"]
