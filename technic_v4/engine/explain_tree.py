"""Alpha explanation tree / SHAP layer for signal attribution."""

from __future__ import annotations

try:
    import shap  # type: ignore
    import xgboost  # noqa: F401  # type: ignore

    HAVE_SHAP = True
except ImportError:  # pragma: no cover
    HAVE_SHAP = False


def explain_alpha(model, X, show: bool = True):
    """Generate SHAP beeswarm plot to explain model-driven alpha signals."""
    if not HAVE_SHAP:
        raise ImportError("shap/xgboost not installed; cannot explain alpha.")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap.plots.beeswarm(shap_values, show=show)


__all__ = ["explain_alpha"]
