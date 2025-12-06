"""Alpha source attribution using SHAP to quantify feature contributions."""

from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    import shap  # type: ignore

    HAVE_SHAP = True
except ImportError:  # pragma: no cover
    HAVE_SHAP = False


def attribute_signal_weights(model, input_features: pd.DataFrame) -> Dict[str, float]:
    """Estimate average feature contributions to alpha prediction."""
    if not HAVE_SHAP:
        raise ImportError("shap is required for alpha source attribution.")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_features)
    means = shap_values.values.mean(axis=0)
    return dict(zip(input_features.columns, means))


__all__ = ["attribute_signal_weights"]
