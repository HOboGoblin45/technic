from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def compute_shap_importance(model, df_features: pd.DataFrame, symbol: Optional[str] = None):
    """
    Placeholder for SHAP computation. Returns {} until wired to real SHAP values.
    """
    return {}


def build_rationale(
    symbol: str,
    row: pd.Series,
    features: Optional[pd.Series] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a short 1–3 sentence rationale string explaining the signal.
    Uses simple templating on existing fields.
    """
    signal = str(row.get("Signal", "") or "").strip()
    tr = row.get("TechRating")
    alpha = row.get("AlphaScore")
    mom = None
    val = None
    if features is not None:
        mom = features.get("ret_21d") or features.get("mom_21")
        val = features.get("value_ep") or features.get("ep")

    regime_label = None
    if regime:
        regime_label = regime.get("label") or f"{regime.get('trend', '')} {regime.get('vol', '')}".strip()
    elif "MarketRegime" in row:
        regime_label = row.get("MarketRegime")

    parts = []
    if signal:
        parts.append(f"{signal} – {symbol}")
    else:
        parts.append(symbol)

    if tr is not None:
        parts.append(f"TechRating {tr:.1f}")
    if alpha is not None:
        parts.append(f"AlphaScore {alpha:.2f}")
    if mom is not None and pd.notna(mom):
        parts.append("positive momentum")
    if val is not None and pd.notna(val):
        parts.append("attractive valuation")
    if regime_label:
        parts.append(f"in a {regime_label} regime")

    # Join succinctly
    rationale = "; ".join(parts)
    return rationale


__all__ = ["build_rationale", "compute_shap_importance"]
