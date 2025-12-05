"""
SHAP-based explainability utilities for scan results and Copilot text.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def explain_top_symbols(model, df_features: pd.DataFrame, symbols: List[str], top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute SHAP values for the top N symbols using a tree explainer.
    Returns {symbol: [(feature, shap_value), ...]}.
    """
    if shap is None or model is None or df_features.empty:
        return {}
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        return {}

    out: Dict[str, List[Tuple[str, float]]] = {}
    for sym in symbols[:top_n]:
        if sym not in df_features.index:
            continue
        row = df_features.loc[[sym]]
        try:
            vals = explainer.shap_values(row)
        except Exception:
            continue
        if isinstance(vals, list):
            vals = vals[0]
        if vals is None:
            continue
        shap_row = vals[0] if hasattr(vals, "__len__") else []
        pairs = list(zip(df_features.columns, shap_row))
        # sort by absolute contribution
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        out[sym] = pairs
    return out


def format_explanation(shap_output: List[Tuple[str, float]], max_items: int = 3) -> str:
    """
    Convert SHAP tuples into human-readable text.
    Example: "Momentum (Ret_21) +0.23, Low ATR -0.12, Earnings yield +0.10"
    """
    if not shap_output:
        return ""
    formatted = []
    for feat, val in shap_output[:max_items]:
        sign = "+" if val >= 0 else "-"
        formatted.append(f"{feat} {sign}{abs(val):.2f}")
    return ", ".join(formatted)
