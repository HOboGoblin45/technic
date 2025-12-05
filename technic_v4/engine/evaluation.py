"""
Evaluation metrics for alpha models and rankings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(pred: pd.Series, actual: pd.Series) -> float:
    """
    Spearman rank correlation between predicted scores and realized returns.
    """
    aligned = pred.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    rho, _ = spearmanr(aligned[0], aligned[1])
    return float(rho)


def precision_at_k(pred: pd.Series, actual: pd.Series, k: int = 10) -> float:
    """
    Precision@k using direction (actual > 0).
    """
    aligned = pred.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    topk = aligned[0].nlargest(k).index
    hits = (aligned[1].loc[topk] > 0).mean()
    return float(hits)


def top_bottom_spread(pred: pd.Series, actual: pd.Series, decile: float = 0.1) -> float:
    """
    Average return of top decile minus bottom decile.
    """
    aligned = pred.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    n = max(1, int(len(aligned[0]) * decile))
    top = aligned[1].loc[aligned[0].nlargest(n).index].mean()
    bot = aligned[1].loc[aligned[0].nsmallest(n).index].mean()
    return float(top - bot)
