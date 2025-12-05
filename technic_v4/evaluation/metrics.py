"""
Evaluation metrics for alpha signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(preds: pd.Series, actual: pd.Series) -> float:
    aligned = preds.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    rho, _ = spearmanr(aligned[0], aligned[1])
    return float(rho)


def precision_at_n(preds: pd.Series, actual: pd.Series, n: int = 10) -> float:
    aligned = preds.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    topn = aligned[0].nlargest(n).index
    return float((aligned[1].loc[topn] > 0).mean())


def hit_rate(preds: pd.Series, actual: pd.Series) -> float:
    aligned = preds.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    return float((np.sign(aligned[0]) == np.sign(aligned[1])).mean())


def average_R(preds: pd.Series, actual: pd.Series) -> float:
    aligned = preds.align(actual, join="inner")
    if aligned[0].empty:
        return np.nan
    return float((aligned[0] * aligned[1]).mean())


def sharpe_ratio(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0 or pd.isna(sigma):
        return np.nan
    return float(mu / sigma * np.sqrt(252))
