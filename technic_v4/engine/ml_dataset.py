"""
Dataset builder for multi-horizon, regime-aware alpha modeling.

This keeps the construction lightweight and GPU-friendly; heavy lifting
is deferred to model wrappers. All functions are pure and safe to call
inside a scan/backtest loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """
    Container for feature/label matrices.
    """

    X: pd.DataFrame
    y: pd.Series
    meta: Dict[str, pd.Series]


def build_features(
    price_panel: pd.DataFrame,
    factors: pd.DataFrame,
    regime_tags: Optional[pd.Series] = None,
    horizons: Iterable[int] = (5,),
) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    """
    Build cross-sectional feature matrix and forward-return labels.

    Parameters
    ----------
    price_panel : DataFrame indexed by [date, symbol] with 'Close'.
    factors     : DataFrame indexed identically with engineered factors.
    regime_tags : optional Series indexed by date with regime labels.
    horizons    : iterable of forward horizons (days) to compute labels for.
    """
    if price_panel.empty or factors.empty:
        return pd.DataFrame(), {}

    closes = price_panel["Close"].unstack()
    returns = closes.pct_change()

    labels: Dict[int, pd.Series] = {}
    for h in horizons:
        fwd = closes.shift(-h) / closes - 1.0
        labels[h] = fwd.stack().rename(f"fwd_{h}d")

    feat = factors.copy()
    if regime_tags is not None:
        regime_df = regime_tags.reindex(factors.index.get_level_values(0)).astype(str)
        feat = feat.assign(regime=regime_df.values)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    labels = {h: lbl.loc[feat.index].dropna() for h, lbl in labels.items()}
    return feat, labels


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple chronological split; caller ensures X/y share index.
    """
    idx = np.arange(len(X))
    cut = int(len(idx) * (1 - val_ratio))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return (
        X.iloc[train_idx].values,
        X.iloc[val_idx].values,
        y.iloc[train_idx].values,
        y.iloc[val_idx].values,
    )
