from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_cross_sectional_dataset(
    snapshots: List[pd.DataFrame],
    feature_columns: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simple cross-sectional dataset builder.

    Assumes snapshots is a list of DataFrames already containing feature_columns
    and target_column. Concatenates and returns X, y.
    """
    if not snapshots:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = pd.concat(snapshots, axis=0, ignore_index=False)
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def build_meta_alpha_dataset(snapshots: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a dataset for meta alpha blending from past scan snapshots.
    Expects snapshots to include columns like factor_alpha, ml_alpha, AlphaScore,
    regime tags, tft_forecast_h*, and a target (e.g., future_return).
    """
    if not snapshots:
        return pd.DataFrame(), pd.Series(dtype=float)
    df = pd.concat(snapshots, axis=0, ignore_index=False)
    feature_cols = []
    for col in ["factor_alpha", "ml_alpha", "AlphaScore"]:
        if col in df.columns:
            feature_cols.append(col)
    feature_cols.extend([c for c in df.columns if c.startswith("tft_forecast_h")])
    # Regime one-hot
    if "RegimeTrend" in df.columns:
        dummies = pd.get_dummies(df["RegimeTrend"], prefix="regime_trend")
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())
    if "RegimeVol" in df.columns:
        dummies = pd.get_dummies(df["RegimeVol"], prefix="regime_vol")
        df = pd.concat([df, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())
    target_col = "future_return" if "future_return" in df.columns else None
    if target_col is None:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    return X, y
