from __future__ import annotations

"""
Offline training script for cross-sectional LGBM alpha model.
Builds a dataset from cached price history and trains a LightGBM regressor
to predict forward returns. Saves model to models/alpha/lgbm_v1.pkl.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.engine.feature_engine import get_latest_features
from technic_v4.data_layer.price_layer import get_stock_history_df
from technic_v4.universe_loader import load_universe

MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")


def compute_forward_return(history_df: pd.DataFrame, horizon: int = 5) -> float:
    """
    Compute forward percent return over horizon days using the last close.
    """
    if history_df is None or history_df.empty:
        return np.nan
    closes = history_df["Close"]
    if len(closes) < horizon + 1:
        return np.nan
    last = closes.iloc[-1]
    fwd_idx = min(len(closes) - 1 + horizon, len(closes) - 1)
    future = closes.shift(-horizon).iloc[-1]
    if last == 0 or pd.isna(future):
        return np.nan
    return (future / last) - 1.0


def build_training_rows(date: pd.Timestamp, symbols: List[str], lookback_days: int = 120, horizon: int = 5) -> List[dict]:
    """
    For a given date and symbols, build training rows with features and target.
    """
    rows = []
    for sym in symbols:
        try:
            hist = get_stock_history_df(symbol=sym, days=lookback_days, use_intraday=False, end_date=date)
        except Exception:
            continue
        if hist is None or hist.empty:
            continue
        feats = get_latest_features(hist, fundamentals=None)
        if feats is None or feats.empty:
            continue
        target = compute_forward_return(hist, horizon=horizon)
        if pd.isna(target):
            continue
        row = feats.to_dict()
        row["symbol"] = sym
        row["asof"] = date
        row["target"] = target
        rows.append(row)
    return rows


def build_training_dataset(start_date: str, end_date: str, horizon: int = 5, lookback_days: int = 120) -> pd.DataFrame:
    """
    Build a cross-sectional dataset over a date range.
    """
    universe = load_universe()
    symbols = [u.symbol for u in universe]
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    all_rows: List[dict] = []
    for dt in dates:
        rows = build_training_rows(dt, symbols, lookback_days=lookback_days, horizon=horizon)
        if rows:
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def train_and_save(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    model_path: Path = MODEL_PATH,
) -> None:
    if df.empty:
        raise ValueError("No training data available")
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    # Time-based split: last 20% for validation
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    model = LGBMAlphaModel()
    model.fit(X_train, y_train)
    # Optionally evaluate on val (not stored here)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"[TRAIN] Saved model to {model_path}")


def main():
    # Configs
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    horizon = 5
    lookback_days = 120

    print("[TRAIN] Building dataset...")
    df = build_training_dataset(start_date, end_date, horizon=horizon, lookback_days=lookback_days)
    if df.empty:
        raise SystemExit("No training data built; aborting.")
    feature_cols = [c for c in df.columns if c not in {"symbol", "asof", "target"}]
    print(f"[TRAIN] Dataset size: {len(df)} rows, {len(feature_cols)} features")
    train_and_save(df, feature_cols, target_col="target", model_path=MODEL_PATH)


if __name__ == "__main__":
    main()
