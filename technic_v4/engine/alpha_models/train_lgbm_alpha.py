from __future__ import annotations

"""
Offline training script for cross-sectional LGBM alpha model.
Builds a dataset from cached price history and trains a LightGBM regressor
to predict forward returns. Saves model artifacts and registers them.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from technic_v4 import model_registry
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.evaluation import metrics as eval_metrics
from technic_v4.engine.feature_engine import get_latest_features
from technic_v4.data_layer.price_layer import get_stock_history_df
from technic_v4.universe_loader import load_universe

MODEL_LATEST = Path("models/alpha/lgbm_v1.pkl")
FEATURES_PATH = Path("models/alpha/lgbm_v1_features.json")


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
    version: str | None = None,
    save_latest: bool = True,
) -> Tuple[Path, dict]:
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
    # Validation metrics
    metrics: dict = {}
    if not X_val.empty:
        preds_val = model.predict(X_val)
        preds_series = pd.Series(preds_val, index=y_val.index)
        metrics["val_ic"] = float(eval_metrics.rank_ic(preds_series, y_val))
        metrics["val_precision_at_10"] = float(
            eval_metrics.precision_at_n(preds_series, y_val, n=min(10, len(preds_series)))
        )
        metrics["val_avgR"] = float(eval_metrics.average_R(preds_series, y_val))
    version = version or datetime.utcnow().strftime("%Y%m%d")
    model_dir = Path("models/alpha")
    model_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = model_dir / f"lgbm_v1_{version}.pkl"
    model.save(str(versioned_path))
    if save_latest:
        model.save(str(MODEL_LATEST))
    try:
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURES_PATH.write_text(
            pd.Series(feature_cols).to_json(orient="values"), encoding="utf-8"
        )
    except Exception:
        pass
    model_registry.register_model(
        model_name="alpha_lgbm_v1",
        version=version,
        metrics=metrics,
        path_pickle=str(versioned_path),
        path_onnx=None,
        feature_names=feature_cols,
        is_active=False,  # promotion handled after gating
    )
    print(f"[TRAIN] Saved model to {versioned_path} (latest copy: {MODEL_LATEST})")
    return versioned_path, metrics


def main():
    # Rolling window: last 300 business days
    end_date = pd.Timestamp.utcnow().normalize()
    start_date = (end_date - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    horizon = 5
    lookback_days = 120

    print("[TRAIN] Building dataset...")
    df = build_training_dataset(start_date, end_date_str, horizon=horizon, lookback_days=lookback_days)
    if df.empty:
        raise SystemExit("No training data built; aborting.")
    feature_cols = [c for c in df.columns if c not in {"symbol", "asof", "target"}]
    print(f"[TRAIN] Dataset size: {len(df)} rows, {len(feature_cols)} features")
    version = datetime.utcnow().strftime("%Y%m%d")
    versioned_path, metrics = train_and_save(df, feature_cols, target_col="target", version=version)

    # Gating vs active baseline
    baseline = model_registry.get_active_model("alpha_lgbm_v1")
    promote = False
    if baseline is None:
        print("[TRAIN] No baseline model; promoting first version.")
        promote = True
    else:
        base_metrics = baseline.get("metrics", {}) if baseline else {}
        base_ic = base_metrics.get("val_ic") or base_metrics.get("ic") or -1e9
        base_prec = base_metrics.get("val_precision_at_10") or base_metrics.get("precision_at_10") or -1e9
        cand_ic = metrics.get("val_ic") or -1e9
        cand_prec = metrics.get("val_precision_at_10") or -1e9
        if cand_ic >= base_ic + 0.01 or cand_prec >= base_prec + 0.05:
            promote = True
            print(f"[TRAIN] Candidate beats baseline (IC {cand_ic:.3f} vs {base_ic:.3f}, P@10 {cand_prec:.3f} vs {base_prec:.3f}). Promoting.")
        else:
            print(f"[TRAIN] Candidate did not beat baseline (IC {cand_ic:.3f} vs {base_ic:.3f}, P@10 {cand_prec:.3f} vs {base_prec:.3f}). Keeping baseline active.")
    # Register (already registered) and set active if promoted
    if promote:
        model_registry.set_active_model("alpha_lgbm_v1", version=version)
    return {"version": version, "metrics": metrics, "promoted": promote}


if __name__ == "__main__":
    main()
