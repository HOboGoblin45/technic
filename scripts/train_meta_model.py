"""
Train a meta-model ("super-ICS") on top of existing scores to predict win probability or forward return.

Expected input parquet must include at least some of these feature columns:
    TechRating, InstitutionalCoreScore, AlphaScore (ML alpha), MuTotal, alpha_blend (if available)
Labels:
    - win_prob: sign of fwd_ret_10d > 0
    - return regression: fwd_ret_10d

Outputs:
    models/alpha/meta_super_ics.pkl (joblib bundle with classifier + calibrator + feature list)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FEATURES = [
    "TechRating",
    "InstitutionalCoreScore",
    "AlphaScore",
    "alpha_blend",
    "MuTotal",
    "factor_alpha",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train meta-model on top of existing scores.")
    p.add_argument("--train-path", type=str, required=True, help="Parquet with scan-derived features.")
    p.add_argument("--model-path", type=str, default="models/alpha/meta_super_ics.pkl", help="Output path for bundle.")
    p.add_argument("--date-col", type=str, default="scan_date", help="Date column for time-based splits.")
    p.add_argument("--label-ret", type=str, default="fwd_ret_10d", help="Forward return column.")
    p.add_argument("--features", type=str, nargs="*", default=None, help="Override feature list.")
    p.add_argument("--train-end", type=str, default="2021-12-31", help="Train end date (inclusive).")
    p.add_argument("--val-end", type=str, default="2022-12-31", help="Validation end date (inclusive).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.train_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        raise SystemExit("Training data empty.")

    date_col = args.date_col
    if date_col not in df.columns:
        # fallback to as_of_date if present
        if "as_of_date" in df.columns:
            date_col = "as_of_date"
        else:
            raise ValueError(f"Date column '{args.date_col}' not found.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    feature_cols: List[str] = args.features if args.features else DEFAULT_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    label_ret = args.label_ret
    if label_ret not in df.columns:
        raise ValueError(f"Label column '{label_ret}' not found.")

    # Build binary label for win probability
    df["label_win"] = pd.to_numeric(df[label_ret], errors="coerce") > 0
    df = df.dropna(subset=feature_cols + [label_ret])

    X = df[feature_cols].fillna(0.0)
    y_win = df["label_win"].astype(int)

    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    mask_train = df[date_col] <= train_end
    mask_val = (df[date_col] > train_end) & (df[date_col] <= val_end)
    mask_test = df[date_col] > val_end

    X_train, y_train = X[mask_train], y_win[mask_train]
    X_val, y_val = X[mask_val], y_win[mask_val]
    X_test, y_test = X[mask_test], y_win[mask_test]

    if X_train.empty:
        raise SystemExit("Train split empty.")

    base_clf = LogisticRegression(max_iter=1000)
    clf = CalibratedClassifierCV(base_estimator=base_clf, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    def eval_split(name: str, Xs, ys):
        if Xs is None or Xs.empty:
            return {}
        probs = clf.predict_proba(Xs)[:, 1]
        brier = brier_score_loss(ys, probs)
        print(f"{name} Brier={brier:.4f}")
        return {"brier": float(brier)}

    metrics = {}
    metrics["train"] = eval_split("train", X_train, y_train)
    metrics["val"] = eval_split("val", X_val, y_val)
    metrics["test"] = eval_split("test", X_test, y_test)

    os.makedirs(Path(args.model_path).parent, exist_ok=True)
    bundle = {"model": clf, "features": feature_cols, "metrics": metrics}
    joblib.dump(bundle, args.model_path)
    print(f"Saved meta-model to {args.model_path}")


if __name__ == "__main__":
    main()
