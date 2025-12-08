import os
import sys
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGB alpha model.")
    p.add_argument(
        "--train-path",
        type=str,
        default="data/training_data.parquet",
        help="Path to training data parquet file.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="models/alpha/xgb_v1.pkl",
        help="Where to save the joblib bundle (model + features).",
    )
    p.add_argument(
        "--onnx-path",
        type=str,
        default="models/alpha/xgb_v1.onnx",
        help="Optional ONNX export path (best effort).",
    )
    p.add_argument(
        "--label",
        type=str,
        default="fwd_ret_5d",
        help="Label column to train on (e.g. fwd_ret_5d, fwd_ret_10d).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    train_path = args.train_path
    model_path = args.model_path
    onnx_path = args.onnx_path
    label_col = args.label

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    df = pd.read_parquet(train_path)
    if df.empty:
        raise ValueError(f"Training data at {train_path} has 0 rows; cannot train.")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data.")

    # Drop rows without labels
    df = df.dropna(subset=[label_col])

    # Use numeric columns as features, excluding IDs and label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Always exclude IDs + all forward-return labels
    EXCLUDE_COLS = {"symbol", "as_of_date", "fwd_ret_5d", "fwd_ret_10d", label_col}
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    X = df[feature_cols].fillna(0.0)
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    print(
        f"Training XGB model on {len(feature_cols)} features and {len(X_train)} rows "
        f"for label '{label_col}'..."
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation R^2: {r2:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {"model": model, "features": feature_cols}
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to {model_path}")

    # Optional ONNX export (best-effort)
    try:
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore

        initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"Exported ONNX model to {onnx_path}")
    except Exception as exc:
        print(f"ONNX export skipped or failed: {exc}")


if __name__ == "__main__":
    main()
