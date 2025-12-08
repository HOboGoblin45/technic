import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

TRAIN_PATH = "data/training_data.parquet"
MODEL_PATH = "models/alpha/xgb_v1.pkl"

EXCLUDE_COLS = ["symbol", "as_of_date", "fwd_ret_5d"]


def main():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}")

    df = pd.read_parquet(TRAIN_PATH)
    if df.empty:
        raise ValueError(f"Training data at {TRAIN_PATH} has 0 rows; cannot train.")

    # Drop rows without labels
    df = df.dropna(subset=["fwd_ret_5d"])

    # Use numeric columns as features, excluding IDs and target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    X = df[feature_cols].fillna(0.0)
    y = df["fwd_ret_5d"]

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

    print(f"Training XGB model on {len(feature_cols)} features and {len(X_train)} rows...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation R^2: {r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {"model": model, "features": feature_cols}
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
