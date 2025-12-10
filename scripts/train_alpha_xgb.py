import os
import sys
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technic_v4.evaluation import metrics as eval_metrics


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
        default="models/alpha/xgb_v2.pkl",
        help="Where to save the joblib bundle (model + features).",
    )
    p.add_argument(
        "--onnx-path",
        type=str,
        default="models/alpha/xgb_v2.onnx",
        help="Optional ONNX export path (best effort).",
    )
    p.add_argument(
        "--label",
        type=str,
        default="fwd_ret_5d",
        help="Label column to train on (e.g. fwd_ret_5d, fwd_ret_10d).",
    )
    p.add_argument(
        "--date-col",
        type=str,
        default="as_of_date",
        help="Column containing the as-of date for time-based splits.",
    )
    p.add_argument(
        "--train-end",
        type=str,
        default="2018-12-31",
        help="End date (inclusive) for the training split.",
    )
    p.add_argument(
        "--val-end",
        type=str,
        default="2021-12-31",
        help="End date (inclusive) for the validation split; test is after this.",
    )
    p.add_argument(
        "--features-out",
        type=str,
        default="models/alpha/xgb_v2_features.json",
        help="Optional path to write the feature list used for training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    train_path = args.train_path
    model_path = args.model_path
    onnx_path = args.onnx_path
    label_col = args.label
    date_col = args.date_col

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    df = pd.read_parquet(train_path)
    if df.empty:
        raise ValueError(f"Training data at {train_path} has 0 rows; cannot train.")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data.")

    # Parse and normalize date column for time-based splits
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in training data.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

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

    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)

    mask_train = df[date_col] <= train_end
    mask_val = (df[date_col] > train_end) & (df[date_col] <= val_end)
    mask_test = df[date_col] > val_end

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    if X_train.empty:
        raise ValueError("Training split is empty; check date boundaries or data coverage.")

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
        f"Training XGB model on {len(feature_cols)} features and {len(X_train)} train rows "
        f"(val={len(X_val)}, test={len(X_test)}) for label '{label_col}'..."
    )
    model.fit(X_train, y_train)

    metrics = {}

    def _eval_split(name: str, X_split: pd.DataFrame, y_split: pd.Series):
        if X_split is None or X_split.empty:
            return
        preds = model.predict(X_split)
        r2 = r2_score(y_split, preds)
        ic = eval_metrics.rank_ic(pd.Series(preds, index=y_split.index), y_split)
        metrics[f"{name}_r2"] = float(r2)
        metrics[f"{name}_ic"] = float(ic)
        print(f"{name.capitalize()} R^2={r2:.4f}, IC={ic:.4f}")

    _eval_split("train", X_train, y_train)
    _eval_split("val", X_val, y_val)
    _eval_split("test", X_test, y_test)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        "model": model,
        "features": feature_cols,
        "metadata": {
            "label": label_col,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "metrics": metrics,
        },
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle to {model_path}")

    if args.features_out:
        try:
            Path(args.features_out).write_text(pd.Series(feature_cols).to_json(orient="values"), encoding="utf-8")
            print(f"Wrote feature list to {args.features_out}")
        except Exception as exc:
            print(f"Feature list write skipped: {exc}")

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
