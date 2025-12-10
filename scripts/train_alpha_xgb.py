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
from technic_v4.engine.regime_engine import classify_spy_regime


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
    p.add_argument(
        "--rolling-start",
        type=str,
        default="2010-01-01",
        help="Earliest date to start rolling windows (inclusive).",
    )
    p.add_argument(
        "--rolling-step-years",
        type=int,
        default=1,
        help="Step size in years for rolling windows.",
    )
    p.add_argument(
        "--rolling-window-train-years",
        type=int,
        default=8,
        help="Years in each rolling training window.",
    )
    p.add_argument(
        "--rolling-window-val-years",
        type=int,
        default=1,
        help="Years in each rolling validation window (immediately after train).",
    )
    p.add_argument(
        "--rolling-window-test-years",
        type=int,
        default=1,
        help="Years in each rolling test window (immediately after val).",
    )
    p.add_argument(
        "--regime-split",
        action="store_true",
        help="Train separate models per regime (TRENDING_UP_LOW_VOL, HIGH_VOL, SIDEWAYS).",
    )
    p.add_argument(
        "--rolling",
        action="store_true",
        help="Enable rolling windows; emits one model per window (suffix includes end year).",
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

    def train_one_split(df_split: pd.DataFrame, suffix: str = "") -> dict:
        X_split = df_split[feature_cols].fillna(0.0)
        y_split = df_split[label_col]

        split_mask_train = df_split[date_col] <= train_end
        split_mask_val = (df_split[date_col] > train_end) & (df_split[date_col] <= val_end)
        split_mask_test = df_split[date_col] > val_end

        X_train, y_train = X_split[split_mask_train], y_split[split_mask_train]
        X_val, y_val = X_split[split_mask_val], y_split[split_mask_val]
        X_test, y_test = X_split[split_mask_test], y_split[split_mask_test]

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
            f"[{suffix or 'base'}] Training XGB on {len(feature_cols)} features and {len(X_train)} train rows "
            f"(val={len(X_val)}, test={len(X_test)}) for label '{label_col}'..."
        )
        model.fit(X_train, y_train)

        metrics = {}

        def _eval_split(name: str, X_s: pd.DataFrame, y_s: pd.Series):
            if X_s is None or X_s.empty:
                return
            preds = model.predict(X_s)
            r2 = r2_score(y_s, preds)
            ic = eval_metrics.rank_ic(pd.Series(preds, index=y_s.index), y_s)
            metrics[f"{name}_r2"] = float(r2)
            metrics[f"{name}_ic"] = float(ic)
            print(f"[{suffix or 'base'}] {name.capitalize()} R^2={r2:.4f}, IC={ic:.4f}")

        _eval_split("train", X_train, y_train)
        _eval_split("val", X_val, y_val)
        _eval_split("test", X_test, y_test)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_out = model_path if not suffix else Path(model_path).with_stem(Path(model_path).stem + f"_{suffix}")
        bundle = {
            "model": model,
            "features": feature_cols,
            "metadata": {
                "label": label_col,
                "train_end": args.train_end,
                "val_end": args.val_end,
                "metrics": metrics,
                "suffix": suffix,
            },
        }
        joblib.dump(bundle, model_out)
        print(f"[{suffix or 'base'}] Saved model bundle to {model_out}")
        return metrics

    all_metrics = {}

    def _maybe_regime_label(df_in: pd.DataFrame) -> pd.DataFrame:
        if "regime_label" in df_in.columns:
            return df_in
        try:
            spy = data_engine.get_price_history("SPY", days=2600, freq="daily")
            if spy is not None and not spy.empty and date_col in df_in.columns:
                spy = spy.sort_index()
                reg_labels = []
                for dt in df_in[date_col]:
                    spy_subset = spy[spy.index <= dt]
                    reg = classify_spy_regime(spy_subset)
                    reg_labels.append(reg.get("label", "UNKNOWN"))
                df_in = df_in.copy()
                df_in["regime_label"] = reg_labels
            else:
                df_in["regime_label"] = "UNKNOWN"
        except Exception:
            df_in["regime_label"] = "UNKNOWN"
        return df_in

    def _run_regime_splits(df_in: pd.DataFrame):
        df_reg = _maybe_regime_label(df_in)
        for reg_lab, df_sub in df_reg.groupby("regime_label"):
            if df_sub.empty:
                continue
            suffix = reg_lab.replace(" ", "_")
            all_metrics[suffix] = train_one_split(df_sub, suffix=suffix)

    if args.rolling:
        start = pd.Timestamp(args.rolling_start)
        end = df[date_col].max()
        step = pd.DateOffset(years=args.rolling_step_years)
        train_years = args.rolling_window_train_years
        val_years = args.rolling_window_val_years
        test_years = args.rolling_window_test_years
        current_start = start
        while current_start < end:
            train_end_win = current_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
            val_end_win = train_end_win + pd.DateOffset(years=val_years)
            test_end_win = val_end_win + pd.DateOffset(years=test_years)
            df_window = df[(df[date_col] >= current_start) & (df[date_col] <= test_end_win)]
            if df_window.empty:
                current_start += step
                continue
            # Override split boundaries for this window
            train_end = train_end_win
            val_end = val_end_win
            window_suffix = f"{train_end.year}_roll"
            if args.regime_split:
                _run_regime_splits(df_window)
            else:
                all_metrics[window_suffix] = train_one_split(df_window, suffix=window_suffix)
            current_start += step
    else:
        if args.regime_split:
            _run_regime_splits(df)
        else:
            all_metrics["base"] = train_one_split(df, suffix="")

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
