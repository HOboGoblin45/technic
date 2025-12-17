from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[3]  # repo root
PRIMARY_DATA = ROOT / "data" / "training_data_v2.parquet"
FALLBACK_DATA = ROOT / "technic_v4" / "scanner_output" / "history" / "replay_ics.parquet"
MODEL_PATH = ROOT / "models" / "meta" / "winprob_10d.pkl"


def _pick_dataset() -> Path:
    for p in (PRIMARY_DATA, FALLBACK_DATA):
        if p.exists():
            return p
    raise FileNotFoundError(f"No dataset found. Looked for: {PRIMARY_DATA}, {FALLBACK_DATA}")


def _bucket_stats(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"prob": probs, "y": y_true})
    df = df.sort_values("prob").reset_index(drop=True)
    df["bin"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")
    stats = df.groupby("bin").agg(count=("y", "size"), avg_prob=("prob", "mean"), win_rate=("y", "mean"))
    stats = stats.reset_index(drop=True)
    return stats


def _chronological_split(df: pd.DataFrame, date_col: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col)
    else:
        df = df.reset_index(drop=True)
    n = len(df)
    if n < 10:
        return df, pd.DataFrame(), pd.DataFrame()
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def _prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    feats = []
    for col in feature_cols:
        if col in df.columns:
            feats.append(col)
    if not feats:
        raise ValueError("No candidate feature columns were found in the dataset.")
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)
    return X


def train_meta_model() -> None:
    dataset_path = _pick_dataset()
    print(f"[META] Using dataset: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    # Normalize column names
    if "symbol" in df.columns and "Symbol" not in df.columns:
        df = df.rename(columns={"symbol": "Symbol"})

    # Target
    if "fwd_ret_10d" not in df.columns:
        raise ValueError("Dataset is missing fwd_ret_10d column.")
    df = df.dropna(subset=["fwd_ret_10d"])
    df["win_10d"] = (df["fwd_ret_10d"] > 0.0).astype(int)

    candidate_features = [
        "InstitutionalCoreScore",
        "TechRating",
        "alpha_blend",
        "AlphaScorePct",
        "QualityScore",
        "fundamental_quality_score",
        "ExplosivenessScore",
        "DollarVolume",
        "ATR14_pct",
        "RiskScore",
        "risk_score",
        "PlayStyle_Explosive",
        "PlayStyle_Stable",
        "PlayStyle_Neutral",
        "regime_trend_TRENDING_UP",
        "regime_trend_TRENDING_DOWN",
        "regime_trend_SIDEWAYS",
        "regime_vol_HIGH_VOL",
        "regime_vol_LOW_VOL",
    ]

    # One-hot playstyle if present
    if "PlayStyle" in df.columns:
        dummies = pd.get_dummies(df["PlayStyle"], prefix="PlayStyle")
        df = pd.concat([df, dummies], axis=1)

    # Determine available features
    available_features = [c for c in candidate_features if c in df.columns]
    X = _prepare_features(df, available_features)
    y = df["win_10d"].astype(int)

    date_col = None
    for c in ("as_of_date", "asof", "Date", "as_of"):
        if c in df.columns:
            date_col = c
            break

    train_df, val_df, test_df = _chronological_split(pd.concat([X, y], axis=1), date_col=None)

    def _split_xy(part: pd.DataFrame):
        if part is None or part.empty:
            return None, None
        return part[available_features], part["win_10d"]

    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)
    X_test, y_test = _split_xy(test_df)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )

    eval_set = []
    if X_val is not None and not X_val.empty:
        eval_set.append((X_val, y_val))

    print(f"[META] Training rows: {len(X_train)}, Val rows: {len(X_val) if X_val is not None else 0}, Test rows: {len(X_test) if X_test is not None else 0}")
    print(f"[META] Features used ({len(available_features)}): {available_features}")

    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def _metrics(split_name: str, Xp, yp):
        if Xp is None or Xp.empty:
            return None
        prob = model.predict_proba(Xp)[:, 1]
        auc = roc_auc_score(yp, prob) if len(np.unique(yp)) > 1 else float("nan")
        ll = log_loss(yp, prob, labels=[0, 1])
        bucket_df = _bucket_stats(prob, yp)
        print(f"[META] {split_name} AUC={auc:.4f} logloss={ll:.4f}  avg_pred={prob.mean():.4f}  win_rate={yp.mean():.4f}")
        print(f"[META] {split_name} bucket stats:\n{bucket_df}")
        return {"auc": auc, "logloss": ll, "avg_pred": float(prob.mean()), "win_rate": float(yp.mean()), "buckets": bucket_df.to_dict(orient="list")}

    metrics = {
        "train": _metrics("train", X_train, y_train),
        "val": _metrics("val", X_val, y_val) if X_val is not None else None,
        "test": _metrics("test", X_test, y_test) if X_test is not None else None,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": available_features, "metrics": metrics}, MODEL_PATH)
    print(f"[META] Saved win_prob_10d model to {MODEL_PATH}")

    # Also write a small JSON summary
    summary_path = MODEL_PATH.with_suffix(".json")
    try:
        clean_metrics = {}
        for k, v in metrics.items():
            if v is None:
                clean_metrics[k] = None
            else:
                cm = v.copy()
                cm.pop("buckets", None)
                clean_metrics[k] = cm
        summary = {
            "dataset": str(dataset_path),
            "rows": int(len(df)),
            "features": available_features,
            "metrics": clean_metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[META] Wrote summary to {summary_path}")
    except Exception:
        pass


if __name__ == "__main__":
    train_meta_model()
