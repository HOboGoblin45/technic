"""
Evaluate alpha signals on historical replay data to demonstrate robustness
and stress-test across eras, playstyles, and regimes.

Usage (from repo root):
    python technic_v4/dev/backtest/evaluate_alpha_history.py \
        --data data/training_data_v2.parquet \
        --label fwd_ret_5d \
        --score InstitutionalCoreScore
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root on sys.path for direct invocation
import sys

# project root (repo root)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technic_v4 import data_engine
from technic_v4.engine import regime_engine
from technic_v4.evaluation import metrics as eval_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest alpha signals on historical replay data.")
    p.add_argument(
        "--data",
        type=str,
        default="data/training_data_v2.parquet",
        help="Parquet path with as_of_date, symbol, score, forward returns.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="fwd_ret_5d",
        help="Forward return column to evaluate (e.g., fwd_ret_5d, fwd_ret_10d).",
    )
    p.add_argument(
        "--score",
        type=str,
        default="InstitutionalCoreScore",
        help="Score column to rank by (fallback to TechRating if missing).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Cross-sectional precision@N cut for daily metrics.",
    )
    p.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip regime stats (avoids SPY fetch / cache warnings).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSON path to save summary metrics.",
    )
    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if "as_of_date" not in df.columns:
        raise ValueError("Dataset must include 'as_of_date' column")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    elif "Symbol" in df.columns:
        df["symbol"] = df["Symbol"].astype(str)
    return df


def compute_daily_metrics(
    df: pd.DataFrame, score_col: str, label_col: str, top_n: int
) -> pd.DataFrame:
    rows: List[dict] = []
    grouped = df.groupby("as_of_date")
    skipped_constant = 0
    for as_of, g in grouped:
        g = g.dropna(subset=[score_col, label_col])
        if g.empty:
            continue
        preds = pd.Series(g[score_col].values, index=g.index)
        actual = pd.Series(g[label_col].values, index=g.index)
        # Skip days with no cross-sectional variation to avoid noisy warnings
        if preds.nunique() < 2 or actual.nunique() < 2:
            skipped_constant += 1
            continue
        ic = eval_metrics.rank_ic(preds, actual)
        n = min(top_n, len(preds))
        top_idx = preds.nlargest(n).index
        prec = float((actual.loc[top_idx] > 0).mean())
        avg_top = float(actual.loc[top_idx].mean())
        rows.append({"as_of_date": as_of, "ic": ic, "precision_at_n": prec, "avg_top": avg_top})
    if skipped_constant:
        print(f"[WARN] Skipped {skipped_constant} days with constant scores or labels.")
    return pd.DataFrame(rows)


def _classify_playstyle_row(row: pd.Series) -> str:
    """Lightweight PlayStyle classifier using RiskScore if PlayStyle missing."""
    risk_val = row.get("PlayStyle")
    if isinstance(risk_val, str) and risk_val:
        return risk_val  # already present

    risk_val = row.get("risk_score", row.get("RiskScore", np.nan))
    try:
        rv = float(risk_val)
    except Exception:
        rv = np.nan
    if pd.isna(rv):
        return "Neutral"
    if rv >= 0.2:
        return "Stable"
    if rv < 0.12:
        return "Explosive"
    return "Neutral"


def ensure_playstyle(df: pd.DataFrame) -> pd.DataFrame:
    if "PlayStyle" in df.columns:
        return df
    df = df.copy()
    df["PlayStyle"] = df.apply(_classify_playstyle_row, axis=1)
    return df


def summarize_split(df: pd.DataFrame, name: str, label_col: str, daily_df: pd.DataFrame) -> dict:
    summary = {"split": name, "rows": len(df)}
    if len(df) == 0:
        return summary
    preds = pd.Series(df[label_col].values, index=df.index)  # placeholder alignment below
    summary["mean_label"] = float(df[label_col].mean())
    if not daily_df.empty:
        summary["mean_ic"] = float(daily_df["ic"].mean())
        summary["p25_ic"] = float(daily_df["ic"].quantile(0.25))
        summary["p75_ic"] = float(daily_df["ic"].quantile(0.75))
        summary["precision_at_n"] = float(daily_df["precision_at_n"].mean())
    return summary


def bucket_metrics(df: pd.DataFrame, score_col: str, label_col: str, buckets: int = 5) -> pd.DataFrame:
    if df.empty or score_col not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(subset=[score_col, label_col])
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df[score_col], q=buckets, labels=False, duplicates="drop")
    grouped = df.groupby("bucket")
    res = grouped[label_col].agg(["mean", "median", "count"])
    res["win_rate"] = grouped[label_col].apply(lambda x: float((x > 0).mean()))
    return res.reset_index()


def playstyle_metrics(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if "PlayStyle" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("PlayStyle")
    res = grouped[label_col].agg(["mean", "median", "count"])
    res["win_rate"] = grouped[label_col].apply(lambda x: float((x > 0).mean()))
    return res.reset_index()


def attach_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each as_of_date with SPY trend/vol regime using a rolling view.
    Uses merge_asof to align the nearest available SPY bar at or before the as_of_date.
    """
    try:
        spy_hist = data_engine.get_price_history("SPY", days=6000, freq="daily")
    except Exception:
        spy_hist = pd.DataFrame()
    if spy_hist is None or spy_hist.empty or "Close" not in spy_hist.columns:
        df["Regime"] = "UNKNOWN"
        return df

    spy = spy_hist.sort_index().copy()
    spy["ret"] = spy["Close"].pct_change()
    spy["ma50"] = spy["Close"].rolling(50).mean()
    spy["ma200"] = spy["Close"].rolling(200).mean()
    spy["vol20"] = spy["ret"].rolling(20).std() * np.sqrt(252)
    spy["vol60"] = spy["ret"].rolling(60).std() * np.sqrt(252)

    def _trend(row):
        if pd.isna(row["ma50"]) or pd.isna(row["ma200"]):
            return "SIDEWAYS"
        if row["ma50"] > row["ma200"] and row["Close"] > row["ma50"]:
            return "TRENDING_UP"
        if row["ma50"] < row["ma200"] and row["Close"] < row["ma50"]:
            return "TRENDING_DOWN"
        return "SIDEWAYS"

    def _vol(row):
        if pd.isna(row["vol20"]) or pd.isna(row["vol60"]) or row["vol60"] == 0:
            return "LOW_VOL"
        ratio = row["vol20"] / row["vol60"]
        if ratio > 1.25:
            return "HIGH_VOL"
        if ratio < 0.8:
            return "LOW_VOL"
        return "LOW_VOL"

    spy["trend"] = spy.apply(_trend, axis=1)
    spy["vol_regime"] = spy.apply(_vol, axis=1)
    spy["Regime"] = spy["trend"] + "_" + spy["vol_regime"]
    spy = spy[["Regime"]].reset_index().rename(columns={"index": "Date"})

    merged = pd.merge_asof(
        df.sort_values("as_of_date"),
        spy,
        left_on="as_of_date",
        right_on="Date",
        direction="backward",
    )
    merged["Regime"] = merged["Regime"].fillna("UNKNOWN")
    merged = merged.drop(columns=["Date"])
    return merged


def regime_metrics(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if "Regime" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("Regime")
    res = grouped[label_col].agg(["mean", "median", "count"])
    res["win_rate"] = grouped[label_col].apply(lambda x: float((x > 0).mean()))
    return res.reset_index()


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.data))

    score_col = args.score if args.score in df.columns else "TechRating"
    if score_col not in df.columns:
        raise ValueError(f"Score column not found: {args.score}")
    label_col = args.label
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    df = df.dropna(subset=[score_col, label_col])
    if df.empty:
        raise SystemExit("Dataset is empty after filtering; aborting.")

    df = ensure_playstyle(df)

    # Daily cross-sectional metrics
    daily_df = compute_daily_metrics(df, score_col, label_col, top_n=args.top_n)

    # Era splits
    train_mask = df["as_of_date"] <= pd.Timestamp("2018-12-31")
    val_mask = (df["as_of_date"] > pd.Timestamp("2018-12-31")) & (df["as_of_date"] <= pd.Timestamp("2021-12-31"))
    test_mask = df["as_of_date"] > pd.Timestamp("2021-12-31")

    print("=== Cross-sectional daily metrics ===")
    if not daily_df.empty:
        print(
            f"IC mean={daily_df['ic'].mean():.4f}, median={daily_df['ic'].median():.4f}, "
            f"Precision@{args.top_n} mean={daily_df['precision_at_n'].mean():.3f}, "
            f"Top-{args.top_n} avg return={daily_df['avg_top'].mean():.4f}"
        )
    else:
        print("No daily metrics computed (possibly missing data).")

    print("\n=== Era splits ===")
    for name, mask in [
        ("train<=2018", train_mask),
        ("val 2019-2021", val_mask),
        ("test 2022+", test_mask),
    ]:
        split_df = df[mask]
        split_daily = daily_df[daily_df["as_of_date"].isin(split_df["as_of_date"])]
        summary = summarize_split(split_df, name, label_col, split_daily)
        print(f"{name}: rows={summary.get('rows')}, mean_label={summary.get('mean_label', float('nan')):.4f}, "
              f"mean_ic={summary.get('mean_ic', float('nan')):.4f}, "
              f"p25_ic={summary.get('p25_ic', float('nan')):.4f}, "
              f"p75_ic={summary.get('p75_ic', float('nan')):.4f}, "
              f"precision_at_n={summary.get('precision_at_n', float('nan')):.3f}")

    print("\n=== Score buckets (quantiles) ===")
    buckets = bucket_metrics(df, score_col, label_col, buckets=5)
    if buckets.empty:
        print("No bucket metrics.")
    else:
        print(buckets.to_string(index=False))

    print("\n=== PlayStyle performance ===")
    ps = playstyle_metrics(df, label_col)
    if ps.empty:
        print("No PlayStyle column present.")
    else:
        print(ps.to_string(index=False))

    regime_summary = None
    if not args.skip_regime:
        print("\n=== Regime performance (SPY trend/vol) ===")
        df_regime = attach_regime(df)
        reg = regime_metrics(df_regime, label_col)
        if reg.empty:
            print("Regime labeling unavailable.")
        else:
            regime_summary = reg
            print(reg.to_string(index=False))

    if args.out:
        import json

        out_path = Path(args.out)
        payload = {
            "daily": {
                "ic_mean": float(daily_df["ic"].mean()) if not daily_df.empty else np.nan,
                "ic_median": float(daily_df["ic"].median()) if not daily_df.empty else np.nan,
                "precision_at_n": float(daily_df["precision_at_n"].mean()) if not daily_df.empty else np.nan,
                "avg_top": float(daily_df["avg_top"].mean()) if not daily_df.empty else np.nan,
            },
            "splits": {
                "train": summarize_split(df[train_mask], "train", label_col, daily_df[daily_df["as_of_date"].isin(df[train_mask]["as_of_date"])]),
                "val": summarize_split(df[val_mask], "val", label_col, daily_df[daily_df["as_of_date"].isin(df[val_mask]["as_of_date"])]),
                "test": summarize_split(df[test_mask], "test", label_col, daily_df[daily_df["as_of_date"].isin(df[test_mask]["as_of_date"])]),
            },
            "buckets": bucket_metrics(df, score_col, label_col, buckets=5).to_dict(orient="list"),
            "playstyle": playstyle_metrics(df, label_col).to_dict(orient="list"),
            "regime": regime_summary.to_dict(orient="list") if regime_summary is not None else {},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        print(f"\nSaved summary JSON to {out_path}")


if __name__ == "__main__":
    main()
