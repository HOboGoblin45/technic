from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SUITE_DIR = ROOT / "evaluation" / "alpha_history_suite"


def _load_one(path: Path) -> Dict[str, Any]:
    """
    Load a single JSON summary produced by run_alpha_score_suite / evaluate_alpha_history.

    We expect filenames like:
        InstitutionalCoreScore__fwd_ret_5d.json

    JSON structure (see evaluate_alpha_history.py):
        {
          "daily": {...},
          "splits": {"train": {...}, "val": {...}, "test": {...}},
          "buckets": {...},      # bucket metrics vs score
          "playstyle": {...},
          "regime": {...},
          "macro": {...},
          "sector": {...},
          "multi": {...},
          "holdout": {...},
        }
    """
    data = json.loads(path.read_text())

    # Derive score + label from filename stem
    stem = path.stem
    if "__" in stem:
        score, label = stem.split("__", 1)
    else:
        score, label = stem, ""

    daily = data.get("daily", {}) or {}
    splits = data.get("splits", {}) or {}
    train = splits.get("train", {}) or {}
    val = splits.get("val", {}) or {}
    test = splits.get("test", {}) or {}

    # Bucket metrics: top vs bottom bucket mean + win_rate
    top_mean = bottom_mean = top_win = bottom_win = None
    buckets = data.get("buckets", {}) or {}
    if buckets:
        try:
            df_b = pd.DataFrame(buckets)
            if not df_b.empty and "bucket" in df_b.columns:
                df_b = df_b.sort_values("bucket")
                bottom = df_b.iloc[0]
                top = df_b.iloc[-1]
                # evaluate_alpha_history.bucket_metrics uses generic 'mean' / 'win_rate'
                bottom_mean = float(bottom.get("mean", float("nan")))
                top_mean = float(top.get("mean", float("nan")))
                bottom_win = float(bottom.get("win_rate", float("nan")))
                top_win = float(top.get("win_rate", float("nan")))
        except Exception:
            # best-effort only
            pass

    row: Dict[str, Any] = {
        "score": score,
        "label": label,
        # Daily cross-sectional metrics
        "daily_ic": daily.get("ic_mean"),
        "daily_precision_at_n": daily.get("precision_at_n"),
        "daily_avg_top": daily.get("avg_top"),
        # Era splits
        "train_rows": train.get("rows"),
        "train_mean_label": train.get("mean_label"),
        "train_precision_at_n": train.get("precision_at_n"),
        "val_rows": val.get("rows"),
        "val_mean_label": val.get("mean_label"),
        "val_precision_at_n": val.get("precision_at_n"),
        "test_rows": test.get("rows"),
        "test_mean_label": test.get("mean_label"),
        "test_precision_at_n": test.get("precision_at_n"),
        # Bucket separation (top vs bottom)
        "bucket_bottom_mean": bottom_mean,
        "bucket_top_mean": top_mean,
        "bucket_bottom_win": bottom_win,
        "bucket_top_win": top_win,
    }
    return row


def _load_all() -> pd.DataFrame:
    if not SUITE_DIR.exists():
        raise FileNotFoundError(f"Suite directory not found: {SUITE_DIR}")

    json_files: List[Path] = sorted(SUITE_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON summaries found in {SUITE_DIR}")

    rows: List[Dict[str, Any]] = []
    for path in json_files:
        try:
            rows.append(_load_one(path))
        except Exception as exc:
            print(f"[SUMMARY] Failed to load {path}: {exc}")
    if not rows:
        raise SystemExit("No valid JSON summaries to summarize.")

    df = pd.DataFrame(rows)
    return df


def _print_overall(df: pd.DataFrame) -> None:
    print("=== Alpha / ICS summary ===")
    print(f"Rows: {len(df)}  (scores x labels combinations)")
    print()

    # Simple view: by test_mean_label and daily_avg_top
    cols = [
        "score",
        "label",
        "test_mean_label",
        "test_precision_at_n",
        "daily_ic",
        "daily_precision_at_n",
        "daily_avg_top",
        "bucket_bottom_mean",
        "bucket_top_mean",
        "bucket_bottom_win",
        "bucket_top_win",
    ]
    view = df[cols].copy()

    # Ensure numeric
    numeric_cols = [
        c
        for c in cols
        if c not in {"score", "label"} and c in view.columns
    ]
    for c in numeric_cols:
        view[c] = pd.to_numeric(view[c], errors="coerce")

    # For each label (fwd_ret_5d, fwd_ret_10d) print ranking by test_mean_label
    for label in sorted(view["label"].dropna().unique()):
        sub = view[view["label"] == label].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("test_mean_label", ascending=False)
        print(f"--- Label: {label} (sorted by test_mean_label) ---")
        print(
            sub[
                [
                    "score",
                    "test_mean_label",
                    "test_precision_at_n",
                    "daily_ic",
                    "daily_precision_at_n",
                    "daily_avg_top",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        print()

    # Also print bucket separation (top vs bottom) as a quick sanity check
    print("--- Bucket separation (top vs bottom bucket mean / win_rate) ---")
    sep = view[
        [
            "score",
            "label",
            "bucket_bottom_mean",
            "bucket_top_mean",
            "bucket_bottom_win",
            "bucket_top_win",
        ]
    ].copy()
    sep = sep.sort_values("bucket_top_mean", ascending=False)
    print(sep.round(4).to_string(index=False))
    print()


def main() -> None:
    df = _load_all()
    _print_overall(df)
    print("[SUMMARY] Done.")


if __name__ == "__main__":
    main()
