from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _default_data_path(root: Path) -> Path:
    """
    Pick the best available backtest dataset, preferring the richer
    training_data_v2 parquet if present, otherwise falling back to the
    replay_ics parquet from scan history.
    """
    candidates: List[Path] = [
        root / "data" / "training_data_v2.parquet",
        root / "technic_v4" / "scanner_output" / "history" / "replay_ics.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No backtest dataset found. Looked for: {', '.join(str(p) for p in candidates)}"
    )


def _load_data(path: Path, score_cols: List[str], label_cols: List[str]) -> pd.DataFrame:
    """
    Load a minimal subset of columns for tuning thresholds.
    """
    preferred_symbol_cols = [["Symbol"], ["symbol"]]
    for sym_cols in preferred_symbol_cols:
        cols = sym_cols + score_cols + label_cols
        try:
            df = pd.read_parquet(path, columns=[c for c in cols if c is not None])
            return df
        except Exception:
            continue
    # If we cannot project columns due to name mismatches, load all and rename if possible.
    df_full = pd.read_parquet(path)
    if "symbol" in df_full.columns and "Symbol" not in df_full.columns:
        df_full = df_full.rename(columns={"symbol": "Symbol"})
    return df_full


def _candidate_thresholds(
    series: pd.Series,
    quantiles: List[float] | None = None,
) -> List[Tuple[float, float]]:
    """
    Given a score series, return a list of (quantile, threshold_value).

    Example: [(0.5, score_at_50th_pct), (0.6, ...), ...]
    """
    if quantiles is None:
        quantiles = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    series = series.dropna()
    if series.empty:
        return []
    out: List[Tuple[float, float]] = []
    for q in quantiles:
        try:
            thr = float(series.quantile(q))
        except Exception:
            continue
        out.append((q, thr))
    return out


def _threshold_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    min_rows: int = 300,
) -> pd.DataFrame:
    """
    For a given score and forward-return label, compute metrics at several
    score-based thresholds.

    Metrics per threshold:
      - mean forward return
      - win rate (label > 0)
      - sample size
      - excess return vs baseline mean
      - excess win rate vs baseline win rate
    """
    sub = df[[score_col, label_col]].dropna()
    if sub.empty:
        return pd.DataFrame()

    baseline_mean = float(sub[label_col].mean())
    baseline_win = float((sub[label_col] > 0).mean())

    thrs = _candidate_thresholds(sub[score_col])
    rows: List[Dict[str, float]] = []

    for q, thr in thrs:
        mask = sub[score_col] >= thr
        df_t = sub.loc[mask]
        n = int(df_t.shape[0])
        if n < min_rows:
            continue
        mean_ret = float(df_t[label_col].mean())
        win_rate = float((df_t[label_col] > 0).mean())
        rows.append(
            {
                "score": score_col,
                "label": label_col,
                "quantile": q,
                "threshold": thr,
                "n": n,
                "mean_ret": mean_ret,
                "win_rate": win_rate,
                "baseline_mean": baseline_mean,
                "baseline_win": baseline_win,
                "excess_ret": mean_ret - baseline_mean,
                "excess_win": win_rate - baseline_win,
            }
        )

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame, score_col: str, label_col: str) -> None:
    if df.empty:
        print(f"[THR] No metrics for {score_col} / {label_col}")
        return

    df = df.sort_values("excess_ret", ascending=False)
    cols = [
        "quantile",
        "threshold",
        "n",
        "mean_ret",
        "baseline_mean",
        "excess_ret",
        "win_rate",
        "baseline_win",
        "excess_win",
    ]
    print(f"=== Threshold suggestions for score='{score_col}', label='{label_col}' ===")
    print(df[cols].round(6).to_string(index=False))
    print()

    # Heuristic "best" pick: highest excess_ret subject to sufficient sample size
    best = df.sort_values("excess_ret", ascending=False).iloc[0]
    print(
        f"[THR] Recommended min {score_col} for {label_col}: "
        f"{best['threshold']:.3f} (>= {best['quantile']*100:.1f}th pct), "
        f"n={int(best['n'])}, mean_ret={best['mean_ret']:.6f}, "
        f"excess_ret={best['excess_ret']:.6f}, win_rate={best['win_rate']:.3f}"
    )
    print()


def main() -> None:
    # Repo root: .../technic-clean/
    root = Path(__file__).resolve().parents[3]
    data_path = _default_data_path(root)

    print(f"[THR] Using data: {data_path}")

    score_cols: List[str] = [
        "InstitutionalCoreScore",
        "TechRating",
        "alpha_blend",
        "AlphaScorePct",
        "ml_alpha_z",
    ]
    label_cols: List[str] = ["fwd_ret_5d", "fwd_ret_10d"]

    df = _load_data(data_path, score_cols=score_cols, label_cols=label_cols)

    available = set(df.columns)
    effective_scores = [c for c in score_cols if c in available]
    effective_labels = [c for c in label_cols if c in available]

    print(f"[THR] Score columns present: {effective_scores}")
    print(f"[THR] Label columns present: {effective_labels}")
    print()

    for score in effective_scores:
        for label in effective_labels:
            metrics_df = _threshold_metrics(df, score_col=score, label_col=label)
            _print_table(metrics_df, score_col=score, label_col=label)

    print("[THR] Threshold suggestion run complete.")


if __name__ == "__main__":
    main()
