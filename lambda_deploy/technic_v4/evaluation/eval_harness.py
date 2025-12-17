from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from technic_v4 import data_engine
from technic_v4.engine.scoring import compute_scores


def _forward_return(symbol: str, as_of: pd.Timestamp, horizon: int) -> Optional[float]:
    """
    Compute forward return over the given horizon from as_of date.
    """
    df = data_engine.get_price_history(symbol, days=horizon + 5, freq="daily")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    df = df.sort_index()
    if as_of not in df.index:
        # attempt to align by date only (ignore time)
        df_match = df[df.index.date == as_of.date()]
        if df_match.empty:
            return None
        as_of_idx = df_match.index[-1]
    else:
        as_of_idx = as_of
    try:
        pos = df.index.get_loc(as_of_idx)
    except Exception:
        return None
    if pos + horizon >= len(df):
        return None
    start = df["Close"].iloc[pos]
    end = df["Close"].iloc[pos + horizon]
    if start == 0:
        return None
    return (end / start) - 1


def backtest_top_n(
    universe: List[str],
    start_date: date,
    end_date: date,
    top_n: int = 10,
    forward_horizon_days: int = 5,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Minimal offline harness: score symbols each day, take top/bottom groups,
    and evaluate forward returns.
    """
    rows: list[dict] = []
    date_range = pd.date_range(start_date, end_date, freq="B")

    for as_of in date_range:
        daily_scores: list[tuple[str, float]] = []
        for sym in universe:
            hist = data_engine.get_price_history(sym, days=lookback_days, freq="daily")
            if hist is None or hist.empty:
                continue
            hist = hist[hist.index <= as_of]
            if hist.empty:
                continue
            scored = compute_scores(hist, trade_style=None, fundamentals=None)
            if scored is None or scored.empty:
                continue
            score = float(scored["TechRating"].iloc[-1])
            daily_scores.append((sym, score))

        if not daily_scores:
            continue

        daily_scores = sorted(daily_scores, key=lambda x: x[1], reverse=True)
        top_group = daily_scores[: top_n]
        bottom_group = daily_scores[-top_n:] if len(daily_scores) >= top_n else []

        for rank, (sym, score) in enumerate(top_group, start=1):
            fwd = _forward_return(sym, as_of, forward_horizon_days)
            rows.append(
                {
                    "as_of_date": as_of.date().isoformat(),
                    "symbol": sym,
                    "rank": rank,
                    "score": score,
                    "fwd_ret": fwd,
                    "group": "top",
                }
            )
        for rank, (sym, score) in enumerate(bottom_group, start=1):
            fwd = _forward_return(sym, as_of, forward_horizon_days)
            rows.append(
                {
                    "as_of_date": as_of.date().isoformat(),
                    "symbol": sym,
                    "rank": rank,
                    "score": score,
                    "fwd_ret": fwd,
                    "group": "bottom",
                }
            )

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute simple metrics from backtest output.
    """
    metrics: Dict[str, float] = {}
    if df.empty:
        return metrics

    valid = df.dropna(subset=["score", "fwd_ret"])
    if len(valid) >= 5:
        ic, _ = spearmanr(valid["score"], valid["fwd_ret"])
        metrics["ic"] = float(ic) if not pd.isna(ic) else 0.0
    else:
        metrics["ic"] = 0.0

    top = valid[valid["group"] == "top"]
    bottom = valid[valid["group"] == "bottom"]
    if not top.empty:
        metrics["top_mean_ret"] = float(top["fwd_ret"].mean())
        metrics["precision_at_n"] = float((top["fwd_ret"] > 0).mean())
    if not bottom.empty:
        metrics["bottom_mean_ret"] = float(bottom["fwd_ret"].mean())

    return metrics


if __name__ == "__main__":
    default_universe = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA", "AMZN"]
    end = date.today()
    start = end - timedelta(days=60)
    results = backtest_top_n(default_universe, start, end, top_n=3, forward_horizon_days=5)
    print(f"Backtest rows: {len(results)}")
    metrics = compute_metrics(results)
    print("Metrics:", metrics)
