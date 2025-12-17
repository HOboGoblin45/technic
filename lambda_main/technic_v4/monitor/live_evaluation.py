from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from technic_v4 import data_engine
from technic_v4.infra.logging import get_logger

logger = get_logger()


def _forward_return(symbol: str, as_of: pd.Timestamp, horizon: int = 5) -> float:
    df = data_engine.get_price_history(symbol, days=horizon + 2, freq="daily")
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan")
    df = df[df.index >= as_of]
    if df.empty:
        return float("nan")
    start = df["Close"].iloc[0]
    end_idx = min(horizon, len(df) - 1)
    end = df["Close"].iloc[end_idx]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return (end - start) / start


def evaluate_log(
    log_path: Path = Path("logs/recommendations.csv"),
    horizon: int = 5,
    warn_threshold: float = -0.01,
) -> Dict[str, Any]:
    if not log_path.exists():
        logger.warning("[live_eval] log file %s not found", log_path)
        return {}
    df = pd.read_csv(log_path, parse_dates=["AsOf"])
    if df.empty:
        return {}
    fwd_rets = []
    bench_rets = []
    for _, row in df.iterrows():
        as_of = row["AsOf"]
        sym = row.get("Symbol")
        if not sym or pd.isna(as_of):
            continue
        r = _forward_return(sym, as_of, horizon=horizon)
        fwd_rets.append(r)
        b = _forward_return("SPY", as_of, horizon=horizon)
        bench_rets.append(b)
    perf = pd.Series(fwd_rets, dtype=float)
    bench = pd.Series(bench_rets, dtype=float)
    mean_perf = perf.mean(skipna=True)
    mean_bench = bench.mean(skipna=True)
    delta = mean_perf - mean_bench if pd.notna(mean_perf) and pd.notna(mean_bench) else float("nan")
    if pd.notna(mean_perf) and mean_perf < warn_threshold:
        logger.warning("[live_eval] mean forward return %.4f below threshold %.4f", mean_perf, warn_threshold)
    return {
        "count": len(perf),
        "mean_return": mean_perf,
        "mean_benchmark": mean_bench,
        "excess": delta,
    }


if __name__ == "__main__":
    stats = evaluate_log()
    logger.info("[live_eval] stats: %s", stats)
