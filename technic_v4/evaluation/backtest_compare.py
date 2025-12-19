from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
from scipy.stats import spearmanr

from technic_v4.config.settings import get_settings
from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4 import data_engine
from technic_v4.engine.factor_engine import zscore


@dataclass
class BacktestConfig:
    universe: List[str]
    start_date: date
    end_date: date
    top_n: int = 10
    forward_horizon_days: int = 5


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


def run_engine_variant(cfg: BacktestConfig, mode: str) -> pd.DataFrame:
    """
    mode="baseline": disable ML/TFT/deep/meta alpha
    mode="new": use current Settings defaults
    """
    settings = get_settings()
    # Snapshot original flags
    orig_flags = {
        "use_ml_alpha": settings.use_ml_alpha,
        "use_tft_features": settings.use_tft_features,
        "use_deep_alpha": settings.use_deep_alpha,
        "use_meta_alpha": settings.use_meta_alpha,
    }
    try:
        if mode == "baseline":
            settings.use_ml_alpha = False
            settings.use_tft_features = False
            settings.use_deep_alpha = False
            settings.use_meta_alpha = False

        rows = []
        dates = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="B")
        for d in dates:
            scan_cfg = ScanConfig(
                max_symbols=len(cfg.universe),
                lookback_days=120,
                min_tech_rating=0.0,
                trade_style="Short-term swing",
                allow_shorts=False,
                only_tradeable=False,
            )
            scan_df, _ = run_scan(config=scan_cfg)
            if scan_df is None or scan_df.empty:
                continue
            scan_df = scan_df.sort_values("TechRating", ascending=False).head(cfg.top_n)
            for rank, (_, r) in enumerate(scan_df.iterrows(), start=1):
                sym = r.get("Symbol")
                score = r.get("TechRating")
                fwd = _forward_return(sym, d, horizon=cfg.forward_horizon_days)
                rows.append(
                    {
                        "as_of_date": d,
                        "symbol": sym,
                        "rank": rank,
                        "score": score,
                        "fwd_ret": fwd,
                        "mode": mode,
                    }
                )
        return pd.DataFrame(rows)
    finally:
        # Restore original flags
        for k, v in orig_flags.items():
            setattr(settings, k, v)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for mode, g in df.groupby("mode"):
        if g.empty:
            continue
        ic = spearmanr(g["score"], g["fwd_ret"], nan_policy="omit").correlation
        precision = (g.sort_values("score", ascending=False).head(10)["fwd_ret"] > 0).mean()
        avg_top = g.sort_values("score", ascending=False).head(10)["fwd_ret"].mean()
        out[f"ic_{mode}"] = ic
        out[f"precision_top_{mode}"] = precision
        out[f"avg_fwd_ret_top_{mode}"] = avg_top
    return out


if __name__ == "__main__":
    # Example usage with a small universe and date range
    universe = [u.strip() for u in ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "JPM", "UNH", "HD"]]
    cfg = BacktestConfig(
        universe=universe,
        start_date=pd.Timestamp("2022-01-03").date(),
        end_date=pd.Timestamp("2022-02-15").date(),
        top_n=5,
        forward_horizon_days=5,
    )
    df_baseline = run_engine_variant(cfg, mode="baseline")
    df_new = run_engine_variant(cfg, mode="new")
    combined = pd.concat([df_baseline, df_new], ignore_index=True)
    metrics = compute_metrics(combined)
    print("Metrics comparison:", metrics)
