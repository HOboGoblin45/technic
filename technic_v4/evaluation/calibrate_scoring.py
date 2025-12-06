from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from scipy.stats import spearmanr

from technic_v4 import data_engine
from technic_v4.engine import feature_engine, scoring

CONFIG_PATH = Path("technic_v4/config/scoring_weights.json")


def calibrate_subscores(
    universe: List[str],
    start_date: date,
    end_date: date,
    forward_horizon_days: int = 5,
    lookback_days: int = 260,
) -> pd.DataFrame:
    """
    Backtests subscores against forward returns to suggest weights.

    Returns a DataFrame with correlation stats and suggested normalized weights
    based on absolute Information Coefficient (Spearman rank correlation).
    """
    rows: list[dict] = []
    span_days = (end_date - start_date).days + forward_horizon_days + lookback_days + 5

    for symbol in universe:
        df = data_engine.get_price_history(symbol, days=span_days, freq="daily")
        if df.empty or "Close" not in df.columns:
            continue
        df = df.sort_index()
        closes = df["Close"].reset_index(drop=True)
        dates = df.index.to_list()

        for idx, as_of in enumerate(dates):
            if as_of.date() < start_date or as_of.date() > end_date:
                continue
            if idx + forward_horizon_days >= len(closes):
                continue

            hist_slice = df.iloc[: idx + 1].tail(lookback_days)
            feats = feature_engine.build_features(hist_slice, fundamentals=None)
            if feats.empty:
                continue

            scores_df = scoring.compute_scores(hist_slice, trade_style=None, fundamentals=None)
            if scores_df.empty:
                continue
            scores_row = scores_df.iloc[0]

            fwd_ret = (closes.iloc[idx + forward_horizon_days] / closes.iloc[idx]) - 1
            rows.append(
                {
                    "symbol": symbol,
                    "as_of_date": as_of.date().isoformat(),
                    f"fwd_ret_{forward_horizon_days}d": fwd_ret,
                    "trend_score": scores_row.get("TrendScore", pd.NA),
                    "momentum_score": scores_row.get("MomentumScore", pd.NA),
                    "volume_score": scores_row.get("VolumeScore", pd.NA),
                    "volatility_score": scores_row.get("VolatilityScore", pd.NA),
                    "oscillator_score": scores_row.get("OscillatorScore", pd.NA),
                    "breakout_score": scores_row.get("BreakoutScore", pd.NA),
                }
            )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    sub_cols = [
        "trend_score",
        "momentum_score",
        "volume_score",
        "volatility_score",
        "oscillator_score",
        "breakout_score",
    ]
    fwd_col = f"fwd_ret_{forward_horizon_days}d"

    # Compute rank IC (Spearman) per subscore
    ic_data = {}
    for col in sub_cols:
        valid = result_df[[col, fwd_col]].dropna()
        if len(valid) < 5:
            ic = 0.0
        else:
            ic, _ = spearmanr(valid[col], valid[fwd_col])
            if pd.isna(ic):
                ic = 0.0
        ic_data[col] = ic

    abs_ics = {k: abs(v) for k, v in ic_data.items()}
    total_ic = sum(abs_ics.values()) or 1.0
    suggested_weights = {f"{k.replace('_score', '')}_weight": v / total_ic for k, v in abs_ics.items()}

    output = pd.DataFrame(
        {
            "subscore": list(ic_data.keys()),
            "ic": list(ic_data.values()),
            "abs_ic": list(abs_ics.values()),
            "suggested_weight": [suggested_weights[f"{k.replace('_score', '')}_weight"] for k in ic_data.keys()],
        }
    )
    return output


def save_weights_to_config(weights: Dict[str, float], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)


if __name__ == "__main__":
    # Simple CLI runner with rough defaults
    default_universe = ["AAPL", "MSFT", "SPY", "QQQ"]
    end = date.today()
    start = end - timedelta(days=180)
    df_weights = calibrate_subscores(default_universe, start, end, forward_horizon_days=5)
    if df_weights.empty:
        print("No data available for calibration.")
    else:
        suggested = dict(zip(df_weights["subscore"].str.replace("_score", "") + "_weight", df_weights["suggested_weight"]))
        print("Suggested weights:", suggested)
        save_weights_to_config(suggested)
        print(f"Saved to {CONFIG_PATH}")
