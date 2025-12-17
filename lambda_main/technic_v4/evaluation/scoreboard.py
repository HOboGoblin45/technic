"""
Scoreboard utilities for logging signals and computing simple metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from technic_v4.evaluation import metrics

SCOREBOARD_DIR = Path("data_cache") / "scoreboard"


def _scoreboard_path(date_str: str) -> Path:
    return SCOREBOARD_DIR / f"{date_str}.json"


def append_daily_signals(df_signals: pd.DataFrame, date_str: str | None = None) -> None:
    """
    Append scan results to scoreboard cache as JSON.
    """
    if df_signals is None or df_signals.empty:
        return
    # Deduplicate columns to avoid ambiguous keys in JSON consumers
    if df_signals.columns.duplicated().any():
        dedup_cols = []
        counts = {}
        for col in df_signals.columns:
            if col not in counts:
                counts[col] = 0
                dedup_cols.append(col)
            else:
                counts[col] += 1
                dedup_cols.append(f"{col}_{counts[col]}")
        df_signals = df_signals.copy()
        df_signals.columns = dedup_cols

    # Trim very wide option-related fields to keep the scoreboard payload lean
    drop_cols = [
        "OptionPicks",
        "OptionTrade",
        "OptionTradeText",
        "OptionQualityScore",
        "OptionIVRiskFlag",
        "OptionTradeText",
    ]
    df_signals = df_signals.drop(columns=[c for c in drop_cols if c in df_signals.columns], errors="ignore")

    SCOREBOARD_DIR.mkdir(parents=True, exist_ok=True)
    if date_str is None:
        date_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    path = _scoreboard_path(date_str)
    payload = df_signals.to_dict(orient="records")
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        return


def _load_all_scores() -> pd.DataFrame:
    if not SCOREBOARD_DIR.exists():
        return pd.DataFrame()
    records: List[dict] = []
    for p in SCOREBOARD_DIR.glob("*.json"):
        try:
            items = json.loads(p.read_text())
            for rec in items:
                rec["date"] = p.stem
                records.append(rec)
        except Exception:
            continue
    return pd.DataFrame(records)


def compute_history_metrics(n: int = 10) -> dict:
    """
    Compute rolling metrics from stored signals.
    """
    df = _load_all_scores()
    if df.empty or "AlphaScore" not in df or "RewardRisk" not in df:
        return {}
    # Use AlphaScore as preds, RewardRisk as proxy actual (placeholder)
    preds = pd.Series(df["AlphaScore"].values, index=df.index)
    actual = pd.Series(df["RewardRisk"].values, index=df.index)
    return {
        "ic": metrics.rank_ic(preds, actual),
        "precision_at_n": metrics.precision_at_n(preds, actual, n=n),
        "hit_rate": metrics.hit_rate(preds, actual),
        "avg_R": metrics.average_R(preds, actual),
    }
