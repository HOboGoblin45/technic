"""Historical misfire pattern scanner."""

from __future__ import annotations

import pandas as pd


def detect_misfires(trade_log: pd.DataFrame) -> pd.Series:
    """Group failed trades by signal_type to find recurring false signal conditions."""
    if "outcome" not in trade_log.columns or "signal_type" not in trade_log.columns:
        raise ValueError("trade_log must contain 'outcome' and 'signal_type' columns")
    return trade_log[trade_log["outcome"] == "fail"].groupby("signal_type").size()


__all__ = ["detect_misfires"]
