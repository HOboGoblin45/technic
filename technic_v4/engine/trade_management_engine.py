from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from technic_v4.infra.logging import get_logger

logger = get_logger()


def suggest_trade_updates(
    symbol: str,
    entry: float,
    stop: float,
    target: float,
    history_df: pd.DataFrame,
    days_since_entry: int,
) -> Dict[str, Any]:
    """
    Basic adaptive trade management suggestions.
    Heuristics:
    - If price moved fast toward/through target, trail stop up near entry.
    - If price stagnates or drops, tighten stop or consider exit.
    """
    updates: Dict[str, Any] = {"new_stop": None, "new_target": None, "notes": ""}
    if history_df is None or history_df.empty:
        return updates

    closes = history_df.get("Close") if "Close" in history_df.columns else None
    if closes is None or closes.empty:
        return updates

    last = float(closes.iloc[-1])
    pct_to_target = (target - entry) / entry if entry else 0
    pct_move = (last - entry) / entry if entry else 0
    notes = []

    # Fast move in favor: if >80% to target within first week, trail stop
    if pct_to_target and pct_move >= 0.8 * pct_to_target and days_since_entry <= 7:
        new_stop = max(stop, entry)  # trail to breakeven or better
        updates["new_stop"] = new_stop
        notes.append("Fast move toward target; trail stop to breakeven.")

    # If price exceeds target, suggest taking partial profits and raising stop
    if last >= target:
        new_stop = max(stop, entry + 0.25 * (target - entry))
        updates["new_stop"] = new_stop
        notes.append("Target reached; consider partial take-profit and trail stop.")

    # Stagnation: if after 10+ days and <25% of target progress
    if days_since_entry >= 10 and pct_move < 0.25 * pct_to_target:
        new_stop = max(stop, entry * 0.97)  # tighten to -3%
        updates["new_stop"] = new_stop
        notes.append("Stagnating; tighten stop to reduce risk.")

    # Adverse move: if drawdown >3% below entry
    if last < entry * 0.97:
        new_stop = max(stop, last * 0.99)
        updates["new_stop"] = new_stop
        notes.append("Price weak; consider tighter stop.")

    if notes:
        updates["notes"] = " ".join(notes)
    return updates


__all__ = ["suggest_trade_updates"]
