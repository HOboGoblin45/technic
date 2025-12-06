"""Market calendar-aware signal filtering."""

from __future__ import annotations


def is_valid_trading_day(dt, holidays) -> bool:
    """Avoid signal execution on known low-liquidity periods or holiday closures."""
    return dt.weekday() < 5 and dt not in holidays


__all__ = ["is_valid_trading_day"]
