"""User-defined stop logic configuration."""

from __future__ import annotations

user_stop_config = {
    "fixed_stop_pct": 0.05,
    "trailing_stop": True,
    "time_limit_minutes": 120,
}

__all__ = ["user_stop_config"]
