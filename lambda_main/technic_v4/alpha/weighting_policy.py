"""Feedback-aware weighting policy."""

from __future__ import annotations

import sqlite3


def boost_from_feedback(strategy_id: str) -> float:
    """Boost signals repeatedly approved by users in feedback.db."""
    try:
        conn = sqlite3.connect("feedback.db")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS feedback (strategy TEXT, vote INT, user TEXT)")
        cursor.execute("SELECT SUM(vote) FROM feedback WHERE strategy = ?", (strategy_id,))
        row = cursor.fetchone()
        conn.close()
        total = row[0] if row and row[0] is not None else 0
        return 1 + (total * 0.05)
    except Exception:
        return 1.0


__all__ = ["boost_from_feedback"]
