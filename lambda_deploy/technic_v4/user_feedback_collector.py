"""User feedback signal collector."""

from __future__ import annotations

import sqlite3


def store_feedback(strategy_id, vote, user_id):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS feedback (strategy TEXT, vote INT, user TEXT)"
    )
    cursor.execute("INSERT INTO feedback VALUES (?, ?, ?)", (strategy_id, vote, user_id))
    conn.commit()
    conn.close()


__all__ = ["store_feedback"]
