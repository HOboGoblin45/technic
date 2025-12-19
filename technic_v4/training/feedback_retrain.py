"""Feedback-weighted retraining dataset builder."""

from __future__ import annotations

import sqlite3
import csv
from pathlib import Path


OUTPUT_PATH = Path("training/curated_feedback_dataset.csv")


def build_weighted_dataset():
    """Export feedback-weighted signals for retraining (weekly job)."""
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS feedback (strategy TEXT, vote INT, user TEXT)")
    cursor.execute("SELECT strategy, SUM(vote) as score FROM feedback GROUP BY strategy")
    rows = cursor.fetchall()
    conn.close()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "weight"])
        for strategy, score in rows:
            weight = 1 + (score or 0) * 0.1
            writer.writerow([strategy, weight])
    return OUTPUT_PATH


__all__ = ["build_weighted_dataset", "OUTPUT_PATH"]
