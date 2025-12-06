"""Slack risk alert emoji formatter."""

from __future__ import annotations


def format_risk(level: str) -> str:
    mapping = {
        "high": "🔥",
        "warn": "⚠️",
        "low": "🧊",
    }
    return mapping.get(level, "")


__all__ = ["format_risk"]
