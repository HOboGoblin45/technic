"""Prototype signal playback studio (placeholder)."""

import json
from pathlib import Path


def load_scan_logs(log_dir: Path = Path("scan_logs")):
    logs = []
    if not log_dir.exists():
        return logs
    for p in log_dir.glob("*.json"):
        try:
            logs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return logs


def playback(logs):
    """Placeholder to iterate logs; UI should render timeline externally."""
    for entry in sorted(logs, key=lambda x: x.get("timestamp", "")):
        yield entry


__all__ = ["load_scan_logs", "playback"]
