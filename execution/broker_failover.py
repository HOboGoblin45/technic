"""Broker failover strategy."""

from __future__ import annotations

import json
from pathlib import Path
from collections import deque


FAIL_THRESH = 2
WINDOW_SECONDS = 10
FAIL_LOG = deque()
FALLBACK_CONF = Path("config/broker_fallback.json")


def record_failure(ts: float) -> str | None:
    """Detect repeated failures and return backup broker name if switching."""
    FAIL_LOG.append(ts)
    # prune
    while FAIL_LOG and ts - FAIL_LOG[0] > WINDOW_SECONDS:
        FAIL_LOG.popleft()
    if len(FAIL_LOG) >= FAIL_THRESH:
        if FALLBACK_CONF.exists():
            cfg = json.loads(FALLBACK_CONF.read_text(encoding="utf-8"))
            return cfg.get("backup")
    return None


__all__ = ["record_failure"]
