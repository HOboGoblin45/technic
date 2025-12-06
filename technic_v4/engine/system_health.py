"""Real-time system health pinger."""

from __future__ import annotations

import logging
import time

try:
    import psutil  # type: ignore

    HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    HAVE_PSUTIL = False

logging.basicConfig(filename="health.log", level=logging.INFO)


def monitor_health(interval_seconds: int = 60) -> None:
    """Log basic system metrics at a fixed interval."""
    if not HAVE_PSUTIL:
        raise ImportError("psutil is required for system health monitoring.")

    while True:
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent()
        disk = psutil.disk_usage("/").percent
        log = f"MEM: {mem:.1f}% CPU: {cpu:.1f}% DISK: {disk:.1f}%"
        logging.info(log)
        time.sleep(interval_seconds)


__all__ = ["monitor_health"]
