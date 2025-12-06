"""API healthcheck & liveness probe."""

from __future__ import annotations


def healthcheck():
    # Placeholder statuses
    return {
        "db": "ok",
        "model_cache": "ok",
        "signal_engine": "ok",
        "task_queue": "ok",
        "status": "healthy",
    }


__all__ = ["healthcheck"]
