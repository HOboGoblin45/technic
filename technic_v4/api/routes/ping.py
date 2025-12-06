"""Live model health ping endpoint stub."""

from __future__ import annotations

def full_check():
    """Return model freshness, drift %, alpha latency, backlog depth."""
    return {
        "model_freshness": "N/A",
        "drift_pct": "N/A",
        "alpha_latency_ms": "N/A",
        "ingest_backlog": "N/A",
        "status": "ok",
    }


__all__ = ["full_check"]
