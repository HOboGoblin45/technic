"""Trade auditor log for postmortem review and traceability."""

from __future__ import annotations


def log_trade_audit(signal_id, inputs, model_state, result):
    """Track every signal, model state, and inputs for audit traceability."""
    return {
        "signal": signal_id,
        "inputs": inputs,
        "model": model_state,
        "result": result,
    }


__all__ = ["log_trade_audit"]
