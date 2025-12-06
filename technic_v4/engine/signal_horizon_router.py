"""Multi-horizon signal orchestration."""

from __future__ import annotations

from typing import Any, Dict

from technic_v4.engine import short_term, mid_term, long_term


def _reconcile(short_sig: Dict[str, Any], long_sig: Dict[str, Any]) -> Dict[str, Any]:
    """If short contradicts long-term signal, defer via confidence-weighted override."""
    if short_sig.get("direction") == long_sig.get("direction"):
        return short_sig
    short_conf = float(short_sig.get("confidence", 0.5))
    long_conf = float(long_sig.get("confidence", 0.5))
    total = short_conf + long_conf or 1.0
    # Weighted vote: more confident horizon wins.
    return short_sig if short_conf / total >= 0.5 else long_sig


def route_signals(strategy: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch signal flow to horizon-specific modules based on strategy config."""
    # Optional reconciliation if both horizons are present in signal.
    if "short_signal" in signal and "long_signal" in signal:
        return _reconcile(signal["short_signal"], signal["long_signal"])

    horizon = strategy.get("horizon", "mid")
    if horizon == "short":
        return short_term.process_signal(signal)
    if horizon == "long":
        return long_term.process_signal(signal)
    return mid_term.process_signal(signal)

__all__ = ["route_signals"]
