"""Risk-adaptive ranker adjusting scores and sizing by profile."""

from __future__ import annotations

from typing import Dict


def adjust_score(base_score: float, signal: Dict, risk_profile: Dict) -> float:
    """Adjust scoring based on volatility, beta, and exposure constraints."""
    vol = signal.get("volatility", 1.0)
    beta = signal.get("beta", 1.0)
    exposure = signal.get("exposure", 0.0)
    target_vol = risk_profile.get("target_vol", 1.0)
    max_exposure = risk_profile.get("max_exposure", 0.25)
    risk_penalty = vol / target_vol
    exposure_penalty = 1 + max(0, exposure - max_exposure)
    beta_penalty = abs(beta - risk_profile.get("target_beta", 1.0))
    return base_score / (risk_penalty * (1 + beta_penalty) * exposure_penalty)


def size_position(adjusted_score: float, risk_profile: Dict) -> float:
    """Derive position size from adjusted score and user risk profile."""
    base_risk = risk_profile.get("risk_pct", 0.01)
    return max(0.0, adjusted_score) * base_risk


__all__ = ["adjust_score", "size_position"]
