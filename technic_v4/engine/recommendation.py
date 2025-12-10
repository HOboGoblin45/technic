"""
Lightweight recommendation text generator for scan rows.

Produces a short, structured narrative using existing fields:
 - PlayStyle
 - ICS / TechRating
 - MuTotal / AlphaScore as expected drift
 - ATR% as risk proxy
 - Sector exposure note if overweight
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _score_phrase(score: float) -> str:
    if score >= 90:
        return "elite, top few percent today"
    if score >= 80:
        return "top ~10% today"
    if score >= 70:
        return "strong, upper quartile"
    if score >= 60:
        return "above average"
    if score >= 50:
        return "mid-pack"
    return "lower tier"


def _drift_phrase(mu: float, alpha: float) -> str:
    bias = mu
    if np.isnan(bias) or abs(bias) < 1e-9:
        bias = alpha
    if np.isnan(bias):
        return "Expected drift: not enough data to infer."
    pct = bias * 100.0
    if pct >= 1.0:
        return f"Expected drift: constructive; models imply roughly +{pct:.1f}% bias over the next couple of weeks."
    if pct >= 0.2:
        return f"Expected drift: mildly positive; models imply about +{pct:.1f}% bias."
    if pct <= -1.0:
        return f"Expected drift: negative; models imply roughly {pct:.1f}% downside bias."
    if pct <= -0.2:
        return f"Expected drift: mildly negative; models imply about {pct:.1f}% bias."
    return "Expected drift: near neutral."


def _historical_analog(playstyle: str) -> str:
    if playstyle.lower() == "stable":
        return "Historical analogs: stable setups like this have tended to deliver modest gains with shallow drawdowns."
    if playstyle.lower() == "explosive":
        return "Historical analogs: explosive runners can post large pops (20-40% in days) but often mean-revert sharply."
    return "Historical analogs: neutral setups have produced steady but unspectacular moves."


def _role_note(playstyle: str, atr: float, sector: str, sector_over: bool) -> str:
    risk = ""
    if atr >= 0.08:
        risk = "High volatility—size down."
    elif atr >= 0.04:
        risk = "Elevated volatility—moderate size."
    else:
        risk = "Low/normal volatility."
    role = ""
    ps = playstyle.lower()
    if ps == "stable":
        role = "Role: suitable as a core or satellite long in a diversified list."
    elif ps == "explosive":
        role = "Role: speculative; use very small size and tight risk controls."
    else:
        role = "Role: balanced; fits as a regular position with standard risk."
    sector_note = f" Sector watch: {sector} overweight in current list—add cautiously." if sector_over else ""
    return f"{role} {risk}.{sector_note}"


def build_recommendation(row: pd.Series, sector_overweights: Dict[str, float], sector_cap: float = 0.3) -> str:
    play = str(row.get("PlayStyle", "Neutral"))
    ics = row.get("InstitutionalCoreScore", np.nan)
    tr = row.get("TechRating", np.nan)
    mu = row.get("MuTotal", np.nan)
    alpha = row.get("AlphaScore", row.get("alpha_blend", np.nan))
    atr = row.get("ATR14_pct", np.nan)
    sector = str(row.get("Sector", "UNKNOWN"))

    # Score descriptor
    score_val = ics if not pd.isna(ics) else tr if not pd.isna(tr) else np.nan
    if pd.isna(score_val):
        score_line = "Score: not available today."
    else:
        score_line = f"Score: {score_val:.0f} ({_score_phrase(float(score_val))})."

    drift_line = _drift_phrase(mu if not pd.isna(mu) else np.nan, alpha if not pd.isna(alpha) else np.nan)
    analog_line = _historical_analog(play)

    overweight = False
    if sector in sector_overweights:
        overweight = sector_overweights[sector] > sector_cap + 0.05
    role_line = _role_note(play, float(atr) if not pd.isna(atr) else 0.0, sector, overweight)

    category_line = f"Category: {play} setup." if play else "Category: setup."

    parts = [
        category_line,
        score_line,
        drift_line,
        analog_line,
        role_line,
    ]
    return " ".join(parts)

