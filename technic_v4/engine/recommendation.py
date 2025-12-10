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

import math
from typing import Any, Dict, Optional

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

    # --- Institutional Core Score (ICS) summary ---
    ics_raw = row.get("InstitutionalCoreScore")
    try:
        ics = float(ics_raw) if ics_raw is not None else math.nan
    except Exception:
        ics = math.nan

    if not math.isnan(ics):
        if ics >= 85:
            ics_label = "institutional\u2011grade core setup"
        elif ics >= 70:
            ics_label = "high\u2011quality, liquid setup"
        elif ics >= 55:
            ics_label = "solid but mid\u2011tier quality"
        else:
            ics_label = "more speculative idea"

        parts.append(f"Institutional Core Score {ics:.0f}/100 ({ics_label}).")

    # --- Fundamental quality summary ---
    quality_raw: Any = None
    for q_col in ("QualityScore", "fundamental_quality_score", "quality_roe_sector_z"):
        if q_col in row and row[q_col] is not None:
            quality_raw = row[q_col]
            break

    quality_desc = None
    if quality_raw is not None:
        try:
            q_val = float(quality_raw)
            # Heuristic: treat 0\u2013100 ranges and z-scores gracefully
            if abs(q_val) <= 5.0:
                # assume z-score like metric
                if q_val >= 1.0:
                    quality_desc = "strong profitability and balance-sheet quality"
                elif q_val >= 0.0:
                    quality_desc = "respectable fundamentals"
                elif q_val <= -1.0:
                    quality_desc = "weak or deteriorating fundamentals"
                else:
                    quality_desc = "mixed fundamental profile"
            else:
                # assume 0-100 style score
                if q_val >= 75:
                    quality_desc = "strong profitability and balance-sheet quality"
                elif q_val >= 55:
                    quality_desc = "respectable fundamentals"
                elif q_val >= 40:
                    quality_desc = "average fundamental quality"
                else:
                    quality_desc = "weak or deteriorating fundamentals"
        except Exception:
            quality_desc = None

    if quality_desc:
        parts.append(f"Fundamental quality: {quality_desc}.")

    # --- Sponsorship / ownership ---
    sponsor = row.get("SponsorshipScore")
    etf_cnt = row.get("etf_holder_count")
    inst_cnt = row.get("inst_holder_count")
    try:
        sponsor_val = float(sponsor) if sponsor is not None else math.nan
    except Exception:
        sponsor_val = math.nan

    if not math.isnan(sponsor_val):
        holder_note = ""
        if etf_cnt is not None and inst_cnt is not None:
            try:
                holder_note = f" Held by ~{int(etf_cnt)} ETFs and ~{int(inst_cnt)} institutions."
            except Exception:
                holder_note = ""
        if sponsor_val >= 80:
            parts.append(f"Strong institutional/ETF sponsorship (score ~{sponsor_val:.0f}/100).{holder_note}")
        elif sponsor_val >= 60:
            parts.append(f"Solid sponsorship footprint (score ~{sponsor_val:.0f}/100).{holder_note}")
        elif sponsor_val <= 30:
            parts.append(f"Low sponsorship; limited fund ownership so far (score ~{sponsor_val:.0f}/100).")

    # --- Insider activity ---
    has_buy = bool(row.get("has_recent_insider_buy", False))
    has_heavy_sell = bool(row.get("has_heavy_insider_sell", False))
    if has_buy and has_heavy_sell:
        parts.append("Insiders active on both sides recently; monitor follow-through.")
    elif has_buy:
        parts.append("Recent cluster of insider buying in the last 90 days.")
    elif has_heavy_sell:
        parts.append("Notable recent insider selling; treat momentum with caution.")

    # --- Earnings / dividend event context ---
    is_pre_earn = bool(row.get("is_pre_earnings_window", False))
    is_post_pos = bool(row.get("is_post_earnings_positive_window", False))
    has_upcoming_earn = bool(row.get("has_upcoming_earnings", False))
    has_pos_surprise = bool(row.get("has_recent_positive_surprise", False))
    has_div_soon = bool(row.get("has_dividend_ex_soon", False))

    next_earn = row.get("next_earnings_date")
    last_earn = row.get("last_earnings_date")
    div_ex = row.get("dividend_ex_date")

    if is_pre_earn:
        if next_earn:
            parts.append(
                f"Note: the stock is within a few days of earnings (next report around {next_earn}); expect gap risk."
            )
        else:
            parts.append(
                "Note: price is trading close to an upcoming earnings date; expect gap risk."
            )
    elif is_post_pos:
        msg = "Recent positive earnings surprise with the stock still in the post-earnings drift window"
        if last_earn:
            msg += f" (last report around {last_earn})"
        parts.append(msg + ".")
    elif has_upcoming_earn and next_earn:
        parts.append(f"Earnings are upcoming around {next_earn}; treat this as an additional source of volatility.")

    if has_pos_surprise and not is_post_pos:
        parts.append("Last earnings beat expectations, which is a mild fundamental tailwind.")

    dy_raw = row.get("dividend_yield")
    dy = None
    try:
        if dy_raw is not None:
            dy = float(dy_raw)
    except Exception:
        dy = None

    if dy is not None and dy > 0:
        pct = dy * 100.0 if dy <= 1.0 else dy  # handle 0.03 vs 3.0 input styles
        if 1.0 <= pct <= 6.0:
            text = f"Dividend yield around {pct:.1f}% adds an income component."
        elif pct > 8.0:
            text = f"Very high dividend yield (~{pct:.1f}%), often a sign of elevated risk; treat the payout as uncertain."
        else:
            text = f"Dividend yield around {pct:.1f}%."
        if has_div_soon and div_ex:
            text += f" Ex-dividend date is coming up around {div_ex}."
        elif has_div_soon:
            text += " Ex-dividend date is coming up soon."
        parts.append(text)

    return " ".join(parts)
