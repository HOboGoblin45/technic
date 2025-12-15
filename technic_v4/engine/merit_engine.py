"""
MERIT Score Engine - Multi-factor Evaluation & Risk-Integrated Technical Score

This module computes a novel composite score (0-100) that combines:
- Technical strength (TechRating percentile)
- Predictive alpha (AlphaScore percentile)
- Win probability (forward-looking)
- Institutional quality (ICS)
- Fundamental quality (QualityScore)
- Liquidity (DollarVolume)
- Volatility safety (inverse ATR%)

Key Innovation: Confluence Bonus rewards when technical and alpha signals AGREE.

Patent-worthy elements:
1. Confluence bonus formula (nonlinear agreement measurement)
2. Risk-integrated scoring (specific penalty structure)
3. Three-pillar integration (tech + alpha + quality)
4. Event-aware adjustments (dynamic penalties)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import pandas as pd
import numpy as np


@dataclass
class MeritConfig:
    """Configuration for MERIT Score computation."""
    
    # Component weights (must sum to ~1.0 before confluence bonus)
    weight_tech: float = 0.26
    weight_alpha: float = 0.22
    weight_win_prob: float = 0.18
    weight_ics: float = 0.14
    weight_quality: float = 0.14
    weight_liquidity: float = 0.04
    weight_vol_safety: float = 0.02
    
    # Confluence bonus multiplier
    confluence_multiplier: float = 0.20  # Range: -10 to +10 points
    
    # Penalty amounts
    penalty_earnings_3d: float = 12.0
    penalty_earnings_7d: float = 6.0
    penalty_low_liquidity: float = 8.0
    penalty_high_atr: float = 8.0
    penalty_very_high_atr: float = 14.0
    penalty_small_cap: float = 6.0
    penalty_micro_cap: float = 20.0
    penalty_ultra_risk: float = 25.0
    
    # Thresholds
    liquidity_threshold: float = 10_000_000  # $10M daily volume
    atr_high_threshold: float = 0.08  # 8% daily ATR
    atr_very_high_threshold: float = 0.12  # 12% daily ATR
    market_cap_small: float = 1_000_000_000  # $1B
    market_cap_micro: float = 300_000_000  # $300M
    
    # Default values for missing data
    default_ics: float = 50.0
    default_quality: float = 50.0
    default_win_prob: float = 0.50
    default_vol_safety: float = 50.0


def _pct_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank (0 to 1) for a series.
    Handles NaN gracefully by assigning median rank.
    """
    if series.isna().all():
        return pd.Series(0.5, index=series.index)
    
    # Use method='average' to handle ties
    ranks = series.rank(method='average', na_option='keep', pct=True)
    
    # Fill NaN with median (0.5)
    ranks = ranks.fillna(0.5)
    
    return ranks


def _safe_num(series: pd.Series, default: float) -> pd.Series:
    """
    Safely convert series to numeric, filling NaN/inf with default.
    """
    result = pd.to_numeric(series, errors='coerce')
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(default)
    return result


def _to_bool(series: pd.Series) -> pd.Series:
    """
    Safely convert series to boolean, treating NaN as False.
    """
    if series.dtype == bool:
        return series.fillna(False)
    
    # Try to convert to bool
    result = series.astype(str).str.lower().isin(['true', '1', 'yes', 't'])
    return result


def _days_to_event(date_series: pd.Series, ref_date: Optional[pd.Timestamp] = None) -> pd.Series:
    """
    Calculate days until event from date series.
    Returns NaN if date is missing or invalid.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    
    try:
        dates = pd.to_datetime(date_series, errors='coerce')
        days = (dates - ref_date).dt.days
        return days
    except Exception:
        return pd.Series(np.nan, index=date_series.index)


def compute_merit(
    df: pd.DataFrame,
    regime: Optional[dict] = None,
    config: Optional[MeritConfig] = None
) -> pd.DataFrame:
    """
    Compute MERIT Score for all rows in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with scan results. Expected columns:
        - TechRating: Technical score
        - AlphaScore or alpha_blend or ml_alpha_z: Predictive alpha
        - win_prob_10d: 10-day win probability (0-1)
        - InstitutionalCoreScore: ICS (0-100)
        - QualityScore: Fundamental quality (0-100)
        - DollarVolume or (Close * Volume): Liquidity
        - ATR14_pct or ATR_pct: Volatility as % of price
        - IsUltraRisky: Boolean flag
        - EarningsSoon or days_to_earnings: Earnings timing
        - market_cap: Market capitalization
        
    regime : dict, optional
        Market regime context (not currently used but reserved for future)
        
    config : MeritConfig, optional
        Configuration object. Uses defaults if not provided.
        
    Returns:
    --------
    pd.DataFrame
        Copy of input DataFrame with added columns:
        - MeritScore: 0-100 composite score
        - MeritBand: Letter grade (A+, A, B, C, D)
        - MeritFlags: Pipe-delimited risk flags
        - MeritSummary: Plain-English 1-sentence summary
        - MeritTechPct, MeritAlphaPct, etc.: Debug percentiles
    """
    if config is None:
        config = MeritConfig()
    
    # Work on a copy
    result = df.copy()
    n = len(result)
    
    if n == 0:
        # Return empty with expected columns
        for col in ['MeritScore', 'MeritBand', 'MeritFlags', 'MeritSummary',
                    'MeritTechPct', 'MeritAlphaPct', 'MeritLiquidityPct', 'MeritVolSafetyPct']:
            result[col] = []
        return result
    
    # ===== STEP 1: Compute Percentiles =====
    
    # Technical percentile
    if 'TechRating' in result.columns:
        tech_pct = _pct_rank(result['TechRating']) * 100
    else:
        tech_pct = pd.Series(50.0, index=result.index)
    
    # Alpha percentile (try multiple column names)
    alpha_col = None
    for col in ['AlphaScore', 'AlphaScorePct', 'alpha_blend', 'ml_alpha_z']:
        if col in result.columns:
            alpha_col = col
            break
    
    if alpha_col:
        alpha_pct = _pct_rank(result[alpha_col]) * 100
    else:
        alpha_pct = pd.Series(50.0, index=result.index)
    
    # Liquidity percentile
    if 'DollarVolume' in result.columns:
        dollar_vol = _safe_num(result['DollarVolume'], 1e6)
    elif 'Close' in result.columns and 'Volume' in result.columns:
        dollar_vol = _safe_num(result['Close'], 100) * _safe_num(result['Volume'], 10000)
    else:
        dollar_vol = pd.Series(1e6, index=result.index)
    
    # Use log scale for liquidity to handle wide range
    log_dollar_vol = np.log10(dollar_vol.clip(lower=1))
    liquidity_pct = _pct_rank(log_dollar_vol) * 100
    
    # Volatility safety percentile (lower ATR% = higher safety)
    atr_col = 'ATR14_pct' if 'ATR14_pct' in result.columns else 'ATR_pct' if 'ATR_pct' in result.columns else None
    if atr_col:
        atr_pct_val = _safe_num(result[atr_col], 0.02)
        vol_safety_pct = 100 - (_pct_rank(atr_pct_val) * 100)
    else:
        atr_pct_val = pd.Series(0.02, index=result.index)
        vol_safety_pct = pd.Series(config.default_vol_safety, index=result.index)
    
    # Win probability percentage
    if 'win_prob_10d' in result.columns:
        win_prob_pct = _safe_num(result['win_prob_10d'], config.default_win_prob) * 100
    else:
        # Fallback to alpha percentile
        win_prob_pct = alpha_pct.copy()
    
    # ICS (already 0-100)
    if 'InstitutionalCoreScore' in result.columns:
        ics = _safe_num(result['InstitutionalCoreScore'], config.default_ics)
    else:
        ics = pd.Series(config.default_ics, index=result.index)
    
    # Quality (already 0-100)
    if 'QualityScore' in result.columns:
        quality = _safe_num(result['QualityScore'], config.default_quality)
    else:
        quality = pd.Series(config.default_quality, index=result.index)
    
    # Store debug percentiles
    result['MeritTechPct'] = tech_pct
    result['MeritAlphaPct'] = alpha_pct
    result['MeritLiquidityPct'] = liquidity_pct
    result['MeritVolSafetyPct'] = vol_safety_pct
    
    # ===== STEP 2: Confluence Bonus =====
    
    # Measure agreement between technical and alpha (0-100, higher = more agreement)
    confluence = 100 - np.abs(tech_pct - alpha_pct)
    
    # Convert to bonus: -10 to +10 points
    # When confluence = 100 (perfect agreement), bonus = +10
    # When confluence = 0 (max disagreement), bonus = -10
    confluence_bonus = (confluence - 50) * config.confluence_multiplier
    
    # ===== STEP 3: Base Score (Weighted Sum) =====
    
    base_score = (
        config.weight_tech * tech_pct +
        config.weight_alpha * alpha_pct +
        config.weight_win_prob * win_prob_pct +
        config.weight_ics * ics +
        config.weight_quality * quality +
        config.weight_liquidity * liquidity_pct +
        config.weight_vol_safety * vol_safety_pct
    ) + confluence_bonus
    
    # ===== STEP 4: Penalties & Flags =====
    
    penalties = pd.Series(0.0, index=result.index)
    # Use dict keyed by index to avoid position/index mismatch
    flags_dict = {idx: [] for idx in result.index}
    
    # Earnings risk
    days_to_earnings = None
    if 'days_to_earnings' in result.columns:
        days_to_earnings = _safe_num(result['days_to_earnings'], 999)
    elif 'EarningsSoon' in result.columns:
        earnings_soon = _to_bool(result['EarningsSoon'])
        days_to_earnings = pd.Series(999, index=result.index)
        days_to_earnings[earnings_soon] = 2  # Assume 2 days if flag is true
    
    if days_to_earnings is not None:
        mask_3d = days_to_earnings <= 3
        mask_7d = (days_to_earnings > 3) & (days_to_earnings <= 7)
        
        penalties[mask_3d] += config.penalty_earnings_3d
        penalties[mask_7d] += config.penalty_earnings_7d
        
        for i in result.index[mask_3d]:
            flags_dict[i].append('EARNINGS_SOON')
        for i in result.index[mask_7d]:
            flags_dict[i].append('EARNINGS_7D')
    
    # Liquidity risk
    mask_low_liq = dollar_vol < config.liquidity_threshold
    penalties[mask_low_liq] += config.penalty_low_liquidity
    for i in result.index[mask_low_liq]:
        flags_dict[i].append('LOW_LIQUIDITY')
    
    # Volatility risk
    mask_very_high_atr = atr_pct_val > config.atr_very_high_threshold
    mask_high_atr = (atr_pct_val > config.atr_high_threshold) & ~mask_very_high_atr
    
    penalties[mask_very_high_atr] += config.penalty_very_high_atr
    penalties[mask_high_atr] += config.penalty_high_atr
    
    for i in result.index[mask_very_high_atr]:
        flags_dict[i].append('VERY_HIGH_ATR')
    for i in result.index[mask_high_atr]:
        flags_dict[i].append('HIGH_ATR')
    
    # Market cap risk
    if 'market_cap' in result.columns:
        market_cap = _safe_num(result['market_cap'], 1e9)
        mask_micro = market_cap < config.market_cap_micro
        mask_small = (market_cap < config.market_cap_small) & ~mask_micro
        
        penalties[mask_micro] += config.penalty_micro_cap
        penalties[mask_small] += config.penalty_small_cap
        
        for i in result.index[mask_micro]:
            flags_dict[i].append('MICRO_CAP')
        for i in result.index[mask_small]:
            flags_dict[i].append('SMALL_CAP')
    
    # Ultra-risky flag
    if 'IsUltraRisky' in result.columns:
        mask_ultra = _to_bool(result['IsUltraRisky'])
        penalties[mask_ultra] += config.penalty_ultra_risk
        for i in result.index[mask_ultra]:
            flags_dict[i].append('ULTRA_RISK')
    
    # Missing data flags (informational, no penalty)
    if 'InstitutionalCoreScore' not in result.columns or result.get('InstitutionalCoreScore', pd.Series()).isna().any():
        for i in result.index[ics == config.default_ics]:
            if 'MISSING_ICS' not in flags_dict[i]:
                flags_dict[i].append('MISSING_ICS')
    
    if 'QualityScore' not in result.columns or result.get('QualityScore', pd.Series()).isna().any():
        for i in result.index[quality == config.default_quality]:
            if 'MISSING_QUALITY' not in flags_dict[i]:
                flags_dict[i].append('MISSING_QUALITY')
    
    # ===== STEP 5: Final Score =====
    
    merit_score = (base_score - penalties).clip(0, 100)
    result['MeritScore'] = merit_score
    
    # ===== STEP 6: Banding =====
    
    def get_band(score):
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        else:
            return 'D'
    
    result['MeritBand'] = merit_score.apply(get_band)
    
    # ===== STEP 7: Flags =====
    
    result['MeritFlags'] = [('|'.join(flags_dict[idx]) if flags_dict[idx] else '') for idx in result.index]
    
    # ===== STEP 8: Summary =====
    
    def build_summary(row):
        """Build plain-English summary for a row."""
        score = row['MeritScore']
        band = row['MeritBand']
        symbol = row.get('Symbol', 'Stock')
        
        # Score description
        if score >= 85:
            score_desc = "exceptional"
        elif score >= 75:
            score_desc = "strong"
        elif score >= 65:
            score_desc = "solid"
        elif score >= 55:
            score_desc = "moderate"
        else:
            score_desc = "weak"
        
        # Technical description
        tech_val = row.get('MeritTechPct', 50)
        if tech_val >= 80:
            tech_desc = "strong technical trend"
        elif tech_val >= 60:
            tech_desc = "positive technical setup"
        elif tech_val >= 40:
            tech_desc = "neutral technicals"
        else:
            tech_desc = "weak technical position"
        
        # Alpha description
        alpha_val = row.get('MeritAlphaPct', 50)
        if alpha_val >= 70:
            alpha_desc = "strong forward edge"
        elif alpha_val >= 55:
            alpha_desc = "positive forward edge"
        elif alpha_val >= 45:
            alpha_desc = "neutral outlook"
        else:
            alpha_desc = "limited upside"
        
        # Quality description
        qual_val = row.get('QualityScore', config.default_quality)
        ics_val = row.get('InstitutionalCoreScore', config.default_ics)
        avg_qual = (qual_val + ics_val) / 2
        
        if avg_qual >= 75:
            qual_desc = "high quality"
        elif avg_qual >= 60:
            qual_desc = "good quality"
        elif avg_qual >= 45:
            qual_desc = "average quality"
        else:
            qual_desc = "lower quality"
        
        # Risk flags
        flags = row.get('MeritFlags', '')
        risk_notes = []
        if 'EARNINGS_SOON' in flags:
            risk_notes.append("earnings imminent")
        elif 'EARNINGS_7D' in flags:
            risk_notes.append("earnings this week")
        if 'LOW_LIQUIDITY' in flags:
            risk_notes.append("low liquidity")
        if 'VERY_HIGH_ATR' in flags or 'HIGH_ATR' in flags:
            risk_notes.append("elevated volatility")
        
        # Build summary
        summary = f"MERIT {score:.0f} ({band}): {score_desc} opportunity with {tech_desc}, {alpha_desc}, {qual_desc}"
        
        if risk_notes:
            summary += f"; caution: {', '.join(risk_notes)}"
        else:
            summary += "; no major risk flags"
        
        return summary
    
    # Apply summary generation (can be slow for large DataFrames, but provides value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result['MeritSummary'] = result.apply(build_summary, axis=1)
    
    return result


__all__ = ['compute_merit', 'MeritConfig']
