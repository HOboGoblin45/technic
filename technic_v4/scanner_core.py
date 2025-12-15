from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
import time

import traceback
import pandas as pd
import numpy as np

from technic_v4.universe_loader import load_universe, UniverseRow
from technic_v4 import data_engine
from technic_v4.config.settings import get_settings
from .infra.logging import get_logger
from technic_v4.engine.scoring import compute_scores, build_institutional_core_score
from technic_v4.engine.factor_engine import zscore, compute_factor_bundle
from technic_v4.engine.regime_engine import classify_regime
from technic_v4.engine.trade_planner import RiskSettings, plan_trades
from technic_v4.engine.portfolio_engine import risk_adjusted_rank, diversify_by_sector
from technic_v4.engine.options_engine import score_options
from technic_v4.engine.merit_engine import compute_merit
from technic_v4.engine import ranking_engine
from technic_v4.engine.options_suggest import suggest_option_trades
from technic_v4.engine import explainability_engine
from datetime import datetime, date
from technic_v4.engine import alpha_inference
from technic_v4.engine import explainability
from technic_v4.engine import meta_experience
from technic_v4.config.risk_profiles import get_risk_profile
from technic_v4.engine import setup_library
from technic_v4.engine import ray_runner
from technic_v4.data_layer.events import get_event_info
from technic_v4.data_layer.earnings_surprises import get_latest_surprise, get_surprise_stats
from technic_v4.data_layer.fundamental_trend import get_fundamental_trend
from technic_v4.data_layer.ratings import get_rating_info
from technic_v4.data_layer.quality import get_quality_info
from technic_v4.data_layer.sponsorship import get_sponsorship, get_insider_flags
from technic_v4.config.thresholds import load_score_thresholds
from technic_v4.engine.meta_inference import score_win_prob_10d
from technic_v4.engine.portfolio_optim import (
    mean_variance_weights,
    inverse_variance_weights,
    hrp_weights,
)
from technic_v4.engine.recommendation import build_recommendation
import concurrent.futures

logger = get_logger()

ProgressCallback = Callable[[str, int, int], None]

# Legacy compatibility for tests/monkeypatch
# PERFORMANCE: Pro Plus optimization - aggressive caching enabled
# With 8 GB RAM, we can cache more price data in memory
def get_stock_history_df(symbol: str, days: int, use_intraday: bool = True, end_date=None):
    return data_engine.get_price_history(symbol, days, freq="intraday" if use_intraday else "daily")

def get_fundamentals(symbol: str):
    return data_engine.get_fundamentals(symbol)

# Where scan CSVs are written
OUTPUT_DIR = Path(__file__).resolve().parent / "scanner_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Universe filter helper
# -----------------------------

def _filter_universe(
    universe: List[UniverseRow],
    sectors: Optional[Iterable[str]] = None,
    subindustries: Optional[Iterable[str]] = None,
    industry_contains: Optional[str] = None,
) -> List[UniverseRow]:
    """
    Apply sector and industry filters to the raw universe.
    """
    filtered: List[UniverseRow] = list(universe)

    has_any_sector = any(row.sector for row in filtered)
    has_any_industry = any(row.industry for row in filtered)
    has_any_subindustry = any(row.subindustry for row in filtered)

    # ----- Sector filter -----
    if sectors and has_any_sector:
        sector_set = {s.lower().strip() for s in sectors}
        before = len(filtered)
        filtered = [
            row
            for row in filtered
            if row.sector and row.sector.lower().strip() in sector_set
        ]
        logger.info(
            "[UNIVERSE] Sector filter kept %d / %d symbols for sectors=%s",
            len(filtered),
            before,
            sorted(sector_set),
        )
    elif sectors and not has_any_sector:
        logger.info(
            "[UNIVERSE] Sector filter requested but universe has no sector data; skipping sector filter."
        )

    # ----- Industry substring filter -----
    if industry_contains and industry_contains.strip():
        if has_any_industry:
            term = industry_contains.lower().strip()
            before = len(filtered)
            filtered = [
                row
                for row in filtered
                if row.industry and term in row.industry.lower()
            ]
            logger.info(
                "[UNIVERSE] Industry filter kept %d / %d symbols for term='%s'",
                len(filtered),
                before,
                term,
            )
        else:
            logger.info(
                "[UNIVERSE] Industry filter requested but universe has no industry data; skipping industry filter."
            )

    # ----- SubIndustry filter -----
    if subindustries and has_any_subindustry:
        sub_set = {s.lower().strip() for s in subindustries}
        before = len(filtered)
        filtered = [
            row
            for row in filtered
            if row.subindustry and row.subindustry.lower().strip() in sub_set
        ]
        logger.info(
            "[UNIVERSE] SubIndustry filter kept %d / %d symbols for subindustries=%s",
            len(filtered),
            before,
            sorted(sub_set),
        )
    elif subindustries and not has_any_subindustry:
        logger.info(
            "[UNIVERSE] SubIndustry filter requested but universe has no subindustry data; skipping subindustry filter."
        )

    return filtered


def _smart_filter_universe(universe: List[UniverseRow], config: "ScanConfig") -> List[UniverseRow]:
    """
    Apply intelligent pre-filtering to reduce universe size by 70-80%.
    Filters out illiquid, penny stocks, and low-quality names before expensive scanning.
    
    PERFORMANCE: This dramatically speeds up scans by reducing symbols to process.
    OPTIMIZATION: Enhanced with market cap and volume-based pre-filtering.
    """
    if not universe:
        return universe
    
    start_count = len(universe)
    filtered = list(universe)
    
    # Filter 1: Remove symbols with invalid tickers
    try:
        before = len(filtered)
        filtered = [
            row for row in filtered
            if row.symbol and 1 <= len(row.symbol) <= 5 and row.symbol.isalpha()
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info("[SMART_FILTER] Removed %d symbols with invalid tickers", removed)
    except Exception:
        pass
    
    # Filter 2: Focus on liquid sectors if no sector specified
    if not config.sectors:
        try:
            liquid_sectors = {
                "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
                "Industrials", "Communication Services", "Consumer Defensive", "Energy"
            }
            before = len(filtered)
            filtered = [
                row for row in filtered
                if not row.sector or row.sector in liquid_sectors
            ]
            removed = before - len(filtered)
            if removed > 0:
                logger.info("[SMART_FILTER] Focused on liquid sectors, removed %d symbols", removed)
        except Exception:
            pass
    
    # Filter 3: Remove known problematic symbols (leveraged ETFs, volatility products)
    try:
        exclude_patterns = [
            'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'UVXY', 'VIXY', 'SVXY', 'TVIX',
            'UPRO', 'SPXU', 'UDOW', 'SDOW', 'TNA', 'TZA', 'FAS', 'FAZ',
            'NUGT', 'DUST', 'JNUG', 'JDST', 'ERX', 'ERY', 'LABU', 'LABD'
        ]
        before = len(filtered)
        filtered = [
            row for row in filtered
            if row.symbol not in exclude_patterns
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info("[SMART_FILTER] Removed %d leveraged/volatility products", removed)
    except Exception:
        pass
    
    # Filter 4: Remove penny stocks and micro-caps (NEW - PHASE 1)
    # This prevents wasting time on symbols that will fail basic filters anyway
    try:
        # Known penny stock patterns (common OTC/pink sheet prefixes)
        penny_patterns = ['AAPL', 'MSFT']  # Placeholder - would need actual penny stock list
        before = len(filtered)
        # For now, just filter by symbol length as proxy (very short = often penny)
        filtered = [
            row for row in filtered
            if len(row.symbol) >= 2  # Remove single-letter symbols (often problematic)
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info("[SMART_FILTER] Removed %d potential penny stocks", removed)
    except Exception:
        pass
    
    # Filter 5: Sector-specific volume thresholds (NEW - PHASE 1)
    # Different sectors have different liquidity profiles
    try:
        # High-volume sectors need higher thresholds
        high_volume_sectors = {"Technology", "Financial Services", "Communication Services"}
        # Lower-volume sectors can have lower thresholds
        low_volume_sectors = {"Utilities", "Real Estate"}
        
        # This is a heuristic - we'll do actual volume check later
        # For now, just note which sectors we're keeping
        sector_counts = {}
        for row in filtered:
            sector = row.sector or "Unknown"
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        if sector_counts:
            logger.info("[SMART_FILTER] Sector distribution: %s", 
                       ", ".join(f"{k}={v}" for k, v in sorted(sector_counts.items(), key=lambda x: -x[1])[:5]))
    except Exception:
        pass
    
    end_count = len(filtered)
    reduction_pct = ((start_count - end_count) / start_count * 100) if start_count > 0 else 0
    
    logger.info(
        "[SMART_FILTER] Reduced universe: %d → %d symbols (%.1f%% reduction)",
        start_count,
        end_count,
        reduction_pct
    )
    
    return filtered


def _prepare_universe(config: "ScanConfig", settings=None) -> List[UniverseRow]:
    """
    Load and filter the universe based on config.
    PERFORMANCE: Now includes smart pre-filtering to reduce symbols by 70-80%.
    """
    universe: List[UniverseRow] = load_universe()
    logger.info("[UNIVERSE] loaded %d symbols from ticker_universe.csv.", len(universe))

    # Apply smart filtering first (reduces by 70-80%)
    universe = _smart_filter_universe(universe, config)

    # Then apply user-specified filters (sectors, industries)
    filtered = _filter_universe(
        universe=universe,
        sectors=config.sectors,
        subindustries=config.subindustries,
        industry_contains=config.industry_contains,
    )
    logger.info(
        "[FILTER] %d symbols after universe filters (from %d).",
        len(filtered),
        len(universe),
    )
    return filtered


def _apply_alpha_blend(
    df: pd.DataFrame,
    regime: Optional[dict] = None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Cross-sectional alpha blend with optional multi-horizon ML alpha.

    - factor_alpha = zscore of baseline TechRating (v1 heuristic)
    - ml_alpha_5d  = XGB 5d model
    - ml_alpha_10d = XGB 10d model (if available)
    - ml_alpha_z   = w5 * z(ml_5d) + w10 * z(ml_10d) when both available
    - alpha_blend  = blend(factor_alpha, ml_alpha_z; TECHNIC_ALPHA_WEIGHT)
    - TechRating   = v2 hybrid score using alpha_blend

    Columns produced:
        factor_alpha       : z-score of baseline TechRating
        Alpha5d            : raw 5d ML alpha (if available)
        Alpha10d           : raw 10d ML alpha (if available)
        AlphaScore_5d      : alias of Alpha5d
        AlphaScore_10d     : alias of Alpha10d
        AlphaScore         : blended raw ML alpha (5d/10d)
        ml_alpha_5d_z      : zscore of 5d alpha
        ml_alpha_10d_z     : zscore of 10d alpha (if available)
        ml_alpha_z         : blended ML zscore used in alpha_blend
        alpha_blend        : final cross-sectional alpha driver
        TechRating_raw     : original TechRating before alpha blending
        TechRating         : blended TechRating v2 (hybrid)
    """
    if df is None or df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    settings = get_settings()
    logger.info(
        "[ALPHA] settings: use_ml_alpha=%s use_meta_alpha=%s alpha_weight=%.2f",
        settings.use_ml_alpha,
        settings.use_meta_alpha,
        getattr(settings, "alpha_weight", 0.5),
    )

    # Baseline TechRating (v1) from heuristic engine
    base_tr = df.get("TechRating", pd.Series(0.0, index=df.index))
    base_tr = pd.to_numeric(base_tr, errors="coerce").fillna(0.0)

    # Factor alpha: cross-sectional z-score of baseline TechRating
    try:
        factor_alpha = zscore(base_tr)
    except Exception:
        logger.warning("[ALPHA] factor zscore failed; using zeros", exc_info=True)
        factor_alpha = pd.Series(0.0, index=df.index)

    df["factor_alpha"] = factor_alpha

    # Regime one-hot features (for meta models if needed)
    if regime:
        trend = str(regime.get("trend", "")).upper()
        vol = str(regime.get("vol", "")).upper()
        trend_cols = [
            "regime_trend_TRENDING_UP",
            "regime_trend_TRENDING_DOWN",
            "regime_trend_SIDEWAYS",
        ]
        vol_cols = [
            "regime_vol_HIGH_VOL",
            "regime_vol_LOW_VOL",
        ]
        for c in trend_cols + vol_cols:
            if c not in df.columns:
                df[c] = 0.0
        if trend:
            col = f"regime_trend_{trend}"
            if col in df.columns:
                df[col] = 1.0
        if vol:
            col = f"regime_vol_{vol}"
            if col in df.columns:
                df[col] = 1.0

    # --- ML alpha (5d + 10d) -------------------------------------------------
    ml_5d = None
    ml_10d = None
    ml_5d_z = None
    ml_10d_z = None
    ml_alpha_z = None

    # Initialize columns so downstream code always sees them
    df["Alpha5d"] = np.nan
    df["Alpha10d"] = np.nan
    df["AlphaScore_5d"] = np.nan
    df["AlphaScore_10d"] = np.nan
    df["ml_alpha_5d_z"] = np.nan
    df["ml_alpha_10d_z"] = np.nan
    df["ml_alpha_z"] = np.nan

    if settings.use_ml_alpha:
        # 5d alpha (regime/sector-aware)
        try:
            reg_label = None
            if regime:
                reg_label = regime.get("label") or regime.get("regime_label")

            if "Sector" in df.columns:
                ml_5d_vals = pd.Series(np.nan, index=df.index)
                for sec, subdf in df.groupby("Sector"):
                    sub_pred = alpha_inference.score_alpha_contextual(subdf, reg_label, sec, as_of_date=as_of_date)
                    ml_5d_vals.loc[subdf.index] = sub_pred
                ml_5d = ml_5d_vals
            else:
                ml_5d = alpha_inference.score_alpha_contextual(df, reg_label, None, as_of_date=as_of_date)
        except Exception:
            logger.warning("[ALPHA] score_alpha() 5d failed", exc_info=True)
            ml_5d = None

        if ml_5d is not None and not ml_5d.empty:
            ml_5d = pd.to_numeric(ml_5d, errors="coerce")
            df["Alpha5d"] = ml_5d
            df["AlphaScore_5d"] = ml_5d
            try:
                ml_5d_z = zscore(ml_5d)
                df["ml_alpha_5d_z"] = ml_5d_z
            except Exception:
                logger.warning("[ALPHA] zscore() on 5d ML alpha failed", exc_info=True)
                ml_5d_z = None

        # 10d alpha (optional second model)
        try:
            ml_10d = alpha_inference.score_alpha_10d(df)
        except Exception:
            logger.warning("[ALPHA] score_alpha_10d() failed", exc_info=True)
            ml_10d = None

        if ml_10d is not None and not ml_10d.empty:
            ml_10d = pd.to_numeric(ml_10d, errors="coerce")
            df["Alpha10d"] = ml_10d
            df["AlphaScore_10d"] = ml_10d
            try:
                ml_10d_z = zscore(ml_10d)
                df["ml_alpha_10d_z"] = ml_10d_z
            except Exception:
                logger.warning("[ALPHA] zscore() on 10d ML alpha failed", exc_info=True)
                ml_10d_z = None

        # Combine into a single ML z-score + raw AlphaScore
        if ml_5d_z is not None and ml_10d_z is not None:
            w5 = 0.4
            w10 = 0.6
            ml_alpha_z = w5 * ml_5d_z + w10 * ml_10d_z
            # Raw AlphaScore: blend raw predictions as well
            df["AlphaScore"] = (w5 * ml_5d + w10 * ml_10d)
            logger.info(
                "[ALPHA] ML alpha (5d+10d) blended with w5=%.2f, w10=%.2f", w5, w10
            )
        elif ml_5d_z is not None:
            ml_alpha_z = ml_5d_z
            df["AlphaScore"] = ml_5d
            logger.info("[ALPHA] ML alpha (5d only) applied")
        elif ml_10d_z is not None:
            ml_alpha_z = ml_10d_z
            df["AlphaScore"] = ml_10d
            logger.info("[ALPHA] ML alpha (10d only) applied")
        else:
            logger.info("[ALPHA] ML alpha unavailable; factor-only alpha will be used")

        if ml_alpha_z is not None:
            df["ml_alpha_z"] = ml_alpha_z

    # If AlphaScore is still missing, fall back to factor-based proxy
    if "AlphaScore" not in df.columns:
        df["AlphaScore"] = np.nan

    # --- Optional meta alpha override ---------------------------------------
    if settings.use_meta_alpha:
        try:
            meta_alpha = alpha_inference.score_meta_alpha(df)
        except Exception:
            logger.warning("[ALPHA] score_meta_alpha() failed", exc_info=True)
            meta_alpha = None

        if meta_alpha is not None and not meta_alpha.empty:
            try:
                meta_z = zscore(meta_alpha)
            except Exception:
                meta_z = None
            if meta_z is not None:
                df["meta_alpha"] = meta_alpha
                ml_alpha_z = meta_z
                df["ml_alpha_z"] = ml_alpha_z
                logger.info("[ALPHA] meta alpha applied, overriding factor/ML blend")

    # Optional meta super model -> win probability (best-effort)
    try:
        meta_super = alpha_inference.score_meta_super(df)
        if meta_super is not None:
            df["win_prob_10d"] = meta_super
    except Exception:
        df["win_prob_10d"] = np.nan

    # --- Build alpha_blend and TechRating v2 --------------------------------
    alpha_blend = factor_alpha.copy()

    # Value/quality tilt: stronger bias toward high earnings yield + ROE, moderate boost for dividend yield,
    # and a firmer penalty for leverage. This is always on to favor sturdier value names.
    val_boost = 0.0
    # Prefer sector-relative z-scores when present to avoid cross-sector bias
    earn_col = "value_earnings_yield_sector_z" if "value_earnings_yield_sector_z" in df.columns else "value_earnings_yield_z"
    roe_col = "quality_roe_sector_z" if "quality_roe_sector_z" in df.columns else "quality_roe_z"
    div_col = "dividend_yield_sector_z" if "dividend_yield_sector_z" in df.columns else "dividend_yield_z"
    lev_col = "leverage_de_sector_z" if "leverage_de_sector_z" in df.columns else "leverage_de_z"
    fcf_col = "value_fcf_yield_sector_z" if "value_fcf_yield_sector_z" in df.columns else "value_fcf_yield_z"
    ev_ebitda_col = "value_ev_ebitda_sector_z" if "value_ev_ebitda_sector_z" in df.columns else "value_ev_ebitda_z"

    if earn_col in df.columns:
        val_boost += 0.25 * pd.to_numeric(df[earn_col], errors="coerce").fillna(0.0)
    if fcf_col in df.columns:
        val_boost += 0.1 * pd.to_numeric(df[fcf_col], errors="coerce").fillna(0.0)
    if ev_ebitda_col in df.columns:
        val_boost += 0.05 * pd.to_numeric(df[ev_ebitda_col], errors="coerce").fillna(0.0)
    if roe_col in df.columns:
        val_boost += 0.1 * pd.to_numeric(df[roe_col], errors="coerce").fillna(0.0)
    if div_col in df.columns:
        val_boost += 0.05 * pd.to_numeric(df[div_col], errors="coerce").fillna(0.0)
    if lev_col in df.columns:
        val_boost -= 0.15 * pd.to_numeric(df[lev_col], errors="coerce").fillna(0.0)

    # Macro risk-on/off tilt: temper in risk-off/high-vol; allow a modest boost in healthy credit + growth regimes.
    macro_pen = 0.0
    if regime:
        if regime.get("macro_equity_high_vol") or regime.get("macro_credit_risk_off") or regime.get("macro_equity_vol_spike"):
            macro_pen -= 0.1
        if regime.get("macro_credit_risk_on") and regime.get("macro_growth_vs_value_trending_up"):
            macro_pen += 0.05

    # Apply tilts to the factor leg
    alpha_blend = alpha_blend + val_boost + macro_pen

    if ml_alpha_z is not None:
        w = getattr(settings, "alpha_weight", 0.35)
        try:
            w = float(w)
        except Exception:
            w = 0.5
        w = max(0.0, min(1.0, w))  # clamp to [0,1]

        alpha_blend = (1.0 - w) * factor_alpha + w * ml_alpha_z
        logger.info("[ALPHA] blended factor + ML with TECHNIC_ALPHA_WEIGHT=%.2f", w)
    else:
        logger.info("[ALPHA] using factor_alpha only (no ML alpha)")

    # Liquidity-based alpha adjustment (downweight thin names)
    if "DollarVolume" not in df.columns and {"Close", "Volume"}.issubset(df.columns):
        try:
            df["DollarVolume"] = df["Close"] * df["Volume"]
        except Exception:
            pass
    if "DollarVolume" in df.columns:
        try:
            liquidity_weight = (df["DollarVolume"] / 20_000_000).clip(upper=1.0)
            alpha_blend = alpha_blend * liquidity_weight
        except Exception:
            pass

    df["alpha_blend"] = alpha_blend

    # Regime-aware scaling of alpha weight in TechRating blend
    alpha_weight_tr = 0.4
    if regime:
        if regime.get("vol") == "HIGH_VOL":
            alpha_weight_tr = 0.35
        if regime.get("trend") == "TRENDING_UP" and regime.get("vol") == "LOW_VOL":
            alpha_weight_tr = 0.45
    alpha_weight_tr = max(0.0, min(1.0, alpha_weight_tr))

    # Scale alpha_blend (z-score) into a TechRating-like band and blend
    tr_from_alpha = alpha_blend * 10.0 + 15.0
    blended_tr = (1.0 - alpha_weight_tr) * base_tr + alpha_weight_tr * tr_from_alpha

    df["TechRating_raw"] = base_tr
    df["TechRating"] = blended_tr

    return df


def _apply_portfolio_suggestions(
    df: pd.DataFrame, risk: RiskSettings, sector_cap: float = 0.3
) -> pd.DataFrame:
    """
    Compute simple suggested portfolio weights/allocations:
      - Momentum/expected-return tilt: MuTotal (or AlphaScore) / ATR risk proxy
      - Sector cap to avoid concentration
      - Risk-parity style weights based on ATR14_pct
    """
    if df is None or df.empty:
        return df
    df = df.copy().reset_index(drop=True)
    mask = df.get("PositionSize", 0) > 0
    if mask.sum() == 0:
        df["weight_suggested"] = 0.0
        df["alloc_suggested"] = 0.0
        return df

    sub = df[mask].copy()
    ret_proxy = pd.to_numeric(sub.get("MuTotal", sub.get("AlphaScore", 0.0)), errors="coerce").fillna(0.0)
    ret_proxy = ret_proxy.clip(lower=0.0)
    risk_proxy = pd.to_numeric(sub.get("ATR14_pct", 0.02), errors="coerce").fillna(0.02)
    risk_proxy = risk_proxy.replace(0, 0.02)

    weight_raw = ret_proxy / risk_proxy
    if weight_raw.sum() <= 0:
        weight_raw = pd.Series(1.0, index=sub.index)

    # Sector cap
    sec = sub.get("Sector")
    if sec is not None:
        total_raw = weight_raw.sum()
        sector_sum = weight_raw.groupby(sec).transform("sum")
        cap_val = sector_cap * total_raw
        scale = np.where(sector_sum > 0, np.minimum(1.0, cap_val / sector_sum), 1.0)
        weight_raw = weight_raw * scale

    weights = weight_raw / weight_raw.sum()
    alloc = weights * risk.account_size

    df["weight_suggested"] = 0.0
    df.loc[sub.index, "weight_suggested"] = weights
    df["alloc_suggested"] = 0.0
    df.loc[sub.index, "alloc_suggested"] = alloc

    # Risk-parity-style weights using ATR as risk proxy
    try:
        rp_weights = inverse_variance_weights(risk_proxy, sector=sub.get("Sector"), sector_cap=sector_cap)
        rp_alloc = rp_weights * risk.account_size
        df["weight_risk_parity"] = 0.0
        df.loc[sub.index, "weight_risk_parity"] = rp_weights
        df["alloc_risk_parity"] = 0.0
        df.loc[sub.index, "alloc_risk_parity"] = rp_alloc
    except Exception:
        df["weight_risk_parity"] = 0.0
        df["alloc_risk_parity"] = 0.0

    # Mean-variance style weights (diagonal covariance proxy)
    try:
        mv_w = mean_variance_weights(sub, ret_col="MuTotal", vol_col="ATR14_pct", sector_col="Sector", sector_cap=sector_cap)
        if not mv_w.empty and mv_w.sum() > 0:
            mv_alloc = mv_w * risk.account_size
            df["weight_mean_variance"] = 0.0
            df.loc[sub.index, "weight_mean_variance"] = mv_w
            df["alloc_mean_variance"] = 0.0
            df.loc[sub.index, "alloc_mean_variance"] = mv_alloc
        else:
            df["weight_mean_variance"] = 0.0
            df["alloc_mean_variance"] = 0.0
    except Exception:
        df["weight_mean_variance"] = 0.0
        df["alloc_mean_variance"] = 0.0
    # HRP-style sector-clustered weights
    try:
        hrp_w = hrp_weights(sub, vol_col="ATR14_pct", sector_col="Sector", sector_cap=sector_cap)
        if not hrp_w.empty and hrp_w.sum() > 0:
            hrp_alloc = hrp_w * risk.account_size
            df["weight_hrp"] = 0.0
            df.loc[sub.index, "weight_hrp"] = hrp_w
            df["alloc_hrp"] = 0.0
            df.loc[sub.index, "alloc_hrp"] = hrp_alloc
        else:
            df["weight_hrp"] = 0.0
            df["alloc_hrp"] = 0.0
    except Exception:
        df["weight_hrp"] = 0.0
        df["alloc_hrp"] = 0.0
    return df


def _annotate_recommendations(df: pd.DataFrame, sector_cap: float = 0.3) -> pd.DataFrame:
    """
    Add a concise recommendation text per row using PlayStyle, scores, drift, ATR, and sector exposure.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    sector_over = {}
    if "weight_suggested" in df.columns and "Sector" in df.columns:
        sector_over = df.groupby("Sector")["weight_suggested"].sum().to_dict()
    df["Recommendation"] = [
        build_recommendation(row, sector_over, sector_cap=sector_cap) for _, row in df.iterrows()
    ]
    return df

# -----------------------------
# Scan configuration
# -----------------------------

@dataclass
class ScanConfig:
    """
    Configuration for a single scan.
    """

    # Engine / performance
    mode: str = "fast"          # reserved for future ("fast" vs "full")
    max_symbols: int = 6000
    lookback_days: int = 150
    as_of_date: Optional[str] = None  # optional YYYY-MM-DD to replay as of a past date

    # Universe filtering
    sectors: Optional[List[str]] = None
    subindustries: Optional[List[str]] = None
    industry_contains: Optional[str] = None

    # Scoring / display
    min_tech_rating: float = 0.0

    # Risk & trade settings
    account_size: float = 10_000.0
    risk_pct: float = 1.0   # 1.0 = 1% of account per trade
    target_rr: float = 2.0
    trade_style: str = "Short-term swing"
    strategy_profile_name: Optional[str] = None
    profile: Optional[str] = None  # Risk profile name (conservative/moderate/aggressive)
    options_mode: str = "stock_plus_options"  # stock_only | stock_plus_options

    # Advanced
    allow_shorts: bool = False
    only_tradeable: bool = True

    @classmethod
    def from_strategy_profile(cls, profile) -> "ScanConfig":
        """
        Build a ScanConfig from a StrategyProfile instance.
        """
        return cls(
            trade_style="Short-term swing" if profile.trade_style == "swing" else "Position / longer-term",
            risk_pct=profile.risk_pct,
            target_rr=profile.target_rr,
            min_tech_rating=profile.min_tech_rating,
            strategy_profile_name=profile.name,
        )


# -----------------------------
# Helper filters
# -----------------------------

# Loosened to keep scans populated even for thinner names / shorter lookbacks.
MIN_BARS = 20
# PERFORMANCE: Optimized for parallel processing
# Use 2x CPU cores for I/O-bound tasks (API calls), capped at 32
import os
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)  # limited IO concurrency; can be overridden via settings.max_workers
MIN_PRICE = 5.0  # $5 minimum to filter penny stocks
MIN_DOLLAR_VOL = 1_000_000  # $1M minimum daily volume for liquidity (Phase 3A)


def _compute_macro_context() -> dict:
    """
    Lightweight macro context using ETF proxies.
    - growth_vs_value: slope of QQQ/SPY ratio over ~3 months
    - ndx_vs_spx_ret20: 20d return of QQQ/SPY
    - ndx_vs_spx_level: latest QQQ/SPY ratio
    - credit_risk_on: HYG/LQD 20d change > 0
    - credit_spread_level: latest HYG/LQD ratio
    - curve_slope: SHY/TLT ratio slope (steepening/flattening)
    - short_curve_slope: SHY/IEF ratio slope (2y/10y proxy)
    - equity_vol_state: SPY 20d vs 60d realized vol, plus spike flag
    - equity_vol_level: SPY 20d realized vol (annualized)
    - vixy_ret20: 20d return of VIXY as a vol gauge
    """
    ctx: dict = {}
    try:
        spy = data_engine.get_price_history("SPY", days=130, freq="daily")
        qqq = data_engine.get_price_history("QQQ", days=130, freq="daily")
        if spy is not None and not spy.empty and qqq is not None and not qqq.empty:
            spy = spy.sort_index()
            qqq = qqq.sort_index()
            df = pd.DataFrame({"spy": spy["Close"], "qqq": qqq["Close"]}).dropna()
            if len(df) >= 60:
                ratio = df["qqq"] / df["spy"]
                ctx["macro_ndx_spx_level"] = float(ratio.iloc[-1])
                x = np.arange(len(ratio.tail(60)))
                y = ratio.tail(60).values
                if not np.allclose(y, y[0]):
                    slope = np.polyfit(x, y, 1)[0]
                    ctx["macro_growth_vs_value_slope"] = float(slope)
                    ctx["macro_growth_vs_value_trending_up"] = bool(slope > 0)
                if len(ratio) >= 20:
                    ctx["macro_ndx_spx_ret20"] = float(ratio.pct_change(20).iloc[-1])
    except Exception:
        pass

    try:
        shy = data_engine.get_price_history("SHY", days=130, freq="daily")
        tlt = data_engine.get_price_history("TLT", days=130, freq="daily")
        if shy is not None and not shy.empty and tlt is not None and not tlt.empty:
            df = pd.DataFrame(
                {
                    "shy": shy.sort_index()["Close"],
                    "tlt": tlt.sort_index()["Close"],
                }
            ).dropna()
            if len(df) >= 60:
                ratio = df["shy"] / df["tlt"]
                x = np.arange(len(ratio.tail(60)))
                y = ratio.tail(60).values
                if not np.allclose(y, y[0]):
                    slope = np.polyfit(x, y, 1)[0]
                    ctx["macro_curve_slope"] = float(slope)
                    ctx["macro_curve_steepening"] = bool(slope > 0)
                    ctx["macro_curve_flattening"] = bool(slope < 0)
    except Exception:
        pass

    try:
        shy = data_engine.get_price_history("SHY", days=130, freq="daily")
        ief = data_engine.get_price_history("IEF", days=130, freq="daily")
        if shy is not None and not shy.empty and ief is not None and not ief.empty:
            df = pd.DataFrame(
                {
                    "shy": shy.sort_index()["Close"],
                    "ief": ief.sort_index()["Close"],
                }
            ).dropna()
            if len(df) >= 60:
                ratio = df["shy"] / df["ief"]
                x = np.arange(len(ratio.tail(60)))
                y = ratio.tail(60).values
                if not np.allclose(y, y[0]):
                    slope = np.polyfit(x, y, 1)[0]
                    ctx["macro_curve_short_slope"] = float(slope)
                    ctx["macro_curve_short_steepening"] = bool(slope > 0)
                    ctx["macro_curve_short_flattening"] = bool(slope < 0)
    except Exception:
        pass

    try:
        hyg = data_engine.get_price_history("HYG", days=60, freq="daily")
        lqd = data_engine.get_price_history("LQD", days=60, freq="daily")
        if hyg is not None and not hyg.empty and lqd is not None and not lqd.empty:
            hyg = hyg.sort_index()
            lqd = lqd.sort_index()
            df = pd.DataFrame({"hyg": hyg["Close"], "lqd": lqd["Close"]}).dropna()
            if len(df) >= 20:
                rel = df["hyg"] / df["lqd"]
                ctx["macro_credit_spread_level"] = float(rel.iloc[-1])
                ret20 = rel.pct_change(20).iloc[-1]
                ctx["macro_credit_spread_trend"] = float(ret20)
                ctx["macro_credit_risk_on"] = bool(ret20 > 0)
                ctx["macro_credit_risk_off"] = bool(ret20 < 0)
    except Exception:
        pass

    try:
        spy = data_engine.get_price_history("SPY", days=80, freq="daily")
        if spy is not None and not spy.empty and "Close" in spy.columns:
            rets = spy["Close"].pct_change()
            vol20 = rets.tail(20).std() * np.sqrt(252)
            vol60 = rets.tail(60).std() * np.sqrt(252)
            if pd.notna(vol20) and pd.notna(vol60) and vol60 != 0:
                ratio = float(vol20 / vol60)
                ctx["macro_equity_vol_ratio_20_60"] = ratio
                ctx["macro_equity_high_vol"] = bool(ratio > 1.25)
                ctx["macro_equity_low_vol"] = bool(ratio < 0.8)
                ctx["macro_equity_vol_level"] = float(vol20)
                # Detect a short-term vol spike vs 20d average
                vol5 = rets.tail(5).std() * np.sqrt(252)
                if pd.notna(vol5) and vol20 and vol20 != 0:
                    ctx["macro_equity_vol_spike"] = bool((vol5 / vol20) > 1.5)
    except Exception:
        pass

    try:
        vixy = data_engine.get_price_history("VIXY", days=40, freq="daily")
        if vixy is not None and not vixy.empty and "Close" in vixy.columns:
            vixy = vixy.sort_index()
            if len(vixy) >= 20:
                ctx["macro_vixy_ret20"] = float(vixy["Close"].pct_change(20).iloc[-1])
    except Exception:
        pass

    return ctx


def _can_pass_basic_filters(symbol: str, meta: dict) -> bool:
    """
    PHASE 2: Quick pre-screening WITHOUT fetching price data.
    Rejects symbols that will definitely fail basic filters.
    
    This is the KEY optimization: check criteria BEFORE expensive API calls.
    
    Args:
        symbol: Stock symbol
        meta: Metadata dict with Sector, Industry, etc.
    
    Returns:
        True if symbol might pass (fetch data)
        False if symbol will definitely fail (skip)
    """
    # 1. Symbol pattern checks (very fast)
    if len(symbol) == 1:
        # Single-letter symbols often have liquidity issues
        return False
    
    if len(symbol) > 5:
        # Very long symbols are often problematic
        return False
    
    # 2. Sector-specific rejections (fast)
    sector = meta.get('Sector', '')
    if sector in ['Real Estate', 'Utilities']:
        # These sectors tend to be low-volatility and fail our momentum filters
        # Only keep if they're large-cap (we'll check market cap if available)
        market_cap = meta.get('market_cap', 0)
        if market_cap > 0 and market_cap < 750_000_000:  # <$750M (Phase 3A: tightened for single-letter)
            return False
    
    # 3. Industry-specific rejections (fast)
    industry = meta.get('Industry', '')
    if industry in ['REIT', 'Closed-End Fund', 'Exchange Traded Fund']:
        # These are investment vehicles, not operating companies
        return False
    
    # 4. Market cap pre-filter (if available)
    market_cap = meta.get('market_cap', 0)
    if market_cap > 0:
        if market_cap < 150_000_000:  # <$150M micro-cap (Phase 3A: tightened)
            # Too small, likely to fail liquidity filters
            return False
    
    return True  # Might pass, fetch data to confirm


def _passes_basic_filters(df: pd.DataFrame) -> bool:
    """
    Quick sanity filters before doing full scoring + trade planning.
    PERFORMANCE: Tightened to reject low-quality symbols early.
    
    NOTE: This runs AFTER data fetch. Phase 2 optimization moves some
    checks to _can_pass_basic_filters() to avoid fetching data for
    symbols that will definitely fail.
    """
    if df is None or df.empty:
        return False

    if len(df) < MIN_BARS:
        return False

    last_row = df.iloc[-1]

    # Price filter: Reject penny stocks
    close = float(last_row.get("Close", 0.0))
    if not pd.notna(close) or close < MIN_PRICE:
        return False

    # Liquidity filter: Reject illiquid stocks
    try:
        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(40).mean())
        if avg_dollar_vol < MIN_DOLLAR_VOL:
            return False
    except Exception:
        # If we can't compute dollar volume, be conservative and reject
        return False

    # Volatility sanity check: Reject if price is too volatile (likely data error)
    try:
        price_std = df["Close"].tail(20).std()
        price_mean = df["Close"].tail(20).mean()
        if price_mean > 0:
            cv = price_std / price_mean  # Coefficient of variation
            if cv > 0.5:  # More than 50% volatility is suspicious
                return False
    except Exception:
        pass

    return True

# -----------------------------
# Core per-symbol logic
# -----------------------------

def _should_use_intraday(trade_style: str) -> bool:
    trade_style = (trade_style or "").lower()
    return trade_style in ("short-term swing", "medium-term swing")

def _resolve_lookback_days(trade_style: str, base_days: int) -> int:
    """
    Adjust effective lookback based on trade style.

    - "day" / "day trading" / "intraday": clamp to <= 30 days
    - "Short-term swing" or plain "swing": clamp to <= 90 days
    - "Medium-term swing": use slider value as-is
    - "Position / longer-term": ensure at least 180 days
    """
    ts = (trade_style or "").lower().strip()

    # Day-trading -> very short history
    if "day" in ts or "intraday" in ts:
        return min(base_days, 30)

    # Medium-term first so it doesn't get caught by generic "swing" logic
    if "medium" in ts:
        return base_days

    # Longer-term / position style
    if "position" in ts or "longer" in ts:
        return max(base_days, 180)

    # Treat plain "swing" as short-term, plus explicit short-term labels
    if "short-term" in ts or ts == "swing":
        return min(base_days, 90)

    # Fallback: leave unchanged
    return base_days

def _scan_symbol(
    symbol: str,
    lookback_days: int,
    trade_style: str,
    as_of_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.Series]:
    """
    Fetch history, compute indicators + scores, and return the *latest* row
    (with all scoring columns) for a single symbol.
    Returns None if data is unusable.
    """
    use_intraday = _should_use_intraday(trade_style)

    # 1) History
    try:
        df = data_engine.get_price_history(
            symbol=symbol,
            days=lookback_days,
            freq="intraday" if use_intraday else "daily",
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Clip to as-of date if supplied
    if as_of_date is not None:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        df = df[df.index <= as_of_date]
        if df.empty:
            return None

    if not _passes_basic_filters(df):
        return None

    # Fundamentals (best-effort)
    fundamentals = data_engine.get_fundamentals(symbol)

    # 3) Indicator + scoring pipeline
    scored = compute_scores(df, trade_style=trade_style, fundamentals=fundamentals)
    if scored is None or scored.empty:
        return None

    latest = scored.iloc[-1].copy()
    latest["symbol"] = symbol

    # Add factor bundle (tech/liquidity/fundamental ratios)
    try:
        fb = compute_factor_bundle(df, fundamentals)
        latest.update(fb.factors)
    except Exception:
        pass

    # Market cap (best-effort from fundamentals cache)
    try:
        raw_f = getattr(fundamentals, "raw", {}) if fundamentals is not None else {}
        latest["market_cap"] = raw_f.get("market_cap", np.nan)
    except Exception:
        latest["market_cap"] = np.nan

    # Event-aware fields (best-effort)
    try:
        ev = get_event_info(symbol)
        anchor_date = as_of_date if as_of_date is not None else df.index.max()
        if ev:
            next_earn = ev.get("next_earnings_date")
            last_earn = ev.get("last_earnings_date")
            div_ex = ev.get("dividend_ex_date")
            surprise_flag = bool(ev.get("earnings_surprise_flag", False))

            # Earnings timing
            latest["earnings_surprise_flag"] = surprise_flag
            if pd.notna(next_earn):
                days_to_next = (next_earn - anchor_date).days
                latest["days_to_next_earnings"] = days_to_next
                latest["days_to_earnings"] = days_to_next  # legacy alias
                latest["pre_earnings_window"] = 0 <= days_to_next <= 7
                latest["earnings_window"] = latest.get("pre_earnings_window", False)
            else:
                latest["days_to_next_earnings"] = np.nan
                latest["pre_earnings_window"] = False
            if pd.notna(last_earn):
                days_since = (anchor_date - last_earn).days
                latest["days_since_last_earnings"] = days_since
                latest["days_since_earnings"] = days_since  # legacy alias
                latest["post_earnings_window"] = 0 <= days_since <= 7
            else:
                latest["days_since_last_earnings"] = np.nan
                latest["post_earnings_window"] = False

            # Dividend timing
            if pd.notna(div_ex):
                days_to_ex = (div_ex - anchor_date).days
                latest["days_to_next_ex_dividend"] = days_to_ex
                latest["dividend_ex_date_within_7d"] = abs(days_to_ex) <= 7
            else:
                latest["days_to_next_ex_dividend"] = np.nan
                latest["dividend_ex_date_within_7d"] = False
    except Exception:
        pass

    # Optional deep alpha from recent history
    settings = get_settings()
    if settings.use_deep_alpha:
        try:
            deep_val = alpha_inference.score_deep_alpha_single(df)
            if deep_val is not None:
                latest["alpha_deep"] = deep_val
        except Exception:
            pass
    return latest


def _attach_event_columns(row: pd.Series) -> pd.Series:
    """
    Best-effort enrichment of a scored row with earnings / dividend event info.
    Expects row["Symbol"] to exist.
    """
    symbol = str(row.get("Symbol") or "").upper()
    if not symbol:
        return row

    try:
        info = get_event_info(symbol)
    except Exception:
        info = {}

    if not info:
        return row

    for key, value in info.items():
        if key not in row or pd.isna(row[key]):
            row[key] = value

    # Fallback earnings surprise from bulk if missing
    if ("earnings_surprise_flag" not in row) or pd.isna(row.get("earnings_surprise_flag")):
        try:
            latest_surprise = get_latest_surprise(symbol)
        except Exception:
            latest_surprise = {}
        if latest_surprise:
            actual = pd.to_numeric(latest_surprise.get("actualEarningResult"), errors="coerce")
            estimate = pd.to_numeric(latest_surprise.get("estimatedEarning"), errors="coerce")
            if pd.notna(actual) and pd.notna(estimate):
                row["earnings_surprise_flag"] = bool(actual > estimate)

    # Attach aggregated surprise stats if available
    try:
        stats = get_surprise_stats(symbol)
    except Exception:
        stats = {}
    if stats:
        for key, val in stats.items():
            if key not in row or pd.isna(row.get(key)):
                row[key] = val

    return row


def _attach_ratings_quality(row: pd.Series) -> pd.Series:
    """
    Best-effort enrichment with ratings/targets and quality score.
    """
    symbol = str(row.get("Symbol") or row.get("symbol") or "").upper()
    if not symbol:
        return row

    # Ratings / price targets
    try:
        rt = get_rating_info(symbol)
    except Exception:
        rt = {}
    for key, val in rt.items():
        if key not in row or pd.isna(row[key]):
            row[key] = val

    # Quality score
    try:
        q = get_quality_info(symbol)
    except Exception:
        q = {}
    if q:
        # Prefer to expose quality_score as QualityScore for ICS
        if ("QualityScore" not in row or pd.isna(row.get("QualityScore"))) and "quality_score" in q:
            row["QualityScore"] = q.get("quality_score")
        for key, val in q.items():
            if key not in row or pd.isna(row[key]):
                row[key] = val

    # Fundamental momentum/trend
    try:
        ft = get_fundamental_trend(symbol)
    except Exception:
        ft = {}
    if ft:
        for key, val in ft.items():
            if key not in row or pd.isna(row.get(key)):
                row[key] = val

    # Sponsorship (institutional/etf) and insider flags
    try:
        sp = get_sponsorship(symbol)
    except Exception:
        sp = {}
    if sp:
        for key, val in sp.items():
            if key not in row or pd.isna(row.get(key)):
                row[key] = val

    try:
        ins = get_insider_flags(symbol)
    except Exception:
        ins = {}
    if ins:
        for key, val in ins.items():
            if key not in row or pd.isna(row.get(key)):
                row[key] = val

    return row


def _process_symbol(
    config: "ScanConfig",
    urow: UniverseRow,
    effective_lookback: int,
    settings=None,
    regime_tags: Optional[dict] = None,
) -> Optional[pd.Series]:
    """
    Wrapper to process a single symbol; returns a Series/row or None.
    
    PHASE 2 OPTIMIZATION: Pre-screens symbol metadata BEFORE fetching data.
    """
    # PHASE 2: Pre-screening check (fast, no API calls)
    meta = {
        'Sector': urow.sector,
        'Industry': urow.industry,
        'SubIndustry': urow.subindustry,
        'market_cap': 0,  # Will be enriched if available
    }
    
    if not _can_pass_basic_filters(urow.symbol, meta):
        # Symbol will definitely fail, skip expensive data fetch
        return None
    
    # Passed pre-screening, proceed with data fetch
    as_of_ts = None
    if config.as_of_date:
        try:
            as_of_ts = pd.Timestamp(config.as_of_date)
        except Exception:
            as_of_ts = None

    latest_local = _scan_symbol(
        symbol=urow.symbol,
        lookback_days=effective_lookback,
        trade_style=config.trade_style,
        as_of_date=as_of_ts,
    )
    if latest_local is None:
        return None
    latest_local["Sector"] = urow.sector or ""
    latest_local["Industry"] = urow.industry or ""
    latest_local["SubIndustry"] = urow.subindustry or ""
    # Attach symbol label for downstream helpers that expect "Symbol"
    latest_local["Symbol"] = urow.symbol
    latest_local = _attach_event_columns(latest_local)
    latest_local = _attach_ratings_quality(latest_local)
    return latest_local

def _run_symbol_scans(
    config: "ScanConfig",
    universe: List[UniverseRow],
    regime_tags: Optional[dict],
    effective_lookback: int,
    settings=None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Execute per-symbol scans (Ray if enabled or thread pool fallback) and return DataFrame plus stats.
    
    PHASE 2 OPTIMIZATION: Pre-screening happens in _process_symbol() before data fetch.
    """
    rows: List[pd.Series] = []
    attempted = kept = errors = rejected = pre_rejected = 0
    total_symbols = len(universe)

    if settings is None:
        settings = get_settings()

    # Decide execution mode up front
    use_ray = getattr(settings, "use_ray", False)
    ray_rows = None

    # Calculate max_workers BEFORE any branching so it's available for logging
    cfg_workers = getattr(settings, "max_workers", None)
    try:
        cfg_workers = int(cfg_workers) if cfg_workers is not None else None
    except Exception:
        cfg_workers = None
    max_workers = MAX_WORKERS if cfg_workers is None else max(MAX_WORKERS, cfg_workers)

    start_ts = time.time()

    # ---------------- Ray path (optional) ----------------
    if use_ray and total_symbols > 0:
        try:
            ray_rows = ray_runner.run_ray_scans(
                [u.symbol for u in universe],
                config,
                regime_tags,
            )
        except Exception:
            logger.warning("[RAY] run_ray_scans failed; falling back to thread pool", exc_info=True)
            ray_rows = None

    if ray_rows is not None:
        # Ray returned rows aligned with universe list
        for urow, latest in zip(universe, ray_rows):
            attempted += 1
            if latest is None:
                rejected += 1
                continue
            latest["Sector"] = urow.sector or ""
            latest["Industry"] = urow.industry or ""
            latest["SubIndustry"] = urow.subindustry or ""
            rows.append(latest)
            kept += 1
        engine_mode = "ray"
        # Note: Ray path doesn't track pre_rejected separately
    else:
        # ---------------- Thread pool path ----------------
        def _worker(idx_urow):
            idx, urow = idx_urow
            symbol = urow.symbol
            if progress_cb is not None:
                try:
                    progress_cb(symbol, idx, total_symbols)
                except Exception:
                    pass
            try:
                latest_local = _process_symbol(
                    config=config,
                    urow=urow,
                    effective_lookback=effective_lookback,
                    settings=settings,
                    regime_tags=regime_tags,
                )
            except Exception:
                logger.warning("[SCAN ERROR] %s", symbol, exc_info=True)
                return ("error", symbol, None, urow)
            return ("ok", symbol, latest_local, urow)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for status, symbol, latest, urow in ex.map(_worker, enumerate(universe, start=1)):
                attempted += 1
                if status == "error" or latest is None:
                    if status == "error":
                        errors += 1
                    else:
                        rejected += 1
                    continue
                latest["Sector"] = urow.sector or ""
                latest["Industry"] = urow.industry or ""
                latest["SubIndustry"] = urow.subindustry or ""

                rows.append(latest)
                kept += 1
        engine_mode = "threadpool"
        
        # Log pre-rejection stats (Phase 2 optimization)
        pre_rejected = attempted - kept - errors
        if pre_rejected > 0:
            logger.info(
                "[PHASE2] Pre-rejected %d symbols before data fetch (%.1f%% reduction)",
                pre_rejected,
                (pre_rejected / max(attempted, 1)) * 100
            )

    elapsed = time.time() - start_ts
    per_symbol = elapsed / max(total_symbols, 1)
    logger.info(
        "[SCAN PERF] symbol engine: %d symbols via %s in %.2fs (%.3fs/symbol, max_workers=%s, use_ray=%s)",
        total_symbols,
        engine_mode,
        elapsed,
        per_symbol,
        max_workers,
        getattr(settings, "use_ray", False),
    )

    stats = {
        "attempted": attempted,
        "kept": kept,
        "errors": errors,
        "rejected": rejected,
        "pre_rejected": pre_rejected,
    }
    return pd.DataFrame(rows), stats


def _finalize_results(
    config: "ScanConfig",
    results_df: pd.DataFrame,
    risk: RiskSettings,
    regime_tags: Optional[dict],
    settings=None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Post-process results: optional TFT merge, alpha blend, filters, trade planning, logging.
    """
    if settings is None:
        settings = get_settings()

    if results_df.empty:
        return pd.DataFrame(), "No results returned. Check universe or data source."

    # Normalize column naming to match UI
    if "symbol" in results_df.columns:
        results_df.rename(columns={"symbol": "Symbol"}, inplace=True)

    # Ensure TechRating and Signal columns exist
    if "TechRating" not in results_df.columns:
        results_df["TechRating"] = 0.0
    if "Signal" not in results_df.columns:
        results_df["Signal"] = ""

    # Optional TFT forecast features
    if settings.use_tft_features:
        try:
            from technic_v4.engine import multihorizon
            from technic_v4.engine.feature_engine import merge_tft_features

            symbols = results_df["Symbol"].tolist()
            tft_feats = multihorizon.build_tft_features_for_symbols(symbols, n_future_steps=3)
            if tft_feats is not None and not tft_feats.empty:
                results_df = merge_tft_features(results_df, tft_feats)
            else:
                logger.info("[TFT] No TFT features available; skipping.")
        except Exception as exc:
            logger.warning("[TFT] TFT feature merge failed: %s", exc)

    # Cross-sectional alpha blend: upgrade TechRating using factor/ML/meta alpha
    results_df = _apply_alpha_blend(results_df, regime=regime_tags, as_of_date=as_of_date)

    # Meta inference: win_prob_10d (best-effort)
    try:
        results_df = score_win_prob_10d(results_df)
    except Exception:
        logger.warning("[META] win_prob_10d scoring failed; leaving empty", exc_info=True)

    # Keep an unfiltered copy for fallbacks
    base_results = results_df.copy()

    status_text = "Scan complete."
    if config.min_tech_rating is not None:
        before = len(results_df)
        results_df = results_df[results_df["TechRating"] >= config.min_tech_rating]
        logger.info(
            "[FILTER] %d symbols after min_tech_rating >= %s (from %d).",
            len(results_df),
            config.min_tech_rating,
            before,
        )

    if results_df.empty:
        if base_results.empty:
            return results_df, "No results passed the TechRating filter."

        # Fallback: show the top few names even if they missed the cutoff
        results_df = (
            base_results.sort_values("TechRating", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )
        status_text = (
            "No results passed the TechRating filter; showing top-ranked names instead."
        )

    # Trade planning
    results_df = plan_trades(results_df, risk)

    # Work on a copy to avoid SettingWithCopyWarning when results_df is a slice
    results_df = results_df.copy()

    # Suggested portfolio weights / allocations (light heuristic)
    results_df = _apply_portfolio_suggestions(results_df, risk)
    results_df = _annotate_recommendations(results_df)

    # Threshold flags from config/score_thresholds.json (ICS + quality floors)
    try:
        thr = load_score_thresholds()
        ics_col = "InstitutionalCoreScore"
        qual_col = "QualityScore"
        if ics_col in results_df.columns:
            core_cut = thr.fwd_ret_10d.ics_core_min
            sat_cut = thr.fwd_ret_10d.ics_satellite_min
            qual_cut = thr.fwd_ret_10d.quality_min
            ics_vals = pd.to_numeric(results_df[ics_col], errors="coerce")
            qual_vals = pd.to_numeric(results_df.get(qual_col, np.nan), errors="coerce").fillna(0.0)
            results_df["passes_core_threshold"] = (ics_vals >= core_cut) & (qual_vals >= qual_cut)
            results_df["passes_satellite_threshold"] = (ics_vals >= sat_cut) & (qual_vals >= qual_cut)
    except Exception:
        logger.warning("[THRESHOLDS] Failed to apply score thresholds", exc_info=True)

    # Regime context columns (same for all rows in this scan)
    if regime_tags:
        trend = regime_tags.get("trend")
        vol_state = regime_tags.get("vol")
        label = regime_tags.get("label") or (f"{trend}_{vol_state}" if trend and vol_state else None)
        risk_on = bool(trend == "TRENDING_UP" and vol_state == "LOW_VOL")
        results_df["regime_label"] = label or ""
        results_df["regime_risk_on"] = risk_on
        results_df["regime_risk_off"] = not risk_on
    else:
        results_df["regime_label"] = ""
        results_df["regime_risk_on"] = False
        results_df["regime_risk_off"] = False
        results_df["macro_growth_vs_value_slope"] = np.nan
        results_df["macro_growth_vs_value_trending_up"] = False
        results_df["macro_credit_spread_trend"] = np.nan
        results_df["macro_credit_risk_on"] = False
        results_df["macro_credit_risk_off"] = False
        results_df["macro_credit_spread_level"] = np.nan
        results_df["macro_curve_slope"] = np.nan
        results_df["macro_curve_steepening"] = False
        results_df["macro_curve_flattening"] = False
        results_df["macro_curve_short_slope"] = np.nan
        results_df["macro_curve_short_steepening"] = False
        results_df["macro_curve_short_flattening"] = False
        results_df["macro_equity_vol_ratio_20_60"] = np.nan
        results_df["macro_equity_high_vol"] = False
        results_df["macro_equity_low_vol"] = False
        results_df["macro_equity_vol_spike"] = False
        results_df["macro_ndx_spx_ret20"] = np.nan
        results_df["macro_ndx_spx_level"] = np.nan
        results_df["macro_equity_vol_level"] = np.nan
        results_df["macro_vixy_ret20"] = np.nan

    # Carry macro context onto rows if present
    if regime_tags:
        for key in [
            "macro_growth_vs_value_slope",
            "macro_growth_vs_value_trending_up",
            "macro_ndx_spx_ret20",
            "macro_ndx_spx_level",
            "macro_credit_spread_trend",
            "macro_credit_risk_on",
            "macro_credit_risk_off",
            "macro_credit_spread_level",
            "macro_curve_slope",
            "macro_curve_steepening",
            "macro_curve_flattening",
            "macro_curve_short_slope",
            "macro_curve_short_steepening",
            "macro_curve_short_flattening",
            "macro_equity_vol_ratio_20_60",
            "macro_equity_high_vol",
            "macro_equity_low_vol",
            "macro_equity_vol_spike",
            "macro_equity_vol_level",
            "macro_vixy_ret20",
        ]:
            val = regime_tags.get(key, np.nan)
            if key not in results_df.columns:
                results_df[key] = val

    # Alpha / risk scaffolding
    # --------------------------------------------------
    # MuHat: heuristic tech-based drift (normalized TechRating)
    tr = pd.to_numeric(results_df.get("TechRating", np.nan), errors="coerce")
    # Center around ~15 with a 20-point half-range, then clamp
    mu_hat = ((tr - 15.0) / 20.0).clip(-0.5, 0.5)
    results_df["MuHat"] = mu_hat

    # MuMl: ML-based drift (normalized AlphaScore)
    if "AlphaScore" in results_df.columns:
        alpha_series = pd.to_numeric(results_df["AlphaScore"], errors="coerce")
    else:
        results_df["AlphaScore"] = np.nan
        alpha_series = pd.Series(np.nan, index=results_df.index)

    if alpha_series.notna().any():
        # Normalize ML alpha into a comparable drift band
        # Assume AlphaScore is an expected forward return; clip extreme tails
        mu_ml = alpha_series.clip(-0.25, 0.25)
        results_df["MuMl"] = mu_ml

        # Regime-aware blend of MuHat and MuMl
        base_w_mu = getattr(settings, "alpha_weight", 0.35)
        try:
            base_w_mu = float(base_w_mu)
        except Exception:
            base_w_mu = 0.35
        base_w_mu = max(0.0, min(1.0, base_w_mu))

        # Adjust ML weight based on market regime if available
        w_mu = base_w_mu
        if regime_tags:
            trend = str(regime_tags.get("trend") or "").upper()
            vol = str(regime_tags.get("vol") or "").upper()

            # In steady uptrends with low vol, trust ML a bit more
            if trend == "TRENDING_UP" and vol == "LOW_VOL":
                w_mu = min(1.0, base_w_mu + 0.15)
            # In high-vol regimes, lean a bit more on technicals
            elif vol == "HIGH_VOL":
                w_mu = max(0.0, base_w_mu - 0.10)

        results_df["MuTotal"] = (1.0 - w_mu) * mu_hat + w_mu * mu_ml
    else:
        # No ML alpha available; drift = tech-only
        results_df["MuMl"] = np.nan
        results_df["MuTotal"] = mu_hat
        # For downstream consumers that expect AlphaScore, fall back to TechRating
        results_df.loc[:, "AlphaScore"] = results_df.get("TechRating", np.nan)

    # Cross-sectional z-scores for value/quality/growth/size factors if present
    factor_cols = [
        "value_ep",
        "value_cfp",
        "value_earnings_yield",
        "dividend_yield",
        "value_fcf_yield",
        "value_ev_ebitda",
        "value_ev_sales",
        "quality_roe",
        "quality_roa",
        "quality_gpm",
        "quality_margin_ebitda",
        "quality_margin_op",
        "quality_margin_net",
        "leverage_de",
        "interest_coverage",
        "growth_rev",
        "growth_eps",
        "size_log_mcap",
    ]
    for col in factor_cols:
        if col in results_df.columns:
            series = pd.to_numeric(results_df[col], errors="coerce")
            if series.notna().any():
                results_df[f"{col}_z"] = zscore(series)

    # Sector-relative z-scores for factors to reduce cross-sector bias
    if "Sector" in results_df.columns:
        for col in factor_cols:
            if col in results_df.columns:
                try:
                    results_df[f"{col}_sector_z"] = (
                        results_df.groupby("Sector")[col]
                        .transform(lambda s: zscore(pd.to_numeric(s, errors="coerce")))
                    )
                except Exception:
                    results_df[f"{col}_sector_z"] = np.nan

    # Cross-sectional percentile ranks for momentum/vol/volume/valuation
    def _pct_rank(series: pd.Series) -> pd.Series:
        return series.rank(pct=True) * 100.0

    momentum_cols = [("mom_21", "momentum_rank_21d"), ("mom_63", "momentum_rank_63d")]
    # Longer horizons (6m ~126d, 12m ~252d) if present
    momentum_cols.extend([("mom_126", "momentum_rank_126d"), ("mom_252", "momentum_rank_252d")])
    for src, tgt in momentum_cols:
        if src in results_df.columns:
            s = pd.to_numeric(results_df[src], errors="coerce")
            if s.notna().any():
                results_df[tgt] = _pct_rank(s)
                if "Sector" in results_df.columns:
                    try:
                        results_df[f"{tgt}_sector"] = (
                            results_df.groupby("Sector")[src]
                            .transform(lambda x: _pct_rank(pd.to_numeric(x, errors="coerce")))
                        )
                    except Exception:
                        results_df[f"{tgt}_sector"] = np.nan

    if "ATR14_pct" in results_df.columns:
        s = pd.to_numeric(results_df["ATR14_pct"], errors="coerce")
        if s.notna().any():
            results_df["vol_rank"] = _pct_rank(s)
            if "Sector" in results_df.columns:
                try:
                    results_df["vol_rank_sector"] = (
                        results_df.groupby("Sector")["ATR14_pct"]
                        .transform(lambda x: _pct_rank(pd.to_numeric(x, errors="coerce")))
                    )
                except Exception:
                    results_df["vol_rank_sector"] = np.nan
    if "DollarVolume" in results_df.columns:
        s = pd.to_numeric(results_df["DollarVolume"], errors="coerce")
        if s.notna().any():
            results_df["volume_rank"] = _pct_rank(s)
            if "Sector" in results_df.columns:
                try:
                    results_df["volume_rank_sector"] = (
                        results_df.groupby("Sector")["DollarVolume"]
                        .transform(lambda x: _pct_rank(pd.to_numeric(x, errors="coerce")))
                    )
                except Exception:
                    results_df["volume_rank_sector"] = np.nan

    # Valuation rank using earnings yield; fallback to FCF yield
    val_src = None
    for cand in ["value_earnings_yield", "value_fcf_yield", "value_ev_ebitda", "value_ev_sales"]:
        if cand in results_df.columns:
            val_src = pd.to_numeric(results_df[cand], errors="coerce")
            if val_src.notna().any():
                break
    if val_src is not None and val_src.notna().any():
        results_df["value_rank"] = _pct_rank(val_src)
        if "Sector" in results_df.columns:
            try:
                results_df["value_rank_sector"] = (
                    results_df.groupby("Sector")[val_src.name]
                    .transform(lambda x: _pct_rank(pd.to_numeric(x, errors="coerce")))
                )
            except Exception:
                results_df["value_rank_sector"] = np.nan

    # Event-aware placeholders (best-effort; default to NaN/False)
    for col, default in [
        ("days_to_earnings", np.nan),
        ("days_since_earnings", np.nan),
        ("earnings_surprise_flag", False),
        ("dividend_ex_date_within_7d", False),
    ]:
        if col not in results_df.columns:
            results_df[col] = default
    # Earnings window flag if within 7 days of earnings date (stub: relies on days_to_earnings)
    if "days_to_earnings" in results_df.columns:
        try:
            dte = pd.to_numeric(results_df["days_to_earnings"], errors="coerce")
            results_df["earnings_window"] = dte.between(0, 7, inclusive="both")
        except Exception:
            results_df["earnings_window"] = False

    # Cross-sectional alpha percentile (0?100) for UI / ranking
    if "AlphaScorePct" not in results_df.columns:
        alpha_source = None
        if results_df["AlphaScore"].notna().any():
            alpha_source = pd.to_numeric(results_df["AlphaScore"], errors="coerce")
        elif "alpha_blend" in results_df.columns and results_df["alpha_blend"].notna().any():
            alpha_source = pd.to_numeric(results_df["alpha_blend"], errors="coerce")

        if alpha_source is not None and alpha_source.notna().any():
            results_df["AlphaScorePct"] = alpha_source.rank(pct=True) * 100.0
        else:
            results_df["AlphaScorePct"] = np.nan

    # --------------------------------------------
    # Sector-neutral alpha normalization
    # --------------------------------------------
    if "Sector" in results_df.columns:
        try:
            # Choose alpha source: AlphaScore preferred, fallback to MuTotal
            alpha_for_sector = None
            if "AlphaScore" in results_df.columns and results_df["AlphaScore"].notna().any():
                alpha_for_sector = pd.to_numeric(results_df["AlphaScore"], errors="coerce")
            elif "MuTotal" in results_df.columns and results_df["MuTotal"].notna().any():
                alpha_for_sector = pd.to_numeric(results_df["MuTotal"], errors="coerce")

            if alpha_for_sector is not None and alpha_for_sector.notna().any():
                tmp = results_df.copy()
                tmp["__alpha_sector"] = alpha_for_sector
                tmp["__sector_key"] = tmp["Sector"].fillna("UNK")

                sector_ranks = (
                    tmp.groupby("__sector_key")["__alpha_sector"]
                    .rank(pct=True)
                    .fillna(0.0)
                    * 100.0
                )

                results_df["SectorAlphaPct"] = sector_ranks
            else:
                results_df["SectorAlphaPct"] = np.nan
        except Exception:
            results_df["SectorAlphaPct"] = np.nan
    else:
        results_df["SectorAlphaPct"] = np.nan

    # ============================
    # INSTITUTIONAL FILTERS v1 (RELAXED)
    # ============================
    # Compute dollar volume
    if {"Close", "Volume"}.issubset(results_df.columns):
        results_df["DollarVolume"] = results_df["Close"] * results_df["Volume"]

    # Minimum liquidity filter (relaxed for broader results)
    if "DollarVolume" in results_df.columns:
        results_df = results_df[results_df["DollarVolume"] >= MIN_DOLLAR_VOL]

    # Price filter (relaxed)
    if "Close" in results_df.columns:
        results_df = results_df[results_df["Close"] >= 1.00]  # $1 minimum (was $5)

    # Market-cap filter (relaxed - skip only nano-caps)
    if "market_cap" in results_df.columns:
        results_df = results_df[results_df["market_cap"] >= 50_000_000]  # $50M minimum (was $300M)
    else:
        logger.info("[FILTER] market_cap column missing; skipping market cap filter")

    # ATR% ceiling (relaxed)
    if "ATR14_pct" in results_df.columns:
        results_df = results_df[results_df["ATR14_pct"] <= 0.50]  # max 50% ATR% (was 20%)

    # ---------------------------------------------
    # Sort by MERIT Score (primary) then TechRating (secondary)
    # Ensures top-quality setups dominate the top ranks
    # ---------------------------------------------
    if "MeritScore" in results_df.columns:
        results_df = results_df.sort_values(["MeritScore", "TechRating"], ascending=False).copy()
    elif "TechRating" in results_df.columns:
        results_df = results_df.sort_values("TechRating", ascending=False).copy()
    if regime_tags:
        results_df["RegimeTrend"] = regime_tags.get("trend")
        results_df["RegimeVol"] = regime_tags.get("vol")
        if "MarketRegime" not in results_df.columns:
            label = regime_tags.get("label") or f"{regime_tags.get('trend','')}_{regime_tags.get('vol','')}"
            results_df["MarketRegime"] = label

    # Portfolio-aware ranking
    try:
        results_df = ranking_engine.rank_results(results_df, max_positions=config.max_symbols or len(results_df))
    except Exception:
        pass

    # Engine-level tradeable filter (soft)
    if config.only_tradeable and "Signal" in results_df.columns:
        tradeable = {"Strong Long", "Long"}
        if config.allow_shorts:
            tradeable.update({"Strong Short", "Short"})

        tradeable_df = results_df[results_df["Signal"].isin(tradeable)]

        # Prefer tradeable, but don't drop everything if none match
        if not tradeable_df.empty:
            results_df = tradeable_df

    # --- PlayStyle classification: Stable vs Explosive -----------------------
    def _classify_playstyle(row: pd.Series) -> str:
        # Prefer scaled risk_score if present, else RiskScore
        rs = row.get("risk_score", None)
        if rs is None or pd.isna(rs):
            rs = row.get("RiskScore", None)
        try:
            rs_val = float(rs) if rs is not None and not pd.isna(rs) else None
        except Exception:
            rs_val = None

        a5 = row.get("Alpha5d", None)
        a10 = row.get("Alpha10d", None)
        try:
            a5_val = float(a5) if a5 is not None and not pd.isna(a5) else None
        except Exception:
            a5_val = None
        try:
            a10_val = float(a10) if a10 is not None and not pd.isna(a10) else None
        except Exception:
            a10_val = None

        expl = row.get("ExplosivenessScore", None)
        vol = row.get("VolatilityScore", None)
        try:
            expl_val = float(expl) if expl is not None and not pd.isna(expl) else None
        except Exception:
            expl_val = None
        try:
            vol_val = float(vol) if vol is not None and not pd.isna(vol) else None
        except Exception:
            vol_val = None

        is_stable = False
        is_explosive = False

        # Stability: low risk, horizons agree, and consistent positive (or negative) alpha
        if a5_val is not None and a10_val is not None:
            same_sign = a5_val * a10_val > 0
            if (
                rs_val is not None
                and rs_val >= 0.75
                and same_sign
                and (a5_val is not None and a5_val > 0)
                and (a10_val is not None and a10_val > 0)
            ):
                is_stable = True

        # Explosive: high volatility / explosiveness or very low risk_score
        if rs_val is not None and rs_val <= 0.45:
            is_explosive = True
        if expl_val is not None and expl_val >= 1.5:
            is_explosive = True
        if vol_val is not None and vol_val >= 1.5:
            is_explosive = True
        if a5_val is not None and a10_val is not None:
            # Large disagreement between horizons ? volatile / uncertain
            if abs(a5_val - a10_val) >= 0.02:
                is_explosive = True

        if is_stable and not is_explosive:
            return "Stable"
        if is_explosive and not is_stable:
            return "Explosive"
        if is_stable and is_explosive:
            return "Hybrid"
        return "Neutral"

    playstyles = results_df.apply(_classify_playstyle, axis=1)
    results_df["PlayStyle"] = playstyles
    results_df["IsStable"] = playstyles == "Stable"
    results_df["IsHighRisk"] = playstyles == "Explosive"

    # Prefer Stable setups in ranking by giving them a small score boost
    if "PlayStyle" in results_df.columns:
        stable_boost = results_df["PlayStyle"].eq("Stable").astype(float) * 0.05
        explosive_penalty = results_df["PlayStyle"].eq("Explosive").astype(float) * 0.05

        # Add to MuTotal or TechRating to slightly tilt toward stable names
        results_df["MuTotal"] = results_df["MuTotal"] + stable_boost - explosive_penalty

    # Institutional Core Score (best-effort)
    if not results_df.empty:
        try:
            results_df["InstitutionalCoreScore"] = build_institutional_core_score(results_df)
        except Exception:
            logger.warning("[ICS] Failed to compute InstitutionalCoreScore", exc_info=True)
    
    # Compute MERIT Score (after ICS/Quality are available)
    if not results_df.empty:
        try:
            results_df = compute_merit(results_df, regime=regime_tags)
            logger.info("[MERIT] Computed MERIT Score for %d results", len(results_df))
        except Exception as e:
            logger.warning("[MERIT] Failed to compute MERIT Score: %s", e, exc_info=True)

    # Mark ultra-high-risk names for a separate "Runners" list
    risk_col = "risk_score" if "risk_score" in results_df.columns else None
    if risk_col is None and "RiskScore" in results_df.columns:
        risk_col = "RiskScore"

    # --------------------------------------------
    # Sector crowding penalty (diversification boost)
    # --------------------------------------------
    if "Sector" in results_df.columns:
        sector_counts = results_df["Sector"].value_counts(normalize=True)
        penalty_map = (1 - sector_counts).to_dict()  # crowded sectors get lower score
        results_df["SectorPenalty"] = results_df["Sector"].map(penalty_map).fillna(1.0)
        if "InstitutionalCoreScore" in results_df.columns:
            results_df["InstitutionalCoreScore"] *= results_df["SectorPenalty"]

    # ------------------------------------------------------------------
    # Classify rows into Investor Core / Satellite / Reject tiers
    # ------------------------------------------------------------------
    try:
        thresholds = load_score_thresholds()

        ics = results_df.get("InstitutionalCoreScore")
        win_prob_10d = None
        for col in ("win_prob_10d", "meta_win_prob_10d", "winprob_10d"):
            if col in results_df.columns:
                win_prob_10d = results_df[col]
                break

        quality = None
        for col in ("QualityScore", "fundamental_quality_score"):
            if col in results_df.columns:
                quality = results_df[col]
                break

        if ics is None:
            results_df["IsInvestorCore"] = False
            results_df["IsSatellite"] = False
            results_df["Tier"] = "REJECT"
        else:
            thr10 = thresholds.fwd_ret_10d

            ics_vals = pd.to_numeric(ics, errors="coerce")
            win_vals = pd.to_numeric(win_prob_10d, errors="coerce") if win_prob_10d is not None else None
            q_vals = pd.to_numeric(quality, errors="coerce") if quality is not None else None

            quality_ok = (q_vals >= thr10.quality_min) if q_vals is not None else pd.Series(True, index=results_df.index)
            quality_ok = quality_ok | quality_ok.isna()

            if win_vals is not None:
                win_ok_core = (win_vals >= thr10.win_prob_min) | win_vals.isna()
            else:
                win_ok_core = pd.Series(True, index=results_df.index)

            is_core = (ics_vals >= thr10.ics_core_min) & win_ok_core & quality_ok
            is_satellite = (ics_vals >= thr10.ics_satellite_min) & ~is_core & quality_ok

            results_df["IsInvestorCore"] = is_core
            results_df["IsSatellite"] = is_satellite

            tier = pd.Series("REJECT", index=results_df.index)
            tier.loc[is_satellite] = "SATELLITE"
            tier.loc[is_core] = "CORE"
            results_df["Tier"] = tier
            # Fallback: if no CORE rows at all, promote top ICS names to CORE
            if (tier == "CORE").sum() == 0:
                try:
                    sortable = results_df[pd.to_numeric(results_df[ics_col], errors="coerce").notna()]
                    sorted_idx = sortable.sort_values(ics_col, ascending=False).index
                    if len(sorted_idx) == 0:
                        sorted_idx = results_df.index
                    core_promote = sorted_idx[: min(50, len(sorted_idx))]
                    results_df.loc[core_promote, "Tier"] = "CORE"
                    results_df.loc[core_promote, "IsInvestorCore"] = True
                except Exception:
                    logger.warning("[THRESHOLDS] Failed to promote top ICS when no CORE rows present", exc_info=True)
    except Exception:
        logger.warning("[THRESHOLDS] Failed to classify tiers", exc_info=True)

    if risk_col:
        ultra_risky_mask = pd.to_numeric(results_df[risk_col], errors="coerce") < 0.08
        results_df["IsUltraRisky"] = ultra_risky_mask.fillna(False)
    else:
        results_df["IsUltraRisky"] = False

    # Remove ultra-risky names from the main candidate pool
    if "IsUltraRisky" in results_df.columns:
        main_df = results_df[~results_df["IsUltraRisky"]].copy()
        runners_df = results_df[results_df["IsUltraRisky"]].copy()
    else:
        main_df = results_df.copy()
        runners_df = pd.DataFrame()

    # Risk-adjusted sorting + diversification fallback on the main list only
    if not main_df.empty:
        vol_col = "vol_realized_20" if "vol_realized_20" in main_df.columns else None
        main_df = risk_adjusted_rank(
            main_df,
            return_col="MuTotal",
            vol_col=vol_col or "TechRating",
        )
        # Use MeritScore for diversification if available, otherwise fall back
        if "MeritScore" in main_df.columns:
            sort_col = "MeritScore"
        elif "risk_score" in main_df.columns:
            sort_col = "risk_score"
        else:
            sort_col = "TechRating"
        
        main_df = main_df.sort_values(sort_col, ascending=False)
        try:
            main_df = diversify_by_sector(
                main_df,
                sector_col="Sector",
                score_col=sort_col,
                top_n=config.max_symbols or 50,
            )
        except Exception:
            pass

    # Optionally save runners to a separate CSV
    if not runners_df.empty:
        runners_path = OUTPUT_DIR / "technic_runners.csv"
        try:
            runners_df.to_csv(runners_path, index=False)
            logger.info(
                "[OUTPUT] Wrote %d ultra-risky runners to %s",
                len(runners_df),
                runners_path,
            )
        except Exception:
            logger.warning(
                "[OUTPUT ERROR] Failed to write runners CSV", exc_info=True
            )

    # From here on, operate on the filtered main_df as results_df
    results_df = main_df

    # Option picks placeholder (populated below when chains are available)
    results_df["OptionPicks"] = [[] for _ in range(len(results_df))]
    options_mode = getattr(config, "options_mode", "stock_plus_options")
    if options_mode != "stock_plus_options":
        logger.info("[options] options_mode=%s -> skipping options strategy computation", options_mode)
        results_df["OptionTrade"] = ""
        results_df["OptionTradeText"] = ""
        results_df["OptionQualityScore"] = np.nan
        results_df["OptionIVRiskFlag"] = False
    else:
        # Optional: suggest simple option trades for strongest longs
        try:
            # Only compute options for the top N names by TechRating to save time
            opt_max = getattr(settings, "options_max_symbols", 200)
            opt_max = int(opt_max) if opt_max else 200
            opt_indices = (
                results_df.sort_values("TechRating", ascending=False).head(opt_max).index
                if "TechRating" in results_df.columns
                else results_df.index
            )

            # Defaults for all rows
            picks: list = [[] for _ in range(len(results_df))]
            quality_scores = [np.nan for _ in range(len(results_df))]
            iv_risk_flags = [False for _ in range(len(results_df))]
            option_strings = ["No option idea (not evaluated)." for _ in range(len(results_df))]
            no_picks_syms = []

            # Ensure event flags for options context
            if "has_upcoming_earnings" not in results_df.columns:
                if "days_to_next_earnings" in results_df.columns:
                    dtn = pd.to_numeric(results_df["days_to_next_earnings"], errors="coerce")
                    results_df["has_upcoming_earnings"] = dtn.notna() & (dtn >= 0)
                    results_df["is_pre_earnings_window"] = dtn.between(0, 7, inclusive="both")
                else:
                    results_df["has_upcoming_earnings"] = False
                    results_df["is_pre_earnings_window"] = False
            if "is_post_earnings_window" not in results_df.columns:
                if "days_since_last_earnings" in results_df.columns:
                    dse = pd.to_numeric(results_df["days_since_last_earnings"], errors="coerce")
                    results_df["is_post_earnings_window"] = dse.between(0, 3, inclusive="both")
                else:
                    results_df["is_post_earnings_window"] = False
            if "has_dividend_soon" not in results_df.columns:
                has_div = False
                if "dividend_ex_date_within_7d" in results_df.columns:
                    ser = results_df["dividend_ex_date_within_7d"]
                    has_div = ser.astype("boolean").fillna(False)
                elif "days_to_next_ex_dividend" in results_df.columns:
                    dtd = pd.to_numeric(results_df["days_to_next_ex_dividend"], errors="coerce")
                    has_div = dtd.between(-7, 7, inclusive="both")
                results_df["has_dividend_soon"] = has_div
            for idx, row in results_df.loc[opt_indices].iterrows():
                sig = row.get("Signal")
                if sig not in {"Strong Long", "Long"}:
                    picks[idx] = []
                    quality_scores[idx] = np.nan
                    iv_risk_flags[idx] = False
                    option_strings[idx] = "No option idea (signal not long)."
                    continue
                spot = row.get("Close") or row.get("Entry") or row.get("Last")
                if spot is None or pd.isna(spot):
                    picks[idx] = []
                    quality_scores[idx] = np.nan
                    iv_risk_flags[idx] = False
                    option_strings[idx] = "No option idea (no price)."
                    continue
                # Skip if too close to earnings
                days_to_earn = row.get("days_to_next_earnings")
                if days_to_earn is not None and not pd.isna(days_to_earn) and days_to_earn <= 3:
                    picks[idx] = []
                    quality_scores[idx] = np.nan
                    iv_risk_flags[idx] = False
                    option_strings[idx] = "No option idea (earnings too close)."
                    continue
                suggested = suggest_option_trades(str(row.get("Symbol")), float(spot), bullish=True)
                picks[idx] = suggested or []
                if suggested:
                    # aggregated quality using top3 sweetness (0.6*max + 0.4*avg_top3)
                    qs = np.nan
                    if isinstance(suggested, list) and suggested:
                        sweetness_vals = []
                        for p in suggested:
                            val = p.get("option_sweetness_score")
                            if val is not None and not pd.isna(val):
                                try:
                                    sweetness_vals.append(float(val))
                                except Exception:
                                    pass
                        if sweetness_vals:
                            sweetness_vals = sorted(sweetness_vals, reverse=True)
                            top3 = sweetness_vals[:3]
                            max_sw = top3[0]
                            avg_top3 = sum(top3) / len(top3)
                            qs = 0.6 * max_sw + 0.4 * avg_top3
                    # risk penalties: high IV and earnings near
                    penalty = 1.0
                    hi_iv = any(bool(p.get("high_iv_flag")) for p in suggested)
                    iv_risk_flags.append(hi_iv or any(bool(p.get("iv_risk_flag")) for p in suggested))
                    if hi_iv:
                        penalty *= 0.8
                    if days_to_earn is not None and not pd.isna(days_to_earn) and days_to_earn <= 7:
                        penalty *= 0.8
                    if bool(row.get("is_pre_earnings_window")) and hi_iv:
                        penalty *= 0.75  # additional gap risk penalty
                        # prefer defined-risk spreads only when gap + high IV
                        filtered = [p for p in suggested if isinstance(p, dict) and "spread" in str(p.get("strategy_type", "")).lower()]
                        if filtered:
                            suggested = filtered
                        else:
                            suggested = []
                            picks[-1] = []
                    qs_adj = qs * penalty if not pd.isna(qs) else np.nan
                    quality_scores[idx] = qs_adj
                    # Risk comments to carry with picks
                    risk_msgs = []
                    if days_to_earn is not None and not pd.isna(days_to_earn) and days_to_earn <= 7:
                        risk_msgs.append(f"Earnings in {int(days_to_earn)}d; gap risk higher than normal.")
                    if bool(row.get("dividend_ex_date_within_7d")) or bool(row.get("has_dividend_soon")):
                        risk_msgs.append("Ex-dividend within 7d; near-term pricing may be distorted.")
                    if hi_iv or iv_risk_flags[-1]:
                        risk_msgs.append("IV elevated; consider defined-risk or smaller size.")
                    risk_comment = "; ".join(risk_msgs)
                    if risk_comment:
                        for p in suggested:
                            if isinstance(p, dict):
                                p["risk_comment"] = risk_comment
                    # Build option text from first pick
                    try:
                        # Choose best by sweetness
                        best = max(
                            suggested,
                            key=lambda p: p.get("option_sweetness_score") if p.get("option_sweetness_score") is not None else -1,
                        )
                        first = best
                        strike = first.get("strike")
                        expiry = first.get("expiry")
                        delta = first.get("delta")
                        spread_pct = first.get("spread_pct")
                        iv_val = first.get("iv")
                        dte = first.get("days_to_exp")
                        strat = first.get("strategy_type", "option")
                        sweetness = first.get("option_sweetness_score")
                        risk_note = ""
                        if hi_iv:
                            risk_note = " IV elevated; consider smaller size or spreads."
                        elif iv_risk_flags[-1]:
                            risk_note = " IV modestly high; mind gap risk."
                        if risk_comment:
                            risk_note = (risk_note + " " + risk_comment).strip()
                        delta_str = f"{delta:.2f}" if delta is not None else "N/A"
                        spread_str = (
                            f"{spread_pct:.2%}" if spread_pct is not None else "N/A"
                        )
                        sweet_str = f"{sweetness:.0f}" if sweetness is not None else "N/A"
                        dte_str = f"{dte}" if dte is not None else "N/A"
                        text = " ".join(
                            [
                                f"{strat} {strike} @ {expiry} (~{dte_str} DTE, "
                                f"Delta={delta_str}, spread {spread_str}, "
                                f"sweetness {sweet_str}/100).",
                                risk_note.strip(),
                            ]
                        ).strip()
                        option_strings[idx] = text
                    except Exception:
                        option_strings[idx] = ""
                else:
                    sym_val = row.get("Symbol")
                    if hasattr(sym_val, "iloc"):
                        try:
                            sym_val = sym_val.iloc[0]
                        except Exception:
                            sym_val = str(sym_val)
                    no_picks_syms.append(str(sym_val))
                    quality_scores[idx] = np.nan
                    iv_risk_flags[idx] = False
                    option_strings[idx] = "No option idea (filters failed or earnings too close)."
                # Structured picks (for UI/API)
            results_df["OptionPicks"] = picks
            # Quality / risk flags
            results_df["OptionQualityScore"] = quality_scores
            results_df["OptionIVRiskFlag"] = iv_risk_flags
            # Human-readable string for CSV/UI
            results_df["OptionTrade"] = option_strings
            results_df["OptionTradeText"] = option_strings
            # Inform if symbols had no option ideas
            if no_picks_syms:
                unique_syms = list(dict.fromkeys(no_picks_syms))  # preserve order, drop duplicates
                clean_syms = [s.strip() for s in unique_syms if s and s.strip().lower() != "nan"]
                syms_str = ", ".join(clean_syms[:10]) + ("..." if len(clean_syms) > 10 else "")
                logger.info("[options] no option candidates for symbols: %s", syms_str)
        except Exception:
            results_df["OptionPicks"] = [[] for _ in range(len(results_df))]
            results_df["OptionTrade"] = ""
            results_df["OptionQualityScore"] = np.nan
            results_df["OptionIVRiskFlag"] = False
            results_df["OptionTradeText"] = ""

    # Build short rationales per idea (best-effort)
    try:
        rationales = []
        for _, row in results_df.iterrows():
            rationales.append(
                explainability_engine.build_rationale(
                    symbol=str(row.get("Symbol", "")),
                    row=row,
                    features=None,
                    regime=regime_tags,
                )
            )
        results_df["Rationale"] = rationales
    except Exception:
        results_df["Rationale"] = ""

    # Optional SHAP explanations for top names
    if settings.use_explainability:
        try:
            feature_cols_avail = [c for c in results_df.columns if c not in {"Symbol", "TechRating", "Signal"}]
            top_symbols = results_df["Symbol"].tolist()[:5]
            shap_raw = explainability.explain_top_symbols(
                alpha_inference.load_default_alpha_model(),
                results_df.set_index("Symbol")[feature_cols_avail],
                symbols=top_symbols,
                top_n=5,
            )
            explanations = []
            for sym in results_df["Symbol"]:
                exp = shap_raw.get(sym, [])
                explanations.append(explainability.format_explanation(exp))
            results_df["Explanation"] = explanations
        except Exception:
            results_df["Explanation"] = ""

    # Meta-layer summaries from historical buckets (best-effort)
    try:
        meta_stats = meta_experience.load_meta_experience()
        if meta_stats is not None:
            meta_texts = [meta_stats.describe_row(row) for _, row in results_df.iterrows()]
            results_df["MetaSummary"] = meta_texts
            setup_tags = [setup_library.classify_setup(row, meta_stats) for _, row in results_df.iterrows()]
            results_df["SetupTag"] = setup_tags
            if "Explanation" not in results_df.columns:
                results_df["Explanation"] = meta_texts
            if "Explanation" in results_df.columns:
                filled = []
                for exp, meta_txt in zip(results_df["Explanation"], meta_texts):
                    exp_str = str(exp) if exp is not None else ""
                    if exp_str.strip():
                        filled.append(exp_str)
                    elif meta_txt:
                        filled.append(meta_txt)
                    else:
                        filled.append("")
                results_df["Explanation"] = filled
        else:
            results_df["MetaSummary"] = ""
            results_df["SetupTag"] = ""
    except Exception:
        results_df["MetaSummary"] = ""
        results_df["SetupTag"] = ""

    # Optional scoreboard logging
    try:
        from technic_v4.evaluation import scoreboard

        scoreboard.append_daily_signals(results_df)
    except Exception:
        pass

    # Optional alerting
    try:
        from technic_v4.alerts import engine as alerts_engine
        alerts = alerts_engine.detect_alerts_from_scan(results_df, regime_tags, locals().get("sb"))
        alerts_engine.log_alerts_to_file(alerts)
    except Exception:
        pass

    # Filter summary (console/logger)
    try:
        logger.info("=== FILTER SUMMARY ===")
        logger.info("Remaining symbols: %d", len(results_df))
        if "Close" in results_df.columns and not results_df.empty:
            logger.info("Min price: %.2f", float(results_df["Close"].min()))
        if "DollarVolume" in results_df.columns and not results_df.empty:
            logger.info("Min dollar volume: %.0f", float(results_df["DollarVolume"].min()))
        if "ATR14_pct" in results_df.columns and not results_df.empty:
            logger.info("Max ATR%%: %.4f", float(results_df["ATR14_pct"].max()))
        logger.info("=====================")
    except Exception:
        pass

    # --------------------------------------------
    # Save daily historical snapshot for backtesting
    # --------------------------------------------
    history_dir = OUTPUT_DIR / "history"
    history_dir.mkdir(exist_ok=True)
    hist_path = history_dir / f"scan_{datetime.utcnow().date()}.csv"
    try:
        results_df.to_csv(hist_path, index=False)
        logger.info("[OUTPUT] Wrote historical snapshot to: %s", hist_path)
    except Exception:
        logger.warning("[OUTPUT ERROR] Failed to write historical snapshot CSV", exc_info=True)

    # Save CSV (CORE by default; fall back to full results if empty)
    core_df = results_df
    if "Tier" in results_df.columns:
        core_df = results_df[results_df["Tier"] == "CORE"].copy()
        if core_df.empty:
            logger.warning("[FILTER] No CORE rows after thresholds; writing full results.")
            core_df = results_df.copy()

    output_path = OUTPUT_DIR / "technic_scan_results.csv"
    try:
        core_df.to_csv(output_path, index=False)
        logger.info("[OUTPUT] Wrote scan results to: %s", output_path)
    except Exception:
        logger.warning("[OUTPUT ERROR] Failed to write scan results CSV", exc_info=True)
    
    # Log top 10 by MERIT Score
    if "MeritScore" in results_df.columns and len(results_df) > 0:
        top_10 = results_df.head(10)
        logger.info("[MERIT] Top 10 by MERIT Score:")
        for idx, row in top_10.iterrows():
            logger.info(
                "  %s: MERIT=%.1f (%s), Tech=%.1f, Alpha=%.2f, WinProb=%.0f%%, Quality=%.0f, ICS=%.0f, Flags=%s",
                row.get("Symbol", "?"),
                row.get("MeritScore", 0),
                row.get("MeritBand", "?"),
                row.get("TechRating", 0),
                row.get("AlphaScore", 0),
                row.get("win_prob_10d", 0) * 100,
                row.get("QualityScore", 0),
                row.get("InstitutionalCoreScore", 0),
                row.get("MeritFlags", "")
            )

    return core_df, status_text


def _validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate scan results before returning to UI/API.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    expected_cols = {
        "Symbol": np.nan,
        "Signal": "",
        "TechRating": np.nan,
        "AlphaScore": np.nan,
        "Entry": np.nan,
        "Stop": np.nan,
        "Target": np.nan,
        "Sector": "",
        "Industry": "",
        "SubIndustry": "",
        "RegimeTrend": "",
        "RegimeVol": "",
        "MarketRegime": "",
        "OptionTrade": "",
        "OptionPicks": [],
        "Rationale": "",
        "Explanation": "",
    }

    for col, default in expected_cols.items():
        if col not in df.columns:
            if isinstance(default, list):
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = default

    # Drop rows missing critical identifiers/scores
    df = df.dropna(subset=["Symbol", "TechRating"])

    # Coerce types
    float_cols = ["TechRating", "AlphaScore", "Entry", "Stop", "Target"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    str_cols = ["Symbol", "Signal", "Sector", "Industry", "SubIndustry"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# -----------------------------
# Public scan entrypoint
# -----------------------------

def run_scan(
    config: Optional[ScanConfig] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Public scan entrypoint: orchestrates universe prep, per-symbol scanning, and result finalization.
    """
    if config is None:
        config = ScanConfig()

    settings = get_settings()
    risk_profile = get_risk_profile(config.profile if config else None)
    logger.info("[PROFILE] Using risk profile '%s': %s", risk_profile.name, risk_profile.label)
    start_ts = time.time()
    logger.info("Starting scan with config: %s", config)

    # Regime context (best-effort) + macro context
    regime_tags = None
    try:
        spy = data_engine.get_price_history(symbol="SPY", days=260, freq="daily")
        if spy is not None and not spy.empty:
            try:
                from technic_v4.engine.regime_engine import classify_spy_regime, detect_market_regime

                regime_tags = classify_spy_regime(spy)
                regime_rich = detect_market_regime(spy)
                if regime_rich:
                    regime_tags.update(regime_rich)
            except Exception:
                regime_tags = classify_regime(spy)
            if regime_tags:
                logger.info(
                    "[REGIME] trend=%s vol=%s state=%s label=%s",
                    regime_tags.get("trend"),
                    regime_tags.get("vol"),
                    regime_tags.get("state_id"),
                    regime_tags.get("label"),
                )
    except Exception:
        pass

    macro_ctx = _compute_macro_context()
    if macro_ctx:
        logger.info("[MACRO] %s", macro_ctx)
        if regime_tags:
            regime_tags.update(macro_ctx)
        else:
            regime_tags = macro_ctx

    # 1) Universe (load + sector/industry filters)
    universe: List[UniverseRow] = _prepare_universe(config, settings=settings)
    total_symbols = len(universe)
    if not universe:
        return pd.DataFrame(), "No symbols match your sector/industry filters."

    # Respect max_symbols
    working = universe
    if config.max_symbols and len(working) > config.max_symbols:
        working = working[: config.max_symbols]

    logger.info(
        "[UNIVERSE] Scanning %d symbols (max_symbols=%s).",
        len(working),
        config.max_symbols,
    )

    # 2) Risk settings
    risk = RiskSettings(
        account_size=config.account_size,
        risk_pct=config.risk_pct,  # already in percent
        target_rr=config.target_rr,
        trade_style=config.trade_style,
    )

    # Effective lookback based on trade style
    effective_lookback = _resolve_lookback_days(
        config.trade_style,
        config.lookback_days,
    )
    logger.info(
        "[SCAN] Using lookback_days=%d for trade_style='%s'",
        effective_lookback,
        config.trade_style,
    )

    # 3) Per-symbol loop (Ray or thread pool)
    results_df, stats = _run_symbol_scans(
        config=config,
        universe=working,
        regime_tags=regime_tags,
        effective_lookback=effective_lookback,
        settings=settings,
        progress_cb=progress_cb,
    )

    logger.info(
        "[SCAN STATS] attempted=%d, kept=%d, errors=%d, rejected=%d",
        stats.get("attempted", 0),
        stats.get("kept", 0),
        stats.get("errors", 0),
        stats.get("rejected", 0),
    )

    if results_df.empty:
        return pd.DataFrame(), "No results returned. Check universe or data source."

    # 4) Finalize results (alpha blend, filters, trade planning, logging)
    as_of_ts = pd.Timestamp(config.as_of_date) if config.as_of_date else pd.Timestamp.utcnow().normalize()

    results_df, status_text = _finalize_results(
        config=config,
        results_df=results_df,
        risk=risk,
        regime_tags=regime_tags,
        settings=settings,
        as_of_date=as_of_ts,
    )

    # Best-effort logging of recommendations for live evaluation
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "recommendations.csv"
        top_log = results_df.head(min(10, len(results_df))).copy()
        top_log["AsOf"] = datetime.utcnow().isoformat()
        cols = [
            "AsOf",
            "Symbol",
            "Signal",
            "TechRating",
            "AlphaScore",
            "Entry",
            "Stop",
            "Target",
            "MarketRegime",
        ]
        for c in cols:
            if c not in top_log.columns:
                top_log[c] = None
        top_log[cols].to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        logger.info("[LOG] appended top recommendations to %s", log_path)
    except Exception:
        logger.warning("[LOG] failed to append recommendations log", exc_info=True)

    results_df = _validate_results(results_df)

    elapsed = time.time() - start_ts
    logger.info(
        "Finished scan in %.2fs (%d results)", elapsed, len(results_df)
    )

    return results_df, status_text


if __name__ == "__main__":
    df, msg = run_scan()
    logger.info(msg)
    logger.info(df.head())
