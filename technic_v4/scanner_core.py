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
from technic_v4.engine.scoring import compute_scores
from technic_v4.engine.factor_engine import zscore
from technic_v4.engine.regime_engine import classify_regime
from technic_v4.engine.trade_planner import RiskSettings, plan_trades
from technic_v4.engine.portfolio_engine import risk_adjusted_rank, diversify_by_sector
from technic_v4.engine.options_engine import score_options
from technic_v4.engine import ranking_engine
from technic_v4.engine.options_suggest import suggest_option_trades
from technic_v4.engine import explainability_engine
from datetime import datetime
from technic_v4.engine import alpha_inference
from technic_v4.engine import explainability
from technic_v4.engine import ray_runner
import concurrent.futures

logger = get_logger()

ProgressCallback = Callable[[str, int, int], None]

# Legacy compatibility for tests/monkeypatch
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


def _prepare_universe(config: "ScanConfig", settings=None) -> List[UniverseRow]:
    """
    Load and filter the universe based on config.
    """
    universe: List[UniverseRow] = load_universe()
    logger.info("[UNIVERSE] loaded %d symbols from ticker_universe.csv.", len(universe))

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


def _apply_alpha_blend(df: pd.DataFrame, regime: Optional[dict] = None) -> pd.DataFrame:
    """
    Cross-sectional alpha blend:
    - factor_alpha = zscore of baseline TechRating (v1 heuristic)
    - ml_alpha     = XGB model prediction (5d fwd return)
    - alpha_blend  = blend(factor_alpha, ml_alpha_z; TECHNIC_ALPHA_WEIGHT)
    - TechRating   = v2 hybrid score using alpha_blend

    Columns produced:
        factor_alpha   : cross-sectional z-score of baseline TechRating
        AlphaScore     : raw ML predictions (for API / scoreboard)
        ml_alpha_z     : z-scored ML alpha (when available)
        meta_alpha     : optional meta-ensemble alpha (if enabled)
        alpha_blend    : final cross-sectional alpha driver
        TechRating_raw : original TechRating before alpha blending
        TechRating     : blended TechRating v2 (hybrid)
    """
    if df is None or df.empty:
        return df

    settings = get_settings()
    logger.info(
        "[ALPHA] use_ml_alpha=%s use_meta_alpha=%s alpha_weight=%.2f",
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

    # Regime one-hot features (for ML/meta models)
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

    # --- ML alpha (XGB) ---------------------------------------------------
    ml_alpha: Optional[pd.Series] = None
    ml_alpha_z: Optional[pd.Series] = None

    if settings.use_ml_alpha:
        try:
            ml_alpha = alpha_inference.score_alpha(df)
        except Exception:
            logger.warning("[ALPHA] score_alpha() failed", exc_info=True)
            ml_alpha = None

    if ml_alpha is not None and not ml_alpha.empty:
        # Raw model prediction (5d forward return) – this is the "true" AlphaScore
        ml_alpha = pd.to_numeric(ml_alpha, errors="coerce")
        df["AlphaScore"] = ml_alpha

        try:
            ml_alpha_z = zscore(ml_alpha)
            df["ml_alpha_z"] = ml_alpha_z
        except Exception:
            logger.warning("[ALPHA] zscore() on ML alpha failed", exc_info=True)
            ml_alpha_z = None
            df["ml_alpha_z"] = np.nan
        logger.info("[ALPHA] ML alpha available for %d rows", ml_alpha.notna().sum())
    else:
        # Ensure AlphaScore exists for downstream consumers (API / eval)
        if "AlphaScore" not in df.columns:
            df["AlphaScore"] = np.nan
        df["ml_alpha_z"] = np.nan
        logger.info("[ALPHA] ML alpha unavailable; factor-only alpha will be used")

    # --- Blend factor alpha and ML alpha ----------------------------------
    alpha_blend = factor_alpha.copy()

    if ml_alpha_z is not None:
        w = getattr(settings, "alpha_weight", 0.5)
        try:
            w = float(w)
        except Exception:
            w = 0.5
        w = max(0.0, min(1.0, w))  # clamp to [0,1]

        alpha_blend = (1.0 - w) * factor_alpha + w * ml_alpha_z
        logger.info("[ALPHA] blended factor + ML with TECHNIC_ALPHA_WEIGHT=%.2f", w)
    else:
        logger.info("[ALPHA] using factor_alpha only (no ML alpha)")

    # --- Optional meta alpha override ------------------------------------
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
                alpha_blend = meta_z
                logger.info("[ALPHA] meta alpha applied, overriding factor/ML blend")

    df["alpha_blend"] = alpha_blend

    # --- Map alpha_blend into TechRating v2 ------------------------------
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
MAX_WORKERS = 6  # limited IO concurrency
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 0.0


def _passes_basic_filters(df: pd.DataFrame) -> bool:
    """Quick sanity filters before doing full scoring + trade planning."""
    if df is None or df.empty:
        return False

    if len(df) < MIN_BARS:
        return False

    last_row = df.iloc[-1]

    close = float(last_row.get("Close", 0.0))
    if not pd.notna(close) or close < MIN_PRICE:
        return False

    try:
        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(40).mean())
    except Exception:
        return True

    if avg_dollar_vol < MIN_DOLLAR_VOL:
        return False

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


def _process_symbol(
    config: "ScanConfig",
    urow: UniverseRow,
    effective_lookback: int,
    settings=None,
    regime_tags: Optional[dict] = None,
) -> Optional[pd.Series]:
    """
    Wrapper to process a single symbol; returns a Series/row or None.
    """
    latest_local = _scan_symbol(
        symbol=urow.symbol,
        lookback_days=effective_lookback,
        trade_style=config.trade_style,
    )
    if latest_local is None:
        return None
    latest_local["Sector"] = urow.sector or ""
    latest_local["Industry"] = urow.industry or ""
    latest_local["SubIndustry"] = urow.subindustry or ""
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
    """
    rows: List[pd.Series] = []
    attempted = kept = errors = rejected = 0
    total_symbols = len(universe)

    if settings is None:
        settings = get_settings()

    # Decide execution mode up front
    use_ray = getattr(settings, "use_ray", False)
    ray_rows = None

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

        max_workers = getattr(settings, "max_workers", None) or MAX_WORKERS
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for status, symbol, latest, urow in ex.map(_worker, enumerate(universe, start=1)):
                attempted += 1
                if status == "error" or latest is None:
                    rejected += 1 if status != "error" else 0
                    errors += 1 if status == "error" else 0
                    continue
                latest["Sector"] = urow.sector or ""
                latest["Industry"] = urow.industry or ""
                latest["SubIndustry"] = urow.subindustry or ""

                rows.append(latest)
                kept += 1
        engine_mode = "threadpool"

    elapsed = time.time() - start_ts
    per_symbol = elapsed / max(total_symbols, 1)
    logger.info(
        "[SCAN PERF] symbol engine: %d symbols via %s in %.2fs (%.3fs/symbol, max_workers=%s, use_ray=%s)",
        total_symbols,
        engine_mode,
        elapsed,
        per_symbol,
        getattr(settings, "max_workers", None),
        getattr(settings, "use_ray", False),
    )

    stats = {
        "attempted": attempted,
        "kept": kept,
        "errors": errors,
        "rejected": rejected,
    }
    return pd.DataFrame(rows), stats


def _finalize_results(
    config: "ScanConfig",
    results_df: pd.DataFrame,
    risk: RiskSettings,
    regime_tags: Optional[dict],
    settings=None,
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
    results_df = _apply_alpha_blend(results_df, regime=regime_tags)

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

    # Alpha / risk scaffolding
    # MuHat: heuristic tech-based drift
    results_df["MuHat"] = results_df["TechRating"] / 100.0

    # MuMl: ML alpha (raw forward-return prediction)
    if "AlphaScore" in results_df.columns:
        alpha_series = pd.to_numeric(results_df["AlphaScore"], errors="coerce")
    else:
        results_df["AlphaScore"] = np.nan
        alpha_series = pd.Series([np.nan] * len(results_df), index=results_df.index)

    if alpha_series.notna().any():
        results_df["MuMl"] = alpha_series

        # Blend heuristic and ML drift using TECHNIC_ALPHA_WEIGHT
        w_mu = getattr(settings, "alpha_weight", 0.5)
        try:
            w_mu = float(w_mu)
        except Exception:
            w_mu = 0.5
        w_mu = max(0.0, min(1.0, w_mu))

        results_df["MuTotal"] = (1.0 - w_mu) * results_df["MuHat"] + w_mu * results_df["MuMl"]
    else:
        # No ML alpha available
        results_df["MuMl"] = np.nan
        results_df["MuTotal"] = results_df["MuHat"]
        # For downstream consumers that expect AlphaScore, fall back to TechRating
        results_df.loc[:, "AlphaScore"] = results_df["TechRating"]

    # Cross-sectional alpha percentile (0–100) for UI / ranking
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

    # Risk-adjusted sorting + diversification fallback
    vol_col = "vol_realized_20" if "vol_realized_20" in results_df.columns else None
    results_df = risk_adjusted_rank(
        results_df,
        return_col="MuTotal",
        vol_col=vol_col or "TechRating",
    )
    sort_col = "risk_score" if "risk_score" in results_df.columns else "TechRating"
    results_df = results_df.sort_values(sort_col, ascending=False)
    try:
        results_df = diversify_by_sector(results_df, sector_col="Sector", score_col=sort_col, top_n=config.max_symbols or 50)
    except Exception:
        pass

    # Option picks placeholder (populated upstream when chains are available)
    results_df["OptionPicks"] = [[] for _ in range(len(results_df))]
    # Optional: suggest simple option trades for strongest longs
    try:
        picks: list = []
        for idx, row in results_df.iterrows():
            sig = row.get("Signal")
            if sig not in {"Strong Long", "Long"}:
                picks.append([])
                continue
            spot = row.get("Close") or row.get("Entry") or row.get("Last")
            if spot is None or pd.isna(spot):
                picks.append([])
                continue
            suggested = suggest_option_trades(str(row.get("Symbol")), float(spot), bullish=True)
            picks.append(suggested or [])
        results_df["OptionTrade"] = picks
    except Exception:
        results_df["OptionTrade"] = [[] for _ in range(len(results_df))]

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

    # Save CSV
    output_path = OUTPUT_DIR / "technic_scan_results.csv"
    try:
        results_df.to_csv(output_path, index=False)
        logger.info("[OUTPUT] Wrote scan results to: %s", output_path)
    except Exception:
        logger.warning("[OUTPUT ERROR] Failed to write scan results CSV", exc_info=True)

    return results_df, status_text


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
        "OptionTrade": [],
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
    start_ts = time.time()
    logger.info("Starting scan with config: %s", config)

    # Regime context (best-effort)
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
    results_df, status_text = _finalize_results(
        config=config,
        results_df=results_df,
        risk=risk,
        regime_tags=regime_tags,
        settings=settings,
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
