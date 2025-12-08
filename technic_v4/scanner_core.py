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
    Upgrade TechRating using cross-sectional factor z-scores.
    Keeps the original TechRating but blends in multi-factor signals.
    """
    if df.empty:
        return df

    factor_cols = [
        "Ret_5",
        "Ret_21",
        "Ret_63",
        "MomentumScore",
        "Reversal_5",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "ADX14",
        "MA10",
        "MA20",
        "MA50",
        "SlopeMA20",
        "TrendStrength50",
        "ATR_pct",
        "VolatilityScore",
        "DollarVolume20",
        "VolumeScore",
        "GapStat10",
        "BreakoutScore",
        "ExplosivenessScore",
        "RSI14",
        "OscillatorScore",
        "value_ep",
        "quality_roe",
        "quality_gpm",
        "dollar_vol_20",
        "vol_realized_20",
        "atr_pct_14",
        "mom_21",
        "mom_63",
        "reversal_5",
        "ma_slope_20",
        "value_cfp",
        "size_log_mcap",
    ]
    available = [c for c in factor_cols if c in df.columns]
    if not available:
        return df

    zed = pd.DataFrame({c: zscore(df[c]) for c in available})
    # Higher is better for most; flip risk measures
    risk_cols = {"atr_pct_14", "vol_realized_20"}
    for rc in risk_cols.intersection(zed.columns):
        zed[rc] = -zed[rc]

    # Base weights: momentum 35%, value 20%, quality 20%, liquidity 10%, risk 15%
    weights = {
        "mom_21": 0.15,
        "mom_63": 0.20,
        "reversal_5": 0.05,
        "ma_slope_20": 0.05,
        "value_ep": 0.10,
        "value_cfp": 0.10,
        "quality_roe": 0.10,
        "quality_gpm": 0.10,
        "dollar_vol_20": 0.05,
        "atr_pct_14": 0.05,
        "vol_realized_20": 0.05,
    }
    if regime:
        trend = str(regime.get("trend", "")).upper()
        vol = str(regime.get("vol", "")).upper()
        if trend == "TRENDING_UP":
            weights["mom_63"] += 0.05
            weights["mom_21"] += 0.02
        elif trend == "TRENDING_DOWN":
            weights["reversal_5"] += 0.05
            weights["value_ep"] += 0.05
        if vol == "HIGH_VOL":
            weights["atr_pct_14"] += 0.05
            weights["vol_realized_20"] += 0.05
            weights["mom_63"] -= 0.03
            weights["mom_21"] -= 0.02
    # Restrict to available cols
    usable_weights = {k: v for k, v in weights.items() if k in zed.columns}
    if not usable_weights:
        return df

    factor_alpha = sum(zed[k] * v for k, v in usable_weights.items())
    factor_alpha = zscore(factor_alpha)

    # Regime dummy features (for ML alpha)
    if regime:
        trend = str(regime.get("trend", "")).upper()
        vol = str(regime.get("vol", "")).upper()
        trend_cols = ["regime_trend_TRENDING_UP", "regime_trend_TRENDING_DOWN", "regime_trend_SIDEWAYS"]
        vol_cols = ["regime_vol_HIGH_VOL", "regime_vol_LOW_VOL"]
        for c in trend_cols + vol_cols:
            df[c] = 0.0
        if trend:
            df[f"regime_trend_{trend}"] = 1.0
        if vol:
            df[f"regime_vol_{vol}"] = 1.0

    # Optional ML alpha blend
    alpha = factor_alpha
    settings = get_settings()
    use_ml_alpha = settings.use_ml_alpha
    logger.info("[ALPHA] use_ml_alpha=%s", use_ml_alpha)
    ml_alpha = None
    if use_ml_alpha:
        try:
            feature_cols_available = [c for c in factor_cols if c in df.columns]
            feature_cols_available += [
                c for c in df.columns
                if c.startswith("regime_trend_") or c.startswith("regime_vol_")
            ]
            logger.info(
                "[ALPHA] feature_cols_available for ML: %s",
                feature_cols_available,
            )
            ml_alpha = alpha_inference.score_alpha(df[feature_cols_available])
        except Exception:
            logger.warning("[ALPHA] error calling score_alpha", exc_info=True)
            ml_alpha = None
        if ml_alpha is not None and not ml_alpha.empty:
            ml_alpha_z = zscore(ml_alpha)
            alpha = 0.5 * factor_alpha + 0.5 * ml_alpha_z
            logger.info("[ALPHA] ML alpha blended with factor alpha")
        else:
            logger.info("[ALPHA] ML alpha missing or empty, skipping blend")

    # Optional meta alpha
    use_meta = settings.use_meta_alpha
    if use_meta:
        try:
            meta_alpha = alpha_inference.score_meta_alpha(df)
        except Exception:
            meta_alpha = None
        if meta_alpha is not None and not meta_alpha.empty:
            alpha = zscore(meta_alpha)
            logger.info("[ALPHA] Meta alpha applied")

    # Regime-aware scaling of alpha weight in TechRating blend
    alpha_weight = 0.4
    if regime:
        if regime.get("vol") == "HIGH_VOL":
            alpha_weight = 0.35
        if regime.get("trend") == "TRENDING_UP" and regime.get("vol") == "LOW_VOL":
            alpha_weight = 0.45

    base_tr = df.get("TechRating", pd.Series(0, index=df.index)).fillna(0)
    blended = (1 - alpha_weight) * base_tr + alpha_weight * (alpha * 10 + 15)  # scale alpha into TR-like range
    df["TechRating_raw"] = base_tr
    df["AlphaScore"] = alpha
    df["TechRating"] = blended
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

    - Short-term swing: use at most 90 days
    - Medium-term swing: use slider value as-is
    - Position / longer-term: ensure at least 180 days
    """
    ts = (trade_style or "").lower()

    if "short-term" in ts:
        return min(base_days, 90)

    if "medium" in ts:
        return base_days

    if "position" in ts or "longer" in ts:
        return max(base_days, 180)

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

    # Try Ray path first if enabled
    ray_rows = ray_runner.run_ray_scans([u.symbol for u in universe], config, regime_tags)
    if ray_rows is not None:
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
    else:

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

    stats = {"attempted": attempted, "kept": kept, "errors": errors, "rejected": rejected}
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

    # Cross-sectional alpha blend: upgrade TechRating using factor z-scores
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

    # Alpha/risk scaffolding
    results_df["MuHat"] = results_df["TechRating"] / 100.0
    results_df["MuMl"] = pd.Series([np.nan] * len(results_df))
    results_df["MuTotal"] = results_df["MuHat"]
    if "AlphaScore" not in results_df.columns:
        results_df["AlphaScore"] = results_df["TechRating"]
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
