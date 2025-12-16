"""
Enhanced scanner_core with Phase 3D-D: Multi-Stage Progress Tracking
Tracks 4 distinct stages with weighted progress and stage-specific ETAs
"""

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

logger = get_logger()

# Import all the existing dependencies
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

# PHASE 3B: Import optimized Ray runner with stateful workers
try:
    from technic_v4.engine.ray_runner_optimized import run_ray_scans_optimized, get_worker_pool
    PHASE3B_AVAILABLE = True
except ImportError:
    PHASE3B_AVAILABLE = False
    logger.warning("[PHASE3B] Optimized Ray runner not available, using standard version")

from technic_v4.data_layer.events import get_event_info
from technic_v4.data_layer.earnings_surprises import get_latest_surprise, get_surprise_stats
from technic_v4.data_layer.fundamental_trend import get_fundamental_trend
from technic_v4.data_layer.ratings import get_rating_info
from technic_v4.data_layer.quality import get_quality_info
from technic_v4.data_layer.sponsorship import get_sponsorship, get_insider_flags
from technic_v4.config.thresholds import load_score_thresholds
import concurrent.futures
from technic_v4.engine.portfolio_optim import (
    mean_variance_weights,
    inverse_variance_weights,
    hrp_weights,
)
from technic_v4.engine.recommendation import build_recommendation
from technic_v4.engine.batch_processor import get_batch_processor
from technic_v4.engine.meta_inference import score_win_prob_10d

# PHASE 3C: Redis caching for 2x speedup
try:
    from technic_v4.cache.redis_cache import redis_cache
    REDIS_AVAILABLE = redis_cache.available
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("[REDIS] Redis cache not available")

# PHASE 3D-B: Enhanced error handling and progress tracking
from technic_v4.errors import ErrorType, ScanError, get_error_message, create_custom_error
from technic_v4.progress import ProgressTracker, MultiStageProgressTracker

# Progress callback type: (stage, current, total, message, metadata)
ProgressCallback = Callable[[str, int, int, str, dict], None]
# Error callback type: (error)
ErrorCallback = Callable[[ScanError], None]

# Import all the existing functions from scanner_core
from technic_v4.scanner_core import (
    _safe_progress_callback,
    get_stock_history_df,
    get_fundamentals,
    OUTPUT_DIR,
    _filter_universe,
    _smart_filter_universe,
    _prepare_universe,
    _apply_alpha_blend,
    _apply_portfolio_suggestions,
    _annotate_recommendations,
    ScanConfig,
    MIN_BARS,
    MAX_WORKERS,
    MIN_PRICE,
    MIN_DOLLAR_VOL,
    _compute_macro_context,
    _can_pass_basic_filters,
    _passes_basic_filters,
    _should_use_intraday,
    _resolve_lookback_days,
    _scan_symbol,
    _attach_event_columns,
    _attach_ratings_quality,
    _process_symbol,
    _run_symbol_scans,
    _finalize_results,
    _validate_results,
)

# PHASE 3D-D: Enhanced run_scan with multi-stage progress tracking
def run_scan_enhanced(
    config: Optional[ScanConfig] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Enhanced scan entrypoint with multi-stage progress tracking.
    
    Tracks 4 distinct stages:
    1. Universe Loading (5%)
    2. Data Fetching (20%)
    3. Symbol Scanning (70%)
    4. Finalization (5%)
    
    Returns:
        Tuple of (results_df, status_text, performance_metrics)
    """
    if config is None:
        config = ScanConfig()

    settings = get_settings()
    risk_profile = get_risk_profile(config.profile if config else None)
    logger.info("[PROFILE] Using risk profile '%s': %s", risk_profile.name, risk_profile.label)
    
    # PHASE 3D-D: Initialize multi-stage progress tracker
    stage_tracker = MultiStageProgressTracker({
        "universe_loading": 0.05,    # 5% - Loading and filtering universe
        "data_fetching": 0.20,        # 20% - Batch pre-fetching price data
        "symbol_scanning": 0.70,      # 70% - Main scanning loop
        "finalization": 0.05          # 5% - Post-processing and saving
    })
    
    overall_start_ts = time.time()
    logger.info("Starting enhanced scan with multi-stage tracking: %s", config)
    
    # Helper to send progress updates
    def send_progress(stage: str, current: int, total: int, message: str = "", extra_metadata: dict = None):
        """Send progress update through callback with stage-specific metadata"""
        if progress_cb:
            metadata = extra_metadata or {}
            
            # Add stage-specific progress info
            if stage_tracker.current_stage == stage:
                progress_info = stage_tracker.update(current)
                metadata.update({
                    'stage_progress_pct': progress_info['stage_progress']['progress_pct'],
                    'overall_progress_pct': progress_info['overall_progress_pct'],
                    'stage_eta': progress_info['stage_progress'].get('estimated_remaining'),
                    'overall_eta': progress_info.get('overall_estimated_remaining'),
                    'stage_throughput': progress_info['stage_progress'].get('throughput', 0),
                })
            
            _safe_progress_callback(
                progress_cb,
                stage=stage,
                current=current,
                total=total,
                message=message,
                metadata=metadata
            )

    # ========== STAGE 1: Universe Loading (5%) ==========
    stage_tracker.start_stage("universe_loading", 100)
    send_progress("universe_loading", 0, 100, "Starting universe loading...")
    
    # Regime context (best-effort) + macro context
    regime_tags = None
    try:
        send_progress("universe_loading", 20, 100, "Computing market regime...")
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

    send_progress("universe_loading", 40, 100, "Computing macro context...")
    macro_ctx = _compute_macro_context()
    if macro_ctx:
        logger.info("[MACRO] %s", macro_ctx)
        if regime_tags:
            regime_tags.update(macro_ctx)
        else:
            regime_tags = macro_ctx

    # Load and filter universe
    send_progress("universe_loading", 60, 100, "Loading universe symbols...")
    universe: List[UniverseRow] = _prepare_universe(config, settings=settings)
    total_symbols = len(universe)
    if not universe:
        # Still complete the stages and return metrics even with no results
        send_progress("universe_loading", 100, 100, "No symbols found")
        stage_tracker.complete_stage()
        
        overall_elapsed = time.time() - overall_start_ts
        
        # Get stage timing breakdown
        stage_timings = {}
        for stage_name in stage_tracker.stage_weights.keys():
            if stage_name in stage_tracker.stages:
                tracker = stage_tracker.stages[stage_name]
                elapsed = time.time() - tracker.start_time
                stage_timings[f"{stage_name}_seconds"] = round(elapsed, 2)
        
        performance_metrics = {
            'total_seconds': round(overall_elapsed, 2),
            'symbols_scanned': 0,
            'symbols_returned': 0,
            'symbols_per_second': 0,
            'speedup': 0,
            'baseline_time': 0,
            'stage_timings': stage_timings,
            'cache_stats': {},
            'scan_stats': {'attempted': 0, 'kept': 0, 'errors': 0, 'rejected': 0},
            'overall_progress_summary': stage_tracker.get_summary()
        }
        
        return pd.DataFrame(), "No symbols match your sector/industry filters.", performance_metrics

    # Respect max_symbols
    working = universe
    if config.max_symbols and len(working) > config.max_symbols:
        working = working[: config.max_symbols]

    send_progress("universe_loading", 80, 100, f"Prepared {len(working)} symbols for scanning")
    logger.info(
        "[UNIVERSE] Scanning %d symbols (max_symbols=%s).",
        len(working),
        config.max_symbols,
    )

    # Risk settings
    risk = RiskSettings(
        account_size=config.account_size,
        risk_pct=config.risk_pct,
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
    
    send_progress("universe_loading", 100, 100, "Universe loading complete")
    stage_tracker.complete_stage()

    # ========== STAGE 2: Data Fetching (20%) ==========
    stage_tracker.start_stage("data_fetching", len(working))
    send_progress("data_fetching", 0, len(working), f"Starting batch data fetch for {len(working)} symbols...")
    
    logger.info("[BATCH PREFETCH] Pre-fetching price data for %d symbols", len(working))
    t_batch_start = time.time()
    
    # Simulate progress during batch fetch (since it's a single operation)
    # We'll update progress based on time elapsed
    fetch_start = time.time()
    price_cache = {}
    
    # Create a wrapper to track batch fetch progress
    def fetch_with_progress():
        nonlocal price_cache
        # Estimate 0.1s per symbol for progress updates
        estimated_time = len(working) * 0.1
        update_interval = 0.5  # Update every 0.5 seconds
        
        # Start the actual fetch in a thread
        import threading
        fetch_complete = threading.Event()
        fetch_result = {}
        
        def do_fetch():
            nonlocal fetch_result
            fetch_result = data_engine.get_price_history_batch(
                symbols=[row.symbol for row in working],
                days=effective_lookback,
                freq="daily"
            )
            fetch_complete.set()
        
        fetch_thread = threading.Thread(target=do_fetch)
        fetch_thread.start()
        
        # Update progress while fetching
        symbols_fetched = 0
        while not fetch_complete.is_set():
            elapsed = time.time() - fetch_start
            # Estimate progress based on time
            estimated_progress = min(elapsed / estimated_time, 0.95)  # Cap at 95%
            symbols_fetched = int(estimated_progress * len(working))
            
            send_progress(
                "data_fetching", 
                symbols_fetched, 
                len(working),
                f"Fetching price data... ({symbols_fetched}/{len(working)})",
                {'batch_mode': True, 'elapsed': elapsed}
            )
            
            fetch_complete.wait(timeout=update_interval)
        
        # Fetch complete
        price_cache = fetch_result
        return price_cache
    
    price_cache = fetch_with_progress()
    
    t_batch_elapsed = time.time() - t_batch_start
    cache_hit_rate = (len(price_cache) / len(working) * 100) if working else 0
    
    send_progress(
        "data_fetching", 
        len(working), 
        len(working),
        f"Data fetch complete: {len(price_cache)}/{len(working)} symbols cached",
        {
            'cache_hit_rate': cache_hit_rate,
            'fetch_time': t_batch_elapsed,
            'symbols_cached': len(price_cache)
        }
    )
    
    logger.info(
        "[BATCH PREFETCH] Cached %d/%d symbols in %.2fs (%.1f%% success rate)", 
        len(price_cache), 
        len(working), 
        t_batch_elapsed,
        cache_hit_rate
    )
    
    stage_tracker.complete_stage()

    # ========== STAGE 3: Symbol Scanning (70%) ==========
    stage_tracker.start_stage("symbol_scanning", len(working))
    
    # Create a wrapper for progress callback that integrates with stage tracker
    def symbol_scan_progress(stage: str, current: int, total: int, message: str = "", metadata: dict = None):
        """Progress callback for symbol scanning that updates stage tracker"""
        if stage == 'symbol_scanning':
            send_progress(stage, current, total, message, metadata)
    
    send_progress("symbol_scanning", 0, len(working), "Starting symbol analysis...")
    
    # Run symbol scans with progress tracking
    results_df, stats = _run_symbol_scans(
        config=config,
        universe=working,
        regime_tags=regime_tags,
        effective_lookback=effective_lookback,
        settings=settings,
        progress_cb=symbol_scan_progress,  # Use our wrapped callback
        price_cache=price_cache,
    )

    logger.info(
        "[SCAN STATS] attempted=%d, kept=%d, errors=%d, rejected=%d",
        stats.get("attempted", 0),
        stats.get("kept", 0),
        stats.get("errors", 0),
        stats.get("rejected", 0),
    )

    if results_df.empty:
        return pd.DataFrame(), "No results returned. Check universe or data source.", {}
    
    stage_tracker.complete_stage()

    # ========== STAGE 4: Finalization (5%) ==========
    stage_tracker.start_stage("finalization", 100)
    send_progress("finalization", 0, 100, "Starting result finalization...")
    
    # Finalize results (alpha blend, filters, trade planning, logging)
    as_of_ts = pd.Timestamp(config.as_of_date) if config.as_of_date else pd.Timestamp.utcnow().normalize()
    
    send_progress("finalization", 30, 100, "Applying alpha blending and filters...")
    results_df, status_text = _finalize_results(
        config=config,
        results_df=results_df,
        risk=risk,
        regime_tags=regime_tags,
        settings=settings,
        as_of_date=as_of_ts,
    )

    send_progress("finalization", 60, 100, "Logging recommendations...")
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

    send_progress("finalization", 80, 100, "Validating results...")
    results_df = _validate_results(results_df)

    # Calculate performance metrics
    overall_elapsed = time.time() - overall_start_ts
    symbols_per_second = len(working) / overall_elapsed if overall_elapsed > 0 else 0
    
    # Calculate speedup vs baseline (no cache)
    baseline_time = len(working) * 2.0  # Assume 2s per symbol without optimizations
    speedup = baseline_time / overall_elapsed if overall_elapsed > 0 else 1.0
    
    # Get final cache stats for performance metrics
    cache_performance = {}
    if REDIS_AVAILABLE:
        try:
            final_cache_stats = redis_cache.get_stats()
            cache_performance = {
                'cache_available': final_cache_stats.get('available', False),
                'cache_hit_rate': final_cache_stats.get('hit_rate', 0),
                'cache_hits': final_cache_stats.get('hits', 0),
                'cache_misses': final_cache_stats.get('misses', 0),
                'total_keys': final_cache_stats.get('total_keys', 0)
            }
        except Exception:
            pass
    
    # Get stage timing breakdown
    stage_timings = {}
    for stage_name in stage_tracker.stage_weights.keys():
        if stage_name in stage_tracker.stages:
            tracker = stage_tracker.stages[stage_name]
            elapsed = time.time() - tracker.start_time
            stage_timings[f"{stage_name}_seconds"] = round(elapsed, 2)
    
    send_progress("finalization", 100, 100, "Scan complete!")
    stage_tracker.complete_stage()
    
    logger.info(
        "Finished enhanced scan in %.2fs (%d results, %.1f sym/s, %.1fx speedup)",
        overall_elapsed, len(results_df), symbols_per_second, speedup
    )
    
    # Log stage breakdown
    logger.info("[STAGE TIMING] Breakdown: %s", stage_timings)

    # Return results with enhanced performance metrics
    performance_metrics = {
        'total_seconds': round(overall_elapsed, 2),
        'symbols_scanned': len(working),
        'symbols_returned': len(results_df),
        'symbols_per_second': round(symbols_per_second, 2),
        'speedup': round(speedup, 2),
        'baseline_time': round(baseline_time, 2),
        'stage_timings': stage_timings,
        'cache_stats': cache_performance,
        'scan_stats': stats,
        'overall_progress_summary': stage_tracker.get_summary()
    }
    
    return results_df, status_text, performance_metrics


# Alias for backward compatibility
run_scan = run_scan_enhanced


if __name__ == "__main__":
    # Test the enhanced scanner
    result = run_scan_enhanced()
    if len(result) == 3:
        df, msg, metrics = result
        logger.info(msg)
        logger.info(df.head())
        logger.info("[PERFORMANCE] %s", metrics)
        logger.info("[STAGE BREAKDOWN] %s", metrics.get('stage_timings', {}))
    else:
        # Backward compatibility
        df, msg = result
        logger.info(msg)
        logger.info(df.head())
