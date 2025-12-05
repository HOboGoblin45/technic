from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Callable

import traceback
import pandas as pd
import numpy as np

from technic_v4.universe_loader import load_universe, UniverseRow
from technic_v4.data_layer.price_layer import get_stock_history_df
from technic_v4.engine.scoring import compute_scores
from technic_v4.engine.factor_engine import zscore
from technic_v4.engine.regime_engine import classify_regime
from technic_v4.data_layer.fundamentals import get_fundamentals
from technic_v4.engine.trade_planner import RiskSettings, plan_trades
from technic_v4.engine.portfolio_engine import risk_adjusted_rank, diversify_by_sector
from technic_v4.engine.options_engine import score_options, OptionPick
from technic_v4.engine import alpha_inference
import concurrent.futures

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
        print(
            f"[UNIVERSE] Sector filter kept {len(filtered)} / {before} "
            f"symbols for sectors={sorted(sector_set)}"
        )
    elif sectors and not has_any_sector:
        print(
            "[UNIVERSE] Sector filter requested but universe has no sector data; "
            "skipping sector filter."
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
            print(
                f"[UNIVERSE] Industry filter kept {len(filtered)} / {before} "
                f"symbols for term='{term}'"
            )
        else:
            print(
                "[UNIVERSE] Industry filter requested but universe has no industry data; "
                "skipping industry filter."
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
        print(
            f"[UNIVERSE] SubIndustry filter kept {len(filtered)} / {before} "
            f"symbols for subindustries={sorted(sub_set)}"
        )
    elif subindustries and not has_any_subindustry:
        print(
            "[UNIVERSE] SubIndustry filter requested but universe has no subindustry data; "
            "skipping subindustry filter."
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
        "mom_21",
        "mom_63",
        "reversal_5",
        "ma_slope_20",
        "atr_pct_14",
        "vol_realized_20",
        "dollar_vol_20",
        "value_ep",
        "value_cfp",
        "quality_roe",
        "quality_gpm",
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
    # Restrict to available cols
    usable_weights = {k: v for k, v in weights.items() if k in zed.columns}
    if not usable_weights:
        return df

    alpha = sum(zed[k] * v for k, v in usable_weights.items())
    alpha = zscore(alpha)

    # Optional ML alpha blend
    use_ml_alpha = str(os.getenv("TECHNIC_USE_ML_ALPHA", "false")).lower() in {"1", "true", "yes"}
    if use_ml_alpha:
        try:
            ml_alpha = alpha_inference.score_alpha(df[[c for c in factor_cols if c in df.columns]])
        except Exception:
            ml_alpha = None
        if ml_alpha is not None and not ml_alpha.empty:
            ml_alpha_z = zscore(ml_alpha)
            alpha = 0.5 * alpha + 0.5 * ml_alpha_z

    base_tr = df.get("TechRating", pd.Series(0, index=df.index)).fillna(0)
    blended = 0.6 * base_tr + 0.4 * (alpha * 10 + 15)  # scale alpha into TR-like range
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

    # Advanced
    allow_shorts: bool = False
    only_tradeable: bool = True


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
        df = get_stock_history_df(
            symbol=symbol,
            days=lookback_days,
            use_intraday=use_intraday,
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if not _passes_basic_filters(df):
        return None

    # Fundamentals (best-effort)
    fundamentals = get_fundamentals(symbol)

    # 3) Indicator + scoring pipeline
    scored = compute_scores(df, trade_style=trade_style, fundamentals=fundamentals)
    if scored is None or scored.empty:
        return None

    latest = scored.iloc[-1].copy()
    latest["symbol"] = symbol
    return latest


# -----------------------------
# Public scan entrypoint
# -----------------------------

def run_scan(
    config: Optional[ScanConfig] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, str]:
    if config is None:
        config = ScanConfig()

    # Regime context (best-effort)
    regime_tags = None
    try:
        spy = get_stock_history_df(symbol="SPY", days=260, use_intraday=False)
        if spy is not None and not spy.empty:
            try:
                from technic_v4.engine.regime_engine import classify_spy_regime

                regime_tags = classify_spy_regime(spy)
            except Exception:
                regime_tags = classify_regime(spy)
            if regime_tags:
                print(f"[REGIME] trend={regime_tags.get('trend')} vol={regime_tags.get('vol')} state={regime_tags.get('state_id')}")
    except Exception:
        pass

    # 1) Load universe
    universe: List[UniverseRow] = load_universe()
    print(f"[UNIVERSE] loaded {len(universe)} symbols from ticker_universe.csv.")

    # 2) Apply filters
    filtered_universe = _filter_universe(
        universe=universe,
        sectors=config.sectors,
        subindustries=config.subindustries,
        industry_contains=config.industry_contains,
    )
    print(
        f"[FILTER] {len(filtered_universe)} symbols after universe filters "
        f"(from {len(universe)})."
    )

    total_symbols = len(filtered_universe)

    if not filtered_universe:
        return pd.DataFrame(), "No symbols match your sector/industry filters."

    universe = filtered_universe

    # Respect max_symbols
    working = universe
    if config.max_symbols and len(working) > config.max_symbols:
        working = working[: config.max_symbols]

    print(
        f"[UNIVERSE] Scanning {len(working)} symbols "
        f"(max_symbols={config.max_symbols})."
    )

    # 3) Risk settings
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
    print(
        f"[SCAN] Using lookback_days={effective_lookback} "
        f"for trade_style='{config.trade_style}'"
    )

    # 4) Per-symbol loop with limited IO parallelism
    rows: List[pd.Series] = []

    attempted = 0
    kept = 0
    errors = 0
    rejected = 0

    def _worker(idx_urow):
        idx, urow = idx_urow
        symbol = urow.symbol
        if progress_cb is not None:
            try:
                progress_cb(symbol, idx, total_symbols)
            except Exception:
                pass
        try:
            latest_local = _scan_symbol(
                symbol=symbol,
                lookback_days=effective_lookback,
                trade_style=config.trade_style,
            )
        except Exception:
            print(f"[SCAN ERROR] {symbol}:")
            traceback.print_exc()
            return ("error", symbol, None, urow)
        return ("ok", symbol, latest_local, urow)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for status, symbol, latest, urow in ex.map(_worker, enumerate(working, start=1)):
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

    print(
        f"[SCAN STATS] attempted={attempted}, kept={kept}, "
        f"errors={errors}, rejected={rejected}"
    )

    if not rows:
        return pd.DataFrame(), "No results returned. Check universe or data source."

    # 5) Build DataFrame from kept rows
    results_df = pd.DataFrame(rows)

    # Normalize column naming to match UI
    if "symbol" in results_df.columns:
        results_df.rename(columns={"symbol": "Symbol"}, inplace=True)

    # Ensure TechRating and Signal columns exist
    if "TechRating" not in results_df.columns:
        results_df["TechRating"] = 0.0
    if "Signal" not in results_df.columns:
        results_df["Signal"] = ""

    # 5b) Cross-sectional alpha blend: upgrade TechRating using factor z-scores
    results_df = _apply_alpha_blend(results_df, regime=regime_tags)

    # Keep an unfiltered copy for fallbacks
    base_results = results_df.copy()

    # 6) TechRating filter
    status_text = "Scan complete."
    if config.min_tech_rating is not None:
        before = len(results_df)
        results_df = results_df[results_df["TechRating"] >= config.min_tech_rating]
        print(
            f"[FILTER] {len(results_df)} symbols after min_tech_rating "
            f">= {config.min_tech_rating} (from {before})."
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

    # 7) Trade planning
    results_df = plan_trades(results_df, risk)

    # 7b) Alpha/risk scaffolding
    results_df["MuHat"] = results_df["TechRating"] / 100.0
    results_df["MuMl"] = pd.Series([np.nan] * len(results_df))
    results_df["MuTotal"] = results_df["MuHat"]
    if "AlphaScore" not in results_df.columns:
        results_df["AlphaScore"] = results_df["TechRating"]
    if regime_tags:
        results_df["RegimeTrend"] = regime_tags.get("trend")
        results_df["RegimeVol"] = regime_tags.get("vol")

    # Engine-level tradeable filter (soft)
    if config.only_tradeable and "Signal" in results_df.columns:
        tradeable = {"Strong Long", "Long"}
        if config.allow_shorts:
            tradeable.update({"Strong Short", "Short"})

        tradeable_df = results_df[results_df["Signal"].isin(tradeable)]

        # Prefer tradeable, but don't drop everything if none match
        if not tradeable_df.empty:
            results_df = tradeable_df

    # ❌ REMOVE the old:
    # if results_df.empty:
    #     return results_df, "No tradeable signals under current filters."

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

    # 8) Save CSV
    output_path = OUTPUT_DIR / "technic_scan_results.csv"
    try:
        results_df.to_csv(output_path, index=False)
        print(f"[OUTPUT] Wrote scan results to: {output_path}")
    except Exception:
        print("[OUTPUT ERROR] Failed to write scan results CSV:")
        traceback.print_exc()

    return results_df, status_text


if __name__ == "__main__":
    df, msg = run_scan()
    print(msg)
    print(df.head())
