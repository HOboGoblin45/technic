"""
Factor engine for multi-factor alpha features.

Computes technical, liquidity, and lightweight fundamental factors for a single
symbol using price history plus optional fundamentals snapshot. Designed to be
called inside the scan loop; keep it fast and resilient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from technic_v4.data_layer.fundamentals import FundamentalsSnapshot


@dataclass
class FactorBundle:
    """Container for per-symbol factor values."""

    factors: Dict[str, float]

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.factors.get(key, default) or default)


def _pct_return(series: pd.Series, window: int) -> float:
    if len(series) < window + 1:
        return np.nan
    recent = series.iloc[-window:]
    start = series.iloc[-window - 1]
    if start == 0:
        return np.nan
    return float((recent.iloc[-1] - start) / start)


def _realized_vol(returns: pd.Series, window: int) -> float:
    if len(returns) < window:
        return np.nan
    return float(returns.iloc[-window:].std() * np.sqrt(252))


def _atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    if {"High", "Low", "Close"} - set(df.columns):
        return np.nan
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    if atr.empty or atr.iloc[-1] == 0 or close.iloc[-1] == 0:
        return np.nan
    return float(atr.iloc[-1] / close.iloc[-1])


def _dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    if {"Close", "Volume"} - set(df.columns):
        return np.nan
    dv = (df["Close"] * df["Volume"]).rolling(window=window).mean()
    if dv.empty:
        return np.nan
    return float(dv.iloc[-1])


def _gap_stat(df: pd.DataFrame, window: int = 10) -> float:
    if {"Open", "Close"} - set(df.columns):
        return np.nan
    gap = (df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1)
    if gap.isna().all():
        return np.nan
    return float(gap.tail(window).mean())


def _ma_slope(series: pd.Series, window: int = 20) -> float:
    if len(series) < window:
        return np.nan
    ma = series.rolling(window=window).mean()
    if ma.isna().all():
        return np.nan
    y = ma.tail(window).values
    x = np.arange(len(y))
    if np.allclose(y, y[0]):
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    last = y[-1] if y[-1] != 0 else 1e-6
    return float(slope / last)


def _fundamental_ratio(raw: Dict, num_keys: tuple[str, ...], denom_key: str) -> float:
    for key in num_keys:
        if raw.get(key):
            num = float(raw[key])
            denom = float(raw.get(denom_key, 0) or 0)
            if denom != 0:
                return num / denom
    return np.nan


def _safe_ratio(raw: Dict, num_keys: tuple[str, ...], denom_keys: tuple[str, ...]) -> float:
    num = None
    for k in num_keys:
        if raw.get(k):
            num = float(raw[k])
            break
    if num is None:
        return np.nan
    denom = None
    for k in denom_keys:
        if raw.get(k):
            denom = float(raw[k])
            break
    if denom is None or denom == 0:
        return np.nan
    return num / denom


def _maybe_div_yield(raw: Dict) -> float:
    for key in ("dividend_yield", "dividendYield", "forward_dividend_yield"):
        if raw.get(key):
            return float(raw[key])
    return np.nan


def _maybe_earnings_yield(raw: Dict) -> float:
    if raw.get("earnings_yield"):
        return float(raw["earnings_yield"])
    for key in ("pe_ratio", "pe", "pe_ttm"):
        pe = raw.get(key)
        try:
            pe = float(pe)
        except Exception:
            pe = None
        if pe and pe != 0:
            return 1.0 / pe
    return np.nan


def compute_factor_bundle(
    prices: pd.DataFrame,
    fundamentals: Optional[FundamentalsSnapshot] = None,
) -> FactorBundle:
    """
    Compute a set of alpha factors for one symbol.

    Parameters
    ----------
    prices : DataFrame with OHLCV columns.
    fundamentals : optional FundamentalsSnapshot.
    """
    df = prices.copy()
    df = df.sort_index()
    closes = df.get("Close")
    rets = closes.pct_change() if closes is not None else pd.Series(dtype=float)
    raw_f = fundamentals.raw if fundamentals is not None else {}

    factors: Dict[str, float] = {}

    # Technical momentum / reversal
    factors["mom_5"] = _pct_return(closes, 5) if closes is not None else np.nan
    factors["mom_21"] = _pct_return(closes, 21) if closes is not None else np.nan
    factors["mom_63"] = _pct_return(closes, 63) if closes is not None else np.nan
    factors["reversal_5"] = -factors["mom_5"] if not np.isnan(factors["mom_5"]) else np.nan
    factors["ma_slope_20"] = _ma_slope(closes, 20) if closes is not None else np.nan

    # Volatility / liquidity
    factors["atr_pct_14"] = _atr_pct(df, 14)
    factors["vol_realized_20"] = _realized_vol(rets, 20) if rets is not None else np.nan
    factors["dollar_vol_20"] = _dollar_volume(df, 20)
    factors["gap_stat_10"] = _gap_stat(df, 10)

    # Fundamentals (best-effort; may be NaN)
    factors["value_ep"] = _fundamental_ratio(raw_f, ("net_income", "netIncome", "ni"), "market_cap")
    factors["value_cfp"] = _fundamental_ratio(raw_f, ("operating_cash_flow", "ocf"), "market_cap")
    factors["value_earnings_yield"] = _maybe_earnings_yield(raw_f)
    factors["dividend_yield"] = _maybe_div_yield(raw_f)
    factors["value_fcf_yield"] = _safe_ratio(raw_f, ("free_cash_flow", "fcf", "freeCashFlow"), ("market_cap", "marketCap"))
    factors["value_ev_ebitda"] = _safe_ratio(raw_f, ("enterprise_value", "ev", "enterpriseValue"), ("ebitda", "EBITDA"))
    factors["value_ev_sales"] = _safe_ratio(raw_f, ("enterprise_value", "ev", "enterpriseValue"), ("revenue", "sales"))
    factors["quality_roe"] = _fundamental_ratio(raw_f, ("return_on_equity", "roe"), "market_cap")
    factors["quality_roa"] = _fundamental_ratio(raw_f, ("return_on_assets", "roa"), "total_assets")
    factors["quality_gpm"] = _fundamental_ratio(raw_f, ("gross_profit", "grossProfit"), "revenue")
    factors["quality_margin_ebitda"] = _safe_ratio(raw_f, ("ebitda", "EBITDA"), ("revenue", "sales"))
    factors["quality_margin_op"] = _safe_ratio(raw_f, ("operating_income", "ebit"), ("revenue", "sales"))
    factors["quality_margin_net"] = _safe_ratio(raw_f, ("net_income", "netIncome", "ni"), ("revenue", "sales"))
    factors["leverage_de"] = _fundamental_ratio(raw_f, ("total_debt", "debt"), "total_equity")
    factors["interest_coverage"] = _fundamental_ratio(raw_f, ("ebit", "operating_income"), "interest_expense")
    factors["growth_rev"] = raw_f.get("revenue_growth") or raw_f.get("sales_growth") or np.nan
    factors["growth_eps"] = raw_f.get("eps_growth") or raw_f.get("earnings_growth") or np.nan
    factors["size_log_mcap"] = (
        float(np.log(raw_f.get("market_cap", raw_f.get("marketCap", 0)) or 0))
        if raw_f.get("market_cap") or raw_f.get("marketCap")
        else np.nan
    )

    return FactorBundle(factors=factors)


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score with NaN-safe behavior."""
    if series.empty:
        return series
    mean = series.mean(skipna=True)
    std = series.std(skipna=True)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - mean) / std
