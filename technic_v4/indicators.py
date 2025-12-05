from __future__ import annotations

from typing import Optional

# indicators.py
import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # --- RSI 14 ---
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # --- ATR 14 % ---
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df["ATR14_pct"] = atr / df["Close"]

    # --- RVOL 20 ---
    # Relative volume: today's volume / 20-day average volume
    if "Volume" in df.columns:
        vol_ma20 = df["Volume"].rolling(window=20).mean()
        df["RVOL20"] = df["Volume"] / vol_ma20
    else:
        # Fallback if volume is missing
        df["RVOL20"] = np.nan

    # --- Percent from 20-day High ---
    df["PctFromHigh20"] = 100 * (df["Close"] - df["Close"].rolling(20).max()) / df["Close"].rolling(20).max()

    # --- Trend Strength 50 ---
    ma50 = df["Close"].rolling(window=50).mean()
    std50 = df["Close"].rolling(window=50).std()
    df["TrendStrength50"] = (df["Close"] - ma50) / std50

    # --- Moving Averages and Slope ---
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["SlopeMA20"] = df["MA20"].diff()

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # --- Bollinger Bands ---
    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["BB_upper"] = ma20 + (2 * std20)
    df["BB_lower"] = ma20 - (2 * std20)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]
    df["BB_pctB"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # --- ADX (Average Directional Index) ---
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()

    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
    minus_di = 100 * (abs(minus_dm).rolling(window=14).mean() / atr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX14"] = dx.rolling(window=14).mean()

    return df



def safe_series(s: pd.Series) -> pd.Series:
    """Ensure a float Series with no inf values."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.astype(float)


def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    s = safe_series(s)
    return s.pct_change(periods=periods)


# ---------- MOVING AVERAGES ----------

def sma(s: pd.Series, window: int) -> pd.Series:
    s = safe_series(s)
    return s.rolling(window=window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    s = safe_series(s)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


# ---------- RSI (RELATIVE STRENGTH INDEX) ----------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder RSI.
    """
    close = safe_series(close)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


# ---------- ATR (AVERAGE TRUE RANGE) & ATR% ----------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = safe_series(high)
    low = safe_series(low)
    close = safe_series(close)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    a = atr(high, low, close, period=period)
    close = safe_series(close)
    return (a / close) * 100.0


# ---------- RELATIVE VOLUME (RVOL) ----------

def relative_volume(
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    RVOL = today's volume / average volume over window.
    """
    volume = safe_series(volume)
    avg_vol = volume.rolling(window=window, min_periods=window).mean()
    return volume / avg_vol


# ---------- VOLATILITY & RANGE METRICS ----------

def rolling_volatility(
    close: pd.Series,
    window: int = 20,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """
    Annualized volatility based on daily returns over a rolling window.
    """
    close = safe_series(close)
    ret = close.pct_change()
    vol = ret.rolling(window=window, min_periods=window).std()
    return vol * np.sqrt(trading_days_per_year)


def percent_from_high(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    How far price is below its rolling high over `lookback` bars, in %.
    0%  = at high
    -10% = 10% below high
    """
    close = safe_series(close)
    rolling_high = close.rolling(window=lookback, min_periods=1).max()
    return (close / rolling_high - 1.0) * 100.0


def percent_from_low(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    How far price is above its rolling low over `lookback` bars, in %.
    0%  = at low
    +10% = 10% above low
    """
    close = safe_series(close)
    rolling_low = close.rolling(window=lookback, min_periods=1).min()
    return (close / rolling_low - 1.0) * 100.0


# ---------- SIMPLE TREND METRICS ----------

def slope(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling linear regression slope of the series.
    Units are 'value per bar'.
    """
    series = safe_series(series)

    def _window_slope(x: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        # x axis: 0..n-1
        xi = np.arange(n)
        # simple linear regression
        A = np.vstack([xi, np.ones(n)]).T
        m, _b = np.linalg.lstsq(A, x, rcond=None)[0]
        return m

    return series.rolling(window=window, min_periods=window).apply(
        lambda arr: _window_slope(arr), raw=True
    )


def trend_strength(
    close: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Combines slope and volatility to estimate how 'clean' the trend is.
    Higher = stronger, more directional trend.
    """
    close = safe_series(close)
    sl = slope(close, window=window)
    vol = rolling_volatility(close, window=window)
    # avoid division by zero
    vol = vol.replace(0, np.nan)
    return sl / vol
