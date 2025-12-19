"""
Market Conditions Tracker
Captures current market state for ML features
"""

from datetime import datetime, time
from typing import Dict, Any, Optional
import numpy as np


def get_current_market_conditions() -> Dict[str, Any]:
    """
    Get current market conditions for ML features
    
    Returns:
        Dictionary with market condition features
    """
    try:
        # Import data engine
        from technic_v4 import data_engine
        
        # Get SPY data for market analysis
        spy = data_engine.get_price_history("SPY", days=30, freq="daily")
        
        if spy is None or spy.empty:
            return _get_default_conditions()
        
        # Calculate market features
        conditions = {
            'spy_trend': _calculate_trend(spy),
            'spy_volatility': _calculate_volatility(spy),
            'spy_momentum': _calculate_momentum(spy),
            'spy_return_5d': _calculate_return(spy, days=5),
            'spy_return_20d': _calculate_return(spy, days=20),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_market_hours': _is_market_open(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get VIX if available
        try:
            vix = data_engine.get_price_history("VIX", days=5, freq="daily")
            if vix is not None and not vix.empty:
                conditions['vix_level'] = float(vix['close'].iloc[-1])
                conditions['vix_change'] = float(vix['close'].pct_change().iloc[-1])
        except:
            conditions['vix_level'] = None
            conditions['vix_change'] = None
        
        return conditions
        
    except Exception as e:
        print(f"Warning: Could not get market conditions: {e}")
        return _get_default_conditions()


def _calculate_trend(df) -> str:
    """
    Calculate market trend (bullish/bearish/neutral)
    
    Args:
        df: Price dataframe with 'close' column
    
    Returns:
        Trend classification
    """
    if df is None or df.empty or len(df) < 20:
        return 'neutral'
    
    # Calculate moving averages
    sma_5 = df['close'].rolling(5).mean().iloc[-1]
    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    current = df['close'].iloc[-1]
    
    # Determine trend
    if current > sma_5 > sma_20:
        return 'bullish'
    elif current < sma_5 < sma_20:
        return 'bearish'
    else:
        return 'neutral'


def _calculate_volatility(df) -> float:
    """
    Calculate market volatility (annualized)
    
    Args:
        df: Price dataframe with 'close' column
    
    Returns:
        Annualized volatility
    """
    if df is None or df.empty or len(df) < 2:
        return 0.15  # Default volatility
    
    # Calculate daily returns
    returns = df['close'].pct_change().dropna()
    
    # Annualize volatility (252 trading days)
    volatility = returns.std() * np.sqrt(252)
    
    return float(volatility)


def _calculate_momentum(df) -> float:
    """
    Calculate price momentum
    
    Args:
        df: Price dataframe with 'close' column
    
    Returns:
        Momentum score (-1 to 1)
    """
    if df is None or df.empty or len(df) < 10:
        return 0.0
    
    # Calculate rate of change over 10 days
    roc = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
    
    # Normalize to -1 to 1 range
    momentum = np.tanh(roc * 10)  # Scale and bound
    
    return float(momentum)


def _calculate_return(df, days: int = 5) -> float:
    """
    Calculate return over specified days
    
    Args:
        df: Price dataframe with 'close' column
        days: Number of days for return calculation
    
    Returns:
        Return as decimal (e.g., 0.05 for 5%)
    """
    if df is None or df.empty or len(df) < days + 1:
        return 0.0
    
    return_pct = (df['close'].iloc[-1] / df['close'].iloc[-(days+1)] - 1)
    
    return float(return_pct)


def _is_market_open() -> bool:
    """
    Check if market is currently open
    
    Returns:
        True if market is open
    """
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    # Note: This is simplified and doesn't account for holidays
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()
    
    return market_open <= current_time <= market_close


def _get_default_conditions() -> Dict[str, Any]:
    """
    Get default market conditions when data unavailable
    
    Returns:
        Dictionary with default values
    """
    return {
        'spy_trend': 'neutral',
        'spy_volatility': 0.15,
        'spy_momentum': 0.0,
        'spy_return_5d': 0.0,
        'spy_return_20d': 0.0,
        'vix_level': None,
        'vix_change': None,
        'time_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'is_market_hours': _is_market_open(),
        'timestamp': datetime.now().isoformat()
    }


def format_market_conditions(conditions: Dict[str, Any]) -> str:
    """
    Format market conditions for display
    
    Args:
        conditions: Market conditions dictionary
    
    Returns:
        Formatted string
    """
    lines = [
        f"Market Conditions ({conditions.get('timestamp', 'N/A')})",
        f"  Trend: {conditions.get('spy_trend', 'N/A')}",
        f"  Volatility: {conditions.get('spy_volatility', 0):.2%}",
        f"  Momentum: {conditions.get('spy_momentum', 0):.2f}",
        f"  5-day return: {conditions.get('spy_return_5d', 0):.2%}",
        f"  20-day return: {conditions.get('spy_return_20d', 0):.2%}",
    ]
    
    if conditions.get('vix_level'):
        lines.append(f"  VIX: {conditions['vix_level']:.2f}")
    
    lines.append(f"  Market hours: {'Yes' if conditions.get('is_market_hours') else 'No'}")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    # Test market conditions
    print("Testing Market Conditions Tracker...")
    
    conditions = get_current_market_conditions()
    
    print("\nCurrent Market Conditions:")
    print(format_market_conditions(conditions))
    
    print("\n✓ Market Conditions test complete!")
