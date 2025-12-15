"""Pytest configuration and shared fixtures"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(100) * 0.5,
        'High': close_prices + np.abs(np.random.randn(100) * 1.5),
        'Low': close_prices - np.abs(np.random.randn(100) * 1.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100),
    })
    
    df.set_index('Date', inplace=True)
    return df


@pytest.fixture
def sample_scan_result():
    """Generate sample scan result for testing"""
    return {
        'Symbol': 'AAPL',
        'Close': 150.0,
        'TechRating': 75.0,
        'Signal': 'Long',
        'RiskScore': 0.8,
        'ATR_pct_14': 0.02,
        'RSI14': 60.0,
        'MA20': 148.0,
        'MA50': 145.0,
        'MA200': 140.0,
        'Volume': 50000000,
        'DollarVolume': 7500000000,
    }


@pytest.fixture
def sample_fundamentals():
    """Generate sample fundamental data for testing"""
    return {
        'market_cap': 2500000000000,  # $2.5T
        'pe_ratio': 25.0,
        'earnings_yield': 0.04,
        'roe': 0.35,
        'gross_margin': 0.42,
        'debt_to_equity': 1.5,
    }


@pytest.fixture
def mock_api_response():
    """Mock API response for testing"""
    return {
        'status': 'ok',
        'results': [
            {
                'symbol': 'AAPL',
                'signal': 'Long',
                'techRating': 75.0,
                'entry': 150.0,
                'stop': 145.0,
                'target': 160.0,
            }
        ]
    }
