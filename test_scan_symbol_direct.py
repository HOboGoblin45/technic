"""
Test what _scan_symbol actually returns
"""

import pandas as pd
from technic_v4.scanner_core import _scan_symbol
from technic_v4 import data_engine

# Get test data
symbol = "AAPL"
price_cache = {symbol: data_engine.get_price_history(symbol, days=90, freq="daily")}

# Call _scan_symbol
result = _scan_symbol(
    symbol=symbol,
    lookback_days=90,
    trade_style="Short-term swing",
    as_of_date=None,
    price_cache=price_cache
)

print(f"_scan_symbol returned type: {type(result)}")
print(f"Length: {len(result) if result is not None else 'None'}")

if result is not None:
    print(f"\nAll keys ({len(result)}):")
    for i, key in enumerate(result.index):
        print(f"  {i+1}. {key}")
    
    # Check for specific columns
    atr_keys = [k for k in result.index if 'ATR' in str(k).upper()]
    print(f"\nATR keys: {atr_keys}")
    
    signal_keys = [k for k in result.index if 'signal' in str(k).lower()]
    print(f"Signal keys: {signal_keys}")
    
    score_keys = [k for k in result.index if 'Score' in str(k)]
    print(f"Score keys: {score_keys}")
