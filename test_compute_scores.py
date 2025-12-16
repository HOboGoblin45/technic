"""
Debug script to check what compute_scores returns
"""

import pandas as pd
from technic_v4 import data_engine
from technic_v4.engine.scoring import compute_scores

# Get some test data
symbol = "AAPL"
df = data_engine.get_price_history(symbol, days=90, freq="daily")

if df is not None and not df.empty:
    print(f"Got {len(df)} bars for {symbol}")
    
    # Call compute_scores
    result = compute_scores(df)
    
    print(f"\ncompute_scores returned type: {type(result)}")
    print(f"Shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    print(f"\nColumns: {list(result.columns) if hasattr(result, 'columns') else 'N/A'}")
    
    # Check for ATR columns
    if hasattr(result, 'columns'):
        atr_cols = [c for c in result.columns if 'ATR' in c.upper()]
        print(f"\nATR-related columns: {atr_cols}")
        
        if atr_cols:
            for col in atr_cols:
                print(f"  {col}: {result[col].iloc[0] if len(result) > 0 else 'N/A'}")
    
    # Extract as Series (like _scan_symbol does)
    if len(result) > 0:
        latest = result.iloc[-1].copy()
        print(f"\nExtracted Series type: {type(latest)}")
        print(f"Series length: {len(latest)}")
        
        atr_keys = [k for k in latest.index if 'ATR' in str(k).upper()]
        print(f"\nATR keys in Series: {atr_keys}")
        
        for key in atr_keys:
            print(f"  {key}: {latest[key]}")
else:
    print(f"Failed to get data for {symbol}")
