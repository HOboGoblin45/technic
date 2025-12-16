"""
Debug script to check what columns are in the DataFrame at each stage
"""

import pandas as pd
from technic_v4.scanner_core import run_scan, ScanConfig

# Monkey-patch to add debug logging
original_finalize = None

def debug_finalize(config, results_df, risk, regime_tags, settings=None, as_of_date=None):
    print(f"\n[DEBUG] _finalize_results called")
    print(f"[DEBUG] results_df shape: {results_df.shape}")
    print(f"[DEBUG] results_df columns ({len(results_df.columns)}): {list(results_df.columns[:30])}")
    
    atr_cols = [c for c in results_df.columns if 'ATR' in c.upper()]
    print(f"[DEBUG] ATR columns: {atr_cols}")
    
    # Call original
    return original_finalize(config, results_df, risk, regime_tags, settings, as_of_date)

# Patch it
import technic_v4.scanner_core as sc
original_finalize = sc._finalize_results
sc._finalize_results = debug_finalize

# Run a small scan
config = ScanConfig(max_symbols=10, lookback_days=90)
df, msg = run_scan(config)

print(f"\n[DEBUG] Final DataFrame shape: {df.shape}")
print(f"[DEBUG] Final DataFrame columns: {list(df.columns[:30])}")
