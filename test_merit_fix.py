"""
Quick test to verify MERIT Score bug fix
"""
import pandas as pd
import numpy as np
from technic_v4.engine.merit_engine import compute_merit

# Create test DataFrame with non-sequential index (this was causing the bug)
test_data = {
    'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'TechRating': [75.0, 80.0, 70.0],
    'AlphaScore': [0.6, 0.7, 0.5],
    'Close': [150.0, 300.0, 2800.0],
    'Volume': [50000000, 30000000, 20000000],
    'ATR14_pct': [0.02, 0.025, 0.03],
    'IsUltraRisky': [False, False, True],
    'market_cap': [2e12, 2.5e12, 1.8e12],
    'QualityScore': [75.0, 80.0, 70.0],
    'InstitutionalCoreScore': [70.0, 75.0, 65.0]
}

# Use non-sequential index (100, 200, 300) to trigger the original bug
df = pd.DataFrame(test_data, index=[100, 200, 300])

print("Testing MERIT Score computation with non-sequential index...")
print(f"DataFrame index: {list(df.index)}")
print()

try:
    result = compute_merit(df)
    print("✅ SUCCESS! MERIT Score computed without errors")
    print()
    print("Results:")
    print(result[['Symbol', 'MeritScore', 'MeritBand', 'MeritFlags']].to_string())
    print()
    print("MERIT bug is FIXED! ✓")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
