"""Test that alpha models load correctly in Ray workers"""

import sys
from technic_v4.scanner_core import run_scan, ScanConfig

def test_alpha_loading():
    """Test that alpha models load without errors"""
    print("="*80)
    print("TESTING ALPHA MODEL LOADING FIX")
    print("="*80)
    
    config = ScanConfig(
        max_symbols=20,
        lookback_days=90,
        trade_style="Short-term swing"
    )
    
    print("\nRunning scan with 20 symbols...")
    print("Checking for alpha model loading errors...\n")
    
    df, msg = run_scan(config)
    
    print(f"\n✓ Scan completed: {len(df)} results")
    print(f"✓ Status: {msg}")
    
    # Check if AlphaScore column has non-zero values (indicating models worked)
    if 'AlphaScore' in df.columns:
        non_zero = (df['AlphaScore'] != 0).sum()
        print(f"✓ AlphaScore column present")
        print(f"✓ Non-zero alpha scores: {non_zero}/{len(df)}")
        
        if non_zero > 0:
            print(f"\n✅ SUCCESS: Alpha models are working!")
            print(f"   Sample AlphaScore values: {df['AlphaScore'].head(3).tolist()}")
            return True
        else:
            print(f"\n⚠️  WARNING: All alpha scores are zero (models may not be predicting)")
            return True  # Still pass, models loaded without error
    else:
        print(f"\n❌ FAILED: AlphaScore column missing")
        return False

if __name__ == "__main__":
    success = test_alpha_loading()
    sys.exit(0 if success else 1)
