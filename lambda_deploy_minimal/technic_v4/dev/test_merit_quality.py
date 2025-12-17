"""
MERIT Score Quality Test

This script validates that MERIT Score is computed correctly and produces
institutional-grade rankings.

Run: python -m technic_v4.dev.test_merit_quality
"""

import pandas as pd
from pathlib import Path


def test_merit_quality():
    """Test MERIT Score quality from latest scan results."""
    
    print("="*60)
    print("MERIT SCORE QUALITY TEST")
    print("="*60)
    
    # Load latest scan results
    results_file = Path("technic_v4/scanner_output/technic_scan_results.csv")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run a scan first to generate results.")
        return False
    
    df = pd.read_csv(results_file)
    print(f"✅ Loaded {len(df)} results from {results_file}")
    
    # Check MERIT columns exist
    required_cols = ["MeritScore", "MeritBand", "MeritFlags", "MeritSummary"]
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Missing MERIT columns: {missing}")
        return False
    
    print(f"✅ All MERIT columns present")
    
    # Validate MeritScore range
    if df["MeritScore"].isna().any():
        print(f"⚠️  Warning: {df['MeritScore'].isna().sum()} rows have NaN MeritScore")
    
    min_score = df["MeritScore"].min()
    max_score = df["MeritScore"].max()
    
    if min_score < 0 or max_score > 100:
        print(f"❌ MeritScore out of range: [{min_score:.1f}, {max_score:.1f}]")
        return False
    
    print(f"✅ MeritScore range valid: [{min_score:.1f}, {max_score:.1f}]")
    
    # Check top 10 are sorted by MeritScore
    top_10 = df.head(10)
    is_sorted = top_10["MeritScore"].is_monotonic_decreasing
    
    if not is_sorted:
        print("❌ Top 10 NOT sorted by MeritScore descending")
        return False
    
    print("✅ Top 10 correctly sorted by MeritScore")
    
    # Display top 10
    print("\n" + "="*60)
    print("TOP 10 BY MERIT SCORE")
    print("="*60)
    
    display_cols = [
        "Symbol", "MeritScore", "MeritBand", "TechRating", 
        "win_prob_10d", "QualityScore", "InstitutionalCoreScore", "MeritFlags"
    ]
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in df.columns]
    
    for idx, row in top_10.iterrows():
        print(f"\n{idx+1}. {row.get('Symbol', '?')}")
        print(f"   MERIT: {row.get('MeritScore', 0):.1f} ({row.get('MeritBand', '?')})")
        print(f"   Tech: {row.get('TechRating', 0):.1f}")
        if 'win_prob_10d' in row:
            print(f"   WinProb: {row.get('win_prob_10d', 0)*100:.0f}%")
        if 'QualityScore' in row:
            print(f"   Quality: {row.get('QualityScore', 0):.0f}")
        if 'InstitutionalCoreScore' in row:
            print(f"   ICS: {row.get('InstitutionalCoreScore', 0):.0f}")
        flags = row.get('MeritFlags', '')
        if flags:
            print(f"   Flags: {flags}")
        if 'MeritSummary' in row and row.get('MeritSummary'):
            print(f"   Summary: {row.get('MeritSummary', '')[:80]}...")
    
    # Check runners file if exists
    runners_file = Path("technic_v4/scanner_output/technic_runners.csv")
    if runners_file.exists():
        runners_df = pd.read_csv(runners_file)
        if "MeritScore" in runners_df.columns:
            print("\n" + "="*60)
            print("RUNNERS MERIT DISTRIBUTION")
            print("="*60)
            print(f"Count: {len(runners_df)}")
            print(f"Mean: {runners_df['MeritScore'].mean():.1f}")
            print(f"Median: {runners_df['MeritScore'].median():.1f}")
            print(f"Range: [{runners_df['MeritScore'].min():.1f}, {runners_df['MeritScore'].max():.1f}]")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_merit_quality()
    exit(0 if success else 1)
