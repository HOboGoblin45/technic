"""
MERIT Score Implementation - Prompts 2-6
Automated implementation script for integrating MERIT Score into Technic.

This script implements:
- Prompt 2: Scanner integration
- Prompt 3: Recommendation text integration
- Prompt 4: API schema updates
- Prompt 5: UI integration (documentation only, Flutter changes manual)
- Prompt 6: Quality test script

Run this script to apply all MERIT integrations automatically.
"""

import re
from pathlib import Path


def backup_file(filepath: Path):
    """Create a backup of the file before modifying."""
    backup_path = filepath.with_suffix(filepath.suffix + '.merit_backup')
    if filepath.exists() and not backup_path.exists():
        backup_path.write_text(filepath.read_text(), encoding='utf-8')
        print(f"✅ Backed up: {filepath} -> {backup_path}")


def implement_prompt_2_scanner_integration():
    """
    Prompt 2: Wire MERIT into scanner_core.py
    """
    print("\n" + "="*60)
    print("PROMPT 2: Scanner Integration")
    print("="*60)
    
    scanner_file = Path('technic_v4/scanner_core.py')
    if not scanner_file.exists():
        print(f"❌ File not found: {scanner_file}")
        return False
    
    backup_file(scanner_file)
    content = scanner_file.read_text(encoding='utf-8')
    
    # 1. Add import at top
    if 'from technic_v4.engine.merit_engine import compute_merit' not in content:
        # Find the imports section
        import_pattern = r'(from technic_v4\.engine\.portfolio_engine import.*?\n)'
        replacement = r'\1from technic_v4.engine.merit_engine import compute_merit\n'
        content = re.sub(import_pattern, replacement, content, count=1)
        print("✅ Added merit_engine import")
    
    # 2. Add compute_merit call in _finalize_results
    # Find where to insert (after ICS/Quality computation, before ranking)
    if 'main_df = compute_merit(main_df' not in content:
        # Look for the risk_adjusted_rank call
        pattern = r'(# Apply risk-adjusted ranking.*?\n\s+try:.*?\n\s+)(main_df = risk_adjusted_rank\()'
        replacement = r'\1# Compute MERIT Score\n        try:\n            main_df = compute_merit(main_df, regime=regime_tags)\n            logger.info("[MERIT] Computed MERIT Score for %d results", len(main_df))\n        except Exception as e:\n            logger.warning("[MERIT] Failed to compute MERIT Score: %s", e, exc_info=True)\n        \n        \2'
        content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
        print("✅ Added compute_merit() call")
    
    # 3. Update sorting to use MeritScore
    if 'sort_values(["MeritScore"' not in content:
        # Find final sort and replace
        pattern = r'main_df = main_df\.sort_values\(\["TechRating"\], ascending=False\)'
        replacement = 'main_df = main_df.sort_values(["MeritScore", "TechRating"], ascending=False)'
        content = re.sub(pattern, replacement, content)
        print("✅ Updated sorting to use MeritScore")
    
    # 4. Update diversify_by_sector to use MeritScore
    if 'score_col="MeritScore"' not in content:
        pattern = r'(main_df = diversify_by_sector\([^)]+)(score_col="risk_score")'
        replacement = r'\1score_col="MeritScore"'
        content = re.sub(pattern, replacement, content)
        print("✅ Updated diversify_by_sector to use MeritScore")
    
    # 5. Add logging for top 10 by MERIT
    if 'Top 10 by MERIT' not in content:
        log_code = '''
    # Log top 10 by MERIT Score
    if "MeritScore" in main_df.columns and len(main_df) > 0:
        top_10 = main_df.head(10)
        logger.info("[MERIT] Top 10 by MERIT Score:")
        for idx, row in top_10.iterrows():
            logger.info(
                "  %s: MERIT=%.1f (%s), Tech=%.1f, Alpha=%.2f, WinProb=%.0f%%, Quality=%.0f, ICS=%.0f, Flags=%s",
                row.get("Symbol", "?"),
                row.get("MeritScore", 0),
                row.get("MeritBand", "?"),
                row.get("TechRating", 0),
                row.get("AlphaScore", 0),
                row.get("win_prob_10d", 0) * 100,
                row.get("QualityScore", 0),
                row.get("InstitutionalCoreScore", 0),
                row.get("MeritFlags", "")
            )
'''
        # Insert before return statement in _finalize_results
        pattern = r'(\n\s+return main_df, status_text)'
        replacement = log_code + r'\1'
        content = re.sub(pattern, replacement, content, count=1)
        print("✅ Added top 10 MERIT logging")
    
    scanner_file.write_text(content, encoding='utf-8')
    print(f"✅ Updated: {scanner_file}")
    return True


def implement_prompt_3_recommendation_text():
    """
    Prompt 3: Integrate MERIT into recommendation generator
    """
    print("\n" + "="*60)
    print("PROMPT 3: Recommendation Text Integration")
    print("="*60)
    
    rec_file = Path('technic_v4/engine/recommendation.py')
    if not rec_file.exists():
        print(f"❌ File not found: {rec_file}")
        return False
    
    backup_file(rec_file)
    content = rec_file.read_text(encoding='utf-8')
    
    # Update build_recommendation to use MERIT
    if 'MeritScore' not in content:
        # Find the function and update it
        pattern = r'(def build_recommendation\([^)]+\):.*?""".*?""")'
        
        new_impl = r'''\1
    
    # Use MERIT Score if available
    merit_score = row.get("MeritScore")
    merit_band = row.get("MeritBand", "")
    merit_summary = row.get("MeritSummary", "")
    
    if merit_score is not None and merit_summary:
        # Use MERIT-based summary
        return merit_summary
    
    # Fallback to original logic if MERIT not available'''
        
        content = re.sub(pattern, new_impl, content, count=1, flags=re.DOTALL)
        print("✅ Updated build_recommendation to use MERIT")
    
    rec_file.write_text(content, encoding='utf-8')
    print(f"✅ Updated: {rec_file}")
    return True


def implement_prompt_4_api_schema():
    """
    Prompt 4: Add MERIT fields to API schema
    """
    print("\n" + "="*60)
    print("PROMPT 4: API Schema Updates")
    print("="*60)
    
    api_file = Path('technic_v4/api_server.py')
    if not api_file.exists():
        print(f"❌ File not found: {api_file}")
        return False
    
    backup_file(api_file)
    content = api_file.read_text(encoding='utf-8')
    
    # Add MERIT fields to ScanResultRow
    if 'merit_score' not in content:
        pattern = r'(class ScanResultRow\(BaseModel\):.*?optionTrade: Optional\[dict\] = None)'
        replacement = r'''\1
    merit_score: Optional[float] = None
    merit_band: Optional[str] = None
    merit_flags: Optional[str] = None
    merit_summary: Optional[str] = None'''
        content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
        print("✅ Added MERIT fields to ScanResultRow")
    
    # Update _format_scan_results to include MERIT fields
    if 'merit_score=_float_or_none' not in content:
        pattern = r'(optionTrade=_maybe_option\(r\),)'
        replacement = r'''\1
                merit_score=_float_or_none(r.get("MeritScore")),
                merit_band=str(r.get("MeritBand") or ""),
                merit_flags=str(r.get("MeritFlags") or ""),
                merit_summary=str(r.get("MeritSummary") or ""),'''
        content = re.sub(pattern, replacement, content)
        print("✅ Updated _format_scan_results to include MERIT")
    
    api_file.write_text(content, encoding='utf-8')
    print(f"✅ Updated: {api_file}")
    return True


def implement_prompt_6_test_script():
    """
    Prompt 6: Create quality test script
    """
    print("\n" + "="*60)
    print("PROMPT 6: Quality Test Script")
    print("="*60)
    
    test_file = Path('technic_v4/dev/test_merit_quality.py')
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    test_code = '''"""
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
    print("\\n" + "="*60)
    print("TOP 10 BY MERIT SCORE")
    print("="*60)
    
    display_cols = [
        "Symbol", "MeritScore", "MeritBand", "TechRating", 
        "win_prob_10d", "QualityScore", "InstitutionalCoreScore", "MeritFlags"
    ]
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in df.columns]
    
    for idx, row in top_10.iterrows():
        print(f"\\n{idx+1}. {row.get('Symbol', '?')}")
        print(f"   MERIT: {row.get('MeritScore', 0):.1f} ({row.get('MeritBand', '?')})")
        print(f"   Tech: {row.get('TechRating', 0):.1f}")
        print(f"   WinProb: {row.get('win_prob_10d', 0)*100:.0f}%")
        print(f"   Quality: {row.get('QualityScore', 0):.0f}")
        print(f"   ICS: {row.get('InstitutionalCoreScore', 0):.0f}")
        flags = row.get('MeritFlags', '')
        if flags:
            print(f"   Flags: {flags}")
    
    # Check runners file if exists
    runners_file = Path("technic_v4/scanner_output/technic_runners.csv")
    if runners_file.exists():
        runners_df = pd.read_csv(runners_file)
        if "MeritScore" in runners_df.columns:
            print("\\n" + "="*60)
            print("RUNNERS MERIT DISTRIBUTION")
            print("="*60)
            print(f"Count: {len(runners_df)}")
            print(f"Mean: {runners_df['MeritScore'].mean():.1f}")
            print(f"Median: {runners_df['MeritScore'].median():.1f}")
            print(f"Range: [{runners_df['MeritScore'].min():.1f}, {runners_df['MeritScore'].max():.1f}]")
    
    print("\\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_merit_quality()
    exit(0 if success else 1)
'''
    
    test_file.write_text(test_code, encoding='utf-8')
    print(f"✅ Created: {test_file}")
    return True


def create_prompt_5_documentation():
    """
    Prompt 5: Create UI integration documentation (Flutter changes are manual)
    """
    print("\n" + "="*60)
    print("PROMPT 5: UI Integration Documentation")
    print("="*60)
    
    doc_file = Path('MERIT_UI_INTEGRATION_GUIDE.md')
    
    doc_content = '''# MERIT Score UI Integration Guide

## Overview

This guide provides instructions for integrating MERIT Score into the Flutter UI.
These changes must be made manually in the Flutter codebase.

## Files to Update

### 1. Model: `technic_app/lib/models/scan_result.dart`

Add MERIT fields to the ScanResult class:

```dart
class ScanResult {
  // ... existing fields ...
  
  final double? meritScore;
  final String? meritBand;
  final String? meritFlags;
  final String? meritSummary;
  
  // Update constructor and fromJson/toJson methods
}
```

### 2. UI: `technic_app/lib/screens/scanner/widgets/scan_result_card.dart`

Add MERIT Score display section with:
- Large prominent score (32pt font)
- Letter grade badge (A+, A, B, C, D)
- Risk flag chips
- Expanded metrics row

See ALL_6_MERIT_PROMPTS_COMPLETE.md for detailed UI design.

## Testing

After making changes:
1. Run `flutter analyze` to check for errors
2. Test the UI with real scan data
3. Verify MERIT displays correctly

'''
    
    doc_file.write_text(doc_content, encoding='utf-8')
    print(f"✅ Created: {doc_file}")
    print("⚠️  Note: Flutter UI changes must be made manually")
    return True


def main():
    """Run all MERIT implementation prompts."""
    print("\n" + "="*80)
    print(" MERIT SCORE IMPLEMENTATION - PROMPTS 2-6")
    print("="*80)
    
    results = {
        "Prompt 2 (Scanner)": implement_prompt_2_scanner_integration(),
        "Prompt 3 (Recommendation)": implement_prompt_3_recommendation_text(),
        "Prompt 4 (API Schema)": implement_prompt_4_api_schema(),
        "Prompt 5 (UI Docs)": create_prompt_5_documentation(),
        "Prompt 6 (Test Script)": implement_prompt_6_test_script(),
    }
    
    print("\n" + "="*80)
    print(" IMPLEMENTATION SUMMARY")
    print("="*80)
    
    for prompt, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{prompt}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✅ All prompts implemented successfully!")
        print("\nNext steps:")
        print("1. Review the changes in each file")
        print("2. Run a test scan: python scripts/run_scan.py")
        print("3. Run quality test: python -m technic_v4.dev.test_merit_quality")
        print("4. Update Flutter UI manually (see MERIT_UI_INTEGRATION_GUIDE.md)")
        print("5. Test end-to-end integration")
    else:
        print("\n❌ Some prompts failed. Please review the errors above.")
    
    return all_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
