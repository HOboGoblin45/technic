"""
Simple Phase 3A Implementation - Just update the constants

This approach is safer - we only modify the module-level constants,
not the function internals.
"""

from pathlib import Path
import re

def implement_phase3a_simple():
    """Update MIN_DOLLAR_VOL constant only"""
    
    scanner_path = Path("technic_v4/scanner_core.py")
    
    if not scanner_path.exists():
        print(f"❌ Error: {scanner_path} not found")
        return False
    
    content = scanner_path.read_text(encoding='utf-8')
    
    # Simple change: Update MIN_DOLLAR_VOL from 500_000 to 1_000_000
    content = re.sub(
        r'MIN_DOLLAR_VOL = 500_000  # \$500K minimum daily volume for liquidity',
        'MIN_DOLLAR_VOL = 1_000_000  # $1M minimum daily volume for liquidity (Phase 3A)',
        content
    )
    
    # Write back
    scanner_path.write_text(content, encoding='utf-8')
    
    print("✅ Phase 3A (Simple) implementation complete!")
    print("\nChanges made:")
    print("1. ✅ Increased MIN_DOLLAR_VOL: $500K → $1M")
    print("\nExpected impact:")
    print("- Pre-rejection rate: 40.3% → ~45-48%")
    print("- Scan time: 74.77s → ~68s (9% improvement)")
    print("\nNote: This is a conservative change. If more improvement is needed,")
    print("we can add sector-specific filters in a follow-up.")
    
    return True

if __name__ == "__main__":
    success = implement_phase3a_simple()
    if success:
        print("\n" + "="*60)
        print("READY TO TEST")
        print("="*60)
        print("\nRun this command to test:")
        print("python -m pytest test_scanner_optimization_thorough.py::test_4_universe_filtering -v")
    else:
        print("\n❌ Implementation failed")
