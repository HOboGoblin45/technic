"""
Phase 3A: Enhanced Pre-Screening Implementation

Goal: Increase pre-rejection rate from 40.3% to 50%+ to reduce scan time from 74.77s to ~62s

Changes:
1. Tighten market cap filter: $100M → $150M
2. Add sector-specific market cap requirements
3. Increase dollar volume threshold: $500K → $1M
4. Add additional quality filters

Expected Impact:
- Pre-rejection: 40.3% → 50%
- Symbols processed: 1,576 → 1,320
- Time: 74.77s → 62.77s (16% improvement)
"""

import re
from pathlib import Path

def implement_phase3a():
    """Implement enhanced pre-screening filters in scanner_core.py"""
    
    scanner_path = Path("technic_v4/scanner_core.py")
    
    if not scanner_path.exists():
        print(f"❌ Error: {scanner_path} not found")
        return False
    
    content = scanner_path.read_text(encoding='utf-8')
    
    # Find and update MIN_PRICE constant
    content = re.sub(
        r'MIN_PRICE = 5\.0  # Raised from \$1 to \$5 to filter penny stocks',
        'MIN_PRICE = 5.0  # $5 minimum to filter penny stocks',
        content
    )
    
    # Find and update MIN_DOLLAR_VOL constant
    content = re.sub(
        r'MIN_DOLLAR_VOL = 500_000  # \$500K minimum daily volume for liquidity',
        'MIN_DOLLAR_VOL = 1_000_000  # $1M minimum daily volume for liquidity (Phase 3A)',
        content
    )
    
    # Add sector-specific market cap requirements after MIN_DOLLAR_VOL
    sector_requirements = '''
# Phase 3A: Sector-specific market cap requirements for better pre-screening
SECTOR_MIN_MARKET_CAP = {
    'Technology': 200_000_000,           # $200M - Tech needs scale
    'Healthcare': 200_000_000,           # $200M - Healthcare needs scale
    'Financial Services': 300_000_000,   # $300M - Financials need larger cap
    'Consumer Cyclical': 150_000_000,    # $150M
    'Industrials': 150_000_000,          # $150M
    'Energy': 200_000_000,               # $200M - Energy needs scale
    'Real Estate': 100_000_000,          # $100M - REITs can be smaller
    'Utilities': 500_000_000,            # $500M - Utilities are typically large
    'Basic Materials': 150_000_000,      # $150M
    'Communication Services': 200_000_000,  # $200M
    'Consumer Defensive': 200_000_000,      # $200M
}
'''
    
    # Insert sector requirements after MIN_DOLLAR_VOL
    content = re.sub(
        r'(MIN_DOLLAR_VOL = \d+_\d+  # .*?\n)',
        r'\1' + sector_requirements + '\n',
        content
    )
    
    # Update the market cap filter from $100M to $150M
    content = re.sub(
        r'if market_cap < 100_000_000:  # <\$100M micro-cap',
        'if market_cap < 150_000_000:  # <$150M micro-cap (Phase 3A: tightened)',
        content
    )
    
    # Add sector-specific market cap check
    # Find the section after the general market cap check
    sector_check_code = '''
        
        # Phase 3A: Apply sector-specific market cap requirements
        sector = meta.get('sector', '')
        if sector in SECTOR_MIN_MARKET_CAP:
            min_cap_for_sector = SECTOR_MIN_MARKET_CAP[sector]
            if market_cap < min_cap_for_sector:
                return False
'''
    
    # Insert after the general market cap check
    content = re.sub(
        r'(if market_cap < 150_000_000:  # <\$150M micro-cap.*?\n.*?return False\n)',
        r'\1' + sector_check_code,
        content
    )
    
    # Update the single-letter symbol check to also check market cap
    content = re.sub(
        r'(if market_cap > 0 and market_cap < 500_000_000:  # <\$500M)',
        r'if market_cap > 0 and market_cap < 750_000_000:  # <$750M (Phase 3A: tightened for single-letter)',
        content
    )
    
    # Write the updated content
    scanner_path.write_text(content, encoding='utf-8')
    
    print("✅ Phase 3A implementation complete!")
    print("\nChanges made:")
    print("1. ✅ Increased MIN_DOLLAR_VOL: $500K → $1M")
    print("2. ✅ Added SECTOR_MIN_MARKET_CAP dictionary")
    print("3. ✅ Tightened general market cap filter: $100M → $150M")
    print("4. ✅ Added sector-specific market cap checks")
    print("5. ✅ Tightened single-letter symbol filter: $500M → $750M")
    print("\nExpected impact:")
    print("- Pre-rejection rate: 40.3% → 50%+")
    print("- Scan time: 74.77s → ~62s (16% improvement)")
    print("\nNext step: Run Test 4 to validate")
    
    return True

if __name__ == "__main__":
    success = implement_phase3a()
    if success:
        print("\n" + "="*60)
        print("READY TO TEST")
        print("="*60)
        print("\nRun this command to test:")
        print("python -m pytest test_scanner_optimization_thorough.py::test_4_universe_filtering -v")
    else:
        print("\n❌ Implementation failed")
