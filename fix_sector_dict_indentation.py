"""Fix the indentation issue after SECTOR_MIN_MARKET_CAP dictionary"""

from pathlib import Path

def fix_indentation():
    scanner_path = Path("technic_v4/scanner_core.py")
    
    if not scanner_path.exists():
        print(f"❌ Error: {scanner_path} not found")
        return False
    
    content = scanner_path.read_text(encoding='utf-8')
    
    # The problem is that after the closing brace of SECTOR_MIN_MARKET_CAP,
    # the next line has too much indentation (8 spaces instead of 4)
    
    # Find and fix the pattern
    content = content.replace(
        "}\n\n        results_df = results_df[results_df[\"DollarVolume\"] >= MIN_DOLLAR_VOL]",
        "}\n\n    # Minimum liquidity filter (relaxed for broader results)\n    MIN_DOLLAR_VOL = 1_000_000  # $1M/day minimum (Phase 3A)\n    if \"DollarVolume\" in results_df.columns:\n        results_df = results_df[results_df[\"DollarVolume\"] >= MIN_DOLLAR_VOL]"
    )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("✅ Fixed indentation after SECTOR_MIN_MARKET_CAP dictionary")
    return True

if __name__ == "__main__":
    fix_indentation()
