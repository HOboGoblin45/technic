"""Remove duplicate MIN_DOLLAR_VOL definition inside function"""

from pathlib import Path

def fix_duplicate():
    scanner_path = Path("technic_v4/scanner_core.py")
    
    if not scanner_path.exists():
        print(f"❌ Error: {scanner_path} not found")
        return False
    
    content = scanner_path.read_text(encoding='utf-8')
    
    # Remove the duplicate MIN_DOLLAR_VOL definition that was incorrectly added inside a function
    content = content.replace(
        "    # Minimum liquidity filter (relaxed for broader results)\n    MIN_DOLLAR_VOL = 1_000_000  # $1M/day minimum (Phase 3A)\n    if \"DollarVolume\" in results_df.columns:\n        results_df = results_df[results_df[\"DollarVolume\"] >= MIN_DOLLAR_VOL]",
        "    # Minimum liquidity filter (relaxed for broader results)\n    if \"DollarVolume\" in results_df.columns:\n        results_df = results_df[results_df[\"DollarVolume\"] >= MIN_DOLLAR_VOL]"
    )
    
    scanner_path.write_text(content, encoding='utf-8')
    print("✅ Removed duplicate MIN_DOLLAR_VOL definition")
    return True

if __name__ == "__main__":
    fix_duplicate()
