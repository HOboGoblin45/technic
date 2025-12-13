"""
Fix overly aggressive institutional filters in scanner_core.py
that are causing 0 results to be returned.
"""

def fix_filters():
    file_path = "technic_v4/scanner_core.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the institutional filters section
    old_filters = '''    # ============================
    # INSTITUTIONAL FILTERS v1
    # ============================
    # Compute dollar volume
    if {"Close", "Volume"}.issubset(results_df.columns):
        results_df["DollarVolume"] = results_df["Close"] * results_df["Volume"]

        # Minimum liquidity filter (institution-grade)
        MIN_DOLLAR_VOL = 5_000_000  # $5M/day minimum
        results_df = results_df[results_df["DollarVolume"] >= MIN_DOLLAR_VOL]

    # Price filter ? investors don't want sub-$5 stocks
    if "Close" in results_df.columns:
        results_df = results_df[results_df["Close"] >= 5.00]

    # Market-cap filter (skip microcaps)
    if "market_cap" in results_df.columns:
        results_df = results_df[results_df["market_cap"] >= 300_000_000]  # $300M minimum
    else:
        print("WARNING: market_cap missing ? add it in feature_engine.py")

    # ATR% ceiling ? block high-volatility junk
        if "ATR14_pct" in results_df.columns:
            results_df = results_df[results_df["ATR14_pct"] <= 0.20]  # max 20% ATR%'''
    
    new_filters = '''    # ============================
    # INSTITUTIONAL FILTERS v1 (RELAXED)
    # ============================
    # Compute dollar volume
    if {"Close", "Volume"}.issubset(results_df.columns):
        results_df["DollarVolume"] = results_df["Close"] * results_df["Volume"]

        # Minimum liquidity filter (relaxed for broader results)
        MIN_DOLLAR_VOL = 500_000  # $500K/day minimum (was $5M)
        results_df = results_df[results_df["DollarVolume"] >= MIN_DOLLAR_VOL]

    # Price filter (relaxed)
    if "Close" in results_df.columns:
        results_df = results_df[results_df["Close"] >= 1.00]  # $1 minimum (was $5)

    # Market-cap filter (relaxed - skip only nano-caps)
    if "market_cap" in results_df.columns:
        results_df = results_df[results_df["market_cap"] >= 50_000_000]  # $50M minimum (was $300M)
    else:
        logger.info("[FILTER] market_cap column missing; skipping market cap filter")

    # ATR% ceiling (relaxed)
    if "ATR14_pct" in results_df.columns:
        results_df = results_df[results_df["ATR14_pct"] <= 0.50]  # max 50% ATR% (was 20%)'''
    
    if old_filters in content:
        content = content.replace(old_filters, new_filters)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed institutional filters in scanner_core.py")
        print("   - MIN_DOLLAR_VOL: $5M → $500K")
        print("   - MIN_PRICE: $5.00 → $1.00")
        print("   - MIN_MARKET_CAP: $300M → $50M")
        print("   - MAX_ATR%: 20% → 50%")
        return True
    else:
        print("❌ Could not find the institutional filters section to replace")
        return False

if __name__ == "__main__":
    fix_filters()
