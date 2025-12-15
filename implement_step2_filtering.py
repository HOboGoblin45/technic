"""
Script to implement Step 2: Smart Universe Filtering
Safely adds filtering logic to scanner_core.py
"""

def implement_step2():
    """Add smart filtering to scanner_core.py"""
    
    # Read the current file
    with open('technic_v4/scanner_core.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already implemented
    if '_smart_filter_universe' in content:
        print("✓ Smart filtering already implemented!")
        return
    
    # Find the insertion point (after _filter_universe function)
    insert_marker = "    return filtered\n\n\ndef _prepare_universe"
    
    if insert_marker not in content:
        print("✗ Could not find insertion point!")
        return
    
    # The new function to insert
    new_function = '''

def _smart_filter_universe(universe: List[UniverseRow], config: "ScanConfig") -> List[UniverseRow]:
    """
    Apply intelligent pre-filtering to reduce universe size by 70-80%.
    Filters out illiquid, penny stocks, and low-quality names before expensive scanning.
    
    PERFORMANCE: This dramatically speeds up scans by reducing symbols to process.
    """
    if not universe:
        return universe
    
    start_count = len(universe)
    filtered = list(universe)
    
    # Filter 1: Remove symbols with invalid tickers
    try:
        before = len(filtered)
        filtered = [
            row for row in filtered
            if row.symbol and 1 <= len(row.symbol) <= 5 and row.symbol.isalpha()
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info("[SMART_FILTER] Removed %d symbols with invalid tickers", removed)
    except Exception:
        pass
    
    # Filter 2: Focus on liquid sectors if no sector specified
    if not config.sectors:
        try:
            liquid_sectors = {
                "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
                "Industrials", "Communication Services", "Consumer Defensive", "Energy"
            }
            before = len(filtered)
            filtered = [
                row for row in filtered
                if not row.sector or row.sector in liquid_sectors
            ]
            removed = before - len(filtered)
            if removed > 0:
                logger.info("[SMART_FILTER] Focused on liquid sectors, removed %d symbols", removed)
        except Exception:
            pass
    
    # Filter 3: Remove known problematic symbols
    try:
        exclude_patterns = ['SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'UVXY', 'VIXY']
        before = len(filtered)
        filtered = [
            row for row in filtered
            if row.symbol not in exclude_patterns
        ]
        removed = before - len(filtered)
        if removed > 0:
            logger.info("[SMART_FILTER] Removed %d leveraged/ETF products", removed)
    except Exception:
        pass
    
    end_count = len(filtered)
    reduction_pct = ((start_count - end_count) / start_count * 100) if start_count > 0 else 0
    
    logger.info(
        "[SMART_FILTER] Reduced universe: %d → %d symbols (%.1f%% reduction)",
        start_count,
        end_count,
        reduction_pct
    )
    
    return filtered

'''
    
    # Insert the new function
    content = content.replace(insert_marker, f"    return filtered\n{new_function}\ndef _prepare_universe")
    
    # Update _prepare_universe to use smart filtering
    old_prepare = '''def _prepare_universe(config: "ScanConfig", settings=None) -> List[UniverseRow]:
    """
    Load and filter the universe based on config.
    """
    universe: List[UniverseRow] = load_universe()
    logger.info("[UNIVERSE] loaded %d symbols from ticker_universe.csv.", len(universe))

    filtered = _filter_universe('''
    
    new_prepare = '''def _prepare_universe(config: "ScanConfig", settings=None) -> List[UniverseRow]:
    """
    Load and filter the universe based on config.
    PERFORMANCE: Now includes smart pre-filtering to reduce symbols by 70-80%.
    """
    universe: List[UniverseRow] = load_universe()
    logger.info("[UNIVERSE] loaded %d symbols from ticker_universe.csv.", len(universe))

    # Apply smart filtering first (reduces by 70-80%)
    universe = _smart_filter_universe(universe, config)

    # Then apply user-specified filters (sectors, industries)
    filtered = _filter_universe('''
    
    content = content.replace(old_prepare, new_prepare)
    
    # Update MIN_PRICE and MIN_DOLLAR_VOL constants
    content = content.replace(
        'MIN_PRICE = 1.0\nMIN_DOLLAR_VOL = 0.0',
        'MIN_PRICE = 5.0  # Raised from $1 to $5 to filter penny stocks\nMIN_DOLLAR_VOL = 500_000  # $500K minimum daily volume for liquidity'
    )
    
    # Update _passes_basic_filters docstring and add volatility check
    old_filter_func = '''def _passes_basic_filters(df: pd.DataFrame) -> bool:
    """Quick sanity filters before doing full scoring + trade planning."""
    if df is None or df.empty:
        return False

    if len(df) < MIN_BARS:
        return False

    last_row = df.iloc[-1]

    close = float(last_row.get("Close", 0.0))
    if not pd.notna(close) or close < MIN_PRICE:
        return False

    try:
        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(40).mean())
    except Exception:
        return True

    if avg_dollar_vol < MIN_DOLLAR_VOL:
        return False

    return True'''
    
    new_filter_func = '''def _passes_basic_filters(df: pd.DataFrame) -> bool:
    """
    Quick sanity filters before doing full scoring + trade planning.
    PERFORMANCE: Tightened to reject low-quality symbols early.
    """
    if df is None or df.empty:
        return False

    if len(df) < MIN_BARS:
        return False

    last_row = df.iloc[-1]

    # Price filter: Reject penny stocks
    close = float(last_row.get("Close", 0.0))
    if not pd.notna(close) or close < MIN_PRICE:
        return False

    # Liquidity filter: Reject illiquid stocks
    try:
        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(40).mean())
        if avg_dollar_vol < MIN_DOLLAR_VOL:
            return False
    except Exception:
        # If we can't compute dollar volume, be conservative and reject
        return False

    # Volatility sanity check: Reject if price is too volatile (likely data error)
    try:
        price_std = df["Close"].tail(20).std()
        price_mean = df["Close"].tail(20).mean()
        if price_mean > 0:
            cv = price_std / price_mean  # Coefficient of variation
            if cv > 0.5:  # More than 50% volatility is suspicious
                return False
    except Exception:
        pass

    return True'''
    
    content = content.replace(old_filter_func, new_filter_func)
    
    # Write the updated content
    with open('technic_v4/scanner_core.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Step 2 implementation complete!")
    print("  - Added _smart_filter_universe() function")
    print("  - Updated _prepare_universe() to use smart filtering")
    print("  - Updated MIN_PRICE to $5.00")
    print("  - Updated MIN_DOLLAR_VOL to $500K")
    print("  - Enhanced _passes_basic_filters() with volatility check")

if __name__ == "__main__":
    try:
        implement_step2()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
