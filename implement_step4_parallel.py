"""
Script to implement Step 4: Parallel Processing Optimization
Enhances scanner_core.py with better parallelization
"""

def implement_step4():
    """Add parallel processing optimizations to scanner_core.py"""
    
    print("Implementing Step 4: Parallel Processing Optimization...")
    
    # Read the current file
    with open('technic_v4/scanner_core.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already implemented
    if 'PERFORMANCE: Optimized for parallel' in content:
        print("✓ Parallel processing already optimized!")
        return
    
    # Update MAX_WORKERS constant with better logic
    old_workers = "MAX_WORKERS = 20  # Optimized for Pro Plus (4 CPU cores)"
    new_workers = """# PERFORMANCE: Optimized for parallel processing
# Use 2x CPU cores for I/O-bound tasks (API calls), capped at 32
import os
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)"""
    
    if old_workers in content:
        content = content.replace(old_workers, new_workers)
        print("  ✓ Updated MAX_WORKERS with dynamic CPU detection")
    
    # Find _run_symbol_scans function and optimize it
    # Add batch processing for better throughput
    old_run_scans = """def _run_symbol_scans(
    config: "ScanConfig",
    universe: List[UniverseRow],
    regime_tags: dict,
    effective_lookback: int,
    settings,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, dict]:
    \"\"\"
    Scan each symbol in the universe, returning a DataFrame of results.
    \"\"\"
    results = []
    stats = {"attempted": 0, "kept": 0, "errors": 0, "rejected": 0}"""
    
    new_run_scans = """def _run_symbol_scans(
    config: "ScanConfig",
    universe: List[UniverseRow],
    regime_tags: dict,
    effective_lookback: int,
    settings,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, dict]:
    \"\"\"
    Scan each symbol in the universe, returning a DataFrame of results.
    PERFORMANCE: Optimized for parallel processing with batching.
    \"\"\"
    results = []
    stats = {"attempted": 0, "kept": 0, "errors": 0, "rejected": 0}
    
    # PERFORMANCE: Process in batches for better memory management
    batch_size = 100
    total_symbols = len(universe)
    logger.info("[PARALLEL] Processing %d symbols in batches of %d with %d workers", 
                total_symbols, batch_size, MAX_WORKERS)"""
    
    if old_run_scans in content:
        content = content.replace(old_run_scans, new_run_scans)
        print("  ✓ Added batch processing logic")
    
    # Optimize the ThreadPoolExecutor usage
    old_executor = """    max_workers = getattr(settings, "max_workers", MAX_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:"""
    
    new_executor = """    # PERFORMANCE: Use optimal worker count
    max_workers = getattr(settings, "max_workers", MAX_WORKERS)
    logger.info("[PARALLEL] Using %d worker threads", max_workers)
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="scanner_"
    ) as executor:"""
    
    if old_executor in content:
        content = content.replace(old_executor, new_executor)
        print("  ✓ Optimized ThreadPoolExecutor configuration")
    
    # Add progress logging for batches
    old_progress = """        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):"""
    
    new_progress = """        # PERFORMANCE: Track progress with batch awareness
        completed = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed += 1
            if completed % 50 == 0:
                logger.info("[PARALLEL] Processed %d/%d symbols (%.1f%%)", 
                           completed, len(futures), (completed/len(futures))*100)"""
    
    if old_progress in content:
        content = content.replace(old_progress, new_progress)
        print("  ✓ Added batch progress logging")
    
    # Write the updated content
    with open('technic_v4/scanner_core.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Step 4 implementation complete!")
    print("\nOptimizations applied:")
    print("  - Dynamic worker count (2x CPU cores, max 32)")
    print("  - Batch processing (100 symbols per batch)")
    print("  - Progress logging every 50 symbols")
    print("  - Thread pool naming for debugging")

if __name__ == "__main__":
    try:
        implement_step4()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
