# Phase 3D-B: Full Implementation Plan

## Current Status ✅
- Infrastructure complete: 36/36 tests passing
- Imports integrated into scanner_core.py
- Type signatures updated
- Backward compatibility helper created
- All verification tests passed

## Implementation Approach

Given the size and complexity of scanner_core.py (2700+ lines), we'll implement progress tracking and error handling incrementally in focused areas.

### Priority 1: Core Progress Tracking (High Impact)

#### 1.1 Add Progress Tracking to `run_scan()` Main Flow
**Location**: Line ~2590 in scanner_core.py
**Changes**:
```python
def run_scan(config, progress_cb):
    # Create multi-stage tracker at the start
    if progress_cb:
        tracker = MultiStageProgressTracker(
            stages={
                'initialization': 1,
                'universe_loading': 1,
                'data_prefetch': len(working),
                'symbol_scanning': len(working),
                'finalization': 1
            },
            callback=lambda stage, curr, tot, msg, meta: 
                _safe_progress_callback(progress_cb, stage, curr, tot, msg, meta)
        )
        
        # Track initialization
        tracker.start_stage('initialization')
        # ... regime detection code ...
        tracker.complete_stage('initialization')
        
        # Track universe loading
        tracker.start_stage('universe_loading')
        universe = _prepare_universe(config)
        tracker.complete_stage('universe_loading')
        
        # Track data prefetch
        tracker.start_stage('data_prefetch')
        price_cache = data_engine.get_price_history_batch(...)
        tracker.complete_stage('data_prefetch')
        
        # Track symbol scanning
        tracker.start_stage('symbol_scanning')
        results_df, stats = _run_symbol_scans(..., progress_cb=tracker.update)
        tracker.complete_stage('symbol_scanning')
        
        # Track finalization
        tracker.start_stage('finalization')
        results_df, status = _finalize_results(...)
        tracker.complete_stage('finalization')
```

**Benefits**:
- Users see real-time progress through all scan stages
- ETA calculations for each stage
- Clear indication of what's happening

#### 1.2 Update `_run_symbol_scans()` Progress Callbacks
**Location**: Line ~1850 in scanner_core.py
**Changes**:
```python
def _worker(idx_urow):
    idx, urow = idx_urow
    symbol = urow.symbol
    if progress_cb is not None:
        # Use safe callback with enhanced parameters
        _safe_progress_callback(
            progress_cb,
            stage='symbol_scanning',
            current=idx,
            total=total_symbols,
            message=f"Scanning {symbol}",
            metadata={
                'symbol': symbol,
                'percentage': (idx / total_symbols) * 100,
                'symbols_per_second': idx / (time.time() - start_ts) if idx > 0 else 0
            }
        )
```

**Benefits**:
- Per-symbol progress updates
- Speed metrics (symbols/second)
- Percentage completion

### Priority 2: Error Handling (Medium Impact)

#### 2.1 Wrap Universe Loading
**Location**: In `run_scan()` around line ~2650
**Changes**:
```python
try:
    universe = _prepare_universe(config, settings=settings)
    if not universe:
        raise ScanError(
            error_type=ErrorType.CONFIGURATION_ERROR,
            message="No symbols match your filters",
            details={'config': config.__dict__}
        )
except Exception as e:
    if isinstance(e, ScanError):
        raise
    raise ScanError(
        error_type=ErrorType.DATA_FETCH_ERROR,
        message=f"Failed to load universe: {str(e)}",
        details={'error': str(e)}
    )
```

#### 2.2 Wrap Data Prefetch
**Location**: In `run_scan()` around line ~2700
**Changes**:
```python
try:
    price_cache = data_engine.get_price_history_batch(
        symbols=[row.symbol for row in working],
        days=effective_lookback,
        freq="daily"
    )
except Exception as e:
    raise ScanError(
        error_type=ErrorType.API_ERROR,
        message=f"Failed to prefetch price data: {str(e)}",
        details={
            'symbols_count': len(working),
            'lookback_days': effective_lookback
        }
    )
```

#### 2.3 Wrap Symbol Scanning
**Location**: In `run_scan()` around line ~2720
**Changes**:
```python
try:
    results_df, stats = _run_symbol_scans(...)
except Exception as e:
    if isinstance(e, ScanError):
        raise
    raise ScanError(
        error_type=ErrorType.CALCULATION_ERROR,
        message=f"Failed during symbol scanning: {str(e)}",
        details={'stats': stats if 'stats' in locals() else {}}
    )
```

### Priority 3: Integration Tests (High Priority)

#### 3.1 Test Progress Callbacks
**File**: `test_scanner_integration.py` (new)
```python
def test_progress_callbacks_during_scan():
    """Test that progress callbacks work during actual scan"""
    progress_updates = []
    
    def capture_progress(stage, current, total, message, metadata):
        progress_updates.append({
            'stage': stage,
            'current': current,
            'total': total,
            'message': message,
            'metadata': metadata
        })
    
    config = ScanConfig(max_symbols=5)  # Small test
    df, msg, metrics = run_scan(config, progress_cb=capture_progress)
    
    # Verify we got updates
    assert len(progress_updates) > 0
    
    # Verify stages
    stages = {u['stage'] for u in progress_updates}
    assert 'initialization' in stages or 'universe_loading' in stages
    
    # Verify metadata includes ETA
    for update in progress_updates:
        if update['current'] > 0:
            assert 'percentage' in update['metadata'] or 'eta_seconds' in update['metadata']
```

#### 3.2 Test Error Handling
**File**: `test_scanner_integration.py`
```python
def test_error_handling_invalid_config():
    """Test that errors are properly wrapped"""
    config = ScanConfig(sectors=['INVALID_SECTOR_THAT_DOES_NOT_EXIST'])
    
    try:
        run_scan(config)
        assert False, "Should have raised ScanError"
    except ScanError as e:
        assert e.error_type == ErrorType.CONFIGURATION_ERROR
        assert 'No symbols match' in str(e)
```

#### 3.3 Test Backward Compatibility
**File**: `test_scanner_integration.py`
```python
def test_old_callback_signature_still_works():
    """Test that old 3-parameter callbacks still work"""
    old_style_calls = []
    
    def old_callback(stage, current, total):
        # Old signature - only 3 params
        old_style_calls.append((stage, current, total))
    
    config = ScanConfig(max_symbols=3)
    df, msg, metrics = run_scan(config, progress_cb=old_callback)
    
    # Should not crash, should have received some calls
    assert len(old_style_calls) > 0
```

## Implementation Steps

### Step 1: Add Progress Tracking (1-2 hours)
1. Modify `run_scan()` to create MultiStageProgressTracker
2. Add stage tracking for each major phase
3. Update `_run_symbol_scans()` to use enhanced callbacks
4. Test with small symbol set

### Step 2: Add Error Handling (1 hour)
1. Wrap universe loading in try-catch
2. Wrap data prefetch in try-catch
3. Wrap symbol scanning in try-catch
4. Test error scenarios

### Step 3: Create Integration Tests (1 hour)
1. Create `test_scanner_integration.py`
2. Add progress callback tests
3. Add error handling tests
4. Add backward compatibility tests
5. Run all tests

### Step 4: Documentation (30 minutes)
1. Update docstrings
2. Create usage examples
3. Document breaking changes (none expected)

## Estimated Total Time
- Implementation: 4-5 hours
- Testing: 1-2 hours
- Documentation: 30 minutes
- **Total**: 5.5-7.5 hours

## Risk Assessment

### Low Risk ✅
- Infrastructure is solid (36/36 tests passing)
- Backward compatibility maintained
- Changes are additive, not destructive
- Helper function handles both old and new signatures

### Medium Risk ⚠️
- Large file size makes changes error-prone
- Multiple code paths (Ray vs thread pool)
- Need to test both paths

### Mitigation
- Implement incrementally
- Test after each change
- Keep old code paths working
- Use feature flags if needed

## Success Criteria

1. ✅ Progress callbacks work during scans
2. ✅ ETA calculations are accurate
3. ✅ Error messages are user-friendly
4. ✅ Backward compatibility maintained
5. ✅ All existing tests still pass
6. ✅ New integration tests pass
7. ✅ No performance degradation

## Alternative: Minimal Implementation

If full implementation is too risky, we can do a minimal version:

1. **Just add progress tracking to `_run_symbol_scans()`** - This gives 80% of the user benefit
2. **Skip error handling wrappers** - Existing error handling is adequate
3. **Create basic integration test** - Just verify callbacks work

This would take 1-2 hours and provide most of the value with minimal risk.

## Recommendation

Given the complexity and size of the codebase, I recommend:

1. **Start with infrastructure validation** (already done ✅)
2. **Implement minimal version first** (1-2 hours)
3. **Test thoroughly** (1 hour)
4. **If successful, add full error handling** (2-3 hours)

This incremental approach minimizes risk while delivering value quickly.

---

**Status**: Infrastructure complete, ready for implementation
**Next Step**: Choose implementation approach (full vs minimal)
**Estimated Time**: 2-7 hours depending on approach
