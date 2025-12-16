# Phase 3D-B: Additional Enhancements Implementation Plan

## Overview
Implementing the 4 optional enhancements requested by the user to complete Phase 3D-B Task 2.

## Current Status
✅ **Core Integration Complete** (8/8 tests passing)
- Progress tracking in `_worker()` function
- Backward-compatible callback system
- Error handling infrastructure ready

## Enhancements to Implement

### 1. Multi-Stage Progress Tracking in `run_scan()`
**Goal**: Track 4 distinct stages with overall progress

**Implementation**:
```python
# In run_scan() function, add MultiStageProgressTracker
tracker = MultiStageProgressTracker(
    stages=['universe_loading', 'data_fetching', 'symbol_scanning', 'finalization'],
    stage_weights=[0.05, 0.20, 0.70, 0.05]  # 70% of time is scanning
)

# Stage 1: Universe Loading
tracker.start_stage('universe_loading', total=1)
universe = _prepare_universe(config, settings=settings)
tracker.update(1, message="Universe loaded")

# Stage 2: Data Fetching (batch prefetch)
tracker.start_stage('data_fetching', total=len(working))
price_cache = data_engine.get_price_history_batch(...)
tracker.update(len(price_cache), message=f"Fetched {len(price_cache)} symbols")

# Stage 3: Symbol Scanning (pass tracker to _run_symbol_scans)
tracker.start_stage('symbol_scanning', total=len(working))
results_df, stats = _run_symbol_scans(..., multi_stage_tracker=tracker)

# Stage 4: Finalization
tracker.start_stage('finalization', total=1)
results_df, status_text = _finalize_results(...)
tracker.update(1, message="Results finalized")
```

**Files to Modify**:
- `technic_v4/scanner_core.py` - `run_scan()` function

### 2. Error Recovery with Automatic Retry Logic
**Goal**: Retry failed operations with exponential backoff

**Implementation**:
```python
def _retry_with_backoff(func, max_retries=3, initial_delay=1.0):
    """
    Retry a function with exponential backoff.
    
    Returns: (success: bool, result: Any, error: Optional[ScanError])
    """
    for attempt in range(max_retries):
        try:
            result = func()
            return True, result, None
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed, create structured error
                error = create_custom_error(
                    error_type=ErrorType.API_ERROR,
                    message=f"Failed after {max_retries} attempts",
                    details={'original_error': str(e), 'attempts': max_retries}
                )
                return False, None, error
            
            # Wait before retry (exponential backoff)
            delay = initial_delay * (2 ** attempt)
            time.sleep(delay)
    
    return False, None, None

# Usage in _scan_symbol():
success, df, error = _retry_with_backoff(
    lambda: data_engine.get_price_history(symbol, days, freq),
    max_retries=3
)
if not success:
    logger.warning("[RETRY] %s: %s", symbol, error.message)
    return None
```

**Files to Modify**:
- `technic_v4/scanner_core.py` - Add `_retry_with_backoff()` helper
- `technic_v4/scanner_core.py` - Wrap API calls in `_scan_symbol()`

### 3. Progress Persistence for Long-Running Scans
**Goal**: Save/resume scan progress

**Implementation**:
```python
class ScanCheckpoint:
    """Manages scan progress checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, scan_id: str, progress: dict):
        """Save current progress."""
        checkpoint_file = self.checkpoint_dir / f"{scan_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'progress': progress
            }, f)
    
    def load(self, scan_id: str) -> Optional[dict]:
        """Load saved progress."""
        checkpoint_file = self.checkpoint_dir / f"{scan_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('progress')
        return None
    
    def clear(self, scan_id: str):
        """Clear checkpoint after successful completion."""
        checkpoint_file = self.checkpoint_dir / f"{scan_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

# Usage in run_scan():
scan_id = f"scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
checkpoint = ScanCheckpoint(OUTPUT_DIR / "checkpoints")

# Try to resume from checkpoint
saved_progress = checkpoint.load(scan_id)
if saved_progress:
    logger.info("[CHECKPOINT] Resuming from saved progress")
    # Skip already processed symbols
    processed_symbols = set(saved_progress.get('processed', []))
    working = [u for u in working if u.symbol not in processed_symbols]

# Save progress periodically
for i, result in enumerate(results):
    if i % 100 == 0:  # Save every 100 symbols
        checkpoint.save(scan_id, {
            'processed': [r['Symbol'] for r in results[:i]],
            'total': len(working)
        })

# Clear checkpoint on success
checkpoint.clear(scan_id)
```

**Files to Create**:
- `technic_v4/checkpoint.py` - New module for checkpoint management

**Files to Modify**:
- `technic_v4/scanner_core.py` - Add checkpoint support to `run_scan()`

### 4. Cancellation Support for Mid-Execution Stops
**Goal**: Allow graceful cancellation of running scans

**Implementation**:
```python
import threading

class ScanCancellation:
    """Thread-safe cancellation token."""
    
    def __init__(self):
        self._cancelled = threading.Event()
    
    def cancel(self):
        """Request cancellation."""
        self._cancelled.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()
    
    def check_and_raise(self):
        """Raise exception if cancelled."""
        if self.is_cancelled():
            raise ScanError(
                error_type=ErrorType.SYSTEM_ERROR,
                message="Scan cancelled by user",
                suggestion="Partial results may be available"
            )

# Usage in run_scan():
def run_scan(
    config: Optional[ScanConfig] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancellation_token: Optional[ScanCancellation] = None,
) -> Tuple[pd.DataFrame, str]:
    
    if cancellation_token is None:
        cancellation_token = ScanCancellation()
    
    # Check for cancellation at key points
    cancellation_token.check_and_raise()
    
    universe = _prepare_universe(config, settings=settings)
    
    cancellation_token.check_and_raise()
    
    # Pass to worker threads
    def _worker_with_cancel(idx_urow):
        cancellation_token.check_and_raise()
        return _worker(idx_urow)
    
    # Use cancellable worker
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_worker_with_cancel, enumerate(universe, start=1)):
            if cancellation_token.is_cancelled():
                ex.shutdown(wait=False, cancel_futures=True)
                break
            # Process result...
```

**Files to Create**:
- `technic_v4/cancellation.py` - New module for cancellation support

**Files to Modify**:
- `technic_v4/scanner_core.py` - Add cancellation checks to `run_scan()` and `_run_symbol_scans()`

## Implementation Order

1. **Multi-Stage Progress Tracking** (Highest Priority)
   - Most visible improvement to user experience
   - Builds on existing progress infrastructure
   - Estimated time: 2-3 hours

2. **Error Recovery with Retry Logic** (High Priority)
   - Improves reliability significantly
   - Relatively straightforward to implement
   - Estimated time: 1-2 hours

3. **Cancellation Support** (Medium Priority)
   - Important for long-running scans
   - Requires careful thread management
   - Estimated time: 2-3 hours

4. **Progress Persistence** (Lower Priority)
   - Nice-to-have for very long scans
   - More complex implementation
   - Estimated time: 3-4 hours

## Testing Strategy

### For Each Enhancement:

1. **Unit Tests**
   - Test individual components in isolation
   - Mock external dependencies

2. **Integration Tests**
   - Test with actual scanner workflow
   - Verify interaction with existing code

3. **End-to-End Tests**
   - Run full scans with enhancements enabled
   - Measure performance impact

### Test Files to Create:

1. `test_multi_stage_progress.py` - Multi-stage tracking tests
2. `test_error_recovery.py` - Retry logic tests
3. `test_cancellation.py` - Cancellation tests
4. `test_checkpoints.py` - Progress persistence tests
5. `test_enhancements_e2e.py` - End-to-end integration tests

## Success Criteria

- ✅ All existing tests continue to pass (8/8)
- ✅ New tests for each enhancement pass
- ✅ Performance overhead < 1% for each enhancement
- ✅ Backward compatibility maintained
- ✅ Documentation updated

## Estimated Total Time

- Implementation: 8-12 hours
- Testing: 4-6 hours
- Documentation: 2-3 hours
- **Total: 14-21 hours**

## Next Steps

1. Get user confirmation to proceed
2. Implement enhancements in priority order
3. Create comprehensive tests for each
4. Update documentation
5. Run full integration test suite
6. Create final completion report

---

**Status**: Ready to implement
**Blocked By**: User confirmation
**Dependencies**: Phase 3D-B Task 1 complete (✅)
