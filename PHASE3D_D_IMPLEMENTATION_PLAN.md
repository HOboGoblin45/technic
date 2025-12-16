# Phase 3D-D: Multi-Stage Progress Tracking Implementation Plan

## Overview
Enhance the scanner to track progress through 4 distinct stages with weighted progress calculation and stage-specific ETAs.

## Stage Breakdown

### Stage 1: Universe Loading (5% of total time)
- Load ticker universe CSV
- Apply smart filters
- Apply user filters (sectors, industries)
- **Progress Points**: 0-5%

### Stage 2: Data Fetching (20% of total time)
- Fetch market regime data
- Batch prefetch price data
- Load macro indicators
- **Progress Points**: 5-25%

### Stage 3: Symbol Scanning (70% of total time)
- Process each symbol individually
- Run technical analysis
- Calculate scores (Tech, Alpha, Merit)
- Apply trade planning
- **Progress Points**: 25-95%

### Stage 4: Finalization (5% of total time)
- Apply final filters
- Sort and rank results
- Write output files
- Generate recommendations
- **Progress Points**: 95-100%

## Implementation Tasks

### 1. Enhance scanner_core.py
- Import MultiStageProgressTracker
- Define stage weights
- Add stage transitions at key points
- Update progress callbacks with stage info

### 2. Update Progress Tracking
- Track sub-progress within each stage
- Calculate weighted overall progress
- Provide stage-specific ETAs
- Include stage metadata in callbacks

### 3. Enhance API
- Include stage information in progress responses
- Show stage completion percentages
- Provide stage-specific messages

### 4. Testing
- Test stage transitions
- Verify weighted progress calculation
- Ensure ETA accuracy per stage
- Test with various scan sizes

## Code Changes Required

### scanner_core.py
```python
# Initialize multi-stage tracker
tracker = MultiStageProgressTracker(
    stages=[
        ("universe_loading", 0.05),
        ("data_fetching", 0.20),
        ("symbol_scanning", 0.70),
        ("finalization", 0.05)
    ]
)

# Stage transitions
tracker.start_stage("universe_loading")
# ... universe loading code ...
tracker.complete_stage("universe_loading")

tracker.start_stage("data_fetching")
# ... data fetching code ...
tracker.complete_stage("data_fetching")

# Per-symbol progress in scanning stage
tracker.start_stage("symbol_scanning")
for i, symbol in enumerate(symbols):
    tracker.update_stage_progress("symbol_scanning", i, len(symbols))
    # ... process symbol ...
tracker.complete_stage("symbol_scanning")
```

## Expected Benefits
1. Users see detailed progress through each stage
2. Better understanding of what's happening
3. More accurate ETAs based on stage weights
4. Easier to identify performance bottlenecks

## Success Criteria
- [ ] All 4 stages tracked with correct weights
- [ ] Overall progress smoothly transitions 0-100%
- [ ] Stage-specific ETAs are accurate
- [ ] API returns stage information
- [ ] Tests pass for multi-stage tracking
