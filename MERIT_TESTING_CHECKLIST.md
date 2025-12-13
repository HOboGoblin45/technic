# MERIT Score Testing Checklist

## Testing Progress

### ✅ Completed Tests

#### Static Analysis
- [x] Flutter analyze: 0 errors, 0 warnings
- [x] Dart type safety validated
- [x] All imports resolved

#### Code Quality
- [x] Backend files compile
- [x] Flutter files compile
- [x] No syntax errors
- [x] Proper error handling in place

---

### ⏳ In Progress

#### Backend Testing
- [ ] **Test Scan Execution** - RUNNING NOW
  - Command: `python scripts/run_scan.py`
  - Expected: Scan completes with MERIT columns
  - Validation: Check for MeritScore, MeritBand, MeritFlags, MeritSummary

---

### 📋 Pending Tests

#### Backend Validation
- [ ] Check CSV output has MERIT columns
- [ ] Verify MERIT score range (0-100)
- [ ] Confirm top 10 sorted by MERIT
- [ ] Validate letter grades (A+, A, B, C, D)
- [ ] Check risk flags are populated
- [ ] Verify confluence bonus applied
- [ ] Run quality test: `python -m technic_v4.dev.test_merit_quality`

#### API Testing
- [ ] Start API server
- [ ] Test `/health` endpoint
- [ ] Test `/v1/scan` with MERIT fields
- [ ] Verify JSON schema correctness
- [ ] Test backward compatibility
- [ ] Check error handling

#### Flutter UI Testing
- [ ] Run Flutter app
- [ ] Trigger scan from UI
- [ ] Verify MERIT Score displays (large, prominent)
- [ ] Check letter grade badges
- [ ] Validate risk flag chips
- [ ] Test metrics row expansion
- [ ] Verify Copilot integration

#### Integration Testing
- [ ] End-to-end: Backend → API → UI
- [ ] Performance with 100+ symbols
- [ ] Error handling (missing data)
- [ ] Edge cases (no MERIT data)

---

## Test Results Log

### Backend Scan Test
**Status**: Running...
**Command**: `python scripts/run_scan.py`
**Started**: [Waiting for completion]
**Expected Output**:
- Scan completes successfully
- CSV file created with MERIT columns
- Console shows MERIT logging
- Top 10 logged with MERIT details

**Validation Steps**:
1. Check `technic_v4/scanner_output/technic_scan_results.csv` exists
2. Verify columns: MeritScore, MeritBand, MeritFlags, MeritSummary
3. Check score range: 0-100
4. Verify sorting: MeritScore descending
5. Review top 10 for institutional quality

---

## Success Criteria

### Backend:
- ✅ Scan completes without errors
- ✅ MERIT columns present in output
- ✅ Scores in valid range (0-100)
- ✅ Top 10 sorted correctly
- ✅ Letter grades assigned
- ✅ Risk flags populated
- ✅ Logging shows MERIT details

### API:
- ✅ Endpoints return MERIT fields
- ✅ JSON schema valid
- ✅ Backward compatible
- ✅ Error handling works

### UI:
- ✅ MERIT displays prominently
- ✅ Letter grades color-coded
- ✅ Risk flags show as chips
- ✅ Metrics expanded correctly
- ✅ No visual glitches

### Integration:
- ✅ End-to-end flow works
- ✅ Performance acceptable
- ✅ Error handling graceful
- ✅ Edge cases handled

---

## Known Issues

None detected so far.

---

## Next Steps After Testing

1. Review test results
2. Fix any issues found
3. Re-test if needed
4. Deploy to Render
5. User acceptance testing
6. App Store submission prep

---

**Current Status**: Backend scan test in progress...
**Next**: Validate scan output and run quality test
