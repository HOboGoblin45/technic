# Test 12 - What to Look For in Terminal

## Terminal Location
Look for the terminal running: `python test_scanner_optimization_thorough.py`

## What to Find

### Test 12 Section
Look for output that starts with:
```
================================================================================
TEST 12: Result Consistency
================================================================================
```

### Expected Output Format
You should see something like:
```
✓ PASS   Result Consistency                       - Results match across runs
         run1_count: 9
         run2_count: 9
         symbols_match: True
         scores_consistent: True
```

OR

```
✗ FAIL   Result Consistency                       - Results differ
         run1_count: X
         run2_count: Y
         difference: ...
```

### Final Summary
At the very end, look for:
```
================================================================================
FINAL TEST SUMMARY
================================================================================
Tests Passed: X/12
Tests Failed: Y/12
Overall Status: PASS/FAIL
```

## What to Copy

Please copy and paste:
1. The entire "TEST 12: Result Consistency" section
2. The "FINAL TEST SUMMARY" section
3. Any error messages if present

## If Test is Still Running

If you see the test is still processing symbols (showing lines like):
```
[PRICE] polygon_rest: PriceSourceStats(source='polygon_rest_daily', symbol='XXXX', ...)
```

Then the test is still running. Please wait a few more minutes and check again.

## If Test Completed with Errors

If you see Python errors or tracebacks, please copy the entire error message.

---

**Once you have the output, paste it in the chat and I'll finalize the report!**
