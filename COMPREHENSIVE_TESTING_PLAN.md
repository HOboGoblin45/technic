# Technic App - Comprehensive Testing Plan

## 🎯 Testing Objective
Validate all functionality of the Technic Flutter app with Render API integration.

---

## Part 1: API Endpoint Testing (Automated)

### Test 1: Health Check
```bash
curl https://technic-m5vn.onrender.com/health
```
**Expected**: `{"status":"ok"}`

### Test 2: Version Info
```bash
curl https://technic-m5vn.onrender.com/version
```
**Expected**: API version and feature flags

### Test 3: Scan Endpoint (POST)
```bash
curl -X POST https://technic-m5vn.onrender.com/v1/scan \
  -H "Content-Type: application/json" \
  -d '{"max_symbols": 5, "trade_style": "Short-term swing", "min_tech_rating": 0.0}'
```
**Expected**: JSON with `status`, `disclaimer`, `results` array

### Test 4: Copilot Endpoint (POST)
```bash
curl -X POST https://technic-m5vn.onrender.com/v1/copilot \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a good swing trade setup?"}'
```
**Expected**: JSON with `answer` field

---

## Part 2: Flutter App Manual Testing

### A. Scanner Page Testing

#### Test 2.1: Basic Scan
1. Open Scanner page
2. Click "Run Scan" button
3. **Verify**:
   - Loading indicator appears
   - Results populate after scan completes
   - Each result shows: ticker, signal, entry, stop, target
   - No crashes or errors

#### Test 2.2: Auto-Scan Prevention ✅ (User Issue #2)
1. Click "Conservative" profile button
   - **Verify**: NO scan triggered
2. Click "Moderate" profile button
   - **Verify**: NO scan triggered
3. Click "Aggressive" profile button
   - **Verify**: NO scan triggered
4. Click "Randomize" button
   - **Verify**: NO scan triggered
5. Open Filters panel, change settings, close
   - **Verify**: NO scan triggered
6. Load a saved preset
   - **Verify**: NO scan triggered
7. Navigate away and back to Scanner
   - **Verify**: NO scan triggered
8. **ONLY** clicking "Run Scan" should trigger a scan

#### Test 2.3: Multi-Sector Selection ✅ (User Issue #1)
1. Open Filters panel (tune icon)
2. Click "Technology" sector
   - **Verify**: Chip highlights blue with checkmark
3. Click "Healthcare" sector
   - **Verify**: BOTH Technology and Healthcare highlighted
4. Click "Financials" sector
   - **Verify**: ALL THREE sectors highlighted
5. Click "Technology" again
   - **Verify**: Technology deselects, other two remain
6. Click "All Sectors"
   - **Verify**: All selections clear

#### Test 2.4: Profile Tooltips Removed ✅ (User Issue #3)
1. Hover mouse over "Conservative" button
   - **Verify**: NO tooltip appears
2. Hover over "Moderate" button
   - **Verify**: NO tooltip appears
3. Hover over "Aggressive" button
   - **Verify**: NO tooltip appears

#### Test 2.5: Filter Combinations
1. Test each filter independently:
   - Lookback days (30, 60, 90, 180)
   - Min Tech Rating (0, 3, 5, 7)
   - Trade Style (Day, Short-term swing, Long-term)
   - Sectors (each individually)
2. Test filter combinations:
   - Technology + Healthcare + 90 days
   - Min Rating 5 + Short-term swing
   - All filters at once

#### Test 2.6: Scan Results Interaction
1. Click on a scan result card
   - **Verify**: Symbol detail page opens
2. Long-press a result
   - **Verify**: Options menu appears
3. Star a symbol
   - **Verify**: Added to My Ideas

#### Test 2.7: Market Pulse
- **Note**: Will be empty due to API limitation
- **Verify**: No crashes when movers section is empty

#### Test 2.8: Scoreboard
- **Note**: Will be empty due to API limitation
- **Verify**: No crashes when scoreboard is empty

### B. Ideas Page Testing

#### Test 2.9: Ideas Display
- **Note**: Will be empty due to API limitation
- **Verify**: Shows empty state message
- **Verify**: No crashes

### C. Copilot Page Testing

#### Test 2.10: Basic Copilot Interaction
1. Type "What is a good swing trade setup?"
2. Click Send
3. **Verify**:
   - Message appears in chat
   - Loading indicator shows
   - AI response appears
   - Response is relevant

#### Test 2.11: Symbol-Specific Questions
1. From Scanner, click "Ask Copilot" on a result
2. **Verify**:
   - Copilot page opens
   - Context is pre-filled
   - Question about that symbol works

#### Test 2.12: Multiple Questions
1. Ask 3-5 different questions in sequence
2. **Verify**:
   - All messages appear in order
   - Scroll works properly
   - No memory leaks

#### Test 2.13: Copilot Error Handling
1. Turn off internet
2. Try to send a message
3. **Verify**: Error message appears gracefully

### D. My Ideas Page Testing

#### Test 2.14: Watchlist Operations
1. Add 3-5 symbols from Scanner
2. Go to My Ideas page
3. **Verify**: All symbols appear
4. Click a symbol
   - **Verify**: Detail page opens
5. Remove a symbol
   - **Verify**: Removed from list
6. Close and reopen app
   - **Verify**: Watchlist persists

### E. Settings Page Testing

#### Test 2.15: Theme Toggle Removed ✅ (User Issue #5)
1. Open Settings page
2. **Verify**: NO theme toggle present
3. **Verify**: App stays in dark mode

#### Test 2.16: Options Mode Toggle
1. Toggle "Stock + Options" / "Stock Only"
2. **Verify**: Setting saves
3. Run a scan
4. **Verify**: Results reflect the mode

#### Test 2.17: Settings Persistence
1. Change settings
2. Close app completely
3. Reopen app
4. **Verify**: Settings are preserved

### F. Navigation Testing

#### Test 2.18: Footer Tab Tooltips ✅ (User Issue #4)
1. Hover over each footer tab icon
   - Scanner
   - Ideas
   - Copilot
   - My Ideas
   - Settings
2. **Verify**: NO tooltips appear on any tab

#### Test 2.19: Tab Navigation
1. Navigate through all tabs in order
2. **Verify**: Each page loads without crashes
3. Navigate backwards
4. **Verify**: State is preserved on each page

#### Test 2.20: Deep Navigation
1. Scanner → Symbol Detail → Back
2. Scanner → Copilot → Back
3. My Ideas → Symbol Detail → Back
4. **Verify**: Navigation stack works correctly

### G. Performance Testing

#### Test 2.21: Large Result Sets
1. Run scan with max symbols = 100
2. **Verify**:
   - App remains responsive
   - Scrolling is smooth
   - No memory issues

#### Test 2.22: Rapid Actions
1. Quickly click between tabs 10 times
2. **Verify**: No crashes or lag

#### Test 2.23: Long Session
1. Use app for 10+ minutes
2. Perform various actions
3. **Verify**: No memory leaks or slowdowns

### H. Error Handling Testing

#### Test 2.24: Network Errors
1. Disconnect internet
2. Try to run scan
3. **Verify**: Graceful error message
4. Reconnect
5. **Verify**: App recovers

#### Test 2.25: API Errors
1. If API returns error
2. **Verify**: User-friendly error message
3. **Verify**: App doesn't crash

#### Test 2.26: Invalid Input
1. Try edge cases in filters
2. **Verify**: Validation works
3. **Verify**: No crashes

### I. Visual/UI Testing

#### Test 2.27: Dark Mode Consistency
1. Check all pages
2. **Verify**: Consistent dark theme
3. **Verify**: Good contrast/readability

#### Test 2.28: Responsive Layout
1. Resize window (if possible)
2. **Verify**: Layout adapts properly

#### Test 2.29: Animations
1. Check page transitions
2. Check loading animations
3. **Verify**: Smooth, no jank

---

## Part 3: Compilation & Code Quality

### Test 3.1: Compilation ✅
- **Status**: PASSED (0 errors, 0 warnings)

### Test 3.2: Hot Reload
1. Make a small UI change
2. Press 'r' for hot reload
3. **Verify**: Change appears instantly

### Test 3.3: Hot Restart
1. Press 'R' for hot restart
2. **Verify**: App restarts cleanly

---

## Testing Checklist Summary

### ✅ Completed (Automated)
- [x] Code compilation
- [x] API health check
- [x] Code fixes applied

### 📋 Manual Testing Required
- [ ] Scanner basic functionality
- [ ] Auto-scan prevention (Issue #2)
- [ ] Multi-sector selection (Issue #1)
- [ ] Profile tooltips removed (Issue #3)
- [ ] Footer tooltips removed (Issue #4)
- [ ] Theme toggle removed (Issue #5)
- [ ] Filter combinations
- [ ] Copilot functionality
- [ ] Watchlist operations
- [ ] Navigation flow
- [ ] Performance under load
- [ ] Error handling
- [ ] Visual consistency

---

## Known Limitations

1. **Movers Section**: Empty (API doesn't return movers data)
2. **Ideas Section**: Empty (API doesn't return ideas data)
3. **Scoreboard**: Empty (API doesn't return scoreboard data)

These require backend API updates to fully function.

---

## Bug Reporting Template

If you find issues during testing:

```
**Bug**: [Brief description]
**Steps to Reproduce**:
1. 
2. 
3. 
**Expected**: [What should happen]
**Actual**: [What actually happened]
**Screenshot**: [If applicable]
**Console Errors**: [Any error messages]
```

---

## Next Steps After Testing

1. Document all findings
2. Fix any critical bugs discovered
3. Update API backend for movers/ideas/scoreboard
4. Retest after fixes
5. Final deployment validation
