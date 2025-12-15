# Comprehensive Testing Guide - Features 7, 8, 9

**Testing Type:** Thorough Testing  
**Estimated Time:** 2-3 hours  
**Features:** Watchlist Notes/Tags, Scan History, Theme Toggle

---

## 🎯 TESTING OVERVIEW

This guide provides step-by-step instructions for thoroughly testing all three completed features. Follow each test case and mark it complete when verified.

---

## 📋 FEATURE 7: WATCHLIST NOTES & TAGS

### **Test Suite 1: Add Notes**

**Test 7.1: Add Note to Watchlist Item**
- [ ] Navigate to Watchlist page
- [ ] Tap on a watchlist item
- [ ] Tap "Add Note" button
- [ ] Enter note text (e.g., "Strong breakout pattern")
- [ ] Tap "Save"
- [ ] **Expected:** Note appears below ticker
- [ ] **Expected:** Note icon shows on card

**Test 7.2: Edit Existing Note**
- [ ] Tap on item with existing note
- [ ] Tap "Edit Note" button
- [ ] Modify note text
- [ ] Tap "Save"
- [ ] **Expected:** Updated note displays
- [ ] **Expected:** Changes persist

**Test 7.3: Note Character Limit**
- [ ] Open note dialog
- [ ] Type 500+ characters
- [ ] **Expected:** Character counter shows "500/500"
- [ ] **Expected:** Cannot type beyond 500 chars
- [ ] **Expected:** Warning message if limit reached

**Test 7.4: Delete Note**
- [ ] Open note dialog for item with note
- [ ] Clear all text
- [ ] Tap "Save"
- [ ] **Expected:** Note removed from display
- [ ] **Expected:** Note icon disappears

### **Test Suite 2: Add Tags**

**Test 7.5: Add Single Tag**
- [ ] Tap on watchlist item
- [ ] Tap "Add Tags" button
- [ ] Select one predefined tag (e.g., "Breakout")
- [ ] Tap "Save"
- [ ] **Expected:** Tag chip appears on card
- [ ] **Expected:** Tag persists

**Test 7.6: Add Multiple Tags**
- [ ] Open tag selector
- [ ] Select 3-4 different tags
- [ ] Tap "Save"
- [ ] **Expected:** All tags display as chips
- [ ] **Expected:** Tags wrap to multiple lines if needed

**Test 7.7: Add Custom Tag**
- [ ] Open tag selector
- [ ] Tap "Custom" option
- [ ] Enter custom tag name (e.g., "My Strategy")
- [ ] Tap "Add"
- [ ] **Expected:** Custom tag appears in list
- [ ] **Expected:** Custom tag can be selected

**Test 7.8: Remove Tags**
- [ ] Open tag selector for item with tags
- [ ] Deselect some tags
- [ ] Tap "Save"
- [ ] **Expected:** Deselected tags removed from display
- [ ] **Expected:** Remaining tags still show

**Test 7.9: All 16 Predefined Tags**
- [ ] Verify all predefined tags available:
  - [ ] Breakout
  - [ ] Momentum
  - [ ] Value
  - [ ] Growth
  - [ ] Dividend
  - [ ] Swing Trade
  - [ ] Day Trade
  - [ ] Long Term
  - [ ] High Risk
  - [ ] Low Risk
  - [ ] Earnings Play
  - [ ] Technical
  - [ ] Fundamental
  - [ ] Watch Closely
  - [ ] Ready to Buy
  - [ ] Consider Selling

### **Test Suite 3: Filter & Search**

**Test 7.10: Filter by Single Tag**
- [ ] Add tags to multiple items
- [ ] Tap filter button
- [ ] Select one tag (e.g., "Breakout")
- [ ] **Expected:** Only items with that tag show
- [ ] **Expected:** Other items hidden

**Test 7.11: Filter by Multiple Tags**
- [ ] Select 2-3 tags in filter
- [ ] **Expected:** Items with ANY selected tag show
- [ ] **Expected:** OR logic working correctly

**Test 7.12: Clear Filter**
- [ ] Apply filter
- [ ] Tap "Clear Filter" or deselect all
- [ ] **Expected:** All items show again

**Test 7.13: Search by Ticker**
- [ ] Enter ticker symbol in search (e.g., "AAPL")
- [ ] **Expected:** Matching items show
- [ ] **Expected:** Non-matching items hidden
- [ ] **Expected:** Case-insensitive search

**Test 7.14: Search by Note Content**
- [ ] Enter text from a note in search
- [ ] **Expected:** Items with matching notes show
- [ ] **Expected:** Partial match works

**Test 7.15: Get All Tags**
- [ ] Navigate to tag filter
- [ ] **Expected:** All unique tags from watchlist show
- [ ] **Expected:** Tags sorted alphabetically
- [ ] **Expected:** No duplicate tags

### **Test Suite 4: Persistence**

**Test 7.16: Notes Persist After Restart**
- [ ] Add notes to several items
- [ ] Close app completely
- [ ] Reopen app
- [ ] Navigate to Watchlist
- [ ] **Expected:** All notes still present
- [ ] **Expected:** Note content unchanged

**Test 7.17: Tags Persist After Restart**
- [ ] Add tags to several items
- [ ] Close app completely
- [ ] Reopen app
- [ ] Navigate to Watchlist
- [ ] **Expected:** All tags still present
- [ ] **Expected:** Tag selections unchanged

### **Test Suite 5: Edge Cases**

**Test 7.18: Empty Note**
- [ ] Try to save empty note
- [ ] **Expected:** Note not saved OR removed if editing

**Test 7.19: No Tags Selected**
- [ ] Open tag selector
- [ ] Don't select any tags
- [ ] Tap "Save"
- [ ] **Expected:** No tags added
- [ ] **Expected:** No error

**Test 7.20: Maximum Tags**
- [ ] Try to add 10+ tags to one item
- [ ] **Expected:** All tags save successfully
- [ ] **Expected:** UI handles many tags gracefully

---

## 📋 FEATURE 8: SCAN HISTORY

### **Test Suite 6: Save Scan History**

**Test 8.1: Run Scan and Auto-Save**
- [ ] Navigate to Scanner page
- [ ] Run a scan with specific filters
- [ ] Wait for scan to complete
- [ ] Navigate to History page
- [ ] **Expected:** New scan appears in history
- [ ] **Expected:** Scan has correct timestamp
- [ ] **Expected:** Scan shows result count

**Test 8.2: Multiple Scans**
- [ ] Run 3-4 different scans
- [ ] Check History page after each
- [ ] **Expected:** All scans saved
- [ ] **Expected:** Most recent scan at top
- [ ] **Expected:** Scans sorted by date (newest first)

**Test 8.3: Scan with Different Filters**
- [ ] Run scan with min MERIT = 70
- [ ] Run scan with min MERIT = 80
- [ ] Run scan with different sectors
- [ ] **Expected:** Each scan saved separately
- [ ] **Expected:** Filter settings preserved

### **Test Suite 7: View Scan History**

**Test 8.4: History List Display**
- [ ] Navigate to History page
- [ ] **Expected:** All saved scans show
- [ ] **Expected:** Each card shows:
  - [ ] Formatted timestamp
  - [ ] Result count
  - [ ] Average MERIT score
  - [ ] Scan type/filters

**Test 8.5: Empty State**
- [ ] Clear all history (or fresh install)
- [ ] Navigate to History page
- [ ] **Expected:** Empty state message shows
- [ ] **Expected:** Helpful text displayed
- [ ] **Expected:** No error

**Test 8.6: Formatted Timestamps**
- [ ] Check timestamp formats:
  - [ ] Today: "Today at 2:30 PM"
  - [ ] Yesterday: "Yesterday at 10:15 AM"
  - [ ] This week: "Monday at 3:45 PM"
  - [ ] Older: "Jan 15 at 9:00 AM"

### **Test Suite 8: Scan History Details**

**Test 8.7: Open Detail Page**
- [ ] Tap on a scan history item
- [ ] **Expected:** Detail page opens
- [ ] **Expected:** Full scan results show
- [ ] **Expected:** All result cards display

**Test 8.8: Detail Page Content**
- [ ] Verify detail page shows:
  - [ ] Scan timestamp
  - [ ] Total results
  - [ ] Average MERIT score
  - [ ] All result cards
  - [ ] Ticker symbols
  - [ ] MERIT scores
  - [ ] Signals

**Test 8.9: Navigate from Detail**
- [ ] Open scan detail
- [ ] Tap on a result card
- [ ] **Expected:** Symbol detail page opens
- [ ] **Expected:** Correct symbol loaded

### **Test Suite 9: Delete Scan History**

**Test 8.10: Delete Single Scan**
- [ ] Swipe left on a scan item (or tap delete)
- [ ] Confirm deletion
- [ ] **Expected:** Scan removed from list
- [ ] **Expected:** Other scans remain

**Test 8.11: Delete All Scans**
- [ ] Delete each scan one by one
- [ ] **Expected:** Empty state shows when all deleted
- [ ] **Expected:** No errors

### **Test Suite 10: Auto-Limit (10 Scans)**

**Test 8.12: Exceed 10 Scan Limit**
- [ ] Run 11+ scans
- [ ] Check History page
- [ ] **Expected:** Only 10 most recent scans show
- [ ] **Expected:** Oldest scan auto-deleted
- [ ] **Expected:** No manual deletion needed

**Test 8.13: Verify Oldest Deleted**
- [ ] Note timestamp of oldest scan
- [ ] Run new scan
- [ ] **Expected:** Oldest scan gone
- [ ] **Expected:** New scan at top

### **Test Suite 11: Persistence**

**Test 8.14: History Persists After Restart**
- [ ] Run several scans
- [ ] Close app completely
- [ ] Reopen app
- [ ] Navigate to History
- [ ] **Expected:** All scans still present
- [ ] **Expected:** Scan data intact

**Test 8.15: Results Persist**
- [ ] Open scan detail
- [ ] Note specific results
- [ ] Restart app
- [ ] Open same scan detail
- [ ] **Expected:** Exact same results show
- [ ] **Expected:** No data loss

### **Test Suite 12: Edge Cases**

**Test 8.16: Scan with Zero Results**
- [ ] Run scan with very strict filters
- [ ] Get zero results
- [ ] **Expected:** Scan still saved
- [ ] **Expected:** Shows "0 results"
- [ ] **Expected:** No crash

**Test 8.17: Scan with Many Results**
- [ ] Run scan with loose filters
- [ ] Get 50+ results
- [ ] **Expected:** All results saved
- [ ] **Expected:** Detail page scrollable
- [ ] **Expected:** Performance acceptable

---

## 📋 FEATURE 9: DARK/LIGHT THEME TOGGLE

### **Test Suite 13: Theme Toggle**

**Test 9.1: Toggle to Light Mode**
- [ ] Navigate to Settings
- [ ] Find Theme toggle
- [ ] Switch to Light mode
- [ ] **Expected:** Immediate theme change
- [ ] **Expected:** All colors update
- [ ] **Expected:** Smooth transition

**Test 9.2: Toggle to Dark Mode**
- [ ] Switch back to Dark mode
- [ ] **Expected:** Immediate theme change
- [ ] **Expected:** All colors revert
- [ ] **Expected:** Smooth transition

**Test 9.3: Toggle Label Updates**
- [ ] Check toggle label in Light mode
- [ ] **Expected:** Shows "Light Mode"
- [ ] Switch to Dark
- [ ] **Expected:** Shows "Dark Mode"

### **Test Suite 14: Theme Persistence**

**Test 9.4: Light Mode Persists**
- [ ] Switch to Light mode
- [ ] Close app completely
- [ ] Reopen app
- [ ] **Expected:** Still in Light mode
- [ ] **Expected:** Theme preference saved

**Test 9.5: Dark Mode Persists**
- [ ] Switch to Dark mode
- [ ] Close app completely
- [ ] Reopen app
- [ ] **Expected:** Still in Dark mode
- [ ] **Expected:** Theme preference saved

### **Test Suite 15: All Pages in Light Mode**

**Test 9.6: Scanner Page - Light**
- [ ] Switch to Light mode
- [ ] Navigate to Scanner
- [ ] **Expected:** Background light
- [ ] **Expected:** Text readable
- [ ] **Expected:** Cards styled correctly
- [ ] **Expected:** Buttons visible

**Test 9.7: Watchlist Page - Light**
- [ ] Navigate to Watchlist
- [ ] **Expected:** Light theme applied
- [ ] **Expected:** Items readable
- [ ] **Expected:** Tags visible
- [ ] **Expected:** Notes readable

**Test 9.8: History Page - Light**
- [ ] Navigate to History
- [ ] **Expected:** Light theme applied
- [ ] **Expected:** Cards styled correctly
- [ ] **Expected:** Timestamps readable

**Test 9.9: Settings Page - Light**
- [ ] Navigate to Settings
- [ ] **Expected:** Light theme applied
- [ ] **Expected:** All sections readable
- [ ] **Expected:** Toggle visible

**Test 9.10: Symbol Detail - Light**
- [ ] Open any symbol detail
- [ ] **Expected:** Light theme applied
- [ ] **Expected:** Charts visible
- [ ] **Expected:** MERIT breakdown readable
- [ ] **Expected:** Trade plan visible

### **Test Suite 16: All Pages in Dark Mode**

**Test 9.11: Scanner Page - Dark**
- [ ] Switch to Dark mode
- [ ] Navigate to Scanner
- [ ] **Expected:** Background dark
- [ ] **Expected:** Text readable (white/light)
- [ ] **Expected:** Cards styled correctly
- [ ] **Expected:** Buttons visible

**Test 9.12: Watchlist Page - Dark**
- [ ] Navigate to Watchlist
- [ ] **Expected:** Dark theme applied
- [ ] **Expected:** Items readable
- [ ] **Expected:** Tags visible
- [ ] **Expected:** Notes readable

**Test 9.13: History Page - Dark**
- [ ] Navigate to History
- [ ] **Expected:** Dark theme applied
- [ ] **Expected:** Cards styled correctly
- [ ] **Expected:** Timestamps readable

**Test 9.14: Settings Page - Dark**
- [ ] Navigate to Settings
- [ ] **Expected:** Dark theme applied
- [ ] **Expected:** All sections readable
- [ ] **Expected:** Toggle visible

**Test 9.15: Symbol Detail - Dark**
- [ ] Open any symbol detail
- [ ] **Expected:** Dark theme applied
- [ ] **Expected:** Charts visible
- [ ] **Expected:** MERIT breakdown readable
- [ ] **Expected:** Trade plan visible

### **Test Suite 17: Dialogs & Modals**

**Test 9.16: Add Note Dialog - Both Themes**
- [ ] Open note dialog in Light mode
- [ ] **Expected:** Dialog styled correctly
- [ ] Switch to Dark mode
- [ ] Open note dialog
- [ ] **Expected:** Dialog styled correctly

**Test 9.17: Tag Selector - Both Themes**
- [ ] Open tag selector in Light mode
- [ ] **Expected:** Tags readable
- [ ] Switch to Dark mode
- [ ] Open tag selector
- [ ] **Expected:** Tags readable

**Test 9.18: Confirmation Dialogs - Both Themes**
- [ ] Trigger delete confirmation in Light
- [ ] **Expected:** Dialog styled correctly
- [ ] Switch to Dark
- [ ] Trigger delete confirmation
- [ ] **Expected:** Dialog styled correctly

### **Test Suite 18: UI Components**

**Test 9.19: Buttons - Both Themes**
- [ ] Check all button types in Light:
  - [ ] Elevated buttons
  - [ ] Outlined buttons
  - [ ] Text buttons
- [ ] Switch to Dark
- [ ] Check all button types
- [ ] **Expected:** All buttons visible and styled

**Test 9.20: Cards - Both Themes**
- [ ] Check card styling in Light
- [ ] **Expected:** Cards have light background
- [ ] **Expected:** Borders visible
- [ ] Switch to Dark
- [ ] **Expected:** Cards have dark background
- [ ] **Expected:** Borders visible

**Test 9.21: Text - Both Themes**
- [ ] Check text readability in Light
- [ ] **Expected:** Dark text on light background
- [ ] Switch to Dark
- [ ] **Expected:** Light text on dark background
- [ ] **Expected:** All text readable

**Test 9.22: Icons - Both Themes**
- [ ] Check icon visibility in Light
- [ ] **Expected:** Icons visible
- [ ] Switch to Dark
- [ ] **Expected:** Icons visible
- [ ] **Expected:** Icon colors appropriate

### **Test Suite 19: Transitions**

**Test 9.23: Smooth Theme Transition**
- [ ] Toggle theme multiple times
- [ ] **Expected:** Smooth color transitions
- [ ] **Expected:** No flashing
- [ ] **Expected:** No layout shifts

**Test 9.24: No Performance Issues**
- [ ] Toggle theme rapidly
- [ ] **Expected:** No lag
- [ ] **Expected:** No crashes
- [ ] **Expected:** Responsive UI

---

## 📊 TESTING SUMMARY

### **Total Test Cases:** 94

**Feature 7 (Notes/Tags):** 20 tests  
**Feature 8 (Scan History):** 17 tests  
**Feature 9 (Theme Toggle):** 24 tests  
**Integration Tests:** 33 tests

### **Completion Tracking:**

- [ ] Feature 7: ___/20 tests passed
- [ ] Feature 8: ___/17 tests passed
- [ ] Feature 9: ___/24 tests passed
- [ ] Integration: ___/33 tests passed

**Overall:** ___/94 tests passed (___%)

---

## 🐛 BUG TRACKING

### **Bugs Found:**

| # | Feature | Test | Description | Severity | Status |
|---|---------|------|-------------|----------|--------|
| 1 |         |      |             |          |        |
| 2 |         |      |             |          |        |
| 3 |         |      |             |          |        |

### **Severity Levels:**
- **Critical:** App crashes, data loss
- **High:** Feature doesn't work
- **Medium:** Feature works but has issues
- **Low:** Minor UI/UX issues

---

## ✅ SIGN-OFF

**Tester:** _______________  
**Date:** _______________  
**Result:** PASS / FAIL / PARTIAL  
**Notes:** _______________

---

## 📝 NOTES

Use this section for any additional observations, suggestions, or feedback during testing.
