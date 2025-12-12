# Technic App - Runtime Testing Checklist

## Test Session Information
- **Date**: January 2025
- **Version**: Phase 3.5 (Refactored UI with Symbol Detail)
- **Platform**: Windows Desktop
- **Build**: Debug

## Testing Status: 🔄 IN PROGRESS

---

## 1. Initial Launch ✅
- [x] App launches without crashes
- [x] No compilation errors
- [ ] Main screen displays correctly
- [ ] Bottom navigation visible
- [ ] Theme applied correctly

**User Confirmation Needed**: Please describe what you see on the screen now. Is it different from the previous version?

---

## 2. Scanner Page Testing
### Basic Functionality
- [ ] Scanner page loads
- [ ] Onboarding card displays (if first time)
- [ ] Risk profile buttons visible (Conservative/Moderate/Aggressive)
- [ ] Randomize button works
- [ ] Filter button accessible
- [ ] Scan button/action works

### Scan Results
- [ ] Results display in cards
- [ ] Each card shows: ticker, signal, entry/stop/target
- [ ] Sparklines render correctly
- [ ] Tap on result navigates to Symbol Detail page ⭐ (NEW)

### Market Pulse
- [ ] Market movers section displays
- [ ] Movers show percentage changes
- [ ] Color coding (green/red) correct

### Scoreboard
- [ ] Performance metrics display
- [ ] Win rates shown
- [ ] Categories visible (Day/Swing/Long-term)

---

## 3. Symbol Detail Page Testing ⭐ (NEW FEATURE)
### Navigation
- [ ] Opens when tapping scan result
- [ ] Opens when tapping My Ideas item
- [ ] Back button works correctly
- [ ] Smooth transition animation

### Content Display
- [ ] Ticker symbol shown prominently
- [ ] Signal badge displays (BUY/SELL/HOLD)
- [ ] Current price visible
- [ ] Chart/sparkline renders

### Technical Metrics
- [ ] Tech Rating displayed
- [ ] R/R Ratio shown
- [ ] Win Probability visible
- [ ] ICS (Institutional Core Score) present

### Trade Plan
- [ ] Entry price shown
- [ ] Stop loss displayed
- [ ] Target price visible
- [ ] All values formatted correctly

### Fundamentals
- [ ] Sector information
- [ ] Industry displayed
- [ ] Quality Score shown
- [ ] Data accurate

### Action Buttons
- [ ] "Ask Copilot" button visible
- [ ] "View Options" button present
- [ ] Buttons respond to taps
- [ ] Appropriate actions triggered

---

## 4. Ideas Page Testing
### Display
- [ ] Ideas feed loads
- [ ] Cards display with sparklines
- [ ] Signal indicators visible
- [ ] Swipe gestures work (if implemented)

### Interaction
- [ ] Tap on idea card works
- [ ] Quick actions accessible
- [ ] Filtering works (if implemented)

---

## 5. Copilot Page Testing
### Interface
- [ ] Chat interface displays
- [ ] Input field visible
- [ ] Send button accessible
- [ ] Message history shows

### Functionality
- [ ] Can type messages
- [ ] Send button works
- [ ] Responses display correctly
- [ ] Typing indicator shows (if implemented)
- [ ] Context from Symbol Detail works

---

## 6. My Ideas Page Testing
### Display
- [ ] Watchlist items display
- [ ] Star icons visible
- [ ] Ticker symbols shown
- [ ] Notes display (if any)

### Interaction
- [ ] Tap on item navigates to Symbol Detail ⭐ (NEW)
- [ ] Delete button works
- [ ] Items persist after restart

---

## 7. Settings Page Testing
### Display
- [ ] Settings page loads
- [ ] All sections visible
- [ ] Profile row displays
- [ ] Theme toggle present

### New Features
- [ ] Profile edit button shows placeholder ⭐ (NEW)
- [ ] Mute alerts button shows placeholder ⭐ (NEW)
- [ ] Refresh rate button opens dialog ⭐ (NEW)
- [ ] Dialog shows options (30s/1m/5m)

### Existing Features
- [ ] Theme switching works
- [ ] Options mode toggle works
- [ ] Settings persist

---

## 8. Navigation Flow Testing
### Tab Navigation
- [ ] Scanner tab works
- [ ] Ideas tab works
- [ ] Copilot tab works
- [ ] My Ideas tab works
- [ ] Settings tab works
- [ ] Active tab highlighted

### Deep Navigation
- [ ] Scanner → Symbol Detail → Back
- [ ] My Ideas → Symbol Detail → Back
- [ ] Symbol Detail → Ask Copilot → Copilot page
- [ ] Symbol Detail → View Options (if implemented)

### State Preservation
- [ ] Scanner state preserved when switching tabs
- [ ] Copilot messages preserved
- [ ] Watchlist persists
- [ ] Settings persist

---

## 9. Theme Testing
### Light Theme
- [ ] All screens readable
- [ ] Contrast sufficient
- [ ] Colors appropriate

### Dark Theme
- [ ] All screens readable
- [ ] Contrast sufficient
- [ ] Colors appropriate
- [ ] Smooth theme transition

---

## 10. Error Handling
### Network Errors
- [ ] Graceful handling when API offline
- [ ] Error messages clear
- [ ] Fallback to mock data works
- [ ] Retry options available

### Data Errors
- [ ] Missing data handled gracefully
- [ ] Invalid data doesn't crash app
- [ ] Loading states display correctly

---

## 11. Performance
### Responsiveness
- [ ] UI responds quickly to taps
- [ ] No lag when scrolling
- [ ] Smooth animations
- [ ] No frame drops

### Memory
- [ ] No memory leaks observed
- [ ] App stable over time
- [ ] No crashes during testing

---

## 12. Visual Polish
### Design Consistency
- [ ] Consistent spacing
- [ ] Aligned elements
- [ ] Proper typography
- [ ] Color scheme cohesive

### Platform Adaptation
- [ ] Looks native on Windows
- [ ] Proper window controls
- [ ] Appropriate sizing

---

## Issues Found

### Critical Issues
(None yet)

### Minor Issues
(To be documented during testing)

### Suggestions
(To be documented during testing)

---

## Next Steps After Testing

1. **If issues found**: Fix and retest
2. **If all passes**: Proceed to Phase 4 (Deployment prep)
3. **Document findings**: Update this checklist with results

---

## User Instructions

Please test the app and provide feedback on:
1. What you see on the main screen
2. Try tapping on different elements
3. Navigate through all tabs
4. Try the new Symbol Detail feature (tap on any scan result or My Ideas item)
5. Test the Settings placeholders
6. Report any issues, crashes, or unexpected behavior

Take screenshots if helpful!
