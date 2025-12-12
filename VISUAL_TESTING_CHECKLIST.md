# Phase 4 Visual Testing Checklist

## Pre-Testing Setup

### 1. Start the Backend API (if needed for live data)
```bash
# Option A: Streamlit (current dev API)
cd technic_v4
streamlit run ui/technic_app.py

# Option B: FastAPI (if available)
cd technic_v4
uvicorn api_server:app --reload --port 8000
```

### 2. Run the Flutter App
```bash
cd technic_app
flutter run
```

**Expected**: App launches without errors

---

## Testing Checklist

### ✅ App Launch
- [ ] App launches successfully
- [ ] No error messages in console
- [ ] Splash screen (if any) displays correctly
- [ ] Initial page loads (Scanner by default)

### ✅ App Shell / Navigation

#### Header
- [ ] Logo displays correctly
- [ ] "technic" title visible
- [ ] Current page subtitle shows correctly
- [ ] **"Live" indicator is GONE** ✨
- [ ] Header background is solid color (not gradient)
- [ ] Header has subtle border at bottom

#### Bottom Navigation
- [ ] All 5 tabs visible (Scan, Ideas, Copilot, My Ideas, Settings)
- [ ] Icons render correctly
- [ ] Labels visible
- [ ] Tapping each tab switches pages
- [ ] Selected tab highlighted properly
- [ ] Navigation bar has subtle shadow

#### Theme
- [ ] Default theme loads (dark or light based on system)
- [ ] Colors look professional (no neon green)
- [ ] Background is solid (not gradient)

---

### ✅ Scanner Page

#### Visual Design
- [ ] Page background is solid color (not gradient)
- [ ] All cards have flat design (no heavy gradients)
- [ ] Shadows are subtle (6px blur, not 18px)
- [ ] Colors are muted/professional (no neon)
- [ ] Spacing looks consistent

#### Market Pulse Card
- [ ] Card renders with flat background
- [ ] Movers list displays
- [ ] Up/down indicators show correctly
- [ ] Percentages formatted properly
- [ ] No gradient background ✨

#### Scoreboard Card
- [ ] Card renders with flat background
- [ ] Win rates display
- [ ] Categories show (Day, Swing, Long)
- [ ] No gradient background ✨

#### Onboarding Card (if visible)
- [ ] Card renders with flat background
- [ ] Welcome message displays
- [ ] Tips/instructions readable
- [ ] No gradient background ✨

#### Filter Panel
- [ ] Filters accessible (button or panel)
- [ ] All filter options visible
- [ ] Dropdowns work
- [ ] Sliders work
- [ ] Switches work
- [ ] "Apply" button works

#### Scan Results
- [ ] Results list displays
- [ ] Each result card shows:
  - [ ] Ticker symbol
  - [ ] Signal type
  - [ ] TechRating
  - [ ] Entry/Stop/Target prices
  - [ ] Sparkline chart (should have gradient - data viz exception)
  - [ ] Sector/Industry
- [ ] Cards have flat background (no gradient)
- [ ] Tapping a result shows details or does nothing (expected)

#### Functionality
- [ ] Pull-to-refresh works (if implemented)
- [ ] Scrolling is smooth
- [ ] No lag or stuttering
- [ ] Search works (if implemented)

---

### ✅ Ideas Page

#### Visual Design
- [ ] Hero banner is flat (no gradient) ✨
- [ ] Page background is solid
- [ ] Cards have flat design
- [ ] Colors are professional

#### Ideas List
- [ ] Ideas display as cards
- [ ] Each idea shows:
  - [ ] Title
  - [ ] Ticker
  - [ ] Strategy type
  - [ ] Rationale/meta text
  - [ ] Sparkline (should have gradient)
- [ ] Cards are flat (no gradient background)
- [ ] Scrolling is smooth

#### Functionality
- [ ] Tapping an idea shows details or Copilot
- [ ] Long-press actions work (if implemented)
- [ ] Filtering works (if implemented)

---

### ✅ Copilot Page

#### Visual Design
- [ ] Hero banner is flat (no gradient) ✨
- [ ] Page background is solid
- [ ] Chat area looks clean
- [ ] Message bubbles styled correctly

#### Chat Interface
- [ ] Text input field visible
- [ ] Send button visible
- [ ] Message history displays (if any)
- [ ] User messages vs assistant messages distinguishable
- [ ] Typing indicator works (if implemented)

#### Functionality
- [ ] **Page loads without errors** (provider fix) ✨
- [ ] Can type in text field
- [ ] Send button works
- [ ] Messages appear in chat
- [ ] Scrolling works
- [ ] Offline message shows if backend unavailable

---

### ✅ My Ideas Page

#### Visual Design
- [ ] Page background is solid
- [ ] Watchlist items display correctly
- [ ] Empty state shows if no saved ideas

#### Watchlist
- [ ] Saved symbols display
- [ ] Each item shows ticker
- [ ] Tapping item does something (details or remove)
- [ ] Star icon to remove works

#### Functionality
- [ ] Can add items from Scanner (star button)
- [ ] Can remove items
- [ ] List persists across app restarts

---

### ✅ Settings Page

#### Visual Design
- [ ] Hero banner is flat (no gradient) ✨
- [ ] Page background is solid
- [ ] Settings sections organized clearly
- [ ] Theme preview (if any) is flat (no gradient) ✨

#### Settings Options
- [ ] **Theme toggle works** (light/dark switch) ✨
- [ ] Options mode toggle works (stock only vs stock+options)
- [ ] Risk profile selector works (if implemented)
- [ ] All settings save properly

#### Functionality
- [ ] Changing theme updates entire app immediately
- [ ] Settings persist across app restarts
- [ ] About/version info displays

---

### ✅ Cross-Cutting Tests

#### Theme Switching
- [ ] Switch to light mode (if in dark)
- [ ] All pages look good in light mode
- [ ] No visual glitches
- [ ] Colors remain professional
- [ ] Switch to dark mode
- [ ] All pages look good in dark mode
- [ ] No visual glitches

#### Performance
- [ ] App feels responsive
- [ ] No lag when scrolling
- [ ] No lag when switching tabs
- [ ] Animations are smooth
- [ ] No memory leaks (app doesn't slow down over time)

#### Edge Cases
- [ ] Rotate device (if on mobile)
- [ ] Layout adapts correctly
- [ ] No overflow errors
- [ ] Test on smallest screen size (iPhone SE)
- [ ] Test on largest screen size (iPad Pro)

---

## Issues Found

### Critical Issues (Must Fix)
_List any critical bugs or visual problems_

1. 
2. 
3. 

### Minor Issues (Nice to Fix)
_List any minor polish items_

1. 
2. 
3. 

### Enhancements (Future)
_List any ideas for improvements_

1. 
2. 
3. 

---

## Testing Summary

**Date**: ___________
**Tester**: ___________
**Device**: ___________
**OS Version**: ___________

**Overall Assessment**:
- [ ] Ready for production
- [ ] Needs minor fixes
- [ ] Needs major fixes

**Notes**:
_Any additional observations_
