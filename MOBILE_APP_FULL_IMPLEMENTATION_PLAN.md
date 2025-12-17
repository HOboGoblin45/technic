# Mobile App Full Implementation Plan

**Goal:** Copy full app from `technic_app/` to `technic_mobile/` and refine with Macintosh aesthetic

---

## Step 1: Copy Full App (Session 1-2)

### 1.1 Copy Models & Data Structures
**From:** `technic_app/lib/models/`  
**To:** `technic_mobile/lib/models/`

Files to copy:
- `scan_result.dart` - Scanner result model
- `scanner_bundle.dart` - Bundle of scan data
- `market_mover.dart` - Market movers data
- `saved_screen.dart` - Saved presets
- All other model files

### 1.2 Copy Services
**From:** `technic_app/lib/services/`  
**To:** `technic_mobile/lib/services/`

Files to copy:
- API service (backend communication)
- Local store (offline storage)
- Any other service files

### 1.3 Copy Providers
**From:** `technic_app/lib/providers/`  
**To:** `technic_mobile/lib/providers/`

Files to copy:
- `app_providers.dart` - Riverpod providers
- Any other provider files

### 1.4 Copy Utilities
**From:** `technic_app/lib/utils/`  
**To:** `technic_mobile/lib/utils/`

Files to copy:
- `helpers.dart` - Helper functions
- Any other utility files

### 1.5 Merge Theme Files
**Action:** Merge `technic_app/lib/theme/app_colors.dart` into existing `technic_mobile/lib/theme/app_theme.dart`
**Status:** ✅ DONE - Colors already corrected

---

## Step 2: Refine UI with Mac Aesthetic (Session 3-4)

### 2.1 Scanner Screen
**From:** `technic_app/lib/screens/scanner/`  
**To:** `technic_mobile/lib/screens/scanner/`

**Refinements:**
- ✨ Cleaner layout with more whitespace
- ✨ Subtle shadows (Mac-style depth)
- ✨ Smooth animations
- ✨ SF Pro typography
- ✨ Simplified icons
- ✨ Card-based design
- ✨ Keep Technic colors

### 2.2 Watchlist Screen
**From:** `technic_app/lib/screens/watchlist/`  
**To:** `technic_mobile/lib/screens/watchlist/`

**Refinements:**
- ✨ Clean list design
- ✨ Swipe actions (iOS-style)
- ✨ Smooth transitions
- ✨ Elegant empty states

### 2.3 Settings Screen
**From:** `technic_app/lib/screens/settings/`  
**To:** `technic_mobile/lib/screens/settings/`

**Refinements:**
- ✨ Grouped settings (iOS-style)
- ✨ Clean toggles and sliders
- ✨ Minimal design

### 2.4 Symbol Detail Screen
**From:** `technic_app/lib/screens/symbol_detail/`  
**To:** `technic_mobile/lib/screens/symbol_detail/`

**Refinements:**
- ✨ Hero animations
- ✨ Smooth scrolling
- ✨ Clean charts
- ✨ Card-based metrics

---

## Step 3: Add Features (Session 5-6)

### 3.1 Core Features
- ✅ Scanner with filters
- ✅ Watchlist management
- ✅ Settings & preferences
- ✅ Symbol details
- ✅ API integration
- ✅ Progress tracking
- ✅ Offline support

### 3.2 Mac Aesthetic Features
- ✨ Smooth page transitions
- ✨ Haptic feedback
- ✨ Pull-to-refresh
- ✨ Contextual menus
- ✨ Keyboard shortcuts (web/desktop)
- ✨ Spotlight-style search

### 3.3 Polish
- ✨ Loading states
- ✨ Error handling
- ✨ Empty states
- ✨ Onboarding
- ✨ Animations
- ✨ Accessibility

---

## Mac Aesthetic Guidelines

### Typography
- **Font:** SF Pro (system font)
- **Hierarchy:** Clear size/weight differences
- **Spacing:** Generous line height

### Layout
- **Whitespace:** 16-24px padding
- **Cards:** 12-16px border radius
- **Spacing:** 8px grid system

### Colors
- **Keep:** Technic institutional palette
- **Backgrounds:** Deep navy (#0A0E27)
- **Cards:** Slate-900 (#141B2D)
- **Primary:** Blue (#3B82F6)
- **Success:** Emerald (#10B981)

### Shadows
- **Subtle:** 0 2px 8px rgba(0,0,0,0.1)
- **Elevated:** 0 4px 16px rgba(0,0,0,0.15)
- **Floating:** 0 8px 24px rgba(0,0,0,0.2)

### Animations
- **Duration:** 200-300ms
- **Easing:** Cubic bezier (ease-in-out)
- **Smooth:** No jarring transitions

### Icons
- **Style:** Simple, recognizable
- **Size:** 20-24px standard
- **Weight:** Medium (not too thin/thick)

---

## Implementation Timeline

### Session 1 (Today)
- ✅ Correct colors
- 🔄 Copy models
- 🔄 Copy services
- 🔄 Copy providers

### Session 2
- Copy scanner screen
- Copy widgets
- Test compilation

### Session 3
- Refine scanner UI
- Apply Mac aesthetic
- Test functionality

### Session 4
- Copy other screens
- Refine each screen
- Test navigation

### Session 5
- Polish animations
- Add loading states
- Error handling

### Session 6
- Final testing
- Performance optimization
- Deploy

---

## Current Status

✅ **Foundation Complete**
- Flutter 3.38.3 environment
- 130+ project files
- Correct Technic colors
- 0 compilation errors

🔄 **Next: Start Step 1**
- Copy models from technic_app
- Copy services
- Copy providers
- Test compilation

---

## Success Criteria

### Functionality
- ✅ All features from technic_app work
- ✅ API integration functional
- ✅ Offline mode works
- ✅ Navigation smooth

### Design
- ✅ Mac aesthetic applied
- ✅ Technic colors maintained
- ✅ Professional appearance
- ✅ Smooth animations

### Performance
- ✅ Fast load times
- ✅ Smooth scrolling
- ✅ Responsive UI
- ✅ No lag

---

## Let's Begin!

Starting with Step 1: Copying the full app structure from `technic_app/` to `technic_mobile/`.
