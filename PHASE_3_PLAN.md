# Phase 3: Page Extraction Plan

## Overview
Extract remaining ~2,782 lines from main.dart into separate page files with proper structure.

## Current State
- **main.dart**: 5,682 lines total
- **Already extracted**: ~2,900 lines (models, services, providers, widgets, utils)
- **Remaining**: ~2,782 lines (pages, shell, helpers)

## Extraction Strategy

### 1. Scanner Page (~1,200 lines)
**File**: `lib/screens/scanner/scanner_page.dart`
**Lines**: ~800-1000 from _ScannerPageState
**Widgets to extract**:
- `lib/screens/scanner/widgets/filter_panel.dart` (~300 lines)
- `lib/screens/scanner/widgets/scan_result_card.dart` (~200 lines)
- `lib/screens/scanner/widgets/market_pulse_card.dart` (~100 lines)
- `lib/screens/scanner/widgets/saved_screen_card.dart` (~100 lines)
- `lib/screens/scanner/widgets/quick_actions.dart` (~100 lines)

### 2. Ideas Page (~400 lines)
**File**: `lib/screens/ideas/ideas_page.dart`
**Lines**: ~300 from _IdeasPageState
**Widgets to extract**:
- `lib/screens/ideas/widgets/idea_card.dart` (~100 lines)

### 3. Copilot Page (~350 lines)
**File**: `lib/screens/copilot/copilot_page.dart`
**Lines**: ~250 from _CopilotPageState
**Widgets to extract**:
- `lib/screens/copilot/widgets/message_bubble.dart` (~50 lines)
- `lib/screens/copilot/widgets/context_card.dart` (~50 lines)

### 4. My Ideas Page (~200 lines)
**File**: `lib/screens/my_ideas/my_ideas_page.dart`
**Lines**: Simple page, mostly complete

### 5. Settings Page (~300 lines)
**File**: `lib/screens/settings/settings_page.dart`
**Lines**: ~250 from SettingsPage
**Widgets to extract**:
- `lib/screens/settings/widgets/profile_row.dart` (~30 lines)

### 6. App Shell (~400 lines)
**File**: `lib/app.dart` - TechnicApp widget
**File**: `lib/shell.dart` - TechnicShell navigation
**Lines**: ~300 for shell, ~100 for app

### 7. Helper Functions & Widgets (~200 lines)
**File**: `lib/widgets/hero_banner.dart` (~80 lines)
**File**: `lib/utils/helpers.dart` (~120 lines)
- `_fmtField`, `_fmtLocalTime`, `_colorFromHex`, etc.

### 8. Legacy Models (if not extracted) (~200 lines)
Move any remaining model classes to models/ directory

## Execution Order
1. ✅ Extract helper functions first (utils/helpers.dart)
2. ✅ Extract shared widgets (hero_banner.dart)
3. ✅ Extract Settings page (simplest)
4. ✅ Extract My Ideas page
5. ✅ Extract Copilot page + widgets
6. ✅ Extract Ideas page + widgets
7. ✅ Extract Scanner page + widgets (most complex)
8. ✅ Extract App shell
9. ✅ Update main.dart to minimal entry point
10. ✅ Test compilation

## Success Criteria
- main.dart < 100 lines (just entry point)
- All pages in separate files
- Zero compilation errors
- Zero analyzer warnings
- Successful build

## Timeline
- Helpers & shared widgets: 30 min
- Simple pages (Settings, My Ideas): 30 min
- Medium pages (Copilot, Ideas): 1 hour
- Complex page (Scanner): 1.5 hours
- App shell & cleanup: 30 min
- Testing & fixes: 30 min

**Total estimated time**: 4-5 hours
