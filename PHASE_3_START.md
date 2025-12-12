# Phase 3: Page Extraction - Starting Now

## Current Status
- **Total lines in main.dart**: 5,682
- **Phase 2 completed**: Extracted ~2,900 lines (models, services, providers, widgets, utils)
- **Remaining to extract**: ~2,782 lines

## What's Still in main.dart
1. **Brand colors & global state** (~50 lines)
2. **TechnicApp widget** (~200 lines) - MaterialApp configuration
3. **TechnicShell widget** (~200 lines) - Navigation shell
4. **ScannerPage** (~1,200 lines) - Most complex page
5. **IdeasPage** (~400 lines)
6. **CopilotPage** (~350 lines)
7. **SettingsPage** (~300 lines)
8. **MyIdeasPage** (~50 lines) - Simplest, already seen at end
9. **Helper functions** (~200 lines) - `_fmtField`, `_fmtLocalTime`, `tone`, etc.
10. **Widget builders** (~500 lines) - `_heroBanner`, `_infoCard`, `_scanResultCard`, etc.
11. **LocalStore class** (~200 lines) - Already at end of file
12. **Mock data** (~200 lines) - Constants for fallback data

## Extraction Strategy - Revised

### Step 1: Create Directory Structure
```
lib/screens/
├── scanner/
│   ├── scanner_page.dart
│   └── widgets/
├── ideas/
│   ├── ideas_page.dart
│   └── widgets/
├── copilot/
│   ├── copilot_page.dart
│   └── widgets/
├── my_ideas/
│   └── my_ideas_page.dart
└── settings/
    └── settings_page.dart
```

### Step 2: Extract in This Order
1. ✅ **Helper functions** → `lib/utils/helpers.dart`
2. ✅ **Shared widget builders** → `lib/widgets/` (hero_banner, etc.)
3. ✅ **LocalStore** → `lib/services/local_store.dart`
4. ✅ **Mock data** → `lib/utils/mock_data.dart`
5. ✅ **MyIdeasPage** → `lib/screens/my_ideas/my_ideas_page.dart` (simplest)
6. ✅ **SettingsPage** → `lib/screens/settings/settings_page.dart`
7. ✅ **CopilotPage** → `lib/screens/copilot/copilot_page.dart`
8. ✅ **IdeasPage** → `lib/screens/ideas/ideas_page.dart`
9. ✅ **ScannerPage** → `lib/screens/scanner/scanner_page.dart` (most complex)
10. ✅ **TechnicShell** → `lib/shell.dart`
11. ✅ **TechnicApp** → `lib/app.dart`
12. ✅ **Update main.dart** → Minimal entry point

### Step 3: Testing After Each Major Extraction
- Run `flutter analyze` after each file
- Fix any import issues immediately
- Ensure no breaking changes

## Time Estimate
- **Per simple page**: 15-20 minutes
- **Per complex page**: 30-45 minutes
- **Testing & fixes**: 30 minutes
- **Total**: 3-4 hours

## Starting Now!
Let's begin with the helper functions and work our way up to the complex pages.
