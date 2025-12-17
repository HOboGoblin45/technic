# Mobile App UI Migration Plan

## Objective
Copy the complete, production-ready UI from `technic_app` to `technic_mobile` to match the exact design shown in screenshots.

## Current State
- `technic_app/`: Complete Flutter app with all screens and functionality
- `technic_mobile/`: Basic structure with backend integration but minimal UI

## Migration Strategy

### Phase 1: Copy Core Structure
1. **Models** - All data models (scan_result, watchlist_item, etc.)
2. **Services** - API service, auth service, storage service
3. **Providers** - State management (theme, alerts, scan history)
4. **Theme** - Complete theme system with colors and styles

### Phase 2: Copy All Screens
1. **Scanner Page** - Main scanning interface with filters
2. **Ideas Page** - Trade ideas display
3. **Copilot Page** - AI chat interface
4. **Watchlist Page** - Saved stocks with alerts
5. **Settings Page** - User preferences
6. **Symbol Detail Page** - Individual stock analysis
7. **Scan History Page** - Past scan results
8. **Auth Pages** - Login/signup
9. **Onboarding** - First-time user experience
10. **Splash Screen** - Loading screen

### Phase 3: Copy Widgets
1. **Scanner Widgets** - Filter panel, result cards, progress overlay
2. **Watchlist Widgets** - Alert dialogs, note dialogs, tag selector
3. **Symbol Detail Widgets** - Charts, trade plans, merit breakdown
4. **Common Widgets** - Empty states, loading skeletons, sparklines

### Phase 4: Integration
1. Update main.dart with proper routing
2. Connect to existing backend integration
3. Test all screens
4. Verify navigation flow

## Files to Copy

### From technic_app/lib/ to technic_mobile/lib/

**Core Files:**
- main.dart
- app_shell.dart
- user_profile.dart
- watchlist_store.dart

**Directories (complete):**
- models/
- providers/
- screens/
- services/
- theme/
- utils/
- widgets/

**Dependencies to Add:**
- Check technic_app/pubspec.yaml for all dependencies
- Add any missing packages to technic_mobile/pubspec.yaml

## Execution Steps

1. Backup current technic_mobile/lib
2. Copy all files from technic_app/lib to technic_mobile/lib
3. Merge pubspec.yaml dependencies
4. Update API endpoints to match backend
5. Test compilation
6. Test all screens
7. Fix any import issues

## Expected Result

Mobile app will have:
- ✅ Exact UI from screenshots
- ✅ All 5 main screens (Scanner, Ideas, Copilot, Watchlist, Settings)
- ✅ Complete navigation
- ✅ All widgets and components
- ✅ Theme system
- ✅ State management
- ✅ Backend integration (already done)

## Timeline

- Phase 1-2: Copy files (10 minutes)
- Phase 3: Update dependencies (5 minutes)
- Phase 4: Testing and fixes (15 minutes)
- **Total: ~30 minutes**
