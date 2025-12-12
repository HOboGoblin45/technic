# technic UI Refactoring Plan - Week 1-2

## Current State
- **File**: `technic_app/lib/main.dart`
- **Size**: 5,682 lines (CRITICAL - needs immediate refactoring)
- **Issues**:
  - All code in single file (unmaintainable)
  - No separation of concerns
  - Difficult to test
  - Hard to add features
  - Old brand colors (#99BFFF instead of #B0CAFF)

## Target State
- **Modular architecture** with clear separation
- **<500 lines per file** maximum
- **Testable components**
- **Updated brand colors** (#B0CAFF, #001D51, #213631, White)
- **Riverpod state management**

---

## Step-by-Step Refactoring Plan

### Phase 1: Create Directory Structure (Day 1)
```
technic_app/lib/
├── main.dart (entry point only, ~50 lines)
├── app.dart (MaterialApp config, ~100 lines)
├── theme/
│   ├── app_colors.dart (brand colors)
│   ├── app_theme.dart (theme configuration)
│   └── text_styles.dart (typography)
├── models/
│   ├── scan_result.dart
│   ├── market_mover.dart
│   ├── idea.dart
│   ├── option_strategy.dart
│   ├── copilot_message.dart
│   ├── scanner_bundle.dart
│   ├── quick_action.dart
│   ├── saved_screen.dart
│   └── scoreboard_slice.dart
├── services/
│   ├── api_service.dart (TechnicApi refactored)
│   ├── storage_service.dart (LocalStore refactored)
│   └── api_config.dart
├── providers/
│   ├── theme_provider.dart
│   ├── scanner_provider.dart
│   ├── copilot_provider.dart
│   └── watchlist_provider.dart
├── screens/
│   ├── scanner/
│   │   ├── scanner_page.dart
│   │   └── widgets/
│   │       ├── filter_panel.dart
│   │       ├── scan_result_card.dart
│   │       ├── market_pulse_card.dart
│   │       └── quick_actions_row.dart
│   ├── ideas/
│   │   ├── ideas_page.dart
│   │   └── widgets/
│   │       └── idea_card.dart
│   ├── copilot/
│   │   ├── copilot_page.dart
│   │   └── widgets/
│   │       └── message_bubble.dart
│   ├── my_ideas/
│   │   └── my_ideas_page.dart
│   └── settings/
│       └── settings_page.dart
├── widgets/
│   ├── sparkline.dart
│   ├── info_card.dart
│   ├── section_header.dart
│   ├── pulse_badge.dart
│   └── hero_banner.dart
└── utils/
    ├── formatters.dart
    └── constants.dart
```

### Phase 2: Extract Models (Day 1-2)
**Priority**: HIGH (foundation for everything else)

**Files to Create**:
1. `lib/models/scan_result.dart` - Extract ScanResult class
2. `lib/models/market_mover.dart` - Extract MarketMover class
3. `lib/models/idea.dart` - Extract Idea class
4. `lib/models/option_strategy.dart` - Extract OptionStrategy class
5. `lib/models/copilot_message.dart` - Extract CopilotMessage class
6. `lib/models/scanner_bundle.dart` - Extract ScannerBundle class
7. `lib/models/quick_action.dart` - Extract QuickAction class
8. `lib/models/saved_screen.dart` - Extract SavedScreen class
9. `lib/models/scoreboard_slice.dart` - Extract ScoreboardSlice class
10. `lib/models/universe_stats.dart` - Extract UniverseStats class

**Each model file should**:
- Include the class definition
- Include fromJson factory
- Include toJson method
- Include any helper methods
- Be ~50-100 lines each

### Phase 3: Create Theme System (Day 2)
**Priority**: HIGH (needed for all UI components)

**Files to Create**:
1. `lib/theme/app_colors.dart`
```dart
// Updated brand colors
class AppColors {
  // Primary (White-dominant)
  static const white = Color(0xFFFFFFFF);
  static const offWhite = Color(0xFFFAFAFA);
  static const lightGray = Color(0xFFF5F5F5);
  
  // Brand Colors (Updated)
  static const skyBlue = Color(0xFFB0CAFF);  // Was #99BFFF
  static const imperialBlue = Color(0xFF001D51);
  static const pineGrove = Color(0xFF213631);
  
  // Dark Mode
  static const darkBg = Color(0xFF0A0A0A);
  static const darkCard = Color(0xFF1A1A1A);
  static const darkElevated = Color(0xFF2A2A2A);
  
  // Semantic Colors
  static const success = Color(0xFF213631);  // Pine Grove
  static const warning = Color(0xFFFFB84D);
  static const error = Color(0xFFFF6B6B);
  static const info = Color(0xFFB0CAFF);  // Sky Blue
}
```

2. `lib/theme/app_theme.dart`
```dart
class AppTheme {
  static ThemeData lightTheme() {
    return ThemeData(
      brightness: Brightness.light,
      primaryColor: AppColors.skyBlue,
      scaffoldBackgroundColor: AppColors.white,
      // ... complete theme
    );
  }
  
  static ThemeData darkTheme() {
    return ThemeData(
      brightness: Brightness.dark,
      primaryColor: AppColors.skyBlue,
      scaffoldBackgroundColor: AppColors.darkBg,
      // ... complete theme
    );
  }
}
```

3. `lib/theme/text_styles.dart`
```dart
class AppTextStyles {
  // iOS-inspired type scale
  static const largeTitle = TextStyle(fontSize: 34, fontWeight: FontWeight.bold);
  static const title1 = TextStyle(fontSize: 28, fontWeight: FontWeight.bold);
  // ... complete scale
}
```

### Phase 4: Extract Services (Day 3)
**Priority**: HIGH (API and storage logic)

**Files to Create**:
1. `lib/services/api_config.dart` - Extract ApiConfig class
2. `lib/services/api_service.dart` - Extract TechnicApi class
3. `lib/services/storage_service.dart` - Extract LocalStore class

### Phase 5: Implement Riverpod (Day 3-4)
**Priority**: HIGH (state management foundation)

**Steps**:
1. Add Riverpod to `pubspec.yaml`
```yaml
dependencies:
  flutter_riverpod: ^2.4.0
```

2. Create providers:
```dart
// lib/providers/theme_provider.dart
final themeProvider = StateNotifierProvider<ThemeNotifier, bool>((ref) {
  return ThemeNotifier();
});

class ThemeNotifier extends StateNotifier<bool> {
  ThemeNotifier() : super(false);
  
  void toggle() => state = !state;
  void setDark(bool isDark) => state = isDark;
}
```

3. Wrap app with ProviderScope
4. Replace all ValueNotifiers with Riverpod providers

### Phase 6: Extract Pages (Day 4-5)
**Priority**: MEDIUM (can be done incrementally)

**Order of Extraction**:
1. Settings Page (simplest, ~200 lines)
2. My Ideas Page (simple, ~150 lines)
3. Copilot Page (~250 lines)
4. Ideas Page (~200 lines)
5. Scanner Page (most complex, ~1500 lines)

### Phase 7: Extract Widgets (Day 5-6)
**Priority**: MEDIUM (reusable components)

**Widgets to Extract**:
1. Sparkline
2. InfoCard
3. SectionHeader
4. PulseBadge
5. HeroBanner
6. ProfileRow
7. MessageBubble

### Phase 8: Update Brand Colors (Day 6)
**Priority**: HIGH (visual refresh)

**Changes**:
- `#99BFFF` → `#B0CAFF` (Sky Blue)
- Keep `#001D51` (Imperial Blue)
- Keep `#213631` (Pine Grove)
- Update all color references throughout

### Phase 9: Testing & Validation (Day 7)
**Priority**: CRITICAL

**Tests to Write**:
- Model serialization tests
- Service tests (with mocks)
- Provider tests
- Widget tests
- Integration tests

### Phase 10: Final Cleanup (Day 7)
**Priority**: MEDIUM

**Tasks**:
- Remove old main.dart
- Update imports
- Run flutter analyze
- Fix any warnings
- Update documentation

---

## Execution Order (Optimized)

### Day 1: Foundation
- ✅ Create directory structure
- ✅ Extract all model classes
- ✅ Create theme system with updated colors
- ✅ Test models

### Day 2: Services & State
- ✅ Extract API service
- ✅ Extract storage service
- ✅ Add Riverpod dependency
- ✅ Create initial providers
- ✅ Test services

### Day 3-4: State Management Migration
- ✅ Create all Riverpod providers
- ✅ Replace ValueNotifiers
- ✅ Update app entry point
- ✅ Test state management

### Day 5-6: Page Extraction
- ✅ Extract Settings Page
- ✅ Extract My Ideas Page
- ✅ Extract Copilot Page
- ✅ Extract Ideas Page
- ✅ Extract Scanner Page (most complex)
- ✅ Extract reusable widgets

### Day 7: Polish & Test
- ✅ Update all brand colors
- ✅ Write comprehensive tests
- ✅ Run flutter analyze
- ✅ Fix warnings
- ✅ Update documentation

---

## Success Criteria

### Code Quality
- ✅ No file exceeds 500 lines
- ✅ All models have tests
- ✅ All services have tests
- ✅ All providers have tests
- ✅ >80% code coverage

### Functionality
- ✅ All existing features work
- ✅ No regressions
- ✅ State persists correctly
- ✅ Navigation works
- ✅ API calls successful

### Performance
- ✅ Build time < 30 seconds
- ✅ Hot reload < 2 seconds
- ✅ App startup < 3 seconds
- ✅ No memory leaks

### Visual
- ✅ Updated brand colors applied
- ✅ Consistent styling
- ✅ Smooth animations
- ✅ No visual glitches

---

## Risk Mitigation

### Backup Strategy
1. Create git branch: `feature/ui-refactoring`
2. Commit after each major step
3. Keep old main.dart until fully tested
4. Have rollback plan

### Testing Strategy
1. Test each extracted component individually
2. Integration test after each phase
3. Full regression test at end
4. User acceptance testing

### Communication
1. Document all changes
2. Update README
3. Create migration guide
4. Note any breaking changes

---

## Next Steps

**Immediate Actions**:
1. Create git branch
2. Create directory structure
3. Start extracting models
4. Create theme system with updated colors

**This Week**:
- Complete model extraction
- Create service layer
- Implement Riverpod
- Extract 2-3 pages

**Next Week**:
- Extract remaining pages
- Extract widgets
- Update brand colors
- Write tests
- Final polish

---

## Notes

- **Preserve all functionality** - No features should be lost
- **Test continuously** - Don't wait until the end
- **Update colors incrementally** - As each component is extracted
- **Document as you go** - Future you will thank you
- **Ask for help** - If stuck, consult documentation or ask questions

**This refactoring is the foundation for everything else. Take time to do it right.**
