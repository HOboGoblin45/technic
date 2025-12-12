# Scanner Page Extraction Strategy

## Challenge
ScannerPage is **1,623 lines** (line 484-2106) with 36 methods - the most complex component in Technic.

## Strategic Approach: Modular Extraction

Instead of extracting as one massive file, we'll create a **modular architecture**:

### Core Files Structure
```
lib/screens/scanner/
├── scanner_page.dart (main page, ~400 lines)
├── scanner_state.dart (state management logic, ~200 lines)
├── widgets/
│   ├── filter_panel.dart (~300 lines)
│   ├── scan_result_card.dart (~150 lines)
│   ├── market_pulse_card.dart (~150 lines)
│   ├── scoreboard_card.dart (~100 lines)
│   ├── quick_actions.dart (~100 lines)
│   ├── preset_manager.dart (~150 lines)
│   └── onboarding_card.dart (~100 lines)
```

### Key Components to Extract

**1. Scanner State (scanner_state.dart)**
- Filter state management
- Scan execution logic
- Preset management
- Progress tracking
- Persistence logic

**2. Main Page (scanner_page.dart)**
- Page structure
- FutureBuilder for scan results
- Refresh logic
- Navigation integration
- Layout composition

**3. Filter Panel (widgets/filter_panel.dart)**
- Sector/industry selection
- Trade style options
- Lookback days slider
- Min rating slider
- Advanced mode toggle
- Options preference

**4. Scan Result Card (widgets/scan_result_card.dart)**
- Individual result display
- Sparkline
- Entry/stop/target
- Tech rating
- Quick actions (Copilot, Save)

**5. Market Pulse Card (widgets/market_pulse_card.dart)**
- Market movers display
- Positive/negative indicators
- Compact layout

**6. Scoreboard Card (widgets/scoreboard_card.dart)**
- Performance metrics
- Win rates
- Strategy breakdown

**7. Quick Actions (widgets/quick_actions.dart)**
- Profile buttons (Conservative, Moderate, Aggressive)
- Randomize button
- Advanced toggle

**8. Preset Manager (widgets/preset_manager.dart)**
- Saved screens list
- Save/load/delete presets
- Preset cards

**9. Onboarding Card (widgets/onboarding_card.dart)**
- Welcome message
- Feature highlights
- Dismiss functionality

## Benefits of This Approach

1. **Maintainability**: Each component is self-contained and < 400 lines
2. **Reusability**: Widgets can be reused elsewhere
3. **Testability**: Each component can be unit tested
4. **Scalability**: Easy to add new features
5. **Team-Ready**: Multiple developers can work in parallel
6. **Performance**: Smaller widgets = better rebuild performance

## Extraction Order

1. ✅ Create widget files first (bottom-up)
2. ✅ Create state management
3. ✅ Create main page (composes widgets)
4. ✅ Test incrementally
5. ✅ Verify zero errors/warnings

## Expected Outcome

- **9 new files** created
- **~1,650 lines** extracted (includes some refactoring)
- **0 errors, 0 warnings**
- **Production-ready** modular architecture
- **Billion-dollar quality** code

Let's execute this plan! 🚀
