# Testing & Quality Assurance Checklist

## Overview

This document provides comprehensive testing checklists for the Technic iOS app before App Store submission.

---

## Unit Tests

### Model Tests
- [x] `ScanResult` - JSON parsing, serialization, tier calculation
- [x] `WatchlistItem` - JSON parsing, copyWith, computed properties
- [x] `CopilotMessage` - Message role handling, metadata
- [x] `OptionStrategy` - Strategy parsing, risk classification

### Service Tests
- [x] `StorageService` - User data, preferences, watchlist persistence
- [ ] `ApiService` - API calls, error handling, retries
- [ ] `AlertService` - Alert creation, triggering, storage
- [ ] `NotificationService` - Local/push notification handling

### Widget Tests
- [x] `PremiumButton` - Variants, sizes, loading states
- [x] `PremiumCard` - Variants, elevation, interaction
- [x] `StockResultCard` - Data display, formatting
- [x] `MetricCard` - Label, value, icon display

---

## Device Testing Checklist

### Physical Devices to Test

| Device | Screen Size | iOS Version | Priority |
|--------|-------------|-------------|----------|
| iPhone SE (3rd gen) | 4.7" | iOS 16+ | High |
| iPhone 13/14 | 6.1" | iOS 16+ | High |
| iPhone 15 Pro | 6.1" | iOS 17+ | Critical |
| iPhone 15 Pro Max | 6.7" | iOS 17+ | Critical |
| iPad Pro 11" | 11" | iPadOS 16+ | Medium |
| iPad Pro 12.9" | 12.9" | iPadOS 16+ | Low |

### Screen Size Compatibility

- [ ] **iPhone SE (4.7")** - Smallest supported screen
  - [ ] All UI elements visible without scrolling issues
  - [ ] Text readable at default size
  - [ ] Buttons easily tappable (44pt minimum)
  - [ ] No content clipping

- [ ] **iPhone Standard (6.1")** - Most common size
  - [ ] Optimal layout and spacing
  - [ ] Safe area insets respected
  - [ ] Dynamic Island compatibility (iPhone 14 Pro+)

- [ ] **iPhone Max (6.7")** - Largest phone screen
  - [ ] Content fills screen appropriately
  - [ ] No excessive whitespace
  - [ ] One-handed use considerations

- [ ] **iPad (if supported)**
  - [ ] Responsive layout adaptation
  - [ ] Split view / Slide Over support
  - [ ] Pointer support for trackpad/mouse

---

## Functional Testing Scenarios

### 1. Fresh Install Flow
- [ ] App launches without crash
- [ ] Onboarding displays correctly
- [ ] Can skip or complete onboarding
- [ ] Proper state after onboarding completion
- [ ] Permission requests appear at right time

### 2. Scanner Feature
- [ ] Scanner page loads without error
- [ ] Filter panel opens/closes smoothly
- [ ] All filter options work correctly
- [ ] Scan executes and returns results
- [ ] Results display correctly
- [ ] Can tap on result to see details
- [ ] Empty state displays when no results
- [ ] Error state displays on API failure
- [ ] Loading indicator shows during scan

### 3. Symbol Detail Page
- [ ] Page loads with stock data
- [ ] Price and change display correctly
- [ ] Chart renders and is interactive
- [ ] Technical indicators display
- [ ] Can add to watchlist
- [ ] Can create price alert
- [ ] AI Copilot integration works

### 4. Watchlist Feature
- [ ] Watchlist loads saved items
- [ ] Can add new symbol
- [ ] Can remove symbol
- [ ] Can add/edit notes
- [ ] Can add/remove tags
- [ ] Items persist after app restart
- [ ] Empty state displays when empty

### 5. Price Alerts
- [ ] Can create new alert
- [ ] Alert conditions save correctly
- [ ] Alert list displays all alerts
- [ ] Can edit existing alert
- [ ] Can delete alert
- [ ] Alerts persist after restart
- [ ] Push notifications received when triggered

### 6. AI Copilot
- [ ] Chat interface loads
- [ ] Can type and send message
- [ ] Response displays correctly
- [ ] Conversation history preserved
- [ ] Can start new conversation
- [ ] Error handling for API failures
- [ ] Loading indicator during response

### 7. Settings
- [ ] Settings page loads
- [ ] Theme toggle works
- [ ] Notification preferences save
- [ ] Account information displays
- [ ] Can sign out
- [ ] Privacy policy link works
- [ ] Terms of service link works

---

## Network Condition Testing

### Offline Behavior
- [ ] App handles no network gracefully
- [ ] Cached data displays when offline
- [ ] Appropriate offline message shown
- [ ] Retry mechanism works when back online
- [ ] No crashes when network unavailable

### Slow Network (3G)
- [ ] Loading indicators display
- [ ] Timeouts handled gracefully
- [ ] User can cancel slow operations
- [ ] App remains responsive during loads

### Network Transition
- [ ] WiFi to Cellular transition smooth
- [ ] Cellular to WiFi transition smooth
- [ ] No data loss during transition
- [ ] Pending operations resume correctly

---

## Performance Testing

### Launch Time
- [ ] **Cold start:** < 2 seconds to interactive
- [ ] **Warm start:** < 1 second to interactive
- [ ] No visible jank during launch
- [ ] Splash screen displays correctly

### Memory Usage
- [ ] **Idle memory:** < 100MB
- [ ] **Active usage:** < 200MB
- [ ] **Peak memory:** < 300MB
- [ ] No memory leaks during navigation
- [ ] Memory released when views dismissed

### Frame Rate
- [ ] **Scroll performance:** 60fps
- [ ] **Animation smoothness:** 60fps
- [ ] **Chart interaction:** 60fps
- [ ] No dropped frames during normal use

### Battery Impact
- [ ] Background activity minimal
- [ ] No excessive CPU when idle
- [ ] Location services (if used) efficient
- [ ] Push notification handling efficient

---

## Background/Foreground Testing

### App State Transitions
- [ ] Home button - app suspends correctly
- [ ] App switcher - app preserves state
- [ ] Return from background - state restored
- [ ] Multitasking - no data loss
- [ ] Force quit - graceful shutdown

### Background Operations
- [ ] Background fetch works
- [ ] Push notifications received
- [ ] Alert checking continues
- [ ] Data syncs in background
- [ ] Proper battery management

### Interruptions
- [ ] Incoming call - app pauses
- [ ] Notification banner - no interference
- [ ] Control Center - app pauses
- [ ] Siri activation - app pauses
- [ ] Return after interruption - state preserved

---

## Push Notification Testing

### Local Notifications
- [ ] Notification permission requested
- [ ] Permission denial handled
- [ ] Alert notifications display
- [ ] Tap notification opens relevant screen
- [ ] Badge count updates correctly
- [ ] Sound plays (if enabled)

### Push Notifications (FCM)
- [ ] Token registration succeeds
- [ ] Push received in foreground
- [ ] Push received in background
- [ ] Push received when terminated
- [ ] Notification tap opens app correctly
- [ ] Topic subscriptions work

---

## Accessibility Testing

### VoiceOver
- [ ] All buttons have labels
- [ ] All images have descriptions
- [ ] Custom widgets accessible
- [ ] Navigation order logical
- [ ] Charts accessible

### Dynamic Type
- [ ] Respects system font size
- [ ] Layout adapts to larger text
- [ ] No text truncation issues
- [ ] Minimum readable at smallest size

### Color/Contrast
- [ ] Sufficient color contrast (4.5:1)
- [ ] Color not sole indicator
- [ ] Dark mode works correctly
- [ ] Reduce motion respected

---

## Security Testing

### Data Protection
- [ ] Sensitive data not logged
- [ ] No hardcoded credentials
- [ ] API keys not in source code
- [ ] Keychain used for secrets
- [ ] Data encrypted at rest

### Network Security
- [ ] All connections use HTTPS
- [ ] Certificate pinning (if applicable)
- [ ] No mixed content
- [ ] API responses validated

### Input Validation
- [ ] SQL injection prevented
- [ ] XSS prevented
- [ ] Input length limits enforced
- [ ] Special characters handled

---

## Pre-Submission Checklist

### App Store Requirements
- [ ] No placeholder content
- [ ] No test data visible
- [ ] No debug logging enabled
- [ ] Privacy manifest complete
- [ ] All required permissions explained
- [ ] Age rating appropriate
- [ ] No private API usage

### Code Quality
- [ ] No compiler warnings
- [ ] All TODOs resolved
- [ ] No force unwraps in production code
- [ ] Error handling complete
- [ ] Memory leaks fixed

### Assets
- [ ] App icons all sizes present
- [ ] Launch screen correct
- [ ] All images optimized
- [ ] No missing assets

### Build Configuration
- [ ] Release build tested
- [ ] Version number incremented
- [ ] Build number updated
- [ ] Correct bundle identifier
- [ ] Correct team/signing

---

## Running Tests

### Unit & Widget Tests
```bash
# Run all tests
flutter test

# Run specific test file
flutter test test/models/scan_result_test.dart

# Run with coverage
flutter test --coverage

# Generate coverage report
genhtml coverage/lcov.info -o coverage/html
```

### Integration Tests
```bash
# Run integration tests
flutter drive --target=test_driver/app.dart

# Run on specific device
flutter drive --target=test_driver/app.dart -d <device_id>
```

### Performance Profiling
```bash
# Run in profile mode
flutter run --profile

# Capture performance trace
flutter run --profile --trace-startup
```

---

## Test Reporting Template

### Test Session Information
- **Date:**
- **Tester:**
- **Device:**
- **iOS Version:**
- **App Version:**
- **Build Number:**

### Summary
- **Total Tests:**
- **Passed:**
- **Failed:**
- **Blocked:**

### Issues Found
| ID | Severity | Description | Steps to Reproduce | Status |
|----|----------|-------------|-------------------|--------|
| 1 | | | | |
| 2 | | | | |

### Notes
-

---

## Automated Test Commands

```bash
# Full test suite
flutter test

# Model tests only
flutter test test/models/

# Service tests only
flutter test test/services/

# Widget tests only
flutter test test/widgets/

# Run tests with verbose output
flutter test --reporter expanded

# Run tests in parallel
flutter test --concurrency=4
```
