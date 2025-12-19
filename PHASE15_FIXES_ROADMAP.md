# Phase 15 Fixes Roadmap

## Overview
This roadmap prioritizes the 94 remaining issues identified during the Phase 15 code review, organized into sprints with clear deliverables.

---

## Sprint 1: Critical Security & Reliability (Priority: URGENT) ✅ COMPLETE

**Estimated Effort:** 1-2 days
**Impact:** Prevents app crashes and security vulnerabilities
**Status:** Completed

### 1.1 Add Authentication Token Handling
**Files to modify:**
- `lib/services/api_client.dart`
- `lib/services/api_service.dart`
- `lib/services/scanner_service.dart`

**Changes:**
```dart
// api_client.dart - Add to _buildHeaders()
Map<String, String> _buildHeaders({String? token}) {
  final headers = Map<String, String>.from(ApiConfig.defaultHeaders);
  if (token != null) {
    headers['Authorization'] = 'Bearer $token';
  }
  return headers;
}
```

### 1.2 Add Timeouts to All Network Requests
**Files to modify:**
- `lib/services/api_service.dart` (6 locations)
- `lib/services/auth_service.dart` (4 locations)
- `lib/services/scanner_service.dart` (all methods)

**Pattern:**
```dart
final response = await _client.get(uri)
    .timeout(const Duration(seconds: 30));
```

### 1.3 Add Error Handling to JSON Parsing
**Files to modify:**
- `lib/services/local_store.dart`
- `lib/services/storage_service.dart`

**Pattern:**
```dart
try {
  final decoded = jsonDecode(jsonString);
  // ... parse data
} catch (e) {
  debugPrint('JSON parse error: $e');
  return defaultValue;
}
```

---

## Sprint 2: Provider Reliability (Priority: HIGH) ✅ COMPLETE

**Estimated Effort:** 1 day
**Impact:** Prevents silent failures in state management
**Status:** Completed

### 2.1 Add Error Handling to Provider Async Methods
**Files to modify:**
- `lib/providers/app_providers.dart` (6 methods)
- `lib/providers/scan_history_provider.dart` (2 methods)
- `lib/providers/theme_provider.dart` (1 method)
- `lib/providers/alert_provider.dart` (1 method)

**Methods to update:**
| File | Method | Line |
|------|--------|------|
| app_providers.dart | `_loadThemeMode()` | 51 |
| app_providers.dart | `_loadOptionsMode()` | 83 |
| app_providers.dart | `_loadUserId()` | 253 |
| app_providers.dart | `_loadWatchlist()` | 327 |
| app_providers.dart | `_saveWatchlist()` | 420 |
| app_providers.dart | `_loadLastTab()` | 442 |
| scan_history_provider.dart | `_loadHistory()` | 24 |
| scan_history_provider.dart | `_saveHistory()` | 94 |
| theme_provider.dart | `_loadTheme()` | 29 |
| alert_provider.dart | `_saveAlerts()` | 116 |

**Pattern:**
```dart
Future<void> _loadData() async {
  try {
    final data = await _storage.loadData();
    state = data;
  } catch (e) {
    debugPrint('Failed to load data: $e');
    state = defaultValue;
  }
}
```

### 2.2 Centralize storageServiceProvider
**Action:** Keep only in `app_providers.dart`, remove from:
- `lib/providers/alert_provider.dart` (line 128)
- `lib/providers/scan_history_provider.dart` (line 100)
- `lib/providers/theme_provider.dart` (line 75)

### 2.3 Fix scanner_provider.dart
**File:** `lib/providers/scanner_provider.dart`

**Option A:** Add ChangeNotifierProvider wrapper
```dart
final scannerProvider = ChangeNotifierProvider<ScannerProvider>((ref) {
  return ScannerProvider(ref.read(scannerServiceProvider));
});
```

**Option B:** Convert to StateNotifier (recommended)

---

## Sprint 3: Code Cleanup (Priority: MEDIUM) ✅ COMPLETE

**Estimated Effort:** 0.5 days
**Impact:** Reduces technical debt and confusion
**Status:** Completed

### 3.1 Remove Duplicate AppColors File
**Action:** Delete `lib/theme/app_colors_fixed.dart`
**Then:** Search and replace any imports:
```bash
# Find files importing the old colors
grep -r "app_colors_fixed" lib/
```

### 3.2 Fix Hardcoded Gradient Colors
**File:** `lib/theme/app_colors.dart`

**Lines to fix:**
- 132-133: Use `darkCard` and `darkCardElevated` constants
- 140-142: Use named constants
- 163: Define `premiumPurple` constant

### 3.3 Remove Duplicate Code in mock_data.dart
**File:** `lib/utils/mock_data.dart`

**Duplicates to remove:**
- `defaultTickers` (use from constants.dart)
- `copilotPrompts` (use from constants.dart)
- Hardcoded colors lines 113, 127 (use AppColors)

---

## Sprint 4: Model Improvements (Priority: MEDIUM) ✅ COMPLETE

**Estimated Effort:** 1 day
**Impact:** Improves type safety and maintainability
**Status:** Completed

### 4.1 Add Null-Safe Type Casting in fromJson
**Files to modify:**
- `lib/models/symbol_detail.dart`
- `lib/models/price_alert.dart`
- `lib/models/scan_result.dart`

**Pattern:**
```dart
// Instead of:
symbol: json['symbol'] as String,

// Use:
symbol: json['symbol']?.toString() ?? '',
price: (json['price'] as num?)?.toDouble() ?? 0.0,
```

### 4.2 Add copyWith Methods
**Priority models:**
1. `lib/models/scan_result.dart`
2. `lib/models/symbol_detail.dart`
3. `lib/models/idea.dart`
4. `lib/models/market_mover.dart`

**Template:**
```dart
ScanResult copyWith({
  String? ticker,
  double? price,
  // ... other fields
}) {
  return ScanResult(
    ticker: ticker ?? this.ticker,
    price: price ?? this.price,
    // ... other fields
  );
}
```

### 4.3 Standardize JSON Key Naming
**File:** `lib/models/scan_result.dart`

**Decision needed:** Choose snake_case or camelCase for API consistency

---

## Sprint 5: Resource Management (Priority: LOW) ✅ COMPLETE

**Estimated Effort:** 0.5 days
**Impact:** Prevents memory leaks in long-running sessions
**Status:** Completed

### 5.1 Add Disposal for Singleton HTTP Clients
**Files:**
- `lib/services/api_client.dart`
- `lib/services/scanner_service.dart`

**Pattern:**
```dart
// Add to singleton classes
static void dispose() {
  _instance?._client.close();
  _instance = null;
}
```

### 5.2 Document Lifecycle Management
Add comments explaining when to call dispose for singletons.

---

## Sprint 6: Configuration Externalization (Priority: LOW) ✅ COMPLETE

**Estimated Effort:** 0.5 days
**Impact:** Easier environment management
**Status:** Completed

### 6.1 Move Hardcoded Values to Environment
**File:** `lib/services/api_config.dart`

**Implemented:**
- Added `String.fromEnvironment` for `API_BASE_URL` with localhost default
- Added `bool.fromEnvironment` for `PRODUCTION` flag
- Added documentation for dart-define usage

### 6.2 Externalize Timeout Values
**File:** `lib/services/api_config.dart`

**Implemented:**
- Imported constants.dart for centralized timeout values
- Changed `connectTimeout`, `receiveTimeout`, `sendTimeout` to reference `constants.apiTimeout`
- Changed `cacheExpiry` to reference `constants.cacheShortDuration`
- Removes duplication between api_config.dart and constants.dart

---

## Summary Timeline

| Sprint | Focus | Priority | Effort | Issues Fixed | Status |
|--------|-------|----------|--------|--------------|--------|
| 1 | Security & Reliability | URGENT | 1-2 days | 15 | ✅ Complete |
| 2 | Provider Reliability | HIGH | 1 day | 14 | ✅ Complete |
| 3 | Code Cleanup | MEDIUM | 0.5 days | 12 | ✅ Complete |
| 4 | Model Improvements | MEDIUM | 1 day | 18 | ✅ Complete |
| 5 | Resource Management | LOW | 0.5 days | 5 | ✅ Complete |
| 6 | Configuration | LOW | 0.5 days | 8 | ✅ Complete |
| **Total** | | | **4.5-5.5 days** | **72** | ✅ All Complete |

---

## Quick Reference: Files by Change Count

| File | Changes Needed |
|------|----------------|
| `lib/services/api_service.dart` | 8 |
| `lib/providers/app_providers.dart` | 7 |
| `lib/services/auth_service.dart` | 5 |
| `lib/services/storage_service.dart` | 4 |
| `lib/models/scan_result.dart` | 4 |
| `lib/services/api_client.dart` | 3 |
| `lib/providers/scan_history_provider.dart` | 3 |
| `lib/models/symbol_detail.dart` | 3 |

---

## Validation Checklist

After completing each sprint:

- [ ] All modified files compile without errors
- [ ] Existing functionality still works
- [ ] New error handling doesn't break happy path
- [ ] No new warnings introduced
- [ ] Changes committed with descriptive messages
