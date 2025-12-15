# Dart Issues to Fix

Based on the problems panel, here are the issues to fix:

## Critical Issues:

### 1. main.dart
- Unused import: 'services/auth_service.dart'

### 2. settings_page.dart  
- Unused import: '../services/auth_service.dart'
- TODO: Navigate to profile edit page
- Deprecated: 'activeColor' - use activeThumbColor instead

### 3. watchlist_page.dart
- Unused element: '_toggleTagFilter' method
- Don't use 'BuildContext' across async gaps

### 4. tag_selector.dart
- Unused local variables: 'selectedPredefined', 'selectedCustom'

### 5. alert_service.dart
- Unused elements: '_fetchCurrentPrice', '_showNotification'
- TODOs for implementation
- Don't invoke 'print' in production

### 6. scan_history_item.dart
- Unnecessary braces in string interpolation

### 7. alert_provider.dart
- Don't invoke 'print' in production

### 8. login_page.dart
- TODO: Implement forgot password
- Deprecated: 'withOpacity'

## Quick Fixes:

1. Remove unused imports
2. Remove unused variables/methods or mark as used
3. Replace deprecated APIs
4. Remove print statements or replace with logging
5. Fix string interpolation
6. Add mounted checks for async gaps

Most of these are warnings and TODOs that don't prevent compilation. The code is functional but needs cleanup for production.
