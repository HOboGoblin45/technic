# Week 3: User Features Development Plan
## Technic Flutter App - Authentication, Settings & Watchlist

**Status:** Ready to Start  
**Estimated Time:** 11-15 hours  
**Priority:** High (Required for production)

---

## 📊 CURRENT STATUS

### ✅ Completed (Weeks 1-2):
- **Week 1:** Scanner page with sort/filter, progress overlay ✅
- **Week 2:** Symbol detail page with charts, MERIT, trade plan ✅
- **Backend:** 98% complete, deployed on Render ✅
- **Frontend:** 60% complete (up from 30%)

### 🎯 Week 3 Goals:
Build essential user features to make the app production-ready:
1. **Authentication** - Login/signup with JWT
2. **Settings** - User preferences and configuration
3. **Watchlist** - Save and track favorite symbols

---

## 🚀 WEEK 3 PRIORITIES

### Priority 1: Authentication System ⭐⭐⭐
**Effort:** 4-6 hours  
**Impact:** Critical - Required for user accounts

**What to Build:**
- Login screen with email/password
- Signup screen with validation
- JWT token management
- Secure token storage
- Auto-login on app start
- Logout functionality

**Files to Create:**
```
lib/screens/auth/
  ├── login_screen.dart          (NEW)
  ├── signup_screen.dart         (NEW)
  └── widgets/
      ├── auth_text_field.dart   (NEW)
      └── auth_button.dart       (NEW)

lib/services/
  └── auth_service.dart          (NEW)

lib/providers/
  └── auth_provider.dart         (NEW)
```

**Dependencies:**
```yaml
flutter_secure_storage: ^9.0.0  # Secure token storage
```

---

### Priority 2: Settings & Preferences ⭐⭐⭐
**Effort:** 3-4 hours  
**Impact:** High - User customization

**What to Build:**
- Default scan parameters
- Notification preferences
- Theme selection (dark/light/system)
- Account management
- Clear cache option
- About/version info

**Files to Update/Create:**
```
lib/screens/settings/
  ├── settings_page.dart         (UPDATE)
  └── widgets/
      ├── settings_section.dart  (NEW)
      ├── settings_tile.dart     (NEW)
      └── theme_selector.dart    (NEW)

lib/services/
  └── settings_service.dart      (NEW)

lib/providers/
  └── settings_provider.dart     (NEW)
```

---

### Priority 3: Watchlist & Saved Scans ⭐⭐⭐
**Effort:** 4-5 hours  
**Impact:** High - Core feature

**What to Build:**
- Watchlist screen with symbol cards
- Add/remove symbols
- Saved scan configurations
- Quick access to favorites
- Sync with backend

**Files to Create:**
```
lib/screens/watchlist/
  ├── watchlist_screen.dart      (NEW)
  └── widgets/
      ├── watchlist_card.dart    (NEW)
      └── add_symbol_dialog.dart (NEW)

lib/services/
  └── watchlist_service.dart     (NEW)

lib/providers/
  └── watchlist_provider.dart    (NEW)

lib/models/
  └── watchlist_item.dart        (NEW)
```

---

## 📋 DETAILED IMPLEMENTATION PLAN

### Day 1-2: Authentication (4-6 hours)

#### Step 1: Auth Service (1.5 hours)
**File:** `lib/services/auth_service.dart`

**Features:**
```dart
class AuthService {
  // Login with email/password
  Future<AuthResponse> login(String email, String password);
  
  // Signup new user
  Future<AuthResponse> signup(String email, String password, String name);
  
  // Logout and clear tokens
  Future<void> logout();
  
  // Check if user is authenticated
  Future<bool> isAuthenticated();
  
  // Get current user
  Future<User?> getCurrentUser();
  
  // Refresh JWT token
  Future<String?> refreshToken();
}
```

**Backend Endpoints:**
- `POST /api/auth/login`
- `POST /api/auth/signup`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `POST /api/auth/refresh`

---

#### Step 2: Secure Storage (0.5 hours)
**Setup:** `flutter_secure_storage`

**Store:**
- JWT access token
- JWT refresh token
- User ID
- User email

---

#### Step 3: Login Screen (1.5 hours)
**File:** `lib/screens/auth/login_screen.dart`

**UI Components:**
- Email text field
- Password text field (obscured)
- Login button
- "Forgot password?" link
- "Sign up" link
- Loading indicator
- Error messages

**Validation:**
- Email format
- Password minimum length
- Empty field checks

---

#### Step 4: Signup Screen (1.5 hours)
**File:** `lib/screens/auth/signup_screen.dart`

**UI Components:**
- Name text field
- Email text field
- Password text field
- Confirm password field
- Terms & conditions checkbox
- Signup button
- "Already have account?" link

**Validation:**
- All fields required
- Email format
- Password strength (8+ chars, 1 number, 1 special)
- Passwords match

---

#### Step 5: Auth Provider (1 hour)
**File:** `lib/providers/auth_provider.dart`

**State Management:**
```dart
class AuthProvider extends StateNotifier<AuthState> {
  // Login
  Future<void> login(String email, String password);
  
  // Signup
  Future<void> signup(String email, String password, String name);
  
  // Logout
  Future<void> logout();
  
  // Auto-login on app start
  Future<void> checkAuthStatus();
}

class AuthState {
  final bool isAuthenticated;
  final User? user;
  final bool isLoading;
  final String? error;
}
```

---

### Day 3-4: Settings & Preferences (3-4 hours)

#### Step 1: Settings Service (1 hour)
**File:** `lib/services/settings_service.dart`

**Features:**
```dart
class SettingsService {
  // Theme
  Future<ThemeMode> getThemeMode();
  Future<void> setThemeMode(ThemeMode mode);
  
  // Scan defaults
  Future<ScanDefaults> getScanDefaults();
  Future<void> setScanDefaults(ScanDefaults defaults);
  
  // Notifications
  Future<NotificationSettings> getNotificationSettings();
  Future<void> setNotificationSettings(NotificationSettings settings);
  
  // Cache
  Future<void> clearCache();
  Future<int> getCacheSize();
}
```

---

#### Step 2: Settings Screen (2-3 hours)
**File:** `lib/screens/settings/settings_page.dart`

**Sections:**

**1. Account**
- User profile
- Email
- Change password
- Delete account

**2. Scan Defaults**
- Default universe
- Min MERIT score
- Max positions
- Risk profile

**3. Appearance**
- Theme selector (Dark/Light/System)
- Font size
- Color scheme

**4. Notifications**
- Scan complete
- New ideas
- Price alerts
- Email notifications

**5. Data & Storage**
- Cache size
- Clear cache button
- Download data

**6. About**
- App version
- Terms of service
- Privacy policy
- Contact support

---

### Day 5-6: Watchlist & Saved Scans (4-5 hours)

#### Step 1: Watchlist Service (1.5 hours)
**File:** `lib/services/watchlist_service.dart`

**Features:**
```dart
class WatchlistService {
  // Get watchlist
  Future<List<WatchlistItem>> getWatchlist();
  
  // Add symbol
  Future<void> addSymbol(String symbol);
  
  // Remove symbol
  Future<void> removeSymbol(String symbol);
  
  // Check if symbol is in watchlist
  Future<bool> isInWatchlist(String symbol);
  
  // Get saved scans
  Future<List<SavedScan>> getSavedScans();
  
  // Save scan configuration
  Future<void> saveScan(String name, ScanConfig config);
  
  // Delete saved scan
  Future<void> deleteSavedScan(String id);
}
```

**Backend Endpoints:**
- `GET /api/watchlist`
- `POST /api/watchlist`
- `DELETE /api/watchlist/{symbol}`
- `GET /api/scans/saved`
- `POST /api/scans/saved`
- `DELETE /api/scans/saved/{id}`

---

#### Step 2: Watchlist Screen (2-3 hours)
**File:** `lib/screens/watchlist/watchlist_screen.dart`

**UI Components:**
- Symbol cards with:
  - Symbol name
  - Current price
  - Price change %
  - MERIT score
  - Quick actions (view detail, remove)
- Add symbol button (FAB)
- Empty state message
- Pull to refresh
- Search/filter

**Features:**
- Real-time price updates
- Sort by: Name, Price, Change %, MERIT
- Swipe to delete
- Tap to view detail

---

#### Step 3: Add Symbol Dialog (0.5 hours)
**File:** `lib/screens/watchlist/widgets/add_symbol_dialog.dart`

**UI:**
- Symbol search field
- Autocomplete suggestions
- Add button
- Cancel button

---

#### Step 4: Saved Scans (1 hour)
**File:** `lib/screens/watchlist/saved_scans_screen.dart`

**UI Components:**
- Saved scan cards with:
  - Scan name
  - Configuration summary
  - Last run date
  - Quick run button
- Add new scan button
- Edit/delete options

---

### Day 7: Integration & Testing (2-3 hours)

#### Step 1: Navigation Integration (1 hour)
- Add auth check to app startup
- Redirect to login if not authenticated
- Add watchlist to bottom navigation
- Add settings to app bar menu

#### Step 2: Testing (1-2 hours)
- Test login flow
- Test signup flow
- Test logout
- Test settings persistence
- Test watchlist CRUD
- Test saved scans

#### Step 3: Bug Fixes & Polish (1 hour)
- Fix any issues found
- Improve error messages
- Add loading states
- Polish animations

---

## 🎨 UI/UX GUIDELINES

### Authentication Screens:
- Clean, minimal design
- Large, clear input fields
- Prominent CTA buttons
- Helpful error messages
- Loading indicators
- Password visibility toggle

### Settings Screen:
- Grouped sections
- Clear labels
- Toggle switches for booleans
- Dropdowns for selections
- Confirmation dialogs for destructive actions

### Watchlist Screen:
- Card-based layout
- Color-coded price changes (green/red)
- Quick actions on cards
- Smooth animations
- Empty state with helpful message

---

## 📦 DEPENDENCIES TO ADD

```yaml
# pubspec.yaml

dependencies:
  # Secure Storage
  flutter_secure_storage: ^9.0.0
  
  # Already have (verify):
  flutter_riverpod: ^2.4.0
  http: ^1.1.0
  shared_preferences: ^2.2.0
```

---

## 🔐 SECURITY CONSIDERATIONS

### Token Management:
- Store tokens in secure storage (not SharedPreferences)
- Implement token refresh logic
- Clear tokens on logout
- Handle expired tokens gracefully

### Password Security:
- Never store passwords locally
- Use HTTPS for all auth requests
- Implement password strength requirements
- Add "forgot password" flow

### API Security:
- Include JWT in Authorization header
- Handle 401 Unauthorized responses
- Implement retry logic for failed requests

---

## 🧪 TESTING CHECKLIST

### Authentication:
- [ ] Login with valid credentials
- [ ] Login with invalid credentials
- [ ] Signup with valid data
- [ ] Signup with invalid data
- [ ] Auto-login on app restart
- [ ] Logout clears tokens
- [ ] Token refresh works
- [ ] Expired token handling

### Settings:
- [ ] Theme changes persist
- [ ] Scan defaults save correctly
- [ ] Notification settings work
- [ ] Cache clear works
- [ ] Settings sync across app

### Watchlist:
- [ ] Add symbol to watchlist
- [ ] Remove symbol from watchlist
- [ ] Watchlist persists
- [ ] Real-time updates work
- [ ] Sort/filter works
- [ ] Empty state displays

---

## 📊 SUCCESS METRICS

### Completion Criteria:
- [ ] Users can login/signup
- [ ] Tokens stored securely
- [ ] Auto-login works
- [ ] Settings persist correctly
- [ ] Watchlist CRUD works
- [ ] Saved scans work
- [ ] All screens responsive
- [ ] No compilation errors
- [ ] Smooth animations

### Performance Targets:
- [ ] Login response: <2s
- [ ] Settings load: <500ms
- [ ] Watchlist load: <1s
- [ ] Smooth 60fps animations

---

## 🎯 WEEK 3 DELIVERABLES

### By End of Week 3:
1. ✅ Complete authentication system
2. ✅ Functional settings page
3. ✅ Working watchlist
4. ✅ Saved scans feature
5. ✅ All features tested
6. ✅ Production-ready code

### Frontend Completion:
- **Current:** 60%
- **After Week 3:** 85%
- **Remaining:** Week 4 polish (15%)

---

## 🚀 NEXT STEPS

### Ready to Start?

**Day 1 (Today):**
1. Add `flutter_secure_storage` dependency
2. Create auth service
3. Build login screen
4. **Estimated: 4 hours**

**Day 2:**
1. Build signup screen
2. Implement auth provider
3. Test auth flow
4. **Estimated: 3 hours**

**Day 3:**
1. Create settings service
2. Build settings screen (Part 1)
3. **Estimated: 3 hours**

**Day 4:**
1. Complete settings screen
2. Test settings persistence
3. **Estimated: 2 hours**

**Day 5:**
1. Create watchlist service
2. Build watchlist screen
3. **Estimated: 3 hours**

**Day 6:**
1. Add symbol dialog
2. Saved scans screen
3. **Estimated: 2 hours**

**Day 7:**
1. Integration & testing
2. Bug fixes & polish
3. **Estimated: 2-3 hours**

---

## 💡 TIPS FOR SUCCESS

### Development Tips:
1. **Start with services** - Build backend integration first
2. **Test as you go** - Don't wait until the end
3. **Use providers** - Leverage Riverpod for state management
4. **Reuse widgets** - Create common components
5. **Handle errors** - Show helpful messages to users

### Code Quality:
- Follow Flutter best practices
- Use const constructors where possible
- Add comments for complex logic
- Keep widgets small and focused
- Use meaningful variable names

### User Experience:
- Show loading indicators
- Provide feedback for actions
- Handle errors gracefully
- Add smooth animations
- Test on different screen sizes

---

## 🎊 LET'S GET STARTED!

Week 3 is all about making Technic a complete, production-ready app with user accounts, personalization, and saved data.

**Ready to build the authentication system?** Let's start with Day 1! 🚀
