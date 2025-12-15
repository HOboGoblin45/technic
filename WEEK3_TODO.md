# Week 3 TODO - User Features
## Technic Flutter App Development

**Target:** Complete authentication, settings, and watchlist features  
**Timeline:** 7 days (11-15 hours total)  
**Current Status:** Ready to start

---

## 📋 MASTER CHECKLIST

### Day 1-2: Authentication System (4-6 hours)
- [ ] **Setup Dependencies**
  - [ ] Add `flutter_secure_storage: ^9.0.0` to pubspec.yaml
  - [ ] Run `flutter pub get`
  - [ ] Test secure storage on device

- [ ] **Auth Service** (1.5 hours)
  - [ ] Create `lib/services/auth_service.dart`
  - [ ] Implement `login()` method
  - [ ] Implement `signup()` method
  - [ ] Implement `logout()` method
  - [ ] Implement `isAuthenticated()` method
  - [ ] Implement `getCurrentUser()` method
  - [ ] Implement `refreshToken()` method
  - [ ] Add error handling
  - [ ] Test with backend API

- [ ] **Secure Storage Setup** (0.5 hours)
  - [ ] Create storage wrapper class
  - [ ] Implement token storage methods
  - [ ] Implement token retrieval methods
  - [ ] Implement token deletion methods
  - [ ] Test storage persistence

- [ ] **Login Screen** (1.5 hours)
  - [ ] Create `lib/screens/auth/login_screen.dart`
  - [ ] Build email text field
  - [ ] Build password text field (with visibility toggle)
  - [ ] Build login button
  - [ ] Add "Forgot password?" link
  - [ ] Add "Sign up" link
  - [ ] Add loading indicator
  - [ ] Add error message display
  - [ ] Implement email validation
  - [ ] Implement password validation
  - [ ] Test UI on different screen sizes

- [ ] **Signup Screen** (1.5 hours)
  - [ ] Create `lib/screens/auth/signup_screen.dart`
  - [ ] Build name text field
  - [ ] Build email text field
  - [ ] Build password text field
  - [ ] Build confirm password field
  - [ ] Add terms & conditions checkbox
  - [ ] Build signup button
  - [ ] Add "Already have account?" link
  - [ ] Implement all field validations
  - [ ] Implement password strength check
  - [ ] Implement password match check
  - [ ] Test UI on different screen sizes

- [ ] **Auth Provider** (1 hour)
  - [ ] Create `lib/providers/auth_provider.dart`
  - [ ] Define `AuthState` class
  - [ ] Implement `login()` method
  - [ ] Implement `signup()` method
  - [ ] Implement `logout()` method
  - [ ] Implement `checkAuthStatus()` for auto-login
  - [ ] Add loading states
  - [ ] Add error states
  - [ ] Test state management

- [ ] **Integration & Testing**
  - [ ] Connect login screen to auth provider
  - [ ] Connect signup screen to auth provider
  - [ ] Test login flow end-to-end
  - [ ] Test signup flow end-to-end
  - [ ] Test auto-login on app restart
  - [ ] Test logout functionality
  - [ ] Test error handling
  - [ ] Fix any bugs found

---

### Day 3-4: Settings & Preferences (3-4 hours)

- [ ] **Settings Service** (1 hour)
  - [ ] Create `lib/services/settings_service.dart`
  - [ ] Implement theme mode methods
  - [ ] Implement scan defaults methods
  - [ ] Implement notification settings methods
  - [ ] Implement cache management methods
  - [ ] Add SharedPreferences integration
  - [ ] Test settings persistence

- [ ] **Settings Models** (0.5 hours)
  - [ ] Create `lib/models/scan_defaults.dart`
  - [ ] Create `lib/models/notification_settings.dart`
  - [ ] Add JSON serialization
  - [ ] Test model conversion

- [ ] **Settings Provider** (0.5 hours)
  - [ ] Create `lib/providers/settings_provider.dart`
  - [ ] Define `SettingsState` class
  - [ ] Implement settings update methods
  - [ ] Add loading states
  - [ ] Test state management

- [ ] **Settings Screen - Account Section** (0.5 hours)
  - [ ] Update `lib/screens/settings/settings_page.dart`
  - [ ] Add user profile display
  - [ ] Add email display
  - [ ] Add change password button
  - [ ] Add delete account button
  - [ ] Style section

- [ ] **Settings Screen - Scan Defaults** (0.5 hours)
  - [ ] Add default universe selector
  - [ ] Add min MERIT score slider
  - [ ] Add max positions input
  - [ ] Add risk profile selector
  - [ ] Style section

- [ ] **Settings Screen - Appearance** (0.5 hours)
  - [ ] Add theme selector (Dark/Light/System)
  - [ ] Add font size selector
  - [ ] Add color scheme preview
  - [ ] Style section
  - [ ] Test theme switching

- [ ] **Settings Screen - Notifications** (0.5 hours)
  - [ ] Add scan complete toggle
  - [ ] Add new ideas toggle
  - [ ] Add price alerts toggle
  - [ ] Add email notifications toggle
  - [ ] Style section

- [ ] **Settings Screen - Data & Storage** (0.5 hours)
  - [ ] Add cache size display
  - [ ] Add clear cache button
  - [ ] Add download data button
  - [ ] Add confirmation dialogs
  - [ ] Style section

- [ ] **Settings Screen - About** (0.5 hours)
  - [ ] Add app version display
  - [ ] Add terms of service link
  - [ ] Add privacy policy link
  - [ ] Add contact support button
  - [ ] Style section

- [ ] **Integration & Testing**
  - [ ] Test all settings save correctly
  - [ ] Test settings persist across app restarts
  - [ ] Test theme changes apply immediately
  - [ ] Test cache clear works
  - [ ] Test all UI interactions
  - [ ] Fix any bugs found

---

### Day 5-6: Watchlist & Saved Scans (4-5 hours)

- [ ] **Watchlist Models** (0.5 hours)
  - [ ] Create `lib/models/watchlist_item.dart`
  - [ ] Create `lib/models/saved_scan.dart`
  - [ ] Add JSON serialization
  - [ ] Test model conversion

- [ ] **Watchlist Service** (1.5 hours)
  - [ ] Create `lib/services/watchlist_service.dart`
  - [ ] Implement `getWatchlist()` method
  - [ ] Implement `addSymbol()` method
  - [ ] Implement `removeSymbol()` method
  - [ ] Implement `isInWatchlist()` method
  - [ ] Implement `getSavedScans()` method
  - [ ] Implement `saveScan()` method
  - [ ] Implement `deleteSavedScan()` method
  - [ ] Add error handling
  - [ ] Test with backend API

- [ ] **Watchlist Provider** (0.5 hours)
  - [ ] Create `lib/providers/watchlist_provider.dart`
  - [ ] Define `WatchlistState` class
  - [ ] Implement CRUD methods
  - [ ] Add loading states
  - [ ] Test state management

- [ ] **Watchlist Screen** (2 hours)
  - [ ] Create `lib/screens/watchlist/watchlist_screen.dart`
  - [ ] Build symbol card widget
  - [ ] Add symbol name display
  - [ ] Add current price display
  - [ ] Add price change % display
  - [ ] Add MERIT score display
  - [ ] Add quick action buttons
  - [ ] Implement pull to refresh
  - [ ] Add empty state message
  - [ ] Add search/filter functionality
  - [ ] Implement sort options
  - [ ] Add swipe to delete
  - [ ] Style all components

- [ ] **Add Symbol Dialog** (0.5 hours)
  - [ ] Create `lib/screens/watchlist/widgets/add_symbol_dialog.dart`
  - [ ] Build symbol search field
  - [ ] Add autocomplete suggestions
  - [ ] Add add button
  - [ ] Add cancel button
  - [ ] Style dialog

- [ ] **Saved Scans Screen** (1 hour)
  - [ ] Create `lib/screens/watchlist/saved_scans_screen.dart`
  - [ ] Build saved scan card widget
  - [ ] Add scan name display
  - [ ] Add configuration summary
  - [ ] Add last run date
  - [ ] Add quick run button
  - [ ] Add edit/delete options
  - [ ] Add empty state message
  - [ ] Style all components

- [ ] **Integration & Testing**
  - [ ] Test add symbol to watchlist
  - [ ] Test remove symbol from watchlist
  - [ ] Test watchlist persistence
  - [ ] Test real-time price updates
  - [ ] Test sort/filter functionality
  - [ ] Test saved scans CRUD
  - [ ] Test all UI interactions
  - [ ] Fix any bugs found

---

### Day 7: Integration & Final Testing (2-3 hours)

- [ ] **Navigation Integration** (1 hour)
  - [ ] Add auth check to app startup
  - [ ] Redirect to login if not authenticated
  - [ ] Add watchlist to bottom navigation
  - [ ] Add settings to app bar menu
  - [ ] Test navigation flow

- [ ] **Comprehensive Testing** (1-2 hours)
  - [ ] Test complete login flow
  - [ ] Test complete signup flow
  - [ ] Test logout and re-login
  - [ ] Test settings persistence
  - [ ] Test theme switching
  - [ ] Test watchlist CRUD operations
  - [ ] Test saved scans
  - [ ] Test error scenarios
  - [ ] Test on different devices
  - [ ] Test on different screen sizes

- [ ] **Bug Fixes & Polish** (1 hour)
  - [ ] Fix all identified bugs
  - [ ] Improve error messages
  - [ ] Add missing loading states
  - [ ] Polish animations
  - [ ] Optimize performance
  - [ ] Clean up code
  - [ ] Add comments

- [ ] **Documentation**
  - [ ] Update README with new features
  - [ ] Document API endpoints used
  - [ ] Create user guide for new features
  - [ ] Update changelog

---

## 🎯 COMPLETION CRITERIA

### Must Have (Required):
- [ ] Users can login with email/password
- [ ] Users can signup for new account
- [ ] Tokens stored securely
- [ ] Auto-login works on app restart
- [ ] Users can logout
- [ ] Settings persist correctly
- [ ] Theme changes work
- [ ] Watchlist CRUD works
- [ ] Saved scans work

### Should Have (Important):
- [ ] Password validation
- [ ] Email validation
- [ ] Error messages are helpful
- [ ] Loading indicators show
- [ ] Animations are smooth
- [ ] UI is responsive
- [ ] Empty states are helpful

### Nice to Have (Optional):
- [ ] Forgot password flow
- [ ] Social login (Google, Apple)
- [ ] Profile picture upload
- [ ] Push notifications
- [ ] Biometric authentication

---

## 📊 PROGRESS TRACKING

### Overall Progress:
- **Day 1-2:** ⬜⬜⬜⬜⬜ 0% (Authentication)
- **Day 3-4:** ⬜⬜⬜⬜⬜ 0% (Settings)
- **Day 5-6:** ⬜⬜⬜⬜⬜ 0% (Watchlist)
- **Day 7:** ⬜⬜⬜⬜⬜ 0% (Integration)

### Frontend Completion:
- **Current:** 60%
- **Target:** 85%
- **Remaining:** 25%

---

## 🚀 READY TO START?

### First Steps:
1. [ ] Review Week 3 plan
2. [ ] Set up development environment
3. [ ] Add flutter_secure_storage dependency
4. [ ] Create auth service file
5. [ ] Start building login screen

### Time Estimate:
- **Minimum:** 11 hours
- **Expected:** 13 hours
- **Maximum:** 15 hours

---

## 💡 TIPS

### Development:
- Start with backend integration (services)
- Test each feature as you build it
- Use Riverpod for state management
- Reuse widgets where possible
- Keep code clean and commented

### Testing:
- Test on real device, not just emulator
- Test different screen sizes
- Test error scenarios
- Test offline behavior
- Test with slow network

### Code Quality:
- Follow Flutter best practices
- Use const constructors
- Add meaningful comments
- Keep widgets small
- Use meaningful names

---

## 🎊 LET'S BUILD WEEK 3!

Ready to make Technic production-ready with user accounts, settings, and watchlist features! 🚀

**Start with Day 1: Authentication System**
