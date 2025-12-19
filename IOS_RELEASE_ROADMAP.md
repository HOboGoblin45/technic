# iOS App Store Release Roadmap

## Overview

This roadmap outlines the path from current state (~30% complete) to iOS App Store submission. Organized into 7 phases with clear deliverables.

**Current Status:** 92% Ready
**Target:** 100% App Store Ready
**Estimated Effort:** 4-6 weeks

---

## Phase 1: Critical iOS Configuration (Priority: URGENT) ✅ COMPLETE

**Effort:** 2-3 days
**Impact:** Unblocks all subsequent phases
**Status:** Completed

**Implemented:**
- ✅ Updated LaunchScreen.storyboard with branded dark theme (#0A0E27 background, Technic blue text)
- ✅ Created PrivacyInfo.xcprivacy with full privacy manifest declarations
- ✅ Updated Info.plist with App Store required keys (encryption, background modes)
- ✅ Added Privacy Manifest to Xcode project resources
- ✅ Created CODE_SIGNING_SETUP.md with detailed signing instructions

### 1.1 Replace Placeholder Launch Images
**Directory:** `ios/Runner/Assets.xcassets/LaunchImage.imageset/`

**Current State:** All images are 1x1 pixel placeholders (68 bytes each)

**Required Assets:**
| File | Size | Purpose |
|------|------|---------|
| LaunchImage.png | 1x | Standard resolution |
| LaunchImage@2x.png | 2x | Retina displays |
| LaunchImage@3x.png | 3x | Super Retina displays |

**Alternative:** Use LaunchScreen.storyboard with vector assets (recommended)

**Files to modify:**
- `ios/Runner/Assets.xcassets/LaunchImage.imageset/Contents.json`
- `ios/Runner/Base.lproj/LaunchScreen.storyboard`

### 1.2 Configure Code Signing
**File:** `ios/Runner.xcodeproj/project.pbxproj`

**Required:**
```
DEVELOPMENT_TEAM = YOUR_TEAM_ID;
CODE_SIGN_IDENTITY = "Apple Distribution";
PROVISIONING_PROFILE_SPECIFIER = "Your App Store Profile";
```

**Steps:**
1. Register App ID in Apple Developer Portal
2. Create Distribution Certificate
3. Create App Store Provisioning Profile
4. Configure in Xcode or project.pbxproj

### 1.3 Add Privacy Manifest (Required since Fall 2024)
**Create:** `ios/Runner/PrivacyInfo.xcprivacy`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSPrivacyTracking</key>
    <false/>
    <key>NSPrivacyTrackingDomains</key>
    <array/>
    <key>NSPrivacyCollectedDataTypes</key>
    <array>
        <dict>
            <key>NSPrivacyCollectedDataType</key>
            <string>NSPrivacyCollectedDataTypeUserID</string>
            <key>NSPrivacyCollectedDataTypeLinked</key>
            <true/>
            <key>NSPrivacyCollectedDataTypeTracking</key>
            <false/>
            <key>NSPrivacyCollectedDataTypePurposes</key>
            <array>
                <string>NSPrivacyCollectedDataTypePurposeAppFunctionality</string>
            </array>
        </dict>
    </array>
    <key>NSPrivacyAccessedAPITypes</key>
    <array>
        <dict>
            <key>NSPrivacyAccessedAPIType</key>
            <string>NSPrivacyAccessedAPICategoryUserDefaults</string>
            <key>NSPrivacyAccessedAPITypeReasons</key>
            <array>
                <string>CA92.1</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
```

### 1.4 Update Info.plist with Required Keys
**File:** `ios/Runner/Info.plist`

**Add:**
```xml
<!-- Required for App Store -->
<key>ITSAppUsesNonExemptEncryption</key>
<false/>

<!-- If using camera/photos -->
<key>NSCameraUsageDescription</key>
<string>Used for profile photos</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Used for profile photos</string>

<!-- If using notifications -->
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>remote-notification</string>
</array>
```

---

## Phase 2: Implement Alert Service (Priority: URGENT) ✅ COMPLETE

**Effort:** 3-4 days
**Impact:** Core feature currently non-functional
**Status:** Completed

**Implemented:**
- ✅ Complete alert checking logic in alert_service.dart with price comparison
- ✅ Added getCurrentPrices() method to api_service.dart
- ✅ Created notification_service.dart for local notifications
- ✅ Configured iOS background fetch in AppDelegate.swift
- ✅ Added BGTaskScheduler for iOS 13+ background tasks
- ✅ Registered background task identifiers in Info.plist

### 2.1 Complete Alert Checking Logic
**File:** `lib/services/alert_service.dart` (line 57)

**Current State:** `// TODO: Implement actual alert checking logic`

**Implementation:**
```dart
Future<void> _checkAlerts() async {
  if (_alerts.isEmpty) return;

  try {
    // Fetch current prices for watched symbols
    final symbols = _alerts.map((a) => a.symbol).toSet().toList();
    final prices = await _apiService.getCurrentPrices(symbols);

    for (final alert in _alerts) {
      final currentPrice = prices[alert.symbol];
      if (currentPrice == null) continue;

      bool triggered = false;
      if (alert.condition == AlertCondition.above && currentPrice >= alert.targetPrice) {
        triggered = true;
      } else if (alert.condition == AlertCondition.below && currentPrice <= alert.targetPrice) {
        triggered = true;
      }

      if (triggered) {
        await _triggerAlert(alert, currentPrice);
      }
    }
  } catch (e) {
    debugPrint('Alert check failed: $e');
  }
}

Future<void> _triggerAlert(PriceAlert alert, double currentPrice) async {
  // Show local notification
  await _notificationService.showAlertNotification(
    title: '${alert.symbol} Alert Triggered',
    body: '${alert.symbol} is now \$${currentPrice.toStringAsFixed(2)}',
  );

  // Mark alert as triggered or remove if one-time
  if (alert.isOneTime) {
    await removeAlert(alert.id);
  }
}
```

### 2.2 Add Price Fetching Endpoint
**File:** `lib/services/api_service.dart`

**Add method:**
```dart
Future<Map<String, double>> getCurrentPrices(List<String> symbols) async {
  final response = await _client.get(
    Uri.parse('${ApiConfig.baseUrl}/prices'),
    headers: {'symbols': symbols.join(',')},
  ).timeout(ApiConfig.connectTimeout);

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return data.map((k, v) => MapEntry(k, (v as num).toDouble()));
  }
  throw ApiException('Failed to fetch prices');
}
```

### 2.3 Configure Background Fetch (iOS)
**File:** `ios/Runner/AppDelegate.swift`

```swift
import UIKit
import Flutter
import BackgroundTasks

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    // Register background task
    BGTaskScheduler.shared.register(
      forTaskWithIdentifier: "com.technic.technicMobile.alertCheck",
      using: nil
    ) { task in
      self.handleAlertCheck(task: task as! BGAppRefreshTask)
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func handleAlertCheck(task: BGAppRefreshTask) {
    // Schedule next check
    scheduleAlertCheck()

    // Perform check via Flutter method channel
    // ...

    task.setTaskCompleted(success: true)
  }

  func scheduleAlertCheck() {
    let request = BGAppRefreshTaskRequest(identifier: "com.technic.technicMobile.alertCheck")
    request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes
    try? BGTaskScheduler.shared.submit(request)
  }
}
```

---

## Phase 3: Push Notifications (Priority: HIGH) ✅ COMPLETE

**Effort:** 3-4 days
**Impact:** Enables real-time alerts when app is closed
**Status:** Completed

**Implemented:**
- ✅ Enabled Firebase dependencies in pubspec.yaml (firebase_core, firebase_messaging)
- ✅ Created FIREBASE_SETUP.md with complete Firebase configuration guide
- ✅ Updated AppDelegate.swift with Firebase initialization and FCM delegate
- ✅ Created Runner.entitlements with APNs capability
- ✅ Updated notification_service.dart with full FCM integration
- ✅ Added background message handler with @pragma('vm:entry-point')
- ✅ Implemented topic subscriptions for symbol alerts
- ✅ Added entitlements to Xcode project build configurations

### 3.1 Enable Firebase Dependencies
**File:** `pubspec.yaml`

**Uncomment/Add:**
```yaml
dependencies:
  firebase_core: ^2.24.2
  firebase_messaging: ^14.7.10
  firebase_analytics: ^10.7.4
```

### 3.2 Configure Firebase for iOS
**Steps:**
1. Create Firebase project at console.firebase.google.com
2. Add iOS app with bundle ID `com.technic.technicMobile`
3. Download `GoogleService-Info.plist`
4. Add to `ios/Runner/` directory
5. Update `ios/Runner/AppDelegate.swift`

**File:** `ios/Runner/AppDelegate.swift`
```swift
import Firebase

override func application(
  _ application: UIApplication,
  didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
) -> Bool {
  FirebaseApp.configure()
  // ... rest of setup
}
```

### 3.3 Create Notification Service
**Create:** `lib/services/notification_service.dart`

```dart
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FirebaseMessaging _fcm = FirebaseMessaging.instance;
  final FlutterLocalNotificationsPlugin _localNotifications =
      FlutterLocalNotificationsPlugin();

  Future<void> initialize() async {
    // Request permission
    await _fcm.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    // Get FCM token
    final token = await _fcm.getToken();
    debugPrint('FCM Token: $token');
    // Send token to backend for push targeting

    // Handle foreground messages
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

    // Handle background/terminated messages
    FirebaseMessaging.onBackgroundMessage(_handleBackgroundMessage);
  }

  void _handleForegroundMessage(RemoteMessage message) {
    // Show local notification when app is in foreground
    showAlertNotification(
      title: message.notification?.title ?? 'Alert',
      body: message.notification?.body ?? '',
    );
  }

  Future<void> showAlertNotification({
    required String title,
    required String body,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'alerts',
      'Price Alerts',
      importance: Importance.high,
      priority: Priority.high,
    );
    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );
    const details = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );

    await _localNotifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
    );
  }
}

@pragma('vm:entry-point')
Future<void> _handleBackgroundMessage(RemoteMessage message) async {
  // Handle background message
  debugPrint('Background message: ${message.messageId}');
}
```

### 3.4 Add APNs Entitlements
**Create:** `ios/Runner/Runner.entitlements`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>aps-environment</key>
    <string>development</string>
</dict>
</plist>
```

---

## Phase 4: App Store Metadata (Priority: HIGH) ✅ COMPLETE

**Effort:** 2-3 days
**Impact:** Required for App Store submission
**Status:** Completed

**Implemented:**
- ✅ Created docs/APP_STORE_DESCRIPTION.md with full App Store listing
- ✅ Created docs/PRIVACY_POLICY.md with GDPR/CCPA compliance
- ✅ Created docs/TERMS_OF_SERVICE.md with financial disclaimers
- ✅ Created docs/SCREENSHOT_GUIDELINES.md with specifications
- ✅ Defined app name, subtitle, keywords, and category
- ✅ Added review notes and demo account information
- ✅ Included promotional text and What's New content

### 4.1 Create App Store Screenshots
**Required sizes:**
| Device | Size | Count |
|--------|------|-------|
| iPhone 6.7" | 1290 x 2796 | 3-10 |
| iPhone 6.5" | 1284 x 2778 | 3-10 |
| iPhone 5.5" | 1242 x 2208 | 3-10 |
| iPad Pro 12.9" | 2048 x 2732 | 3-10 |

**Recommended screens to capture:**
1. Scanner results with stock picks
2. AI Copilot conversation
3. Symbol detail with charts
4. Watchlist view
5. Ideas/recommendations

### 4.2 Write App Store Description
**Create:** `docs/app_store_description.md`

```markdown
# Technic - Quant Trading Companion

## Short Description (30 chars)
AI-Powered Stock Scanner

## Full Description (4000 chars max)
Technic is your intelligent trading companion, combining quantitative
analysis with AI insights to help you discover high-potential stock setups.

**KEY FEATURES:**

📊 Smart Stock Scanner
- Scan thousands of stocks using proven technical patterns
- Filter by sector, market cap, volume, and custom criteria
- Get ranked results based on technical strength

🤖 AI Copilot
- Chat with our AI to understand market movements
- Get explanations of complex trading setups
- Ask questions about specific stocks in plain English

📈 Real-Time Charts
- Interactive price charts with key indicators
- Support and resistance levels
- Volume analysis and momentum signals

🔔 Price Alerts
- Set custom price targets
- Get notified when stocks hit your levels
- Never miss an entry or exit point

💡 Trading Ideas
- Curated daily trade ideas
- Risk/reward analysis included
- Track ideas and build your watchlist

**WHO IS THIS FOR?**
- Active traders looking for an edge
- Swing traders seeking quality setups
- Anyone wanting to level up their trading

Download Technic and start trading smarter today.

## Keywords (100 chars)
stocks,trading,scanner,AI,technical analysis,charts,alerts,watchlist,investing,market
```

### 4.3 Create Privacy Policy
**Create:** `docs/privacy_policy.md` (also host on web)

**Required sections:**
- Data we collect (user ID, watchlist, preferences)
- How we use data
- Third-party services (Firebase, analytics)
- Data retention
- User rights (GDPR/CCPA compliance)
- Contact information

### 4.4 Create Terms of Service
**Create:** `docs/terms_of_service.md`

**Required sections:**
- Disclaimer (not financial advice)
- Acceptable use
- Account terms
- Intellectual property
- Limitation of liability
- Termination

---

## Phase 5: Testing & Quality Assurance (Priority: HIGH) ✅ COMPLETE

**Effort:** 3-4 days
**Impact:** Prevents rejection and crashes
**Status:** Completed

**Implemented:**
- ✅ Created test/models/scan_result_test.dart with comprehensive tests
- ✅ Created test/models/watchlist_item_test.dart
- ✅ Created test/models/copilot_message_test.dart
- ✅ Created test/services/storage_service_test.dart
- ✅ Created test/widgets/premium_button_test.dart
- ✅ Created test/widgets/premium_card_test.dart
- ✅ Created docs/TESTING_CHECKLIST.md with device/performance testing guide

### 5.1 Create Unit Tests
**Directory:** `test/`

**Priority test files:**
```
test/
├── models/
│   ├── scan_result_test.dart
│   ├── symbol_detail_test.dart
│   └── price_alert_test.dart
├── services/
│   ├── api_client_test.dart
│   ├── alert_service_test.dart
│   └── storage_service_test.dart
└── providers/
    ├── scanner_provider_test.dart
    └── alert_provider_test.dart
```

### 5.2 Create Widget Tests
**File:** `test/widget_test.dart`

Test critical UI flows:
- Scanner form submission
- Alert creation
- Watchlist add/remove
- Theme switching

### 5.3 Device Testing Checklist
**Test on physical devices:**
- [ ] iPhone SE (smallest screen)
- [ ] iPhone 15 Pro (current flagship)
- [ ] iPhone 15 Pro Max (largest screen)
- [ ] iPad (if supporting)

**Test scenarios:**
- [ ] Fresh install
- [ ] Upgrade from previous version
- [ ] Low memory conditions
- [ ] Poor network connectivity
- [ ] Background/foreground transitions
- [ ] Push notification handling

### 5.4 Performance Testing
- [ ] App launch time < 2 seconds
- [ ] Scanner results load < 3 seconds
- [ ] Smooth scrolling (60 fps)
- [ ] Memory usage < 200MB typical

---

## Phase 6: iOS-Specific Features (Priority: MEDIUM)

**Effort:** 3-5 days
**Impact:** Better user experience and App Store ranking
**Completion:** 92% → 97%

### 6.1 Add Apple Sign-In
**File:** `pubspec.yaml`
```yaml
dependencies:
  sign_in_with_apple: ^5.0.0
```

**Implementation in:** `lib/services/auth_service.dart`

### 6.2 Add Biometric Authentication
**File:** `pubspec.yaml`
```yaml
dependencies:
  local_auth: ^2.1.6
```

**Add to:** `lib/screens/auth/` for secure app access

### 6.3 Configure Deep Linking
**File:** `ios/Runner/Info.plist`
```xml
<key>CFBundleURLTypes</key>
<array>
    <dict>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>technic</string>
        </array>
    </dict>
</array>
```

**Configure in:** `lib/router/` for handling `technic://` URLs

### 6.4 Add Haptic Feedback
**File:** `lib/utils/haptics.dart`
```dart
import 'package:flutter/services.dart';

class Haptics {
  static void light() => HapticFeedback.lightImpact();
  static void medium() => HapticFeedback.mediumImpact();
  static void heavy() => HapticFeedback.heavyImpact();
  static void success() => HapticFeedback.mediumImpact();
  static void error() => HapticFeedback.heavyImpact();
}
```

---

## Phase 7: Final Submission Prep (Priority: URGENT)

**Effort:** 2-3 days
**Impact:** App Store approval
**Completion:** 97% → 100%

### 7.1 Build Release Version
```bash
# Clean build
flutter clean
flutter pub get

# Build iOS release
flutter build ios --release

# Archive in Xcode
# Product > Archive
```

### 7.2 TestFlight Beta Testing
1. Upload build to App Store Connect
2. Add internal testers
3. Fix any reported issues
4. Add external testers (optional)
5. Collect feedback for 3-5 days minimum

### 7.3 App Store Connect Setup
**Configure in App Store Connect:**
- [ ] App name and subtitle
- [ ] App description
- [ ] Keywords
- [ ] Screenshots for all device sizes
- [ ] App icon (1024x1024)
- [ ] Privacy policy URL
- [ ] Support URL
- [ ] Marketing URL (optional)
- [ ] Age rating questionnaire
- [ ] Pricing (Free or paid)
- [ ] App Review information

### 7.4 Pre-Submission Checklist
- [ ] All placeholder content removed
- [ ] No debug code or test API keys
- [ ] Privacy manifest complete
- [ ] All required entitlements configured
- [ ] No private API usage
- [ ] Proper error handling throughout
- [ ] Accessibility labels on key elements
- [ ] No hardcoded localhost URLs
- [ ] Version and build numbers updated

### 7.5 Submit for Review
1. Select build in App Store Connect
2. Complete App Review Information
3. Add any review notes (demo account if needed)
4. Submit for review
5. Monitor for questions/rejection feedback

---

## Summary Timeline

| Phase | Focus | Effort | Completion | Status |
|-------|-------|--------|------------|--------|
| 1 | iOS Configuration | 2-3 days | 0% → 45% | ✅ Complete |
| 2 | Alert Service | 3-4 days | 45% → 55% | ✅ Complete |
| 3 | Push Notifications | 3-4 days | 55% → 70% | ✅ Complete |
| 4 | App Store Metadata | 2-3 days | 70% → 85% | ✅ Complete |
| 5 | Testing & QA | 3-4 days | 85% → 92% | ✅ Complete |
| 6 | iOS Features | 3-5 days | 92% → 97% | Pending |
| 7 | Submission | 2-3 days | 97% → 100% | Pending |
| **Total** | | **18-26 days** | **100%** | |

---

## Quick Reference: Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `ios/Runner/Assets.xcassets/LaunchImage.imageset/*` | Replace | 1 |
| `ios/Runner.xcodeproj/project.pbxproj` | Modify | 1 |
| `ios/Runner/PrivacyInfo.xcprivacy` | Create | 1 |
| `ios/Runner/Info.plist` | Modify | 1, 6 |
| `lib/services/alert_service.dart` | Modify | 2 |
| `lib/services/api_service.dart` | Modify | 2 |
| `ios/Runner/AppDelegate.swift` | Modify | 2, 3 |
| `pubspec.yaml` | Modify | 3, 6 |
| `ios/Runner/GoogleService-Info.plist` | Create | 3 |
| `lib/services/notification_service.dart` | Create | 3 |
| `ios/Runner/Runner.entitlements` | Create | 3 |
| `docs/app_store_description.md` | Create | 4 |
| `docs/privacy_policy.md` | Create | 4 |
| `docs/terms_of_service.md` | Create | 4 |
| `test/**/*_test.dart` | Create | 5 |

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| App Store rejection | Follow guidelines strictly, test thoroughly |
| Backend API not ready | Use mock data for demo, add error handling |
| Firebase quota limits | Monitor usage, implement caching |
| Long review times | Submit early, plan for 1-2 week review |
| iOS-specific bugs | Test on multiple real devices |

---

## Post-Launch Checklist

- [ ] Monitor crash reports in App Store Connect
- [ ] Respond to user reviews
- [ ] Track analytics for usage patterns
- [ ] Plan v1.1 with user feedback
- [ ] Set up automated build pipeline (fastlane)
