# Release Build Guide

## Overview

This guide covers the complete process for building, testing, and releasing Technic to the iOS App Store.

---

## Prerequisites

### Development Environment
- macOS 13.0+ (Ventura or later)
- Xcode 15.0+
- Flutter 3.10+
- CocoaPods 1.14+
- Active Apple Developer Program membership

### Required Accounts
- Apple Developer Account (with App Store Connect access)
- Firebase Console access (for push notifications)

### Required Certificates
- Apple Distribution Certificate
- App Store Provisioning Profile
- APNs Key (for push notifications)

---

## Step 1: Prepare for Release

### 1.1 Update Version Numbers

Edit `pubspec.yaml`:
```yaml
version: 1.0.0+1  # format: major.minor.patch+buildNumber
```

**Version Guidelines:**
- **Major** (1.x.x): Breaking changes, major features
- **Minor** (x.1.x): New features, enhancements
- **Patch** (x.x.1): Bug fixes, minor improvements
- **Build** (+1): Increment for each App Store upload

### 1.2 Update iOS Version

The iOS version is automatically synced from `pubspec.yaml`, but verify in:
- `ios/Runner/Info.plist` (CFBundleShortVersionString, CFBundleVersion)

### 1.3 Verify Bundle Identifier

Confirm bundle ID matches App Store Connect:
```
com.technic.technicMobile
```

Check in `ios/Runner.xcodeproj/project.pbxproj`:
```
PRODUCT_BUNDLE_IDENTIFIER = com.technic.technicMobile;
```

---

## Step 2: Configure Release Settings

### 2.1 Update Entitlements for Production

Edit `ios/Runner/Runner.entitlements`:
```xml
<!-- Change from development to production -->
<key>aps-environment</key>
<string>production</string>
```

### 2.2 Verify Code Signing

In Xcode > Runner > Signing & Capabilities:
- Team: Your Apple Developer Team
- Signing Certificate: Apple Distribution
- Provisioning Profile: App Store profile

Or in `project.pbxproj`:
```
CODE_SIGN_IDENTITY = "Apple Distribution";
DEVELOPMENT_TEAM = YOUR_TEAM_ID;
PROVISIONING_PROFILE_SPECIFIER = "Technic App Store";
```

### 2.3 Verify Capabilities

Ensure these are enabled in Xcode:
- [x] Push Notifications
- [x] Sign in with Apple
- [x] Associated Domains
- [x] Background Modes (fetch, remote-notification, processing)

---

## Step 3: Clean Build

### 3.1 Clean Flutter Project

```bash
# Navigate to project
cd technic_mobile

# Clean Flutter build artifacts
flutter clean

# Get dependencies
flutter pub get

# Generate any code (if using build_runner)
flutter pub run build_runner build --delete-conflicting-outputs
```

### 3.2 Clean iOS Build

```bash
# Navigate to iOS directory
cd ios

# Remove Pods
rm -rf Pods
rm -f Podfile.lock

# Reinstall pods
pod install --repo-update

# Return to project root
cd ..
```

### 3.3 Verify Dependencies

```bash
# Check for outdated packages
flutter pub outdated

# Check for any issues
flutter analyze
```

---

## Step 4: Build Release

### 4.1 Build iOS Release

```bash
# Build release IPA
flutter build ios --release

# Or build for specific target
flutter build ios --release --target=lib/main.dart
```

### 4.2 Open in Xcode

```bash
# Open Xcode workspace
open ios/Runner.xcworkspace
```

### 4.3 Archive in Xcode

1. Select **Product > Destination > Any iOS Device**
2. Select **Product > Archive**
3. Wait for archive to complete
4. Organizer window will open automatically

---

## Step 5: Upload to App Store Connect

### 5.1 Validate Archive

In Xcode Organizer:
1. Select the archive
2. Click **Validate App**
3. Choose distribution options:
   - [x] App Store Connect
   - [x] Upload your app's symbols
4. Wait for validation to complete
5. Fix any validation errors

### 5.2 Distribute to App Store

1. Click **Distribute App**
2. Select **App Store Connect**
3. Choose **Upload**
4. Select options:
   - [x] Upload your app's symbols
   - [x] Manage Version and Build Number (optional)
5. Click **Upload**
6. Wait for upload to complete

### 5.3 Alternative: Using Transporter

1. Export archive as `.ipa`
2. Open Transporter app
3. Drag `.ipa` file to Transporter
4. Click **Deliver**

---

## Step 6: TestFlight Beta Testing

### 6.1 Process Build

After upload, in App Store Connect:
1. Wait for build processing (10-30 minutes)
2. Build appears in TestFlight section
3. Complete Export Compliance questionnaire
4. Add missing compliance information if prompted

### 6.2 Internal Testing

1. Go to **TestFlight > Internal Testing**
2. Create new group or use existing
3. Add team members (up to 100)
4. Enable build for testing
5. Testers receive email invitation

### 6.3 External Testing (Optional)

1. Go to **TestFlight > External Testing**
2. Create testing group
3. Add testers by email (up to 10,000)
4. Submit for Beta App Review
5. Wait for approval (usually 24-48 hours)

### 6.4 Collect Feedback

- Monitor TestFlight feedback in App Store Connect
- Check crash reports
- Review tester comments
- Test on multiple device types
- Minimum testing period: 3-5 days recommended

---

## Step 7: App Store Submission

### 7.1 Prepare App Store Listing

In App Store Connect > App Information:
- App name: Technic
- Subtitle: AI-Powered Stock Scanner
- Category: Finance
- Privacy Policy URL: https://technic.app/privacy
- Support URL: https://technic.app/support

### 7.2 Prepare Version Information

- Version description (What's New)
- Keywords (max 100 characters)
- Promotional text (optional)
- Screenshots for all required sizes
- App Preview videos (optional)

### 7.3 App Review Information

- Demo account credentials (if needed)
- Contact information
- Notes for reviewer
- Attachment (if needed)

### 7.4 Submit for Review

1. Select build from dropdown
2. Complete all required fields
3. Answer App Review questions
4. Click **Submit for Review**
5. App status changes to "Waiting for Review"

---

## Release Checklist

### Before Archive
- [ ] Version number updated
- [ ] Build number incremented
- [ ] Release notes written
- [ ] All features tested
- [ ] No debug code present
- [ ] Analytics/crash reporting configured
- [ ] API endpoints point to production

### Before Upload
- [ ] Archive validated successfully
- [ ] No warnings or errors
- [ ] Correct signing identity
- [ ] Correct provisioning profile

### Before Submission
- [ ] All screenshots uploaded
- [ ] App description complete
- [ ] Keywords optimized
- [ ] Privacy policy URL valid
- [ ] Support URL valid
- [ ] Age rating completed
- [ ] Pricing set

### After Submission
- [ ] Monitor email for review updates
- [ ] Respond promptly to any questions
- [ ] Prepare for potential rejection reasons
- [ ] Plan post-launch monitoring

---

## Troubleshooting

### Archive Fails

```bash
# Check for code signing issues
security find-identity -v -p codesigning

# Verify provisioning profiles
ls ~/Library/MobileDevice/Provisioning\ Profiles/
```

### Validation Errors

**"Missing Push Notification Entitlement"**
- Verify `aps-environment` in entitlements
- Check APNs capability in Xcode

**"Invalid Bundle ID"**
- Ensure bundle ID matches App Store Connect exactly
- Check for typos in project.pbxproj

**"Missing Privacy Manifest"**
- Add PrivacyInfo.xcprivacy to project
- Ensure it's included in Copy Bundle Resources

### Upload Fails

**"Authentication Error"**
- Check Apple ID credentials
- Verify App Store Connect access
- Try generating new app-specific password

**"Build Processing Failed"**
- Check email for specific error
- Common: missing required icons, invalid binary

---

## Quick Reference Commands

```bash
# Full release build
flutter clean && flutter pub get && flutter build ios --release

# Open in Xcode
open ios/Runner.xcworkspace

# Check Flutter doctor
flutter doctor -v

# Analyze code
flutter analyze

# Run tests
flutter test
```

---

## Contact & Support

For build issues:
- Flutter: https://flutter.dev/docs
- Xcode: https://developer.apple.com/documentation/xcode
- App Store Connect: https://developer.apple.com/app-store-connect/
