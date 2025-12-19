# iOS Code Signing Setup Guide

This guide walks through configuring code signing for App Store distribution.

## Prerequisites

1. **Apple Developer Account** ($99/year)
   - Enroll at: https://developer.apple.com/programs/enroll/

2. **Xcode** (latest version recommended)

3. **Mac with macOS** (required for iOS development)

## Step 1: Create App ID

1. Go to [Apple Developer Portal](https://developer.apple.com/account/resources/identifiers/list)
2. Click **+** to register a new identifier
3. Select **App IDs** → Continue
4. Select **App** → Continue
5. Fill in:
   - Description: `Technic Mobile`
   - Bundle ID: `com.technic.technicMobile` (Explicit)
6. Enable capabilities as needed:
   - [x] Push Notifications
   - [x] Associated Domains (for deep linking)
   - [x] Sign In with Apple (if using)
7. Click **Continue** → **Register**

## Step 2: Create Distribution Certificate

1. Go to [Certificates](https://developer.apple.com/account/resources/certificates/list)
2. Click **+** to create new certificate
3. Select **Apple Distribution** → Continue
4. Follow instructions to create CSR using Keychain Access:
   - Open Keychain Access
   - Certificate Assistant → Request Certificate from CA
   - Enter email, select "Saved to disk"
5. Upload CSR → Download certificate
6. Double-click to install in Keychain

## Step 3: Create Provisioning Profile

1. Go to [Profiles](https://developer.apple.com/account/resources/profiles/list)
2. Click **+** to create new profile
3. Select **App Store Connect** → Continue
4. Select your App ID: `com.technic.technicMobile` → Continue
5. Select your Distribution Certificate → Continue
6. Name: `Technic Mobile App Store`
7. Generate → Download

## Step 4: Configure Xcode Project

### Option A: Automatic Signing (Recommended)

1. Open `ios/Runner.xcworkspace` in Xcode
2. Select **Runner** project in navigator
3. Select **Runner** target → Signing & Capabilities
4. Check **Automatically manage signing**
5. Select your Team from dropdown
6. Xcode will handle certificate and profile

### Option B: Manual Signing

Edit `ios/Runner.xcodeproj/project.pbxproj`:

```
DEVELOPMENT_TEAM = YOUR_TEAM_ID;
CODE_SIGN_IDENTITY = "Apple Distribution";
CODE_SIGN_STYLE = Manual;
PROVISIONING_PROFILE_SPECIFIER = "Technic Mobile App Store";
```

**Find your Team ID:**
- Go to [Apple Developer Membership](https://developer.apple.com/account/#/membership)
- Team ID is displayed on the page

## Step 5: Build for App Store

```bash
# Clean and build
flutter clean
flutter pub get

# Build iOS release
flutter build ios --release

# Open in Xcode for archive
open ios/Runner.xcworkspace
```

In Xcode:
1. Select **Any iOS Device** as build target
2. **Product** → **Archive**
3. Once complete, **Distribute App** → **App Store Connect**

## Troubleshooting

### "No signing certificate" error
- Ensure certificate is installed in Keychain
- Check certificate hasn't expired
- Verify you're using the correct team

### "Provisioning profile doesn't include" error
- Regenerate provisioning profile
- Download and install latest profile
- Clean build folder: **Product** → **Clean Build Folder**

### Code signing issues in CI/CD
For automated builds, use Fastlane Match:
- https://docs.fastlane.tools/actions/match/

## Environment Variables for CI

```bash
# For fastlane/CI builds
export APPLE_TEAM_ID="YOUR_TEAM_ID"
export MATCH_PASSWORD="your_match_password"
export FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD="app-specific-password"
```

## Entitlements

If using Push Notifications, ensure `Runner.entitlements` exists:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>aps-environment</key>
    <string>production</string>
</dict>
</plist>
```

## Quick Reference

| Setting | Value |
|---------|-------|
| Bundle ID | `com.technic.technicMobile` |
| App Name | Technic |
| Minimum iOS | 13.0 |
| Architectures | arm64 |

## Next Steps After Signing

1. Create app in [App Store Connect](https://appstoreconnect.apple.com)
2. Upload build via Xcode or Transporter
3. Submit for TestFlight testing
4. Submit for App Store review
