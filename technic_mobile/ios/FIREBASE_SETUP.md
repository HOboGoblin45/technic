# Firebase Setup for iOS Push Notifications

This guide explains how to configure Firebase Cloud Messaging (FCM) for push notifications.

## Prerequisites

1. Apple Developer Account with Push Notifications capability
2. Firebase account at [console.firebase.google.com](https://console.firebase.google.com)

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click **Add project**
3. Enter project name: `Technic Mobile`
4. Follow the wizard (disable/enable Google Analytics as needed)

## Step 2: Add iOS App to Firebase

1. In Firebase Console, click **Add app** → **iOS**
2. Enter iOS bundle ID: `com.technic.technicMobile`
3. App nickname: `Technic iOS`
4. Click **Register app**

## Step 3: Download GoogleService-Info.plist

1. Download `GoogleService-Info.plist` from Firebase Console
2. Place it in `ios/Runner/GoogleService-Info.plist`
3. In Xcode, right-click Runner folder → **Add Files to "Runner"**
4. Select `GoogleService-Info.plist`, ensure "Copy items if needed" is checked
5. Select target: Runner

## Step 4: Configure APNs in Firebase

### Generate APNs Key (Recommended)

1. Go to [Apple Developer Portal](https://developer.apple.com/account/resources/authkeys/list)
2. Click **+** to create a new key
3. Name: `Technic Push Notifications`
4. Check **Apple Push Notifications service (APNs)**
5. Click **Continue** → **Register**
6. Download the `.p8` key file (save it securely!)
7. Note the **Key ID** shown

### Upload to Firebase

1. In Firebase Console → Project Settings → Cloud Messaging
2. Under **Apple app configuration**, click **Upload**
3. Upload the `.p8` file
4. Enter Key ID and Team ID (from Apple Developer Portal)

## Step 5: Update Xcode Project

The following files should already be configured:

### Info.plist
```xml
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>remote-notification</string>
</array>
```

### Runner.entitlements
```xml
<key>aps-environment</key>
<string>development</string>
```

## Step 6: Verify Setup

1. Build and run on a physical iOS device (simulators don't support push)
2. Accept notification permissions when prompted
3. Check Xcode console for FCM token
4. Use Firebase Console → Cloud Messaging to send a test message

## Troubleshooting

### No FCM Token
- Ensure GoogleService-Info.plist is in the Runner folder
- Verify bundle ID matches Firebase configuration
- Check that APNs key is uploaded to Firebase

### Notifications Not Received
- Must test on physical device, not simulator
- Check APNs environment (development vs production)
- Verify entitlements file is included in build

### "APNs device token not set" Error
- Ensure Info.plist has `remote-notification` in UIBackgroundModes
- Request notification permissions before getting FCM token

## File Checklist

- [ ] `ios/Runner/GoogleService-Info.plist` - From Firebase
- [ ] `ios/Runner/Runner.entitlements` - APNs capability
- [ ] `ios/Runner/Info.plist` - Background modes configured
- [ ] `ios/Runner/AppDelegate.swift` - Firebase initialization

## Testing Push Notifications

### Using Firebase Console
1. Go to Cloud Messaging → Compose notification
2. Enter title and body
3. Select target: Single device (paste FCM token)
4. Send test message

### Using curl (with FCM Server Key)
```bash
curl -X POST \
  https://fcm.googleapis.com/fcm/send \
  -H 'Authorization: key=YOUR_SERVER_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "to": "DEVICE_FCM_TOKEN",
    "notification": {
      "title": "Test Alert",
      "body": "AAPL is now $150.00"
    }
  }'
```
