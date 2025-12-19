# Pre-Submission Checklist

## Overview

Complete this checklist before submitting Technic to the App Store. Each item must be verified to ensure a smooth review process.

---

## Code Quality

### Debug Code Removal

- [ ] Remove all `debugPrint` statements (or disable in release)
- [ ] Remove all `print` statements
- [ ] Remove console logging
- [ ] Remove test scaffolding
- [ ] Remove TODO comments
- [ ] Remove FIXME comments
- [ ] Disable verbose logging

### API Configuration

- [ ] All API endpoints point to production servers
- [ ] No localhost URLs in code
- [ ] No staging/development URLs
- [ ] API keys are production keys
- [ ] Firebase configuration is production
- [ ] Sentry/analytics in production mode

### Error Handling

- [ ] All network calls have error handling
- [ ] User-friendly error messages displayed
- [ ] No stack traces shown to users
- [ ] Graceful degradation when offline
- [ ] Timeout handling implemented

---

## Security

### Sensitive Data

- [ ] No hardcoded API keys in source
- [ ] No hardcoded passwords
- [ ] No test credentials in code
- [ ] Secure storage for tokens
- [ ] Keychain used for sensitive data

### Network Security

- [ ] All connections use HTTPS
- [ ] No HTTP endpoints allowed
- [ ] SSL certificate validation enabled
- [ ] No certificate pinning bypass in production

### Data Privacy

- [ ] Privacy manifest complete (PrivacyInfo.xcprivacy)
- [ ] All data collection disclosed
- [ ] User consent obtained where required
- [ ] Data deletion capability exists
- [ ] GDPR/CCPA compliance verified

---

## iOS Configuration

### Info.plist

- [ ] Bundle display name correct
- [ ] Bundle identifier matches App Store
- [ ] Version number updated
- [ ] Build number incremented
- [ ] All required usage descriptions present:
  - [ ] NSFaceIDUsageDescription
  - [ ] (others as needed)
- [ ] ITSAppUsesNonExemptEncryption set to NO
- [ ] Background modes configured correctly
- [ ] URL schemes configured

### Entitlements

- [ ] aps-environment set to `production`
- [ ] Sign in with Apple configured
- [ ] Associated Domains configured
- [ ] All required capabilities enabled

### Code Signing

- [ ] Distribution certificate valid
- [ ] Provisioning profile valid (not expired)
- [ ] Team ID correct
- [ ] Automatic signing or manual profile selected

---

## Assets

### App Icons

- [ ] 1024x1024 App Store icon present
- [ ] All required icon sizes in Assets.xcassets
- [ ] No alpha channel in icons
- [ ] Icons render correctly at all sizes

### Launch Screen

- [ ] Launch screen configured
- [ ] No placeholder images
- [ ] Brand colors applied
- [ ] Works on all device sizes

### Images

- [ ] All placeholder images replaced
- [ ] Images optimized for size
- [ ] @2x and @3x versions present
- [ ] No copyrighted images without license

---

## User Interface

### Device Compatibility

- [ ] Works on iPhone SE (smallest)
- [ ] Works on iPhone 15 Pro Max (largest)
- [ ] Works on all supported iOS versions
- [ ] Handles Dynamic Type
- [ ] Handles Dark Mode
- [ ] Safe area insets respected

### Accessibility

- [ ] VoiceOver support
- [ ] Accessibility labels on interactive elements
- [ ] Sufficient color contrast
- [ ] Text scales with accessibility settings
- [ ] No information conveyed by color alone

### Localization

- [ ] All strings externalized (if localizing)
- [ ] Date/time formatting uses locale
- [ ] Number formatting uses locale
- [ ] Currency formatting correct

---

## Functionality

### Core Features

- [ ] Scanner works correctly
- [ ] AI Copilot responds appropriately
- [ ] Charts render correctly
- [ ] Watchlist add/remove works
- [ ] Alerts create/edit/delete works
- [ ] Settings persist correctly

### Authentication

- [ ] Login works
- [ ] Signup works
- [ ] Apple Sign-In works
- [ ] Logout works
- [ ] Token refresh works
- [ ] Biometric authentication works

### Push Notifications

- [ ] Permission request works
- [ ] Notifications received in foreground
- [ ] Notifications received in background
- [ ] Notification tap opens correct screen
- [ ] Token registration successful

### Deep Links

- [ ] technic:// scheme works
- [ ] Universal links work (if configured)
- [ ] Links open correct screens
- [ ] Invalid links handled gracefully

---

## Performance

### App Launch

- [ ] Cold start < 2 seconds
- [ ] Warm start < 1 second
- [ ] No white screen during launch
- [ ] Splash screen displays correctly

### Memory

- [ ] No memory leaks
- [ ] Memory usage < 200MB typical
- [ ] Handles low memory warnings
- [ ] Images released when off-screen

### Battery

- [ ] No excessive background activity
- [ ] Location services (if used) efficient
- [ ] Network calls batched where possible

### Network

- [ ] Works on slow connections (3G)
- [ ] Handles network timeouts
- [ ] Caching implemented where appropriate
- [ ] Offline mode works correctly

---

## Testing

### Automated Tests

- [ ] Unit tests pass
- [ ] Widget tests pass
- [ ] Integration tests pass
- [ ] Code coverage acceptable

### Manual Testing

- [ ] Fresh install tested
- [ ] Upgrade from previous version tested
- [ ] All user flows tested
- [ ] Edge cases tested
- [ ] Error states tested

### Device Testing

- [ ] Tested on physical iPhone
- [ ] Tested on multiple iOS versions
- [ ] Tested on different screen sizes
- [ ] Tested with slow network

---

## App Store Metadata

### Required Assets

- [ ] Screenshots for 6.7" iPhone
- [ ] Screenshots for 6.5" iPhone
- [ ] Screenshots for 5.5" iPhone
- [ ] App Preview videos (optional)
- [ ] App icon 1024x1024

### Text Content

- [ ] App name (30 chars max)
- [ ] Subtitle (30 chars max)
- [ ] Description (4000 chars max)
- [ ] Keywords (100 chars max)
- [ ] What's New text
- [ ] Promotional text (optional)

### URLs

- [ ] Privacy Policy URL valid
- [ ] Support URL valid
- [ ] Marketing URL valid (optional)

### Review Information

- [ ] Contact info current
- [ ] Demo account provided
- [ ] Review notes written
- [ ] Attachments uploaded (if needed)

---

## Legal & Compliance

### App Store Guidelines

- [ ] No private API usage
- [ ] No undocumented features
- [ ] Accurate app description
- [ ] Screenshots match actual app
- [ ] No misleading claims

### Financial App Requirements

- [ ] "Not financial advice" disclaimer
- [ ] No guaranteed returns claims
- [ ] Risk disclosures present
- [ ] Terms of Service in place

### Data Protection

- [ ] Privacy Policy published
- [ ] Terms of Service published
- [ ] Data collection disclosed
- [ ] User consent mechanisms in place

---

## Final Verification

### Build

- [ ] Release build compiles without errors
- [ ] Release build runs without crashes
- [ ] Archive creates successfully
- [ ] Validation passes in Xcode

### Upload

- [ ] Build uploads successfully
- [ ] Build processing completes
- [ ] Export compliance answered
- [ ] Build available in App Store Connect

### Submission

- [ ] All metadata complete
- [ ] Build selected
- [ ] Reviewed all information
- [ ] Ready to submit

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| QA | | | |
| Product Manager | | | |

---

## Notes

Add any additional notes or issues discovered during the checklist review:

```
[Add notes here]
```

---

## Version History

| Date | Version | Reviewer | Notes |
|------|---------|----------|-------|
| | 1.0.0 | | Initial release |
