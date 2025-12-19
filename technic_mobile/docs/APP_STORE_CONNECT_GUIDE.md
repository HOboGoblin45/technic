# App Store Connect Setup Guide

## Overview

This guide walks through the complete App Store Connect setup for Technic.

---

## Initial Setup

### 1. Create App in App Store Connect

1. Log in to [App Store Connect](https://appstoreconnect.apple.com)
2. Click **My Apps**
3. Click **+** > **New App**
4. Fill in required information:

| Field | Value |
|-------|-------|
| Platform | iOS |
| Name | Technic |
| Primary Language | English (U.S.) |
| Bundle ID | com.technic.technicMobile |
| SKU | technic-ios-001 |
| User Access | Full Access |

---

## App Information

### General Information

Navigate to **App Information**:

| Field | Value |
|-------|-------|
| Name | Technic |
| Subtitle | AI-Powered Stock Scanner |
| Category | Finance |
| Secondary Category | Utilities (optional) |
| Content Rights | Does not contain third-party content |

### Privacy Policy

| Field | Value |
|-------|-------|
| Privacy Policy URL | https://technic.app/privacy |

### Age Rating

Complete the Age Rating questionnaire:
- Gambling: No
- Contests: No
- Violence: No
- Sexual Content: No
- Drugs: No
- Alcohol: No
- Profanity: No
- User Generated Content: No
- Horror: No

**Result: Rated 4+**

---

## Pricing & Availability

### Price

| Field | Value |
|-------|-------|
| Price | Free |
| In-App Purchases | Configure separately if applicable |

### Availability

- [x] All territories
- Or select specific countries/regions

### Pre-Orders

- [ ] Enable pre-orders (optional)

---

## App Privacy

### Data Collection

Based on Technic's features, declare the following data types:

**Contact Info**
- Email Address: Collected for account creation
- Linked to identity: Yes
- Used for tracking: No

**Identifiers**
- User ID: Used for app functionality
- Linked to identity: Yes
- Used for tracking: No

**Usage Data**
- Product Interaction: Used for analytics
- Linked to identity: No
- Used for tracking: No

**Diagnostics**
- Crash Data: Used for app functionality
- Performance Data: Used for analytics
- Linked to identity: No

### Data Purposes

For each data type, specify:
- App Functionality
- Analytics
- NOT used for Third-Party Advertising
- NOT used for Developer's Advertising

---

## Version Information

### Version Details

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Copyright | © 2025 Technic Inc. |

### What's New

```
Initial release of Technic - your AI-powered stock scanning companion.

• Smart stock scanner with technical filters
• AI Copilot for market insights
• Interactive price charts
• Price alerts with push notifications
• Curated trading ideas
• Personalized watchlist
```

### Promotional Text (Optional)

```
Discover high-potential stock setups with AI-powered scanning. Get real-time alerts and intelligent market insights.
```

---

## App Review Information

### Contact Information

| Field | Value |
|-------|-------|
| First Name | [Your name] |
| Last Name | [Your name] |
| Phone | [Your phone] |
| Email | support@technic.app |

### Demo Account

```
Email: demo@technic.app
Password: TechnicDemo2024!
```

### Notes for Reviewer

```
Technic is a stock scanning and analysis tool for educational purposes.
Key points for review:

1. The app does NOT execute trades or connect to brokerage accounts
2. All financial data is for informational purposes only
3. Push notifications are used solely for user-configured price alerts
4. Sign in with Apple is available as primary authentication
5. Biometric authentication (Face ID) is optional for app unlock

Network Requirements:
- The app requires an internet connection to fetch real-time stock data
- Background fetch is used for checking price alerts

If you need any additional information, please contact us at support@technic.app
```

### Attachment

Upload annotated screenshots showing:
- How to create an account
- Key features
- Any complex functionality

---

## Screenshots

### Required Sizes

Upload screenshots for each size:

| Device | Resolution | Required |
|--------|-----------|----------|
| iPhone 6.7" | 1290 x 2796 | Yes |
| iPhone 6.5" | 1284 x 2778 | Yes |
| iPhone 5.5" | 1242 x 2208 | Yes |
| iPad Pro 12.9" | 2048 x 2732 | If supporting iPad |

### Recommended Screenshots (in order)

1. **Scanner Results** - Hero shot showing stock picks
2. **AI Copilot** - Chat interface with insights
3. **Symbol Detail** - Interactive chart view
4. **Price Alerts** - Alert management
5. **Trading Ideas** - Curated recommendations
6. **Watchlist** - Personalized stock list

### App Previews (Optional)

30-second videos showing key features:
- Resolution matches screenshot requirements
- Audio optional but recommended
- Can upload up to 3 per device size

---

## App Icon

### Requirements

| Specification | Value |
|---------------|-------|
| Size | 1024 x 1024 pixels |
| Format | PNG (no alpha) |
| Color Space | sRGB |
| Rounded Corners | Apple applies automatically |

### Design Guidelines

- Simple, recognizable design
- Works at small sizes
- No text (unless part of logo)
- Avoid photographs
- Use brand colors

---

## In-App Purchases (If Applicable)

### Setup IAP

If offering premium features:

1. Navigate to **Features > In-App Purchases**
2. Click **+** to create new IAP
3. Choose type:
   - Consumable
   - Non-Consumable
   - Auto-Renewable Subscription
   - Non-Renewing Subscription

### Subscription Example

| Field | Value |
|-------|-------|
| Reference Name | Technic Pro Monthly |
| Product ID | com.technic.pro.monthly |
| Price | $9.99/month |
| Subscription Group | Technic Pro |

---

## TestFlight Configuration

### Internal Testing

1. Go to **TestFlight > Internal Testing**
2. Create group: "Internal Team"
3. Add testers (up to 100)
4. Enable automatic distribution

### External Testing

1. Go to **TestFlight > External Testing**
2. Create group: "Beta Testers"
3. Add beta description and feedback email
4. Submit for Beta App Review

### Test Information

```
Test Description:
Help us test the latest version of Technic! We're looking for feedback on:
- Stock scanner functionality
- AI Copilot responses
- Price alert reliability
- Overall performance and stability

Please report any bugs or issues through the TestFlight feedback feature.

Feedback Email: beta@technic.app
```

---

## App Analytics

### Enable Analytics

1. Go to **Analytics**
2. Review available metrics:
   - Downloads
   - Sessions
   - Active Devices
   - Retention
   - Crashes

### Sales & Trends

Monitor:
- App Units
- In-App Purchases
- Proceeds
- Updates

---

## Submission Checklist

Before submitting:

- [ ] All app information complete
- [ ] Privacy declarations accurate
- [ ] Age rating questionnaire complete
- [ ] Pricing configured
- [ ] All territories selected
- [ ] Screenshots for all required sizes
- [ ] App icon uploaded
- [ ] Version description written
- [ ] Demo account credentials provided
- [ ] Contact information current
- [ ] Review notes explain key features
- [ ] Build selected and processed

---

## After Submission

### Review Timeline

- **Waiting for Review**: Queued for review
- **In Review**: Being reviewed (usually 24-48 hours)
- **Pending Developer Release**: Approved, waiting for release
- **Ready for Sale**: Live on App Store

### Handling Rejection

If rejected:
1. Read rejection reason carefully
2. Make required changes
3. Respond in Resolution Center
4. Resubmit for review

### Common Rejection Reasons

1. **Guideline 2.1** - App Completeness (crashes, bugs)
2. **Guideline 2.3** - Accurate Metadata (screenshots, description)
3. **Guideline 4.2** - Minimum Functionality
4. **Guideline 5.1** - Privacy (data collection disclosure)

---

## Post-Launch

### Monitor Performance

- Check crash reports daily
- Monitor user reviews
- Track analytics

### Respond to Reviews

- Reply to negative reviews professionally
- Thank users for positive feedback
- Address common issues in updates

### Plan Updates

- Regular bug fix releases
- Feature updates based on feedback
- iOS version compatibility updates
