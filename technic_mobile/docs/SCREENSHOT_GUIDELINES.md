# App Store Screenshot Guidelines

## Overview

This document provides specifications and guidelines for creating App Store screenshots for Technic.

---

## Required Screenshot Sizes

### iPhone Screenshots (Required)

| Display Size | Resolution | Device Examples |
|-------------|------------|-----------------|
| 6.7" | 1290 x 2796 | iPhone 15 Pro Max, 14 Pro Max |
| 6.5" | 1284 x 2778 | iPhone 15 Plus, 14 Plus |
| 6.1" | 1179 x 2556 | iPhone 15 Pro, 14 Pro |
| 5.5" | 1242 x 2208 | iPhone 8 Plus (legacy) |

**Note:** You can use 6.7" screenshots for all iPhone sizes (App Store will scale).

### iPad Screenshots (If supporting iPad)

| Display Size | Resolution | Device Examples |
|-------------|------------|-----------------|
| 12.9" | 2048 x 2732 | iPad Pro 12.9" |
| 11" | 1668 x 2388 | iPad Pro 11" |

---

## Screenshot Requirements

### Quantity
- **Minimum:** 3 screenshots per device size
- **Maximum:** 10 screenshots per device size
- **Recommended:** 5-6 screenshots

### Format
- **File Type:** PNG or JPEG
- **Color Space:** sRGB
- **Orientation:** Portrait (recommended) or Landscape

### Content Rules
- Must be actual screenshots from the App (no mockups)
- No alpha/transparency
- No excessive text overlay
- Device frames are optional
- Must represent actual App functionality

---

## Recommended Screenshot Sequence

### Screenshot 1: Scanner Results (Hero Shot)
**Purpose:** Show the core value proposition

**Content:**
- Scanner results screen with stock picks
- Multiple stocks visible with technical scores
- Show sector, volume, and momentum data
- Highlight one stock with good metrics

**Caption:** "Discover High-Potential Stocks"

**Technical Notes:**
- Ensure data looks realistic
- Use dark theme for consistency
- Show variety in stock sectors

---

### Screenshot 2: AI Copilot
**Purpose:** Showcase AI-powered insights

**Content:**
- Copilot conversation screen
- Show a question about market/stock
- Display intelligent, helpful response
- Include follow-up suggestions

**Caption:** "AI-Powered Market Insights"

**Technical Notes:**
- Show natural conversation flow
- Include specific stock mentions
- Display timestamp for realism

---

### Screenshot 3: Symbol Detail Chart
**Purpose:** Highlight charting capabilities

**Content:**
- Full symbol detail screen
- Interactive candlestick chart
- Key technical indicators visible
- Price, change, and volume data

**Caption:** "Interactive Technical Charts"

**Technical Notes:**
- Use a stock with clear trend
- Show recent price movement
- Include support/resistance if visible

---

### Screenshot 4: Price Alerts
**Purpose:** Demonstrate notification features

**Content:**
- Alerts list screen
- Show multiple active alerts
- Display different alert types
- Include triggered alert example

**Caption:** "Never Miss a Trade Setup"

**Technical Notes:**
- Mix of price above/below alerts
- Show different stocks
- Include alert status indicators

---

### Screenshot 5: Trading Ideas
**Purpose:** Show curated recommendations

**Content:**
- Ideas/recommendations screen
- Display trade idea cards
- Show entry price, target, stop
- Include sector and rationale

**Caption:** "Curated Daily Trade Ideas"

**Technical Notes:**
- Use realistic price levels
- Show variety of setups
- Include risk/reward info

---

### Screenshot 6: Watchlist
**Purpose:** Highlight personalization

**Content:**
- Watchlist screen
- User's saved stocks
- Quick price updates
- Easy access to details

**Caption:** "Your Personalized Watchlist"

**Technical Notes:**
- Show 5-8 stocks
- Mix of gainers/losers
- Include add button visibility

---

## Design Guidelines

### Visual Consistency
- Use dark theme throughout all screenshots
- Maintain consistent data/time across shots
- Use same device frame style (if using frames)
- Keep caption style uniform

### Caption Style
```
Font: SF Pro Display (Bold)
Size: 72-96pt
Color: White (#FFFFFF)
Position: Top of screenshot
Background: Subtle gradient or overlay
```

### Color Palette (for overlays/captions)
| Color | Hex | Use |
|-------|-----|-----|
| Technic Blue | #1E88E5 | Accents, highlights |
| Background Dark | #0A0E27 | Backgrounds |
| Text White | #FFFFFF | Captions |
| Success Green | #4CAF50 | Positive indicators |
| Alert Red | #F44336 | Negative indicators |

### Device Frames
- Optional but recommended for professional look
- Use official Apple device frames
- Match device to screenshot size
- Keep frame color consistent (Space Black recommended)

---

## Screenshot Capture Process

### 1. Prepare the App
```bash
# Run in release mode for best performance
flutter run --release
```

### 2. Set Up Test Data
- Populate watchlist with 5-8 stocks
- Create 3-4 sample alerts
- Have AI Copilot conversation ready
- Ensure scanner has recent results

### 3. Capture Screenshots

**On Simulator:**
```bash
# iPhone 15 Pro Max
xcrun simctl io booted screenshot screenshot1.png
```

**On Device:**
- Press Side Button + Volume Up simultaneously
- Screenshots save to Photos

### 4. Post-Processing
1. Crop to exact dimensions
2. Add captions and overlays
3. Apply device frames (optional)
4. Export as PNG (no compression)

---

## Localization Notes

### Localized Screenshots
If supporting multiple languages, create separate screenshot sets:
- Translate all captions
- Use localized app content
- Maintain same visual style

### Priority Languages
1. English (US) - Primary
2. English (UK)
3. Spanish
4. German
5. French
6. Japanese
7. Chinese (Simplified)

---

## Validation Checklist

Before submission, verify:

- [ ] All required sizes provided
- [ ] Minimum 3 screenshots per size
- [ ] Screenshots are actual app content
- [ ] No placeholder or test data visible
- [ ] Dark theme used consistently
- [ ] Captions are clear and readable
- [ ] No App Store guideline violations
- [ ] File sizes within limits (<10MB each)
- [ ] Correct aspect ratios
- [ ] sRGB color space

---

## Tools & Resources

### Recommended Tools
- **Figma/Sketch:** Add frames and captions
- **Rotato:** 3D device mockups
- **Screenshots Pro:** Automated screenshot creation
- **AppLaunchpad:** Screenshot generator

### Apple Resources
- [App Store Screenshot Specifications](https://developer.apple.com/help/app-store-connect/reference/screenshot-specifications)
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)

---

## Example File Structure

```
screenshots/
├── iphone_6.7/
│   ├── 01_scanner_results.png
│   ├── 02_ai_copilot.png
│   ├── 03_symbol_detail.png
│   ├── 04_price_alerts.png
│   ├── 05_trading_ideas.png
│   └── 06_watchlist.png
├── iphone_5.5/
│   └── (same structure)
├── ipad_12.9/
│   └── (same structure)
└── raw/
    └── (original captures)
```
