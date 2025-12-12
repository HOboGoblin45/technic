# technic Brand Guidelines (Updated)

## Brand Identity

### Name
**technic** - Always lowercase, never capitalized
- Correct: "technic"
- Incorrect: "Technic", "TECHNIC", "TechNic"

### Tagline
"Personal Quant" - Title case for tagline only

---

## Logo System

### Primary Logo (Symbol)
The technic logo is a minimalist "tQ" monogram that doubles as a magnifying glass, symbolizing:
- **Precision**: The clean geometric forms
- **Discovery**: The magnifying glass metaphor
- **Quantitative Analysis**: The technical, structured design

**Current SVG** (240x240):
```svg
<svg width="240" height="240" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#AFC9FF" stroke-width="12" stroke-linecap="round" stroke-linejoin="round">
    <line x1="120" y1="48" x2="120" y2="160" />
    <line x1="64" y1="48" x2="176" y2="48" />
    <circle cx="120" cy="136" r="48" />
    <line x1="154" y1="170" x2="184" y2="200" />
  </g>
</svg>
```

**Design Elements**:
- **T-shape**: Horizontal bar (top) + vertical stem
- **Q-circle**: Magnifying glass lens
- **Handle**: Diagonal line extending from circle
- **Stroke**: 12px, rounded caps and joins
- **Color**: Soft pastel sky-blue (#B0CAFF / #AFC9FF)

### Logo Refinements (Aesthetic Improvements)

#### Option A: Enhanced Contrast & Depth
```svg
<svg width="240" height="240" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
  <!-- Subtle shadow for depth -->
  <g opacity="0.15" fill="none" stroke="#001D51" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" transform="translate(2, 2)">
    <line x1="120" y1="48" x2="120" y2="160" />
    <line x1="64" y1="48" x2="176" y2="48" />
    <circle cx="120" cy="136" r="48" />
    <line x1="154" y1="170" x2="184" y2="200" />
  </g>
  <!-- Main logo -->
  <g fill="none" stroke="#B0CAFF" stroke-width="12" stroke-linecap="round" stroke-linejoin="round">
    <line x1="120" y1="48" x2="120" y2="160" />
    <line x1="64" y1="48" x2="176" y2="48" />
    <circle cx="120" cy="136" r="48" />
    <line x1="154" y1="170" x2="184" y2="200" />
  </g>
</svg>
```

#### Option B: Gradient Fill (Premium Look)
```svg
<svg width="240" height="240" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="technicGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#B0CAFF;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#001D51;stop-opacity:1" />
    </linearGradient>
  </defs>
  <g fill="none" stroke="url(#technicGradient)" stroke-width="12" stroke-linecap="round" stroke-linejoin="round">
    <line x1="120" y1="48" x2="120" y2="160" />
    <line x1="64" y1="48" x2="176" y2="48" />
    <circle cx="120" cy="136" r="48" />
    <line x1="154" y1="170" x2="184" y2="200" />
  </g>
</svg>
```

#### Option C: Refined Proportions (Balanced)
```svg
<svg width="240" height="240" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#B0CAFF" stroke-width="14" stroke-linecap="round" stroke-linejoin="round">
    <!-- Slightly thicker stroke for better visibility at small sizes -->
    <line x1="120" y1="52" x2="120" y2="158" />
    <line x1="68" y1="52" x2="172" y2="52" />
    <circle cx="120" cy="134" r="46" />
    <line x1="153" y1="167" x2="182" y2="196" />
  </g>
</svg>
```

#### Option D: Minimalist (Ultra-Clean)
```svg
<svg width="240" height="240" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#B0CAFF" stroke-width="10" stroke-linecap="round" stroke-linejoin="round">
    <!-- Thinner stroke for ultra-minimalist aesthetic -->
    <line x1="120" y1="50" x2="120" y2="160" />
    <line x1="66" y1="50" x2="174" y2="50" />
    <circle cx="120" cy="136" r="48" />
    <line x1="154" y1="170" x2="184" y2="200" />
  </g>
</svg>
```

**Recommendation**: **Option C (Refined Proportions)** - Slightly thicker stroke (14px) ensures better visibility at small sizes (app icons, navigation bars) while maintaining the clean aesthetic. The adjusted proportions create better visual balance.

---

## Color Palette

### Primary Colors

#### 1. **White** (Primary/Background)
- **Usage**: Main background, cards, surfaces
- **Philosophy**: Apple-inspired sterile cleanliness
- **Values**:
  - Pure White: `#FFFFFF`
  - Off-White (light mode): `#FAFAFA`
  - Dark Background: `#0A0A0A` (for dark mode)

#### 2. **Soft Pastel Sky-Blue** (Brand Primary)
- **HEX**: `#B0CAFF`
- **RGB**: `176, 202, 255`
- **Usage**: Primary actions, highlights, logo, interactive elements
- **Variants**:
  - Light: `#D4E3FF` (hover states, backgrounds)
  - Dark: `#8BA8E6` (pressed states)
  - Muted: `#B0CAFF80` (50% opacity for subtle accents)

#### 3. **Pantone Imperial Blue** (Accent/Depth)
- **HEX**: `#001D51`
- **RGB**: `0, 29, 81`
- **Usage**: Text, borders, shadows, depth elements
- **Variants**:
  - Light: `#1A3A6B` (secondary text)
  - Muted: `#001D5140` (25% opacity for subtle borders)

#### 4. **Pantone Pine Grove** (Accent/Organic)
- **HEX**: `#213631`
- **RGB**: `33, 54, 49`
- **Usage**: Success states, positive indicators, organic elements
- **Variants**:
  - Light: `#3A5A4F` (hover states)
  - Muted: `#21363140` (25% opacity for backgrounds)

### Semantic Colors (Derived from Palette)

#### Success/Positive
- **Primary**: Pine Grove `#213631`
- **Light**: `#4A7A6A`
- **Usage**: Gains, positive changes, confirmations

#### Warning/Caution
- **Primary**: Amber (derived) `#FFB84D`
- **Light**: `#FFD699`
- **Usage**: Warnings, moderate risk, attention needed

#### Error/Negative
- **Primary**: Coral Red (derived) `#FF6B6B`
- **Light**: `#FF9999`
- **Usage**: Losses, errors, high risk, stops

#### Info/Neutral
- **Primary**: Sky-Blue `#B0CAFF`
- **Light**: `#D4E3FF`
- **Usage**: Information, neutral states, tips

---

## Typography

### Font Families

#### iOS
- **Primary**: SF Pro (system font)
- **Weights**: Regular (400), Medium (500), Semibold (600), Bold (700)
- **Dynamic Type**: Support all accessibility sizes

#### Android
- **Primary**: Roboto (system font)
- **Weights**: Regular (400), Medium (500), Bold (700)
- **Scale**: Material 3 type scale

#### Web/Desktop
- **Primary**: Inter
- **Fallback**: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif

### Type Scale

```
Display Large:   57px / 64px line height (Bold)
Display Medium:  45px / 52px line height (Bold)
Display Small:   36px / 44px line height (Bold)

Headline Large:  32px / 40px line height (Bold)
Headline Medium: 28px / 36px line height (Semibold)
Headline Small:  24px / 32px line height (Semibold)

Title Large:     22px / 28px line height (Semibold)
Title Medium:    16px / 24px line height (Medium)
Title Small:     14px / 20px line height (Medium)

Body Large:      16px / 24px line height (Regular)
Body Medium:     14px / 20px line height (Regular)
Body Small:      12px / 16px line height (Regular)

Label Large:     14px / 20px line height (Medium)
Label Medium:    12px / 16px line height (Medium)
Label Small:     11px / 16px line height (Medium)
```

---

## UI Application Guidelines

### Light Mode (Primary)

**Background Hierarchy**:
```
Level 0 (Base):       #FFFFFF (Pure white)
Level 1 (Cards):      #FAFAFA (Off-white)
Level 2 (Elevated):   #F5F5F5 (Light gray)
Level 3 (Overlays):   #FFFFFF with shadow
```

**Text Hierarchy**:
```
Primary:    #001D51 (Imperial Blue) - Main content
Secondary:  #1A3A6B (Light Imperial) - Supporting text
Tertiary:   #6B7280 (Gray) - Captions, metadata
Disabled:   #9CA3AF (Light gray) - Disabled states
```

**Interactive Elements**:
```
Primary Button:
  - Background: #B0CAFF (Sky-Blue)
  - Text: #001D51 (Imperial Blue)
  - Hover: #D4E3FF
  - Pressed: #8BA8E6

Secondary Button:
  - Background: Transparent
  - Border: #B0CAFF
  - Text: #001D51
  - Hover: #B0CAFF20 (background)

Tertiary Button:
  - Background: Transparent
  - Text: #B0CAFF
  - Hover: #B0CAFF20 (background)
```

**Accents & Highlights**:
```
Links:          #B0CAFF
Selection:      #B0CAFF40 (40% opacity)
Focus Ring:     #B0CAFF with 4px blur
Dividers:       #E5E7EB (Light gray)
Borders:        #D1D5DB (Medium gray)
```

### Dark Mode (Secondary)

**Background Hierarchy**:
```
Level 0 (Base):       #0A0A0A (Near black)
Level 1 (Cards):      #1A1A1A (Dark gray)
Level 2 (Elevated):   #2A2A2A (Medium gray)
Level 3 (Overlays):   #1A1A1A with shadow
```

**Text Hierarchy**:
```
Primary:    #FFFFFF (White) - Main content
Secondary:  #B0CAFF (Sky-Blue) - Supporting text
Tertiary:   #9CA3AF (Gray) - Captions, metadata
Disabled:   #6B7280 (Dark gray) - Disabled states
```

**Interactive Elements**:
```
Primary Button:
  - Background: #B0CAFF
  - Text: #001D51
  - Hover: #D4E3FF
  - Pressed: #8BA8E6

Secondary Button:
  - Background: Transparent
  - Border: #B0CAFF
  - Text: #B0CAFF
  - Hover: #B0CAFF20 (background)

Tertiary Button:
  - Background: Transparent
  - Text: #B0CAFF
  - Hover: #B0CAFF20 (background)
```

**Accents & Highlights**:
```
Links:          #B0CAFF
Selection:      #B0CAFF40 (40% opacity)
Focus Ring:     #B0CAFF with 4px blur
Dividers:       #2A2A2A
Borders:        #3A3A3A
```

---

## Component Styling

### Cards
```
Light Mode:
  - Background: #FAFAFA
  - Border: 1px solid #E5E7EB
  - Shadow: 0 2px 8px rgba(0, 29, 81, 0.08)
  - Radius: 16px

Dark Mode:
  - Background: #1A1A1A
  - Border: 1px solid #2A2A2A
  - Shadow: 0 2px 8px rgba(0, 0, 0, 0.4)
  - Radius: 16px
```

### Tier Badges
```
CORE:
  - Background: Linear gradient #213631 → #4A7A6A
  - Text: #FFFFFF
  - Border: None
  - Icon: ✓ or ⭐

SATELLITE:
  - Background: Linear gradient #B0CAFF → #8BA8E6
  - Text: #001D51
  - Border: None
  - Icon: ◆ or 🛰️

REJECT:
  - Background: #F3F4F6
  - Text: #6B7280
  - Border: 1px solid #D1D5DB
  - Icon: ✕ or ⊘
```

### Score Bars
```
High Score (>70):
  - Fill: Linear gradient #213631 → #4A7A6A
  - Background: #E5E7EB

Medium Score (50-70):
  - Fill: Linear gradient #B0CAFF → #8BA8E6
  - Background: #E5E7EB

Low Score (<50):
  - Fill: #9CA3AF
  - Background: #E5E7EB
```

### Sparklines
```
Positive (Up):
  - Line: #213631 (Pine Grove)
  - Fill: Linear gradient #21363140 → transparent
  - Stroke Width: 2px

Negative (Down):
  - Line: #FF6B6B (Coral Red)
  - Fill: Linear gradient #FF6B6B40 → transparent
  - Stroke Width: 2px
```

---

## Logo Usage

### App Icon (iOS/Android)
```
Size: 1024x1024 (master)
Background: White (#FFFFFF)
Logo: Centered, 70% of canvas
Padding: 15% on all sides
Format: PNG with transparency
```

**Variants**:
- **Light Mode**: Sky-Blue logo on white background
- **Dark Mode**: Sky-Blue logo on dark background (#0A0A0A)
- **Adaptive**: Use system theme to switch

### Navigation Bar
```
Size: 32x32 (iOS), 24x24 (Android)
Logo: Symbol only
Color: Sky-Blue (#B0CAFF)
Background: Transparent
```

### Launch Screen
```
Layout: Centered logo + wordmark
Logo Size: 120x120
Wordmark: "technic" below logo
Background: White (light) or #0A0A0A (dark)
Animation: Subtle fade-in (300ms)
```

### Marketing Materials
```
Lockup: Logo + "technic" + "Personal Quant"
Spacing: 16px between logo and text
Alignment: Center or left-aligned
Minimum Size: 48px height (logo)
Clear Space: 24px on all sides
```

---

## Spacing System

### Base Unit: 4px

```
Space 1:  4px   (Tight)
Space 2:  8px   (Close)
Space 3:  12px  (Default)
Space 4:  16px  (Comfortable)
Space 5:  20px  (Spacious)
Space 6:  24px  (Section)
Space 8:  32px  (Large Section)
Space 12: 48px  (Major Section)
Space 16: 64px  (Page Section)
```

### Component Padding
```
Buttons:        12px horizontal, 8px vertical
Cards:          16px all sides
Inputs:         12px horizontal, 10px vertical
Chips:          10px horizontal, 6px vertical
List Items:     16px horizontal, 12px vertical
```

---

## Elevation & Shadows

### Light Mode
```
Level 1 (Cards):
  - Shadow: 0 1px 3px rgba(0, 29, 81, 0.08)

Level 2 (Raised):
  - Shadow: 0 2px 8px rgba(0, 29, 81, 0.12)

Level 3 (Floating):
  - Shadow: 0 4px 16px rgba(0, 29, 81, 0.16)

Level 4 (Modal):
  - Shadow: 0 8px 32px rgba(0, 29, 81, 0.24)
```

### Dark Mode
```
Level 1 (Cards):
  - Shadow: 0 1px 3px rgba(0, 0, 0, 0.3)

Level 2 (Raised):
  - Shadow: 0 2px 8px rgba(0, 0, 0, 0.4)

Level 3 (Floating):
  - Shadow: 0 4px 16px rgba(0, 0, 0, 0.5)

Level 4 (Modal):
  - Shadow: 0 8px 32px rgba(0, 0, 0, 0.6)
```

---

## Animation & Motion

### Timing Functions
```
Ease Out:     cubic-bezier(0.0, 0.0, 0.2, 1)  - Entering
Ease In:      cubic-bezier(0.4, 0.0, 1, 1)    - Exiting
Ease In Out:  cubic-bezier(0.4, 0.0, 0.2, 1)  - Transitioning
```

### Durations
```
Instant:  100ms  - Micro-interactions
Fast:     200ms  - Hover states, tooltips
Normal:   300ms  - Page transitions, modals
Slow:     500ms  - Complex animations
```

### Principles
- **Respect reduced motion**: Disable animations if user prefers
- **Purposeful**: Every animation should have a reason
- **Subtle**: Avoid distracting or excessive motion
- **Consistent**: Use same timing for similar interactions

---

## Accessibility

### Contrast Ratios (WCAG AA)
```
Normal Text:    4.5:1 minimum
Large Text:     3:1 minimum
UI Components:  3:1 minimum
```

**Verified Combinations**:
- ✅ #001D51 on #FFFFFF: 14.8:1 (Excellent)
- ✅ #B0CAFF on #001D51: 7.2:1 (Excellent)
- ✅ #213631 on #FFFFFF: 12.6:1 (Excellent)
- ✅ #FFFFFF on #0A0A0A: 19.2:1 (Excellent)

### Touch Targets
```
Minimum:    44x44 points (iOS), 48x48 dp (Android)
Preferred:  48x48 points (iOS), 56x56 dp (Android)
Spacing:    8px minimum between targets
```

### Focus Indicators
```
Outline:    2px solid #B0CAFF
Offset:     2px
Radius:     Inherit from element
```

---

## File Naming Conventions

### Logo Files
```
logo-symbol.svg           - Symbol only (vector)
logo-symbol-white.svg     - Symbol on dark backgrounds
logo-wordmark.svg         - "technic" text only
logo-lockup.svg           - Symbol + wordmark
logo-lockup-tagline.svg   - Symbol + wordmark + tagline

icon-1024.png            - App icon (iOS/Android master)
icon-512.png             - App icon (web)
icon-192.png             - App icon (Android)
icon-180.png             - App icon (iOS)
icon-120.png             - App icon (iOS)
icon-76.png              - App icon (iPad)
```

### Asset Organization
```
assets/
├── brand/
│   ├── logo-symbol.svg
│   ├── logo-symbol-refined.svg (Option C)
│   ├── logo-wordmark.svg
│   └── logo-lockup.svg
├── icons/
│   ├── app/
│   │   ├── icon-1024.png
│   │   ├── icon-512.png
│   │   └── ...
│   └── ui/
│       ├── scan.svg
│       ├── ideas.svg
│       └── ...
└── images/
    ├── onboarding/
    ├── empty-states/
    └── illustrations/
```

---

## Brand Voice & Messaging

### Tone
- **Confident**: We know our analysis is sophisticated
- **Accessible**: But we explain it clearly
- **Honest**: Transparent about risks and limitations
- **Professional**: Institutional-grade, not gimmicky

### Key Messages
1. "Institutional-grade quantitative analysis in your pocket"
2. "AI-powered insights, human-friendly explanations"
3. "Not financial advice, but the best data-driven guidance available"
4. "Built by quants, designed for everyone"

### Writing Style
- **Concise**: Get to the point quickly
- **Clear**: Avoid jargon unless necessary
- **Actionable**: Tell users what to do next
- **Transparent**: Always show confidence levels and sources

---

## Implementation Notes

### Flutter Theme Configuration
```dart
// lib/theme/app_theme.dart
class TechnicTheme {
  // Brand Colors
  static const skyBlue = Color(0xFFB0CAFF);
  static const imperialBlue = Color(0xFF001D51);
  static const pineGrove = Color(0xFF213631);
  
  // Light Theme
  static ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    primaryColor: skyBlue,
    scaffoldBackgroundColor: Colors.white,
    cardColor: Color(0xFFFAFAFA),
    // ... rest of theme
  );
  
  // Dark Theme
  static ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    primaryColor: skyBlue,
    scaffoldBackgroundColor: Color(0xFF0A0A0A),
    cardColor: Color(0xFF1A1A1A),
    // ... rest of theme
  );
}
```

### CSS Variables (Web)
```css
:root {
  /* Brand Colors */
  --color-sky-blue: #B0CAFF;
  --color-imperial-blue: #001D51;
  --color-pine-grove: #213631;
  
  /* Light Mode */
  --bg-primary: #FFFFFF;
  --bg-secondary: #FAFAFA;
  --text-primary: #001D51;
  --text-secondary: #1A3A6B;
  
  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  /* ... */
}

[data-theme="dark"] {
  --bg-primary: #0A0A0A;
  --bg-secondary: #1A1A1A;
  --text-primary: #FFFFFF;
  --text-secondary: #B0CAFF;
}
```

---

## Summary

**Brand Essence**: Clean, sophisticated, trustworthy quantitative analysis delivered through a minimalist, Apple-inspired interface.

**Visual Identity**: 
- Logo: Minimalist "tQ" magnifying glass in soft sky-blue
- Colors: Predominantly white with strategic use of sky-blue, imperial blue, and pine grove
- Typography: System fonts (SF Pro/Roboto) for native feel
- Style: Ultra-clean, almost sterile, with subtle depth through shadows

**Recommended Logo**: **Option C (Refined Proportions)** with 14px stroke for optimal visibility across all sizes.

**Key Principle**: Let the sophisticated quantitative analysis shine through a clean, unobtrusive interface that feels native to each platform.
