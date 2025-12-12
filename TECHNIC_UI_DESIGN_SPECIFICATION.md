# Technic UI Design Specification
## Inspired by Best-in-Class Finance Apps

Based on analysis of: Robinhood, Webull, Copilot Money, Trading 212, and premium finance apps.

---

## Design Philosophy

**Goal**: Surpass the reference apps in cleanliness and refinement while maintaining Technic's sophisticated quantitative edge.

**Core Principles**:
1. **Ultra-Minimal**: Less is more - every pixel serves a purpose
2. **Data-First**: Information hierarchy optimized for quick decisions
3. **Institutional Trust**: Professional, reliable, sophisticated
4. **Effortless UX**: Zero learning curve, intuitive interactions
5. **Performance**: Instant feedback, smooth animations, no lag

---

## Key Observations from Reference Apps

### What Works (To Adopt)

#### 1. **Robinhood** - Simplicity Master
- ✅ **Clean typography**: Large, bold numbers for prices
- ✅ **Minimal chrome**: No unnecessary UI elements
- ✅ **Color discipline**: Green/red only for gains/losses
- ✅ **Flat design**: No gradients, minimal shadows
- ✅ **Generous whitespace**: Breathing room between elements
- ✅ **Clear hierarchy**: Price → Chart → Details

#### 2. **Webull** - Professional Trading
- ✅ **Dark theme mastery**: Deep blacks, subtle grays
- ✅ **Data density**: Lots of info without clutter
- ✅ **Chart prominence**: Large, interactive charts
- ✅ **Tabbed navigation**: Clean organization (Chart/Options/News)
- ✅ **Pill-shaped buttons**: Modern, clean action buttons

#### 3. **Copilot Money** - Modern Fintech
- ✅ **Card-based layout**: Clean separation of content
- ✅ **Subtle shadows**: Depth without heaviness
- ✅ **Rounded corners**: 12-16px radius for modern feel
- ✅ **Icon consistency**: Monochrome, simple icons
- ✅ **Color accents**: Single accent color used sparingly

#### 4. **Trading 212** - Clean Information
- ✅ **List efficiency**: Compact, scannable lists
- ✅ **Inline sparklines**: Quick visual reference
- ✅ **Pill badges**: Clean status indicators
- ✅ **Search prominence**: Easy discovery
- ✅ **White backgrounds**: Clean, professional (light mode)

### What to Avoid (Current Technic Issues)

- ❌ **Neon colors**: Lime green (#B6FF3B), bright yellows
- ❌ **Emoji icons**: 🎯, 💡, 🚀 - unprofessional
- ❌ **Heavy gradients**: Multiple color transitions
- ❌ **Playful elements**: "Live" badges, decorative graphics
- ❌ **Inconsistent spacing**: Random padding/margins
- ❌ **Over-designed cards**: Too many shadows/borders
- ❌ **Cluttered layouts**: Too much happening at once

---

## Technic Design System v2.0

### Color Palette

#### Dark Theme (Primary)
```
Background Hierarchy:
- App Background:    #0A0E27 (deep navy, almost black)
- Card Background:   #141B2D (slate-900 equivalent)
- Card Elevated:     #1A2332 (subtle lift)
- Borders:           #2D3748 (slate-700, very subtle)

Text:
- Primary:           #F7FAFC (slate-50, high contrast)
- Secondary:         #A0AEC0 (slate-400, readable)
- Tertiary:          #718096 (slate-500, de-emphasized)

Accent Colors:
- Primary Blue:      #3B82F6 (blue-500, trust/action)
- Success Green:     #10B981 (emerald-500, NOT neon)
- Danger Red:        #EF4444 (red-500, losses/stops)
- Warning Amber:     #F59E0B (amber-500, caution)
- Info Teal:         #14B8A6 (teal-500, neutral info)

Chart Colors:
- Bullish Candle:    #10B981 (muted green)
- Bearish Candle:    #EF4444 (muted red)
- Line Chart:        #3B82F6 (primary blue)
- Volume Bars:       #4B5563 (gray-600, subtle)
```

#### Light Theme (Secondary)
```
Background Hierarchy:
- App Background:    #F8FAFC (slate-50)
- Card Background:   #FFFFFF (pure white)
- Card Elevated:     #F1F5F9 (slate-100)
- Borders:           #E2E8F0 (slate-200)

Text:
- Primary:           #1E293B (slate-800)
- Secondary:         #475569 (slate-600)
- Tertiary:          #94A3B8 (slate-400)

(Same accent colors as dark theme)
```

### Typography

#### Font Family
```
Primary: SF Pro (iOS) / Roboto (Android) / Inter (Web)
Monospace: SF Mono / Roboto Mono (for prices/numbers)
```

#### Type Scale
```
Display (Prices):    32px / 700 weight / -0.5px tracking
Heading 1:           24px / 700 weight / -0.25px tracking
Heading 2:           20px / 600 weight / normal tracking
Heading 3:           18px / 600 weight / normal tracking
Body Large:          16px / 400 weight / normal tracking
Body:                14px / 400 weight / normal tracking
Caption:             12px / 400 weight / normal tracking
Label:               11px / 500 weight / 0.5px tracking (uppercase)
```

#### Number Formatting
```
Prices:              Monospace, tabular figures
Percentages:         +5.25% (with + for gains)
Large Numbers:       $1.2M (abbreviated with suffix)
```

### Spacing System

```
4px Grid System:
- xs:  4px   (tight spacing, inline elements)
- sm:  8px   (compact spacing, related items)
- md:  12px  (default spacing, list items)
- lg:  16px  (section spacing, card padding)
- xl:  24px  (major sections, screen padding)
- 2xl: 32px  (screen-level spacing)
- 3xl: 48px  (hero sections)
```

### Component Specifications

#### Cards
```
Style: Flat with subtle border
Background: Card background color
Border: 1px solid border color
Border Radius: 12px
Padding: 16px
Shadow: None (or 0 2px 4px rgba(0,0,0,0.05) for light theme)
Spacing: 12px between cards
```

#### Buttons

**Primary Button**:
```
Background: Primary blue (#3B82F6)
Text: White
Height: 44px (minimum touch target)
Border Radius: 12px
Font: 16px / 600 weight
Padding: 12px 24px
Shadow: None
Hover: Darken 10%
Active: Darken 20%
```

**Secondary Button**:
```
Background: Transparent
Border: 1px solid border color
Text: Primary text color
(Same dimensions as primary)
```

**Text Button**:
```
Background: Transparent
Text: Primary blue
No border, no padding
Underline on hover
```

#### Icons
```
Size: 20px or 24px (consistent)
Style: Outline (not filled)
Color: Secondary text color
Source: SF Symbols (iOS) / Material Icons (Android)
NO emoji, NO playful graphics
```

#### Badges/Pills
```
Height: 24px
Border Radius: 12px (fully rounded)
Padding: 6px 12px
Font: 12px / 500 weight
Background: Accent color at 10% opacity
Text: Accent color at 100%
Example: "BUY" badge = green bg 10%, green text 100%
```

#### Charts
```
Background: Transparent
Grid Lines: Border color at 20% opacity
Axis Labels: Tertiary text color, 11px
Candlesticks: 
  - Bullish: Success green
  - Bearish: Danger red
  - Wick: Same color at 60% opacity
Line Charts: Primary blue, 2px width
Area Fill: Primary blue at 10% opacity (gradient to 0%)
```

#### Lists
```
Item Height: 64px (comfortable tap target)
Padding: 12px 16px
Separator: 1px border color
Hover: Card elevated background
Active: Darken 5%
```

---

## Screen-by-Screen Specifications

### Scanner Page

#### Layout Structure
```
┌─────────────────────────────────────┐
│ [Technic Logo]          [Settings] │ ← Header (56px)
├─────────────────────────────────────┤
│                                     │
│  Risk Profile Pills                 │ ← 48px height
│  [Conservative] [Moderate] [Aggr.]  │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ [Scan] Button               │   │ ← Prominent, 56px
│  └─────────────────────────────┘   │
│                                     │
│  Last scanned: 2 minutes ago        │ ← Caption text
│                                     │
│  ┌─────────────────────────────┐   │
│  │ AAPL                    BUY │   │
│  │ $175.43  +2.5%              │   │
│  │ [Sparkline]                 │   │
│  │ Entry: $174 • Target: $182  │   │
│  └─────────────────────────────┘   │
│                                     │
│  [More results...]                  │
│                                     │
└─────────────────────────────────────┘
```

#### Key Changes
- **Remove**: "Live" indicator, emoji icons, onboarding card (after first use)
- **Add**: Manual "Scan" button (primary, prominent)
- **Simplify**: Risk profile pills (flat, no gradients)
- **Clean**: Result cards (minimal, data-first)
- **Persist**: Results stay when switching tabs

### Ideas Page

#### Layout Structure
```
┌─────────────────────────────────────┐
│ Ideas                    [Filter]   │ ← Header
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │ MSFT                        │   │
│  │ Momentum Breakout           │   │ ← Strategy name
│  │                             │   │
│  │ [Large Sparkline Chart]     │   │ ← Prominent
│  │                             │   │
│  │ Entry: $380 • R/R: 3.2x     │   │
│  │                             │   │
│  │ [Ask Copilot] [Save]        │   │ ← Action buttons
│  └─────────────────────────────┘   │
│                                     │
│  [Swipe for next idea]              │
│                                     │
└─────────────────────────────────────┘
```

#### Key Changes
- **Card Stack**: One idea at a time, swipeable
- **Larger Charts**: Make sparklines more prominent
- **Clear Actions**: Obvious buttons for Copilot/Save
- **Strategy Labels**: Show "why" this is an idea

### Symbol Detail Page

#### Layout Structure
```
┌─────────────────────────────────────┐
│ [←] AAPL                    [Star]  │ ← Navigation
├─────────────────────────────────────┤
│                                     │
│  $175.43                            │ ← Large price
│  +$4.32 (+2.54%) Today              │ ← Change
│                                     │
│  ┌─────────────────────────────┐   │
│  │                             │   │
│  │   [Interactive Chart]       │   │ ← Full-width
│  │                             │   │
│  └─────────────────────────────┘   │
│                                     │
│  [1D] [1W] [1M] [3M] [1Y] [ALL]     │ ← Time periods
│                                     │
│  Trade Plan                         │ ← Section header
│  Entry: $174.00                     │
│  Stop:  $170.50                     │
│  Target: $182.00                    │
│  R/R: 3.2x                          │
│                                     │
│  Metrics                            │
│  Tech Rating: 8.5/10                │
│  Win Prob: 68%                      │
│  ICS: Core (9.2)                    │
│                                     │
│  [Ask Copilot] [View Options]       │ ← Actions
│                                     │
└─────────────────────────────────────┘
```

#### Key Changes
- **Robinhood-style**: Price → Chart → Details hierarchy
- **Clean Metrics**: Simple key/value pairs, no fancy cards
- **Action Buttons**: Bottom, clear, accessible

### Copilot Page

#### Layout Structure
```
┌─────────────────────────────────────┐
│ Copilot                             │ ← Simple header
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Analyzing AAPL...           │   │ ← System message
│  └─────────────────────────────┘   │
│                                     │
│           ┌─────────────────────┐   │
│           │ What's the outlook? │   │ ← User (right)
│           └─────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ AAPL shows strong momentum  │   │ ← Assistant (left)
│  │ with support at $170...     │   │
│  └─────────────────────────────┘   │
│                                     │
│  [Suggested prompts...]             │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Type your question...       │   │ ← Input
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
```

#### Key Changes
- **iMessage-style**: Clean bubble chat
- **Subtle Differentiation**: User vs Assistant bubbles
- **Suggested Prompts**: Help users get started
- **No Decorations**: Just clean, functional chat

### Settings Page

#### Layout Structure
```
┌─────────────────────────────────────┐
│ Settings                            │
├─────────────────────────────────────┤
│                                     │
│  Profile                            │ ← Section
│  ┌─────────────────────────────┐   │
│  │ [Avatar] John Doe           │   │
│  │ john@example.com            │   │
│  └─────────────────────────────┘   │
│                                     │
│  Preferences                        │
│  Theme              [Dark ▼]        │ ← Dropdown
│  Options Mode       [Both ▼]        │
│                                     │
│  Notifications                      │
│  Alerts             [Toggle]        │
│  Refresh Rate       [1m ▼]          │
│                                     │
│  About                              │
│  Version 1.0.0                      │
│  Data Sources                       │
│  Privacy Policy                     │
│                                     │
└─────────────────────────────────────┘
```

#### Key Changes
- **Grouped Lists**: iOS Settings-style
- **No Hero Banners**: Simple, functional
- **No Badges**: Just clean text and toggles
- **Minimal**: Only essential settings

---

## Animation & Interaction

### Principles
- **Fast**: 200-300ms max for transitions
- **Subtle**: Ease-in-out curves, no bouncing
- **Purposeful**: Animations guide attention
- **Smooth**: 60fps minimum, no jank

### Specific Animations
```
Page Transitions:     300ms ease-in-out, slide
Button Press:         100ms scale(0.95)
Card Tap:             200ms background fade
List Item Swipe:      250ms ease-out
Chart Updates:        400ms ease-in-out
Loading Spinner:      Subtle, small, centered
Pull to Refresh:      Native platform behavior
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. ✅ New color system
2. ✅ Typography scale
3. ✅ Spacing system
4. ✅ Remove all neon colors
5. ✅ Remove all emoji icons

### Phase 2: Components (Week 2)
1. ✅ Redesign cards (flat, minimal)
2. ✅ Redesign buttons (clean, accessible)
3. ✅ Redesign badges (pill-shaped, subtle)
4. ✅ Update icons (monochrome, consistent)
5. ✅ Chart styling (professional colors)

### Phase 3: Screens (Week 3)
1. ✅ Scanner page overhaul
2. ✅ Ideas page redesign
3. ✅ Symbol detail refinement
4. ✅ Copilot chat cleanup
5. ✅ Settings simplification

### Phase 4: Polish (Week 4)
1. ✅ Animation tuning
2. ✅ Accessibility audit
3. ✅ Performance optimization
4. ✅ User testing
5. ✅ Final refinements

---

## Success Metrics

### Quantitative
- [ ] Flutter analyze: 0 errors, 0 warnings
- [ ] 60fps on all screens
- [ ] < 100ms interaction response time
- [ ] WCAG AA accessibility compliance
- [ ] < 50MB app size

### Qualitative
- [ ] "Cleaner than Robinhood" - User feedback
- [ ] "Looks like a billion-dollar app" - Investor ready
- [ ] "Easiest trading app I've used" - Simplicity test
- [ ] "Feels professional and trustworthy" - Brand perception
- [ ] "I want to use this every day" - Engagement

---

## Design Checklist

Before considering any screen "done":

- [ ] No neon colors (lime green, bright yellow, etc.)
- [ ] No emoji icons or playful graphics
- [ ] No heavy gradients or shadows
- [ ] Consistent spacing (4px grid)
- [ ] Proper typography hierarchy
- [ ] Accessible contrast ratios (4.5:1 minimum)
- [ ] Touch targets ≥ 44x44px
- [ ] Smooth animations (60fps)
- [ ] Works in light AND dark mode
- [ ] Looks good on all screen sizes
- [ ] Passes "show to investor" test

---

## Conclusion

**Target**: Surpass Robinhood, Webull, and Copilot in cleanliness and refinement.

**Approach**: Ultra-minimal, data-first, institutional trust.

**Timeline**: 4 weeks to transform from "50% there" to "best-in-class".

**Outcome**: A finance app so clean and refined that it sets a new standard for the industry.

---

*This specification will guide all UI/UX decisions for Technic v1.0 and beyond.*
