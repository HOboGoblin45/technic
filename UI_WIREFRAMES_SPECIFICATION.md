# Technic UI/UX Specification: World-Class Mobile Experience

## Design Philosophy

**Goal**: Create an institutional-grade trading app that feels as polished as Robinhood but with the sophistication of Bloomberg Terminal, optimized for mobile-first usage.

**Core Principles**:
1. **Clarity over Complexity**: Surface sophisticated features through progressive disclosure
2. **Speed over Perfection**: Optimize for quick decision-making (< 3 taps to any action)
3. **Trust through Transparency**: Always show confidence levels, data sources, and disclaimers
4. **Platform Native**: Feels like it belongs on iOS/Android, not a web wrapper

---

## Screen-by-Screen Specification

### **1. Scanner Page - The Command Center**

#### Layout Structure
```
┌─────────────────────────────────────┐
│ ⚡ technic          [Live] [Profile]│ ← Header (70px)
├─────────────────────────────────────┤
│                                     │
│ 🔍 Search: AAPL, NVDA, tech...     │ ← Search Bar (56px)
│                                     │
├─────────────────────────────────────┤
│ 📊 Quick Scan                       │
│ ┌─────────┬─────────┬─────────┐   │
│ │ Risk:   │ Horizon:│ Options:│   │ ← Quick Filters (120px)
│ │Balanced │ Swing   │ Stock+  │   │
│ └─────────┴─────────┴─────────┘   │
│ [Run Scan] [Advanced ▼]            │
├─────────────────────────────────────┤
│                                     │
│ 💎 CORE PICKS (12)                 │ ← Tier Section
│                                     │
│ ┌─────────────────────────────────┐│
│ │ NVDA  [CORE] ICS: 87/100       ││
│ │ ▲ Breakout Long  •  Win: 68%   ││ ← Result Card
│ │ ▁▂▃▅▆█ Entry: 488 → Target: 520││   (140px each)
│ │ [Ask Copilot] [Options ▼]      ││
│ └─────────────────────────────────┘│
│                                     │
│ ┌─────────────────────────────────┐│
│ │ AAPL  [CORE] ICS: 82/100       ││
│ │ ▲ Momentum Swing  •  Win: 64%  ││
│ │ ▁▃▄▅▆█ Entry: 178 → Target: 186││
│ │ [Ask Copilot] [Options ▼]      ││
│ └─────────────────────────────────┘│
│                                     │
│ 🛰️ SATELLITE (8)                   │
│ [Show More ▼]                      │
│                                     │
├─────────────────────────────────────┤
│ [Scan] [Ideas] [Copilot] [★] [⚙️] │ ← Bottom Nav (70px)
└─────────────────────────────────────┘
```

#### Component Specifications

**A. Search Bar**
- **Height**: 56px
- **Style**: Rounded (14px radius), subtle shadow
- **Placeholder**: "Search: AAPL, NVDA, tech stocks..."
- **Autocomplete**: Dropdown with recent + trending symbols
- **Voice Input**: Microphone icon (iOS only)
- **Behavior**: 
  - Tap → Focus + show keyboard
  - Type → Real-time autocomplete (debounced 300ms)
  - Select → Navigate to Symbol Detail page

**B. Quick Filters Panel**
- **Height**: 120px (collapsed), 400px (expanded)
- **Layout**: 3 columns on phone, 5 columns on tablet
- **Filters**:
  1. **Risk Profile**: Conservative / Balanced / Aggressive
  2. **Time Horizon**: Short-term / Swing / Position
  3. **Options Mode**: Stock Only / Stock + Options
  4. **Sectors**: Multi-select chips (expandable)
  5. **Advanced**: Lookback days, min rating, max symbols
- **Behavior**:
  - Default: Show top 3 filters only
  - Tap "Advanced ▼" → Expand to show all filters
  - Changes auto-save to user profile
  - "Run Scan" button: Primary CTA (always visible)

**C. Result Card (Core/Satellite)**
- **Height**: 140px (collapsed), 280px (expanded with options)
- **Layout**:
  ```
  ┌─────────────────────────────────────┐
  │ NVDA          [CORE]  ICS: 87/100  │ ← Header Row
  │ ▲ Breakout Long  •  Win: 68%       │ ← Signal Row
  │ ▁▂▃▅▆█ (sparkline)                 │ ← Chart Row
  │ Entry: 488  Stop: 472  Target: 520 │ ← Trade Plan Row
  │ [Ask Copilot] [Options ▼] [★]      │ ← Action Row
  └─────────────────────────────────────┘
  ```
- **Tier Badge**:
  - CORE: Green gradient (#B6FF3B → #9EF01A)
  - SATELLITE: Blue gradient (#5EEAD4 → #99BFFF)
  - Size: 60px × 24px, rounded pill
- **ICS Score**: 
  - Format: "87/100"
  - Color: Green (>80), Yellow (65-80), Gray (<65)
  - Tap → Show ICS breakdown tooltip
- **Win Probability**:
  - Format: "Win: 68%"
  - Color: Green (>60%), Yellow (50-60%), Gray (<50%)
  - Tap → Show confidence interval
- **Sparkline**:
  - Height: 30px
  - Data: Last 90 days
  - Color: Green (up), Red (down)
  - Tap → Navigate to Symbol Detail
- **Action Buttons**:
  - "Ask Copilot": Opens Copilot with pre-filled question
  - "Options ▼": Expands to show options strategies
  - "★": Add to watchlist (filled if already saved)

**D. Options Expansion (when tapped)**
```
┌─────────────────────────────────────┐
│ 📈 Options Strategies (3)           │
│                                     │
│ ✓ Call Spread $490/$500            │ ← Best Strategy
│   Exp: 30 DTE  •  Delta: 0.65      │
│   Max Profit: $800  •  Risk: $200  │
│   Sweetness: 85/100                │
│   [View Details]                    │
│                                     │
│ • Call $495                         │ ← Alternative
│   Exp: 30 DTE  •  Delta: 0.70      │
│   [View Details]                    │
│                                     │
│ ⚠️ High IV (85th percentile)        │ ← Risk Warning
│   Consider defined-risk spreads     │
└─────────────────────────────────────┘
```

#### Interaction Patterns

**Gestures**:
- **Tap Card**: Navigate to Symbol Detail page
- **Long Press Card**: Quick actions menu (Save, Share, Dismiss)
- **Swipe Left**: Dismiss from results
- **Swipe Right**: Add to watchlist
- **Pull Down**: Refresh scan
- **Scroll Up**: Load more results (infinite scroll)

**States**:
- **Loading**: Skeleton cards with shimmer animation
- **Empty**: Illustration + "No results found. Try adjusting filters."
- **Error**: Error message + "Retry" button
- **Offline**: Cached results + "Showing cached data" banner

---

### **2. Symbol Detail Page - Deep Dive**

#### Layout Structure
```
┌─────────────────────────────────────┐
│ ← NVDA                    [★] [⋮]  │ ← Header
├─────────────────────────────────────┤
│                                     │
│ $488.50  ▲ +2.3% (+$10.95)         │ ← Price Header
│ Updated 2 min ago                   │
│                                     │
├─────────────────────────────────────┤
│ [1D] [5D] [1M] [3M] [6M] [1Y] [ALL]│ ← Chart Timeframe
│                                     │
│     ┌─────────────────────────┐    │
│ 500 │         ╱╲    ╱╲        │    │
│     │        ╱  ╲  ╱  ╲       │    │ ← Price Chart
│ 480 │    ╱╲╱    ╲╱    ╲      │    │   (200px)
│     │   ╱                ╲    │    │
│ 460 │  ╱                  ╲   │    │
│     └─────────────────────────┘    │
│     Jan    Feb    Mar    Apr       │
│                                     │
├─────────────────────────────────────┤
│ 📊 Technic Scores                   │
│                                     │
│ ┌──────────┬──────────┬──────────┐ │
│ │TechRating│   ICS    │ Quality  │ │ ← Score Grid
│ │   82     │  87/100  │   7.8    │ │   (100px)
│ │  ████░   │  █████░  │  ████░   │ │
│ └──────────┴──────────┴──────────┘ │
│                                     │
│ ┌──────────┬──────────┬──────────┐ │
│ │Win Prob  │   ATR%   │ Momentum │ │
│ │   68%    │   3.2%   │   +18%   │ │
│ │  ████░   │  ███░░   │  █████░  │ │
│ └──────────┴──────────┴──────────┘ │
│                                     │
├─────────────────────────────────────┤
│ 🎯 Trade Setup                      │
│                                     │
│ Signal: Breakout Long               │
│ Entry: $488  •  Stop: $472          │
│ Target: $520  •  R:R 2.0:1          │
│ Position Size: 20 shares ($9,760)   │
│                                     │
│ [Ask Copilot About This Setup]     │
│                                     │
├─────────────────────────────────────┤
│ 📈 Options Strategies               │
│ [View 3 Strategies →]              │
│                                     │
├─────────────────────────────────────┤
│ 📰 Events & Catalysts               │
│                                     │
│ • Earnings in 12 days (May 24)     │
│ • Dividend Ex-Date: None            │
│ • Insider Activity: 2 buys (30d)   │
│                                     │
├─────────────────────────────────────┤
│ 🏢 Fundamentals                     │
│                                     │
│ Sector: Technology                  │
│ Industry: Semiconductors            │
│ Market Cap: $1.2T                   │
│                                     │
│ P/E: 45.2  •  P/B: 12.8            │
│ ROE: 28.5%  •  Debt/Equity: 0.3    │
│                                     │
├─────────────────────────────────────┤
│ 🧠 Factor Breakdown                 │
│                                     │
│ Momentum:    ████████░░  82/100    │
│ Value:       ███░░░░░░░  35/100    │
│ Quality:     ███████░░░  78/100    │
│ Growth:      █████████░  92/100    │
│ Volatility:  ████░░░░░░  45/100    │
│                                     │
├─────────────────────────────────────┤
│ 💬 Ask Copilot                      │
│                                     │
│ "What are the key risks for NVDA?" │
│ "Compare NVDA to AMD"               │
│ "Explain the breakout setup"       │
│                                     │
└─────────────────────────────────────┘
```

#### Component Specifications

**A. Price Header**
- **Height**: 80px
- **Layout**: Price + Change + Timestamp
- **Price**: 
  - Font: 32px, bold
  - Color: White (default), Green (up), Red (down)
- **Change**:
  - Format: "▲ +2.3% (+$10.95)"
  - Color: Green (up), Red (down)
  - Arrow: ▲ (up), ▼ (down)
- **Timestamp**:
  - Format: "Updated 2 min ago"
  - Font: 12px, gray
  - Updates every 60 seconds

**B. Interactive Chart**
- **Height**: 200px (phone), 300px (tablet)
- **Type**: Candlestick (default), Line (optional)
- **Timeframes**: 1D, 5D, 1M, 3M, 6M, 1Y, ALL
- **Features**:
  - Pinch to zoom
  - Pan to scroll
  - Tap to show crosshair + price
  - Long press to show OHLC tooltip
- **Overlays**:
  - Entry/Stop/Target lines (if trade setup exists)
  - Moving averages (20, 50, 200 day)
  - Volume bars (bottom)
- **Indicators** (optional, toggle):
  - RSI, MACD, Bollinger Bands

**C. Score Grid**
- **Layout**: 2 rows × 3 columns
- **Each Cell**:
  - Label: 10px, uppercase, gray
  - Value: 24px, bold, white
  - Progress Bar: 4px height, colored
- **Colors**:
  - Green: >70
  - Yellow: 50-70
  - Gray: <50
- **Tap Behavior**: Show detailed breakdown in bottom sheet

**D. Trade Setup Card**
- **Height**: 160px
- **Layout**: Signal + Levels + Position Size + CTA
- **Signal**:
  - Font: 18px, bold
  - Icon: ▲ (long), ▼ (short)
- **Levels**:
  - Entry, Stop, Target with prices
  - R:R ratio calculated
  - Color-coded: Green (entry/target), Red (stop)
- **Position Size**:
  - Calculated based on user's account size + risk %
  - Format: "20 shares ($9,760)"
- **CTA**: "Ask Copilot About This Setup"
  - Opens Copilot with pre-filled question
  - Primary button style

**E. Events & Catalysts**
- **Height**: Variable (based on events)
- **Layout**: Bullet list
- **Event Types**:
  - Earnings (with countdown)
  - Dividends (ex-date, amount)
  - Insider Activity (buys/sells, 30d)
  - Analyst Ratings (upgrades/downgrades)
  - News (major headlines, 7d)
- **Icons**: 📅 (earnings), 💰 (dividend), 👔 (insider), 📰 (news)
- **Tap Behavior**: Expand to show details

**F. Factor Breakdown**
- **Height**: 200px
- **Layout**: Horizontal bars with labels
- **Factors**:
  - Momentum (price trends)
  - Value (P/E, P/B, etc.)
  - Quality (ROE, margins)
  - Growth (revenue, earnings)
  - Volatility (ATR, realized vol)
- **Bars**:
  - Width: Proportional to score (0-100)
  - Color: Gradient (red → yellow → green)
  - Label: Factor name + score
- **Tap Behavior**: Show factor definition + calculation

---

### **3. Ideas Page - Swipeable Card Stack**

#### Layout Structure
```
┌─────────────────────────────────────┐
│ 💡 Ideas (12)          [Filter ▼]  │ ← Header
├─────────────────────────────────────┤
│                                     │
│     ┌─────────────────────────┐    │
│     │                         │    │
│     │   AAPL                  │    │
│     │   Momentum Swing        │    │
│     │                         │    │ ← Top Card
│     │   ▁▃▄▅▆█               │    │   (400px)
│     │                         │    │
│     │   Entry: 178 → 186     │    │
│     │   Win: 64%  •  ICS: 82 │    │
│     │                         │    │
│     │   [Ask Copilot]        │    │
│     └─────────────────────────┘    │
│                                     │
│   ┌───────────────────────────┐    │ ← Next Card
│   │ MSFT  •  Breakout Long    │    │   (Peek 40px)
│   └───────────────────────────┘    │
│                                     │
│ ← Swipe Left (Dismiss)              │
│ → Swipe Right (Save)                │
│                                     │
├─────────────────────────────────────┤
│ [Scan] [Ideas] [Copilot] [★] [⚙️] │
└─────────────────────────────────────┘
```

#### Component Specifications

**A. Idea Card**
- **Size**: 340px × 400px
- **Style**: Elevated card with shadow
- **Layout**:
  ```
  ┌─────────────────────────────────┐
  │ AAPL          [CORE]  ICS: 82  │ ← Header
  │ Momentum Swing                  │ ← Strategy
  │                                 │
  │ ▁▃▄▅▆█                         │ ← Sparkline
  │                                 │
  │ Why This Idea:                  │
  │ Strong momentum + quality       │ ← Rationale
  │ fundamentals. Institutional     │   (80px)
  │ buying increasing.              │
  │                                 │
  │ Trade Plan:                     │
  │ Entry: $178  →  Target: $186   │ ← Plan
  │ Stop: $174  •  R:R 2.0:1       │   (60px)
  │                                 │
  │ Win: 64%  •  Quality: 7.8      │ ← Metrics
  │                                 │
  │ [Ask Copilot] [View Details]   │ ← Actions
  └─────────────────────────────────┘
  ```
- **Swipe Gestures**:
  - **Left**: Dismiss (fade out + remove from stack)
  - **Right**: Save to watchlist (fly to star icon)
  - **Up**: View full details (navigate to Symbol Detail)
  - **Tap**: Flip card to show back (options strategies)

**B. Card Back (Options)**
```
┌─────────────────────────────────┐
│ AAPL Options                    │
│                                 │
│ ✓ Call Spread $180/$185        │
│   30 DTE  •  Delta: 0.65       │
│   Max Profit: $400             │
│   Max Risk: $100               │
│   Sweetness: 82/100            │
│                                 │
│ • Call $182                     │
│   30 DTE  •  Delta: 0.70       │
│                                 │
│ [View All Strategies]          │
│ [Flip Back]                    │
└─────────────────────────────────┘
```

**C. Filter Panel**
- **Trigger**: Tap "Filter ▼" button
- **Display**: Bottom sheet (300px height)
- **Filters**:
  - Strategy Type: All / Breakout / Momentum / Pullback / Reversal
  - Risk Level: All / Stable / Neutral / Explosive
  - Sector: Multi-select
  - Time Horizon: Short-term / Swing / Position
- **Apply**: Real-time filtering (no "Apply" button needed)

#### Interaction Patterns

**Card Stack Behavior**:
- Show 1 card at a time (top card)
- Peek next card (40px visible at bottom)
- Smooth animations (300ms ease-out)
- Haptic feedback on swipe actions
- Auto-advance after dismiss/save

**Empty State**:
```
┌─────────────────────────────────┐
│                                 │
│        💡                       │
│                                 │
│   No Ideas Yet                  │
│                                 │
│   Run a scan to generate        │
│   personalized trade ideas      │
│                                 │
│   [Run Scan]                    │
│                                 │
└─────────────────────────────────┘
```

---

### **4. Copilot Page - AI Assistant**

#### Layout Structure
```
┌─────────────────────────────────────┐
│ 🤖 Copilot              [Clear]     │ ← Header
├─────────────────────────────────────┤
│                                     │
│ ┌─────────────────────────────────┐│
│ │ Context: NVDA                   ││ ← Context Pill
│ │ Breakout Long  •  ICS: 87      ││   (Dismissible)
│ └─────────────────────────────────┘│
│                                     │
│ ┌─────────────────────────────────┐│
│ │ What are the key risks for     ││ ← User Message
│ │ this NVDA setup?               ││   (Right-aligned)
│ └─────────────────────────────────┘│
│                                     │
│ ┌─────────────────────────────────┐│
│ │ Based on the current setup,    ││
│ │ here are the key risks:        ││
│ │                                ││
│ │ 1. **Earnings Risk**: NVDA has ││ ← Assistant Message
│ │    earnings in 12 days. High   ││   (Left-aligned)
│ │    IV (85th percentile) suggests││
│ │    market expects volatility.  ││
│ │                                ││
│ │ 2. **Technical Risk**: Price is││
│ │    extended (+18% in 21 days). ││
│ │    Pullback to $475 support    ││
│ │    possible.                   ││
│ │                                ││
│ │ 3. **Sector Risk**: Semis are  ││
│ │    overbought (RSI: 72).       ││
│ │                                ││
│ │ **Recommendation**: Consider   ││
│ │ defined-risk options (spreads) ││
│ │ to limit downside.             ││
│ │                                ││
│ │ [Show Options] [View Chart]    ││
│ └─────────────────────────────────┘│
│                                     │
│ ┌─────────────────────────────────┐│
│ │ 💬 Type your question...       ││ ← Input Field
│ │                          [Send]││   (60px)
│ └─────────────────────────────────┘│
│                                     │
│ Suggested:                          │
│ • Explain this setup                │ ← Suggested
│ • What are the risks?               │   Prompts
│ • Compare to sector peers           │   (Tappable)
│                                     │
├─────────────────────────────────────┤
│ [Scan] [Ideas] [Copilot] [★] [⚙️] │
└─────────────────────────────────────┘
```

#### Component Specifications

**A. Context Pill**
- **Height**: 60px
- **Style**: Rounded pill with gradient background
- **Layout**: Symbol + Signal + Key Metric
- **Dismiss**: X button (top-right)
- **Behavior**:
  - Auto-populated when navigating from Scanner/Ideas
  - Persists across messages
  - Can be manually cleared

**B. Message Bubbles**
- **User Messages**:
  - Alignment: Right
  - Background: Primary color gradient
  - Text: White
  - Max Width: 80% of screen
  - Border Radius: 18px (left), 4px (bottom-right)
- **Assistant Messages**:
  - Alignment: Left
  - Background: Dark gray (#1A1A1A)
  - Text: White
  - Max Width: 85% of screen
  - Border Radius: 18px (right), 4px (bottom-left)
  - Markdown Support: Bold, italic, lists, code blocks
  - Action Buttons: Inline CTAs (e.g., "Show Options")

**C. Input Field**
- **Height**: 60px (collapsed), 120px (expanded for multi-line)
- **Style**: Rounded rectangle with border
- **Placeholder**: "Type your question..."
- **Features**:
  - Auto-expand for long messages
  - Send button (always visible)
  - Voice input button (iOS only)
  - Emoji picker (optional)
- **Behavior**:
  - Focus → Keyboard appears, scroll to bottom
  - Send → Show typing indicator, disable input
  - Response → Re-enable input, scroll to bottom

**D. Typing Indicator**
```
┌─────────────────────────────────┐
│ ● ● ●  Copilot is thinking...  │ ← Animated dots
└─────────────────────────────────┘
```

**E. Suggested Prompts**
- **Display**: Horizontal scrollable chips
- **Style**: Outlined chips with icon
- **Prompts**:
  - "Explain this setup"
  - "What are the risks?"
  - "Compare to sector peers"
  - "Show me similar setups"
  - "What's the options play?"
- **Behavior**: Tap → Auto-fill input + send

**F. Error State**
```
┌─────────────────────────────────┐
│ ⚠️ Copilot is temporarily      │
│    unavailable.                 │
│                                 │
│ Showing cached guidance until   │
│ service recovers.               │
│                                 │
│ [Retry]                         │
└─────────────────────────────────┘
```

#### Interaction Patterns

**Message Actions**:
- **Long Press Message**: Copy text
- **Tap Action Button**: Execute action (e.g., show chart)
- **Swipe Message**: Delete (user messages only)

**Context Management**:
- **Auto-Context**: Populated from Scanner/Ideas navigation
- **Manual Context**: User can type symbol in message
- **Clear Context**: Tap X on context pill
- **Context History**: Last 5 symbols (accessible via dropdown)

---

### **5. My Ideas (Watchlist) Page**

#### Layout Structure
```
┌─────────────────────────────────────┐
│ ⭐ My Ideas (8)      [Sort ▼] [+]  │ ← Header
├─────────────────────────────────────┤
│                                     │
│ ┌─────────────────────────────────┐│
│ │ NVDA          $488.50  ▲ +2.3% ││
│ │ Breakout Long  •  ICS: 87      ││ ← Watchlist Card
│ │ ▁▃▄▅▆█                         ││   (100px)
│ │ Added 2 days ago               ││
│ │ [View] [Remove]                ││
│ └─────────────────────────────────┘│
│                                     │
│ ┌─────────────────────────────────┐│
│ │ AAPL          $178.20  ▲ +1.1% ││
│ │ Momentum Swing  •  ICS: 82     ││
│ │ ▁▂▃▅▆█                         ││
│ │ Added 1 week ago               ││
│ │ [View] [Remove]                ││
│ └─────────────────────────────────┘│
│                                     │
│ [Show More ▼]                      │
│                                     │
├─────────────────────────────────────┤
│ 📊 Watchlist Performance            │
│                                     │
│ Avg Return: +5.2%                   │
│ Win Rate: 62.5% (5/8)              │
│ Best: NVDA (+12.3%)                │
│ Worst: TSLA (-3.1%)                │
│                                     │
└─────────────────────────────────────┘
```

#### Component Specifications

**A. Watchlist Card**
- **Height**: 100px
- **Layout**: Symbol + Price + Signal + Sparkline + Actions
- **Price**:
  - Real-time (updates every 60s)
  - Color: Green (up), Red (down)
- **Signal**: Original signal when added
- **Sparkline**: 30-day price history
- **Metadata**: "Added X days ago"
- **Actions**:
  - "View": Navigate to Symbol Detail
  - "Remove": Delete from watchlist (with undo)

**B. Sort Options**
- **Trigger**: Tap "Sort ▼"
- **Options**:
  - Recently Added (default)
  - Alphabetical (A-Z)
  - Price Change (High to Low)
  - ICS Score (High to Low)
  - Win Probability (High to Low)
- **Behavior**: Instant re-sort (no "Apply" button)

**C. Add Symbol Button (+)**
- **Trigger**: Tap "+" button
- **Display**: Modal with search
- **Search**: Autocomplete symbol search
- **Add**: Tap symbol → Add to watchlist → Close modal

**D. Watchlist Performance**
- **Height**: 120px
- **Metrics**:
  - Average Return: % change since added
  - Win Rate: % of symbols with positive return
  - Best Performer: Symbol + return
  - Worst Performer: Symbol + return
- **Update**: Daily (at market close)

**E. Empty State**
```
┌─────────────────────────────────┐
│                                 │
│        ⭐                       │
│                                 │
│   No Saved Ideas Yet            │
│                                 │
│   Star symbols from Scanner     │
│   or Ideas to track them here   │
│                                 │
│   [Browse Ideas]                │
│                                 │
└─────────────────────────────────┘
```

---

### **6. Settings Page**

#### Layout Structure
```
┌─────────────
