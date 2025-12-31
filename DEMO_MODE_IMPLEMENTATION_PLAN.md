# 🎬 Demo Mode for Investor Presentation

## 🎯 Goal
Create a demo mode that shows predetermined, impressive results for:
1. **Scanner** - Shows curated high-quality stock picks
2. **Copilot** - Provides intelligent explanations of the results
3. **Recommendations** - Displays professional investment recommendations

## 📋 Implementation Strategy

### 1. Demo Data Files
Create JSON files with predetermined results:
- `demo_scan_results.json` - Curated stock picks with excellent metrics
- `demo_copilot_responses.json` - Pre-written intelligent responses
- `demo_recommendations.json` - Professional investment recommendations

### 2. Backend Demo Mode
Add a demo mode flag to the API that returns predetermined data:
- Environment variable: `DEMO_MODE=true`
- API endpoints return demo data instead of live scans
- Consistent, impressive results every time

### 3. Mobile App Demo Toggle
Add a demo mode switch in the app:
- Settings screen toggle
- Visual indicator when in demo mode
- Seamless switching between demo and live

## 🎨 Demo Data Design

### Scanner Results (5-7 High-Quality Stocks)
**Criteria for Demo Stocks**:
- Well-known, reputable companies
- Strong technical ratings (80-95)
- High MERIT scores (85-95)
- Clear buy signals
- Diverse sectors (Tech, Healthcare, Finance)
- Realistic but impressive metrics

**Example Stocks**:
1. **AAPL** (Apple) - Tech leader, A+ setup
2. **MSFT** (Microsoft) - Enterprise strength, A+ setup
3. **NVDA** (NVIDIA) - AI momentum, A setup
4. **JPM** (JPMorgan) - Financial stability, A setup
5. **UNH** (UnitedHealth) - Healthcare quality, A+ setup
6. **V** (Visa) - Payment processing, A setup
7. **GOOGL** (Alphabet) - Search dominance, A setup

### Copilot Responses
**Pre-written responses for common questions**:
- "Why is AAPL recommended?"
- "What's the risk with NVDA?"
- "Explain the top 3 picks"
- "What sectors look strong?"
- "How should I allocate my portfolio?"

### Investment Recommendations
**Professional-grade recommendations**:
- Portfolio allocation suggestions
- Risk management strategies
- Entry/exit points
- Position sizing
- Diversification advice

## 🔧 Implementation Steps

### Step 1: Create Demo Data Files
Location: `technic_v4/demo/`
- `demo_scan_results.json`
- `demo_copilot_responses.json`
- `demo_recommendations.json`

### Step 2: Update Backend API
File: `api.py`
- Add `DEMO_MODE` environment variable check
- Create demo endpoints or modify existing ones
- Return demo data when in demo mode

### Step 3: Update Mobile App
Files:
- `lib/services/api_service.dart` - Add demo mode support
- `lib/screens/settings/settings_screen.dart` - Add demo toggle
- `lib/providers/app_providers.dart` - Manage demo state

### Step 4: Add Visual Indicators
- Banner at top: "🎬 DEMO MODE"
- Different color scheme (subtle)
- Clear indication this is demonstration data

## 📊 Demo Scan Results Structure

```json
{
  "results": [
    {
      "symbol": "AAPL",
      "techRating": 92.5,
      "meritScore": 94.0,
      "resultTier": "A+",
      "signal": "Strong Long",
      "entry": 185.50,
      "stop": 178.20,
      "target": 198.75,
      "rewardRisk": 2.8,
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "rationale": "Apple shows exceptional technical strength with strong institutional support. Recent product launches and services growth provide solid fundamental backing. Risk-reward ratio of 2.8:1 offers attractive upside potential.",
      "meritFlags": "High Quality, Strong Momentum, Institutional Favorite",
      "qualityScore": 92,
      "alphaScore": 0.85,
      "riskScore": 0.88
    }
  ],
  "status": "Found 7 A+ grade setups",
  "symbols_scanned": 2210,
  "symbols_returned": 7,
  "scan_time": "2.3s"
}
```

## 💬 Demo Copilot Responses

```json
{
  "responses": {
    "why_aapl": {
      "question": "Why is AAPL recommended?",
      "answer": "Apple (AAPL) is recommended based on multiple strong factors:\n\n1. **Technical Strength** (92.5/100): Price action shows consistent uptrend with strong support levels. Recent breakout above $180 confirms bullish momentum.\n\n2. **Quality Metrics** (94/100): Apple maintains exceptional profit margins, strong cash flow, and minimal debt. Their services segment provides recurring revenue stability.\n\n3. **Institutional Support**: Heavy institutional ownership (>60%) provides price stability and validates the investment thesis.\n\n4. **Risk Management**: Stop loss at $178.20 limits downside to 3.9%, while target at $198.75 offers 7.1% upside - a favorable 2.8:1 reward-risk ratio.\n\n5. **Sector Leadership**: As a technology leader with diversified revenue streams, Apple benefits from both growth and defensive characteristics.\n\nRecommended position size: 15-20% of portfolio for balanced risk exposure."
    },
    "top_3_picks": {
      "question": "Explain the top 3 picks",
      "answer": "Here are the top 3 recommendations from today's scan:\n\n**1. AAPL (Apple) - MERIT Score: 94**\n- Tech leader with strong fundamentals\n- Entry: $185.50, Target: $198.75 (7.1% upside)\n- Best for: Core portfolio position\n\n**2. UNH (UnitedHealth) - MERIT Score: 93**\n- Healthcare stability with growth\n- Entry: $528.30, Target: $558.20 (5.7% upside)\n- Best for: Defensive allocation\n\n**3. MSFT (Microsoft) - MERIT Score: 91**\n- Enterprise strength + AI exposure\n- Entry: $378.40, Target: $398.50 (5.3% upside)\n- Best for: Technology diversification\n\nAll three offer A+ setups with strong risk-reward ratios and complement each other across sectors."
    },
    "portfolio_allocation": {
      "question": "How should I allocate my portfolio?",
      "answer": "Based on today's scan results, here's a balanced allocation strategy:\n\n**Core Holdings (60%)**:\n- AAPL: 20% - Technology leader\n- MSFT: 20% - Enterprise strength\n- UNH: 20% - Healthcare stability\n\n**Growth Positions (30%)**:\n- NVDA: 15% - AI momentum\n- GOOGL: 15% - Digital advertising\n\n**Financial Stability (10%)**:\n- JPM: 5% - Banking strength\n- V: 5% - Payment processing\n\nThis allocation provides:\n- Sector diversification (Tech 55%, Healthcare 20%, Finance 10%, Other 15%)\n- Balance between growth and stability\n- Manageable risk with quality names\n- Strong institutional backing across all positions\n\nRecommended: Start with 50% of intended allocation, add on strength."
    }
  }
}
```

## 🎯 Demo Recommendations

```json
{
  "recommendations": [
    {
      "title": "Core Portfolio Strategy",
      "description": "Build a foundation with quality large-caps",
      "allocation": {
        "AAPL": 20,
        "MSFT": 20,
        "UNH": 20
      },
      "rationale": "These three stocks provide stability, growth, and diversification across key sectors. All show A+ technical setups with strong fundamentals.",
      "risk_level": "Moderate",
      "time_horizon": "6-12 months"
    },
    {
      "title": "Growth Opportunity",
      "description": "Capitalize on AI and technology trends",
      "allocation": {
        "NVDA": 15,
        "GOOGL": 15
      },
      "rationale": "Both companies are leaders in AI and digital transformation. Higher growth potential with managed risk through quality metrics.",
      "risk_level": "Moderate-High",
      "time_horizon": "3-6 months"
    },
    {
      "title": "Defensive Allocation",
      "description": "Financial stability and payment processing",
      "allocation": {
        "JPM": 5,
        "V": 5
      },
      "rationale": "Provides portfolio stability and consistent returns. Both benefit from economic growth while offering downside protection.",
      "risk_level": "Low-Moderate",
      "time_horizon": "12+ months"
    }
  ]
}
```

## 🚀 Quick Start Guide

### For Demonstration:

1. **Enable Demo Mode**:
   ```bash
   # Set environment variable
   export DEMO_MODE=true
   
   # Or in .env file
   DEMO_MODE=true
   ```

2. **Launch Mobile App**:
   ```bash
   cd technic_mobile
   flutter run -d emulator-5554
   ```

3. **Demo Flow**:
   - Open app
   - Navigate to Scanner
   - Tap "Scan" button
   - See 7 high-quality A+ stocks instantly
   - Tap any stock for details
   - Ask Copilot: "Explain the top 3 picks"
   - View professional recommendations

### Benefits for Investor Demo:

✅ **Consistent Results**: Same impressive data every time
✅ **Fast Performance**: Instant results (no API delays)
✅ **Professional Quality**: Curated, realistic data
✅ **Comprehensive**: Scanner + Copilot + Recommendations
✅ **Impressive Metrics**: A+ setups, high MERIT scores
✅ **Clear Value Prop**: Shows full capability of the platform

## 📱 Visual Demo Mode Indicators

### Banner
```
┌─────────────────────────────────────┐
│ 🎬 DEMO MODE - Presentation Data   │
└─────────────────────────────────────┘
```

### Settings Toggle
```
Demo Mode
[ON] OFF

When enabled, shows curated demonstration
data for presentations. Disable for live
market scanning.
```

## 🎬 Demo Script for Investors

**Opening** (30 seconds):
"Let me show you how Technic identifies high-quality investment opportunities..."

**Scanner Demo** (1 minute):
- Tap Scan button
- Show 7 A+ stocks appear instantly
- Highlight MERIT scores (90+)
- Point out diverse sectors

**Stock Detail** (1 minute):
- Tap AAPL
- Show technical rating (92.5)
- Explain entry/stop/target
- Highlight risk-reward ratio (2.8:1)

**Copilot Demo** (1 minute):
- Ask: "Explain the top 3 picks"
- Show intelligent, detailed response
- Demonstrate AI-powered insights

**Recommendations** (1 minute):
- Show portfolio allocation
- Explain diversification strategy
- Highlight risk management

**Closing** (30 seconds):
"This is how Technic helps investors make informed decisions with confidence."

---

**Total Demo Time**: 4-5 minutes
**Impact**: High - Shows complete value proposition
**Repeatability**: Perfect - Same results every time
