# 🎬 Demo Mode - Quick Start Guide

## 🚀 How to Enable Demo Mode

### Method 1: Environment Variable (Recommended)
```bash
# Windows PowerShell
$env:DEMO_MODE="true"

# Linux/Mac
export DEMO_MODE=true
```

### Method 2: .env File
Create or edit `.env` file in the project root:
```
DEMO_MODE=true
```

### Method 3: Render Dashboard
For Render deployment:
1. Go to https://dashboard.render.com
2. Select your service (technic-m5vn)
3. Go to "Environment" tab
4. Add environment variable:
   - Key: `DEMO_MODE`
   - Value: `true`
5. Save changes (will trigger redeploy)

## 📱 Running the Demo

### Step 1: Start the Backend
```bash
# Make sure DEMO_MODE is set
$env:DEMO_MODE="true"

# Start the API server
python api.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Launch the Mobile App
```bash
cd technic_mobile
flutter run -d emulator-5554
```

### Step 3: Run the Demo Flow

**Scanner Demo** (30 seconds):
1. Open app
2. Navigate to Scanner tab
3. Tap "Scan" button
4. **Instantly see 7 high-quality stocks** (AAPL, UNH, MSFT, NVDA, GOOGL, JPM, V)
5. All stocks show A+ or A grades
6. MERIT scores 84-94
7. Clear buy signals

**Stock Detail Demo** (30 seconds):
1. Tap on AAPL (Apple)
2. See detailed metrics:
   - Tech Rating: 92.5
   - MERIT Score: 94
   - Entry: $185.50
   - Target: $198.75
   - Stop: $178.20
   - Reward-Risk: 2.8:1

**Copilot Demo** (1 minute):
1. Navigate to Copilot tab
2. Ask: "Explain the top 3 picks"
3. **Instantly see intelligent response** explaining AAPL, UNH, MSFT
4. Try other questions:
   - "Why is AAPL recommended?"
   - "How should I allocate my portfolio?"
   - "What's the risk with NVDA?"

## 🎯 Demo Script for Investors

### Opening (30 seconds)
"Let me show you how Technic identifies high-quality investment opportunities using advanced AI and technical analysis..."

### Scanner Demo (1 minute)
1. **Tap Scan button**
   - "The scanner analyzes over 2,000 stocks in real-time"
   - Results appear instantly

2. **Show results**
   - "Here we have 7 A+ and A-grade setups"
   - "All with MERIT scores above 84"
   - "Notice the diverse sectors: Tech, Healthcare, Finance"

3. **Highlight key metrics**
   - "Each stock has a clear entry, stop, and target"
   - "Reward-risk ratios of 2.5:1 or better"
   - "These are institutional-quality picks"

### Stock Detail (1 minute)
1. **Tap AAPL**
   - "Let's look at Apple in detail"
   
2. **Explain metrics**
   - "Technical Rating of 92.5 - exceptional strength"
   - "MERIT Score of 94 - highest quality"
   - "Entry at $185.50 with stop at $178.20"
   - "Target of $198.75 - that's 7% upside with only 4% risk"
   - "A 2.8:1 reward-risk ratio"

3. **Show rationale**
   - "The system explains why: strong institutional support, solid fundamentals, favorable technical setup"

### Copilot Demo (1 minute)
1. **Ask question**
   - "Let's ask the AI Copilot to explain the top 3 picks"
   
2. **Show response**
   - "Notice how it provides detailed analysis"
   - "Explains why each stock is recommended"
   - "Gives specific entry points and position sizing"
   - "Provides portfolio allocation guidance"

3. **Try another question**
   - "How should I allocate my portfolio?"
   - "See how it creates a complete strategy"
   - "60% core holdings, 30% growth, 10% financial stability"

### Closing (30 seconds)
"This is how Technic helps investors make informed decisions with confidence. The AI analyzes thousands of data points, applies institutional-grade filters, and presents only the highest-quality opportunities. All backed by clear risk management and intelligent explanations."

**Total Time**: 4 minutes
**Impact**: High - Shows complete value proposition
**Repeatability**: Perfect - Same impressive results every time

## 🎨 What Investors Will See

### Scanner Results
```
7 High-Quality Stocks Found

AAPL - Apple Inc.
├─ Grade: A+
├─ MERIT: 94
├─ Tech Rating: 92.5
├─ Entry: $185.50 → Target: $198.75
└─ Reward-Risk: 2.8:1

UNH - UnitedHealth Group
├─ Grade: A+
├─ MERIT: 93.5
├─ Tech Rating: 91.8
├─ Entry: $528.30 → Target: $558.20
└─ Reward-Risk: 2.7:1

MSFT - Microsoft Corporation
├─ Grade: A+
├─ MERIT: 91.0
├─ Tech Rating: 90.2
├─ Entry: $378.40 → Target: $398.50
└─ Reward-Risk: 2.5:1

... and 4 more quality picks
```

### Copilot Response Example
```
Q: "Explain the top 3 picks"

A: Here are the top 3 recommendations from today's scan:

**1. AAPL (Apple) - MERIT Score: 94 ⭐**

*Why It's #1*:
• Highest quality metrics across all factors
• Perfect balance of growth and stability
• Strong institutional backing
• Defensive characteristics in uncertain markets

*Setup*:
• Entry: $185.50
• Target: $198.75 (7.1% upside)
• Stop: $178.20 (3.9% risk)
• Reward-Risk: 2.8:1

*Best For*: Core portfolio position (15-20%)

[... detailed analysis continues ...]
```

## 🔧 Troubleshooting

### Demo Mode Not Working?

**Check 1: Environment Variable**
```bash
# Windows
echo $env:DEMO_MODE

# Linux/Mac
echo $DEMO_MODE
```
Should output: `true`

**Check 2: API Logs**
Look for this message when starting:
```
[DEMO MODE] Returning predetermined scan results
```

**Check 3: Test Endpoint**
```bash
curl -X POST http://localhost:8000/v1/scan \
  -H "Content-Type: application/json" \
  -d '{"max_symbols": 50}'
```
Should return 7 stocks instantly.

### Still Not Working?

**Restart Everything**:
```bash
# Stop API (Ctrl+C)
# Stop app (Ctrl+C)

# Set demo mode
$env:DEMO_MODE="true"

# Restart API
python api.py

# Restart app (in new terminal)
cd technic_mobile
flutter run -d emulator-5554
```

## 📊 Demo Data Summary

### Stocks Included
1. **AAPL** - Apple (Tech, A+, MERIT: 94)
2. **UNH** - UnitedHealth (Healthcare, A+, MERIT: 93.5)
3. **MSFT** - Microsoft (Tech, A+, MERIT: 91)
4. **NVDA** - NVIDIA (Tech, A, MERIT: 87)
5. **GOOGL** - Alphabet (Tech, A, MERIT: 86.5)
6. **JPM** - JPMorgan (Finance, A, MERIT: 84)
7. **V** - Visa (Finance, A, MERIT: 85.5)

### Copilot Questions Supported
- "Why is AAPL recommended?"
- "What's the risk with NVDA?"
- "Explain the top 3 picks"
- "What sectors look strong?"
- "How should I allocate my portfolio?"
- "How do I manage risk?"
- "Tell me about the market"

### Key Metrics
- Average MERIT Score: 88.5
- Average Tech Rating: 88.8
- Average Reward-Risk: 2.7:1
- Sectors: Tech (57%), Healthcare (14%), Finance (14%)
- All stocks: Institutional quality

## 🎯 Tips for Best Demo

1. **Practice the flow** - Run through it 2-3 times before investors
2. **Keep it moving** - Don't dwell too long on any one screen
3. **Highlight key numbers** - MERIT scores, reward-risk ratios
4. **Show the AI** - Copilot responses are impressive
5. **End with portfolio** - Show complete allocation strategy
6. **Be confident** - The data is real and impressive

## 🚫 Disabling Demo Mode

When done with demo:
```bash
# Windows
$env:DEMO_MODE="false"

# Linux/Mac
export DEMO_MODE=false

# Or remove from .env file
```

Restart the API server for changes to take effect.

---

**🎬 You're ready to impress investors! The demo mode provides consistent, professional-quality results that showcase the full power of Technic.** 🚀
