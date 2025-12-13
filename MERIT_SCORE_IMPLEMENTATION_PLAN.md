# MERIT Score Implementation Plan

## Executive Summary

This document outlines the implementation of a novel **MERIT Score** (Multi-factor Evaluation & Risk-Integrated Technical Score) - a proprietary composite metric that combines technical analysis, predictive alpha, fundamental quality, and risk assessment into a single 0-100 score.

**Purpose**: Create a unique, patent-worthy metric that differentiates Technic from competitors while maintaining simplicity for end users.

---

## What is the MERIT Score?

**MERIT** = **M**ulti-factor **E**valuation & **R**isk-**I**ntegrated **T**echnical Score

A holistic 0-100 score that answers: **"How good is this trading opportunity RIGHT NOW?"**

### Key Innovation:
Unlike simple weighted averages, MERIT includes:
1. **Confluence Bonus**: Rewards when technical and alpha signals AGREE
2. **Risk Integration**: Penalizes high volatility and event risk
3. **Quality Overlay**: Incorporates institutional-grade fundamental metrics
4. **Event Awareness**: Adjusts for earnings, liquidity, market cap

---

## MERIT Formula (v1)

### Step 1: Compute Percentiles
```python
TechPct = percentile_rank(TechRating) * 100
AlphaPct = percentile_rank(AlphaScore) * 100
LiquidityPct = percentile_rank(log10(DollarVolume)) * 100
VolSafetyPct = 100 - percentile_rank(ATR14_pct) * 100  # Lower volatility = higher safety
WinProbPct = win_prob_10d * 100 (if available)
ICS = InstitutionalCoreScore (if available, else 50)
Quality = QualityScore (if available, else 50)
```

### Step 2: Calculate Confluence Bonus
```python
Confluence = 100 - abs(TechPct - AlphaPct)  # 0-100, higher when tech & alpha agree
ConfluenceBonus = (Confluence - 50) * 0.20  # Range: -10 to +10
```

### Step 3: Base Score (Weighted Sum)
```python
base = (
    0.26 * TechPct +           # Technical strength (26%)
    0.22 * AlphaPct +           # Predictive alpha (22%)
    0.18 * WinProbPct +         # Win probability (18%)
    0.14 * ICS +                # Institutional quality (14%)
    0.14 * Quality +            # Fundamental quality (14%)
    0.04 * LiquidityPct +       # Liquidity (4%)
    0.02 * VolSafetyPct         # Volatility safety (2%)
) + ConfluenceBonus
```

### Step 4: Apply Penalties
```python
penalties = 0

# Earnings risk
if days_to_earnings <= 3: penalties += 12
elif days_to_earnings <= 7: penalties += 6

# Liquidity risk
if DollarVolume < 10_000_000: penalties += 8

# Volatility risk
if ATR14_pct > 0.12: penalties += 14
elif ATR14_pct > 0.08: penalties += 8

# Market cap risk
if market_cap < 1_000_000_000: penalties += 6
if market_cap < 300_000_000: penalties += 20

# Ultra-risky flag
if IsUltraRisky: penalties += 25
```

### Step 5: Final Score
```python
MeritScore = clamp(base - penalties, 0, 100)
```

### Step 6: Banding
```python
if MeritScore >= 90: MeritBand = "A+"
elif MeritScore >= 80: MeritBand = "A"
elif MeritScore >= 70: MeritBand = "B"
elif MeritScore >= 60: MeritBand = "C"
else: MeritBand = "D"
```

---

## Implementation Phases

### Phase 1: Core Engine (Prompt 1)
**File**: `technic_v4/engine/merit_engine.py`

**Deliverables**:
- `MeritConfig` dataclass with weights and penalty settings
- `compute_merit(df, regime, config)` function
- Output columns:
  - `MeritScore` (0-100)
  - `MeritBand` ("A+", "A", "B", "C", "D")
  - `MeritFlags` (pipe-delimited: "EARNINGS_SOON|LOW_LIQUIDITY")
  - `MeritSummary` (1-sentence plain English)
  - Debug columns: `MeritTechPct`, `MeritAlphaPct`, etc.

**Key Requirements**:
- Fully vectorized (no row-by-row loops)
- Graceful handling of missing columns
- Never crashes

---

### Phase 2: Scanner Integration (Prompt 2)
**Files**: `technic_v4/scanner_core.py`, ranking modules

**Changes**:
1. Call `compute_merit()` after ICS/Quality computation
2. Sort results by `MeritScore` (primary) then `TechRating` (secondary)
3. Update `diversify_by_sector()` to use `score_col="MeritScore"`
4. Include MERIT columns in CSV outputs
5. Add logging for top 10 by MERIT

**Validation**:
- Top 10 should be "institutional grade" (no junk)
- MERIT should correlate with but improve upon TechRating

---

### Phase 3: Recommendation Text (Prompt 3)
**Files**: `technic_v4/engine/recommendation.py` or similar

**Changes**:
1. Lead with MERIT: "Strong Long — AAPL. MERIT 86 (A)."
2. Explain 3 pillars:
   - Technical setup
   - Forward edge (alpha/win prob)
   - Institutional quality
3. Mention event flags if present
4. Add action suggestion (entry/stop/target)
5. Include options idea if available

**Example Output**:
```
Strong Long — AAPL. MERIT 86 (A).

Technical: Strong uptrend with momentum confirmation (TechRating 78).
Forward Edge: 65% win probability over 10 days, positive alpha forecast.
Quality: Institutional-grade fundamentals (ICS 82, Quality 75).

Idea: Buy on pullback toward $175 with stop at $170; target $185 (2.1:1 R/R).
Options: Bull call spread $175/$180 expiring Jan 2024 (45 DTE, sweetness 82/100).
```

---

### Phase 4: API Integration (Prompt 4)
**Files**: `technic_v4/api_server.py`, `api_contract.py`

**Changes**:
1. Add MERIT fields to `ScanResultRow` model:
   ```python
   merit_score: Optional[float] = None
   merit_band: Optional[str] = None
   merit_flags: Optional[str] = None
   merit_summary: Optional[str] = None
   ```
2. Ensure `/v1/scan` endpoint includes these fields
3. Maintain backward compatibility

---

### Phase 5: UI Integration (Prompt 5)
**Files**: Flutter app (`technic_app/lib/`)

**Changes**:
1. **Card Header**:
   - Line 1: Symbol + Signal badge
   - Line 2: **"MERIT 86 (A)"** (large, prominent)
   - Line 3: Chips for TechRating, WinProb, Quality, ICS
   - Line 4: `merit_summary` text

2. **Risk/Event Chips**:
   - Red/orange for EARNINGS_SOON
   - Warning for LOW_LIQUIDITY
   - Caution for HIGH_ATR

3. **Copilot Integration**:
   - Use `merit_summary` + flags for explanations
   - Structure: "What's the trade?", "Why Technic likes it?", "What could go wrong?", "How to manage it?"

---

### Phase 6: Quality Testing (Prompt 6)
**File**: `technic_v4/dev/test_merit_quality.py`

**Tests**:
1. Load `technic_scan_results.csv`
2. Assert `MeritScore` exists and is in [0, 100]
3. Assert top 10 sorted by `MeritScore` descending
4. Print top 10 with key columns
5. Check `technic_runners.csv` (should have lower MERIT)

---

## Why MERIT is Novel & Patent-Worthy

### 1. Unique Combination
- **No other platform** combines technical, alpha, quality, liquidity, and event risk into ONE score
- Most tools show these separately or use simple averages
- MERIT uses **nonlinear confluence bonus** to reward agreement

### 2. Risk-Integrated
- Not just "what's strong" but "what's strong AND safe"
- Volatility and event penalties are built-in
- Institutional-grade risk management

### 3. Forward-Looking
- Incorporates ML predictions (win probability, alpha)
- Not just backward-looking technicals
- Bridges technical analysis with predictive analytics

### 4. Institutional Quality
- Integrates ICS and QualityScore
- Filters out junk automatically (via penalties)
- Aligns with how professional traders think

### 5. User-Friendly
- Single 0-100 score (like a credit score)
- Letter grades (A+, A, B, C, D)
- Plain-English summary
- Event flags for risk awareness

---

## Patent Strategy

### Patentable Elements:
1. **Confluence Bonus Formula**: Novel way to measure multi-factor agreement
2. **Risk-Integrated Scoring**: Specific penalty structure for events/volatility
3. **Three-Pillar Integration**: Technical + Alpha + Quality in one metric
4. **Event-Aware Adjustments**: Dynamic penalties based on earnings/liquidity

### Prior Art Differentiation:
- **vs. TradingView**: They show indicators separately, no composite
- **vs. Bloomberg**: They have ratings but not this specific formula
- **vs. Robinhood**: No quantitative scoring at all
- **vs. Technic's ICS**: ICS is linear weighted sum; MERIT adds confluence bonus and event penalties

### Patent Application:
- **Title**: "System and Method for Multi-Factor Risk-Integrated Stock Scoring"
- **Claims**: Specific formula, confluence calculation, penalty structure
- **Embodiment**: Software implementation in trading platform

---

## Success Metrics

### Technical Metrics:
- ✅ MERIT computes without errors for 5,000+ symbols
- ✅ Top 10 by MERIT are institutional-grade (no penny stocks)
- ✅ MERIT correlates with but improves upon TechRating
- ✅ Confluence bonus correctly rewards agreement

### User Metrics:
- ✅ Users understand MERIT score (survey: >80% comprehension)
- ✅ MERIT becomes primary decision metric (usage tracking)
- ✅ Reduced time to decision (vs. analyzing multiple metrics)
- ✅ Increased confidence in trades (user feedback)

### Business Metrics:
- ✅ MERIT becomes marketing differentiator
- ✅ Patent application filed within 6 months
- ✅ Competitive advantage established
- ✅ User retention improves (MERIT users vs. non-MERIT)

---

## Timeline

### Week 1: Core Implementation
- Day 1-2: Implement `merit_engine.py` (Prompt 1)
- Day 3-4: Integrate into scanner (Prompt 2)
- Day 5: Testing and validation

### Week 2: User-Facing Features
- Day 1-2: Recommendation text (Prompt 3)
- Day 3: API integration (Prompt 4)
- Day 4-5: UI integration (Prompt 5)

### Week 3: Testing & Refinement
- Day 1-2: Quality testing (Prompt 6)
- Day 3-4: User testing and feedback
- Day 5: Calibration and adjustments

### Week 4: Launch & Documentation
- Day 1-2: Final testing
- Day 3: Deploy to production
- Day 4: User documentation
- Day 5: Marketing materials

---

## Next Steps

### Immediate Actions:
1. **Review this plan** with development team
2. **Run Prompt 1** to create `merit_engine.py`
3. **Test on sample data** to validate formula
4. **Iterate on weights** based on backtesting

### Follow-Up:
1. Run remaining prompts in sequence
2. Conduct user testing
3. Gather feedback and refine
4. Prepare patent application materials

---

## Appendix: BlackBox AI Prompts

The 6 prompts are ready to paste into BlackBox AI in sequence:

1. **Prompt 1**: Create `merit_engine.py` with core computation
2. **Prompt 2**: Wire MERIT into scan pipeline and ranking
3. **Prompt 3**: Integrate MERIT into recommendation generator
4. **Prompt 4**: Expose MERIT in API schema/endpoints
5. **Prompt 5**: UI integration (cards + Copilot)
6. **Prompt 6**: Add quality test script

Each prompt is self-contained and can be executed independently once prerequisites are met.

---

**Document Status**: Ready for implementation
**Last Updated**: After ChatGPT feature review
**Next Action**: Execute Prompt 1 to create merit_engine.py
