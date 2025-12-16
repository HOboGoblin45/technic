# ✅ ML Alpha Utilization Confirmed

## Summary

ML alpha is **FULLY ENABLED** and being used throughout the Technic scanner system.

---

## Configuration Status

### **Settings (technic_v4/config/settings.py)**

```python
use_ml_alpha: bool = field(default=True)  # ✅ ENABLED BY DEFAULT
alpha_weight: float = field(default=0.35)  # 35% ML, 65% factor
```

**Environment Variable Override:**
- `TECHNIC_USE_ML_ALPHA=1` (can be set to override default)
- `TECHNIC_ALPHA_WEIGHT=0.35` (can be adjusted 0.0-1.0)

---

## How ML Alpha is Used

### **1. Alpha Inference (scanner_core.py lines 500-650)**

The scanner computes **multi-horizon ML alpha**:

```python
# 5-day alpha (regime/sector-aware)
ml_5d = alpha_inference.score_alpha_contextual(df, regime, sector)

# 10-day alpha (optional second model)
ml_10d = alpha_inference.score_alpha_10d(df)

# Blended ML alpha
ml_alpha_z = 0.4 * ml_5d_z + 0.6 * ml_10d_z
```

**Columns Created:**
- `Alpha5d` - 5-day ML prediction
- `Alpha10d` - 10-day ML prediction  
- `AlphaScore` - Blended raw ML alpha
- `ml_alpha_5d_z` - Z-scored 5d alpha
- `ml_alpha_10d_z` - Z-scored 10d alpha
- `ml_alpha_z` - Final blended ML z-score

---

### **2. Alpha Blending (scanner_core.py lines 650-750)**

ML alpha is blended with factor alpha:

```python
# Factor alpha: z-score of baseline TechRating
factor_alpha = zscore(TechRating)

# Blend with ML alpha (35% ML, 65% factor by default)
alpha_blend = (1.0 - 0.35) * factor_alpha + 0.35 * ml_alpha_z
```

**Result:**
- `alpha_blend` - Final cross-sectional alpha driver
- Used to upgrade `TechRating` from v1 (heuristic) to v2 (hybrid)

---

### **3. TechRating v2 Upgrade (scanner_core.py lines 750-800)**

The blended alpha upgrades TechRating:

```python
# Scale alpha_blend into TechRating-like band
tr_from_alpha = alpha_blend * 10.0 + 15.0

# Blend with baseline TechRating (40% alpha weight)
TechRating_v2 = 0.6 * TechRating_v1 + 0.4 * tr_from_alpha
```

**Columns:**
- `TechRating_raw` - Original v1 heuristic score
- `TechRating` - Upgraded v2 hybrid score (factor + ML)

---

### **4. Expected Return (MuTotal) (scanner_core.py lines 1800-1900)**

ML alpha drives expected return calculations:

```python
# Heuristic drift from TechRating
MuHat = (TechRating - 15.0) / 20.0

# ML-based drift from AlphaScore
MuMl = AlphaScore.clip(-0.25, 0.25)

# Regime-aware blend (35% ML by default)
MuTotal = 0.65 * MuHat + 0.35 * MuMl
```

**Used For:**
- Portfolio optimization
- Risk-adjusted ranking
- Trade sizing

---

### **5. Regime-Aware Adjustments**

ML alpha weight adjusts based on market regime:

```python
# Base weight: 35%
w_mu = 0.35

# In steady uptrends + low vol: trust ML more
if trend == "TRENDING_UP" and vol == "LOW_VOL":
    w_mu = 0.50  # +15%

# In high vol: lean on technicals
elif vol == "HIGH_VOL":
    w_mu = 0.25  # -10%
```

---

## Verification in Logs

When scanner runs, you'll see:

```
[ALPHA] settings: use_ml_alpha=True use_meta_alpha=False alpha_weight=0.35
[ALPHA] ML alpha (5d+10d) blended with w5=0.40, w10=0.60
[ALPHA] blended factor + ML with TECHNIC_ALPHA_WEIGHT=0.35
```

---

## Impact on Results

### **Columns in Scan Results:**

**ML Alpha Columns:**
- `Alpha5d` - 5-day ML prediction
- `Alpha10d` - 10-day ML prediction
- `AlphaScore` - Blended ML alpha
- `AlphaScorePct` - Cross-sectional percentile (0-100)
- `ml_alpha_z` - Z-scored ML alpha

**Hybrid Columns:**
- `factor_alpha` - Pure technical z-score
- `alpha_blend` - Factor + ML blend
- `TechRating` - Upgraded with ML alpha
- `MuTotal` - Expected return (ML-enhanced)

**Quality Indicators:**
- `win_prob_10d` - Meta model win probability
- `InstitutionalCoreScore` - Uses ML alpha
- `MeritScore` - Uses ML alpha + quality

---

## How to Adjust ML Alpha Weight

### **Option 1: Environment Variable (Recommended)**

Set in Render:
```bash
TECHNIC_ALPHA_WEIGHT=0.50  # 50% ML, 50% factor
```

### **Option 2: Code Change**

Edit `technic_v4/config/settings.py`:
```python
alpha_weight: float = field(default=0.50)  # Change from 0.35
```

### **Weight Guidelines:**

| Weight | ML % | Factor % | Best For |
|--------|------|----------|----------|
| 0.20 | 20% | 80% | Conservative, trust technicals |
| 0.35 | 35% | 65% | **Default balanced** |
| 0.50 | 50% | 50% | Equal weight |
| 0.65 | 65% | 35% | Trust ML more |
| 0.80 | 80% | 20% | Aggressive ML-driven |

---

## Disable ML Alpha (Not Recommended)

If you want to disable ML alpha:

```bash
# Set environment variable
TECHNIC_USE_ML_ALPHA=0
```

**Result:**
- Scanner falls back to pure factor alpha
- `AlphaScore` will be NaN
- `TechRating` will be v1 (heuristic only)
- `MuTotal` will use `MuHat` only

---

## Current Status

✅ **ML Alpha: ENABLED**  
✅ **Alpha Weight: 35% (balanced)**  
✅ **Multi-horizon: 5d + 10d models**  
✅ **Regime-aware: Adjusts in high/low vol**  
✅ **Sector-aware: Contextual predictions**  
✅ **Quality-enhanced: Win probability meta model**

---

## Next Steps

1. **Deploy to Render** - ML alpha will be active
2. **Check logs** - Look for `[ALPHA]` messages
3. **Verify results** - Check `AlphaScore` column in CSV
4. **Monitor performance** - Compare ML vs factor alpha

---

## Summary

**Your scanner is FULLY utilizing ML alpha!**

- ✅ Enabled by default
- ✅ Blended with factor alpha (35/65 split)
- ✅ Upgrades TechRating from v1 to v2
- ✅ Drives expected return calculations
- ✅ Adjusts for market regime
- ✅ Sector-aware predictions

**No changes needed - ML alpha is working as designed!**
