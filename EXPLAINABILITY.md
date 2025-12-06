## Technic Model Drivers (Glossary)

These feature names appear in explanations (SHAP “model drivers”) and in ranking outputs. Higher values generally help the score unless noted.

- **1m momentum (mom_21)**: 1‑month return.
- **3m momentum (mom_63)**: 3‑month return.
- **5d reversal (reversal_5)**: Short-term mean reversion; high value can hurt trend setups.
- **MA20 slope (ma_slope_20)**: Slope of 20‑day moving average (trend strength).
- **E/P value (value_ep)**: Earnings yield (inverse of P/E); higher = cheaper.
- **CF/P value (value_cfp)**: Cash flow yield; higher = cheaper.
- **ROE quality (quality_roe)**: Return on equity; higher = better quality.
- **Gross margin (quality_gpm)**: Profitability measure; higher = better quality.
- **ATR% (atr_pct_14)**: Average True Range as % of price; lower = calmer volatility.
- **Realized vol (vol_realized_20)**: 20‑day historical volatility; lower = calmer volatility.
- **Dollar volume (dollar_vol_20)**: Average daily dollar turnover; higher = more liquid.
- **Breakout / Explosiveness scores**: Capture strong upside follow‑through behavior.
- **Trend strength (TrendStrength50)**: Longer-term trend measure.
- **MomentumScore / VolatilityScore**: Composite momentum/volatility factors used in TechRating.

Interpretation tip: In explanations, a positive contribution (e.g., “1m momentum +0.30”) means that factor pushed the model score up; negative means it dragged the score down.
