# Technic Runbook

## Overview
Technic is a modular scan engine for multi-factor equity/option idea generation. Core pieces:
- **data_engine**: unified facade over cached daily bars, Polygon fallback, fundamentals, and options chains.
- **feature_engine**: builds per-symbol feature vectors (returns, vol, trend, volume, fundamentals, optional TFT forecasts).
- **scoring**: TechRating + subscores with configurable weights; optional ML alpha (LGBM/XGB/ensemble) via lpha_inference.
- **regime_engine**: trend/vol regime tags for context-aware scoring.
- **options_suggest/ranking_engine**: optional option strategies and portfolio-aware ranking.
- **scanner_core.run_scan**: orchestrates universe prep → per-symbol processing → finalize/validation.
- **trade_management_engine**: basic adaptive stop/target suggestions.
- **api_server**: FastAPI wrapper exposing /v1/scan.

## How to run a scan locally
### CLI (direct)
`
python -m technic_v4.scanner_core
`
Returns a DataFrame and status text; writes CSV to 	echnic_v4/scanner_output/.

### Programmatic
`python
from technic_v4.scanner_core import run_scan, ScanConfig
cfg = ScanConfig(max_symbols=50, trade_style="Short-term swing")
df, status = run_scan(cfg)
print(status)
print(df.head())
`

### API server
`
python -m technic_v4.api_server  # starts FastAPI on 0.0.0.0:8000
`
POST /v1/scan with JSON body:
`json
{
  "max_symbols": 25,
  "trade_style": "Short-term swing",
  "min_tech_rating": 0
}
`
Response: status + esults list (symbol, signal, techRating, alphaScore, entry/stop/target, rationale, sector/industry).

## Backtests
- **Baseline vs New Engine:** python -m technic_v4.evaluation.backtest_compare
  - Compares baseline (no ML/TFT) vs full engine on a fixed universe/date range.
- **Eval harness:** python -m technic_v4.evaluation.eval_harness (or import acktest_top_n).
- **Latency bench:** python -m technic_v4.evaluation.latency_bench to time scans across universe sizes.

## Shadow mode (baseline vs new live logging)
- Enable via Settings env: TECHNIC_ENABLE_SHADOW_MODE=true.
- Use evaluation/shadow_mode.run_shadow_scan to capture baseline+new DataFrames and log to logs/shadow_signals.csv (append). UI still uses the "new" engine; shadow logs are for analysis.

## Retraining alpha models (high level)
1. Prepare training data (features via eature_engine.build_features; targets = forward returns).
2. Run training script (e.g., 	rain_loop.py or 	rain_lgbm_alpha.py) to fit model.
3. Register model via model_registry.register_model(name, path, metrics); ensure registry JSON under models/registry.json.
4. Promote best model (set active) in registry; lpha_inference loads the active model for inference.

## Deployment
- Container/Docker: ensure dependencies installed; run FastAPI with uvicorn:
  `ash
  uvicorn technic_v4.api_server:app --host 0.0.0.0 --port 8000
  `
- Env/Settings: use TECHNIC_ prefix. Common flags:
  - TECHNIC_USE_ML_ALPHA, TECHNIC_USE_TFT_FEATURES, TECHNIC_USE_META_ALPHA, TECHNIC_USE_ONNX_ALPHA
  - TECHNIC_DATA_CACHE_DIR, TECHNIC_MODELS_DIR
  - TECHNIC_ENABLE_SHADOW_MODE

## Troubleshooting
- **Empty results / missing data:** check data_cache availability; data_engine logs when falling back to Polygon.
- **API rate limits:** reduce max_symbols or scan frequency; consider cache pre-warm.
- **Unexpected outputs:** verify scoring weights (	echnic_v4/config/scoring_weights.json), Settings flags, and regime detection.
- **Logs:** centralized logger prints to stdout; recommendation logs in logs/recommendations.csv; shadow logs in logs/shadow_signals.csv.

## Extending
- Add new features in eature_engine; consume in scoring/ML.
- Add new alpha models in lpha_models and register via model_registry.
- UI remains minimal: display Symbol, Signal, TechRating/AlphaScore, Entry/Stop/Target, and short Rationale.
