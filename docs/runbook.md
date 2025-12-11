# Technic Runbook

## Overview
Technic is a backend-first alpha scanner. Core pieces: a cache-first **data_engine** (Polygon fallback for OHLCV, fundamentals, options), **feature_engine** (returns/vol/trend/volume/fundamentals, optional TFT forecasts), **scoring** (TechRating + configurable subscore weights) plus optional ML alpha (LGBM/XGB/ensemble), **regime_engine** (trend/vol tags), **options** helpers, **ranking** for portfolio-aware sorting, **explainability** (rationales/SHAP hook), and **trade_management_engine** for adaptive stops/targets.

## Running a scan locally
```python
from technic_v4.scanner_core import run_scan, ScanConfig
cfg = ScanConfig(max_symbols=50, trade_style="Short-term swing")
df, status = run_scan(cfg)
print(status)
print(df.head())
```
CLI: `python -m technic_v4.scanner_core`

## HTTP API usage
Start the API server:
```bash
uvicorn technic_v4.api_server:app --host 0.0.0.0 --port 8502
```
Request:
```json
POST /v1/scan
Headers: {"X-API-Key": "<key>"}
Body: {"max_symbols":25, "trade_style":"Short-term swing", "min_tech_rating":0}
```
Response includes status, disclaimer, and results (symbol, signal, techRating, alphaScore, entry/stop/target, rationale, optionTrade).

## Backtesting & shadow mode
- `python -m technic_v4.evaluation.backtest_compare` compares baseline vs full engine on a fixed universe/date range.
- `evaluation/shadow_mode.py` can log baseline vs new engine side-by-side (enable `TECHNIC_ENABLE_SHADOW_MODE=true` and append logs to `logs/shadow_signals.csv`).
- `python -m technic_v4.dev.backtest.run_alpha_score_suite` runs a small backtest suite for **InstitutionalCoreScore**, **TechRating**, **alpha_blend**, **AlphaScorePct**, and **ml_alpha_z** against the historical training dataset (`data/training_data_v2.parquet` or `technic_v4/scanner_output/history/replay_ics.parquet`). JSON summaries are written to `evaluation/alpha_history_suite/` and the key metrics are printed to stdout.
- `python -m technic_v4.dev.backtest.summarize_alpha_score_suite` reads all JSON summaries in `evaluation/alpha_history_suite/` (from `run_alpha_score_suite`) and prints a comparative report of **InstitutionalCoreScore**, **TechRating**, **alpha_blend**, **AlphaScorePct**, and **ml_alpha_z** across 5d/10d horizons, showing test-period forward returns, precision@N, daily IC, and top-vs-bottom bucket separation.
- `python -m technic_v4.dev.backtest.suggest_score_thresholds` loads the same training/replay dataset and, for each score (InstitutionalCoreScore, TechRating, alpha_blend, AlphaScorePct, ml_alpha_z) and forward-return label (fwd_ret_5d, fwd_ret_10d), evaluates a range of score quantile thresholds (e.g., 50th, 60th, 70th, 80th, 85th, 90th). For each threshold it reports sample size, mean forward return, win rate, and excess performance versus the baseline, then suggests a recommended minimum score per (score, label) pair. Use these to set min ICS/TechRating/alpha_blend cutoffs for Investor Core vs Satellite lists.

## Latency benchmarks
- `python -m technic_v4.evaluation.latency_bench` runs scans across universe sizes and logs elapsed time and result counts.

## Tests & CI
Run tests: `pytest -q` or `make test`. Coverage includes scanner orchestration smoke tests and eval harness smoke tests.

## Deployment
- Docker: `docker compose up --build` (Dockerfile runs uvicorn on 8502).
- Consider reverse proxy + HTTPS; healthcheck endpoint `/health`.
- Env via `TECHNIC_` prefix (e.g., `TECHNIC_API_KEY`, `TECHNIC_USE_ML_ALPHA`, cache/model dirs).

## Retraining alpha models
1) Build features/targets. 2) Train (e.g., train_lgbm_alpha / train_loop). 3) Register via `model_registry` (metrics in models/registry.json). 4) Backtest & shadow. 5) Promote active model (alpha_inference loads active).

## Paper portfolio
`python -m technic_v4.evaluation.paper_portfolio` simulates static vs adaptive exits using `trade_management_engine` suggestions.

## Troubleshooting
- **No results**: check data cache; data_engine logs cache/API fallback.
- **API errors**: verify `TECHNIC_API_KEY`, request body, and server logs.
- **Slow scans**: lower `max_symbols`, disable ML/TFT via settings, check latency bench.
- Logs: stdout logger; recommendations in `logs/recommendations.csv`; shadow logs in `logs/shadow_signals.csv`; scan CSV in `technic_v4/scanner_output/`.
