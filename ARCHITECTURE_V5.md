## Technic v5 Architecture (Target State)

This document maps the current codebase to the planned v5 modular “engines” and defines responsibilities/IO so we can evolve incrementally without breaking existing behavior.

### Current Snapshot (v4.x)
- **data_layer/**: polygon client, price_layer (history), fundamentals cache, options_data (chains), relative_strength, market_cache.
- **engine/**: scoring (indicator-based TechRating), trade_planner (entries/stops/position size), options_selector, factor_engine (momentum/vol/liquidity/value/quality factors), regime_engine (trend/vol tags), portfolio_engine (risk-adjusted score/diversification), evaluation helpers.
- **scanner_core.py**: loads universe, filters by sector/industry/subindustry, fetches history, computes scores, applies alpha blend, plans trades, returns DataFrame.
- **ui/**: `technic_app.py` Streamlit mini-API (scanner/movers/ideas/scoreboard) and front-end surface.
- **api_contract.py**: Pydantic contracts + optional FastAPI adapter.
- **technic_app/** (Flutter): mobile client consuming scanner/movers/ideas endpoints.

### Target Modules (v5)
- **data_engine**: unified loaders for equity/option data (Polygon Massive), fundamentals, alt-data (news/sentiment/insider/macro). Caching + async/Ray hooks.
- **feature_engine**: factor computation (TA-Lib technicals, cross-sectional ranks, IV/HV gap, sentiment embeddings), feature store per symbol-date.
- **regime_engine**: HMM/vol-trend classifier with probabilities + tags exposed to models and scoring.
- **alpha_models**: tree baselines (LightGBM/XGBoost/CatBoost), deep time-series (LSTM/TCN/TFT) with multi-horizon outputs; SHAP hooks; ONNX export.
- **options_engine**: chain ingestion + scoring (delta/liquidity/IV surface/skew), strategy recommender (call/put spreads, defined-risk), ties to underlying alpha.
- **ranking_engine**: portfolio-aware ranking (utility, sector caps, correlation proxy), optional mean-variance weights (cvxpy), risk-adjusted score.
- **explainability_engine**: SHAP/LIME drivers → natural-language rationales for Copilot; factor/regime/option context.
- **evaluation_engine**: IC/precision@N/top-bottom spread, walk-forward CV, backtests/tear sheets.
- **inference_engine**: batch/stream scoring with Ray, ONNXRuntime-GPU fast path.

### Integration Path (incremental, non-breaking)
1) Document architecture (this file) and keep v4 behavior intact.
2) Add data_engine/feature_engine facades that wrap existing data_layer + factor_engine, exposing typed outputs for scanner_core.
3) Introduce alpha_models baseline (LightGBM) optional inference: populate MuMl/MuTotal when model artifact exists; fallback to current TechRating.
4) Extend options_engine to pull Polygon chains/Greeks and attach option picks to scan output; keep empty list fallback.
5) Enhance ranking_engine usage in scanner_core (utility + diversification); expose regime tags throughout.
6) Add explainability hooks (SHAP top drivers) into scan payload; surface in UI/Copilot text.
7) Add evaluation_engine utilities for offline validation; wire minimal tests.
8) Add inference_engine stubs for Ray/ONNX batch scoring; gated by availability.

All steps should keep Python 3.10 compatibility, prefer GPU (device="cuda") when used, and degrade gracefully to CPU.***
