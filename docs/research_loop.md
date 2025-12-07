# Technic Research Loop (Phase 8)

## Goal
Maintain a disciplined, low-risk experimentation cycle: introduce improvements behind flags, evaluate with data, and only promote when they outperform.

## Cycle
1. Pick 1–2 ideas per quarter (e.g., new alpha model, new factor, new regime classifier).
2. Prototype behind Settings flags; do not change production defaults.
3. Evaluate using:
   - `python -m technic_v4.evaluation.backtest_compare`
   - Shadow mode logging (`evaluation/shadow_mode.py`, TECHNIC_ENABLE_SHADOW_MODE=true)
   - Paper portfolio simulation (optional: `python -m technic_v4.evaluation.paper_portfolio`)
4. Decide per idea: promote / keep experimental / kill.
5. Document changes in CHANGELOG.md and runbook.md as needed.

## Guardrails
- Never promote a model without outperformance on IC + Precision@N.
- Always run at least a short shadow-mode period before changing defaults.
- Keep production defaults stable; only flip flags after evidence and documentation.
