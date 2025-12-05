# Technic API (FastAPI)

Server entry: `uvicorn api:app --reload` (from repo root, venv activated).

## Health
- `GET /health`
  - Response: `{"status": "ok"}`

## Scan
- `GET /scan`
  - Query params:
    - `max_symbols` (int, 1–500, default 50)
    - `min_tech_rating` (float, default 0.0)
    - `sort_by` (column name, default `RewardRisk`)
    - `ascending` (bool, default false)
    - `include_log` (bool, default true)
  - Response (ScanResponse):
    - `results`: [{ticker, signal, rrr, entry, stop, target, techRating, riskScore, sector, industry}]
    - `movers`: [{ticker, delta, note, isPositive}]
    - `ideas`: [{title, ticker, meta, plan}]
    - `log` (optional, omitted if include_log=false)

## Options
- `GET /options/{ticker}`
  - Query params:
    - `direction` (call|put, default call)
    - `trade_style` (string, default Swing)
    - `tech_rating` (float, optional)
    - `risk_score` (float, optional)
    - `price_target` (float, optional)
  - Response (OptionsResponse):
    - `ticker`, `direction`, `trade_style`
    - `candidates`: list of contracts with fields:
      - ticker, contract_type, strike, expiration, dte, delta, iv, open_interest, volume,
        bid/ask/mid/last, spread_pct, breakeven, underlying, moneyness, score, reason

## Symbol detail
- `GET /symbol/{ticker}`
  - Query params: `days` (int, default 90), `intraday` (bool, default false)
  - Response (SymbolResponse):
    - `ticker`, `last`, `history` (list of {date, Open, High, Low, Close, Volume}),
    - `fundamentals` (dict),
    - `options_available` (bool)

## Universe stats
- `GET /universe_stats`
  - Response (UniverseStats):
    - `sectors`: [string]
    - `subindustries`: [string]

## Copilot
- `POST /copilot`
  - Body: `{"question": "..."}` 
  - Response: `{"answer": "..."}`

## Notes
- All endpoints return JSON.
- Ensure `.venv` is active and dependencies installed (`pip install -r requirements.txt`).
- If using environment keys (Polygon, OpenAI), set them before running.
