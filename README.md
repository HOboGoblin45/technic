# Technic v4

## Quickstart (UI)
1) Install deps: `pip install -r requirements.txt`
2) Set `POLYGON_API_KEY`
3) Run: `streamlit run technic_v4/ui/technic_app.py`

### Optional: scoreboard sync API
- Run API: `uvicorn scoreboard_service:app --reload --port 8000`
- Set env: `SCOREBOARD_SYNC_URL=http://localhost:8000/scoreboard`, plus optional `SCOREBOARD_SYNC_TOKEN` and `SCOREBOARD_USER_ID`

## Tests
- `python -m pytest tests`
  - To see warnings: `python -m pytest -q -W default -W once`
  - Warnings are filtered by default via `pytest.ini`.

## Environment variables
- `POLYGON_API_KEY` (required for prices/history)
- `FMP_API_KEY` (optional fundamentals)
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_USER_ID` (optional cloud sync)
- `SCOREBOARD_SYNC_URL`, `SCOREBOARD_SYNC_TOKEN`, `SCOREBOARD_USER_ID` (optional scoreboard sync)
- `TECHNIC_API_KEY` (optional; set to require auth on FastAPI endpoints)

Tip: on PowerShell, you can set in your profile:
```powershell
notepad $PROFILE  # then add lines like:
$env:POLYGON_API_KEY = "your_key"
$env:FMP_API_KEY = "your_key"
```
Reload with `. $PROFILE` or open a new shell.

## Health check
- Check env + basic commands: `python scripts/healthcheck.py`
- Run tests: `python -m pytest -q`

## Branding
- See `BRANDING.md` and `branding_notes.md`
- Assets in `assets/brand/`; PNG icons in `assets/brand/png/`

## API (FastAPI)

Optional REST surface for Flutter/clients:
```
python -m technic_v4.api_contract
```

Endpoints (honors `TECHNIC_API_KEY` if set):
- `GET /health`
- `POST /scan` body example:
  ```json
  {"limit":50,"offset":0,"max_symbols":100,"lookback_days":150,"min_tech_rating":0}
  ```
- `POST /options` body example:
  ```json
  {"symbol":"AAPL","direction":"call","limit":3}
  ```
- `GET /symbol/{symbol}` returns flags for history/fundamentals and a fundamentals snapshot.

Deps: `fastapi`, `uvicorn` (already in `requirements.txt`).
