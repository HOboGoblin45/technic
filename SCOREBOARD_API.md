# Scoreboard API (FastAPI)

## What it is
`scoreboard_service.py` is a tiny FastAPI app that persists per-user scoreboards to JSON on disk. UI can sync to it by setting:
- `SCOREBOARD_SYNC_URL` (e.g., `http://localhost:8000/scoreboard`)
- `SCOREBOARD_SYNC_TOKEN` (optional bearer token)
- `SCOREBOARD_USER_ID` (user key, default `default`)

## Run locally
```
uvicorn scoreboard_service:app --reload --port 8000
```

Or with token auth:
```
$env:SCOREBOARD_API_TOKEN="yoursecret"
uvicorn scoreboard_service:app --reload --port 8000
```

Default storage: `./data_cache/scoreboards/<user_id>.json`
Override with `SCOREBOARD_STORAGE_DIR`.

## Endpoints
- `GET /health` → `{"status": "ok"}`
- `GET /scoreboard/{user_id}` → `{"scoreboard": [...]}`
- `POST /scoreboard/{user_id}` with body `{"scoreboard": [...]}` → saves and returns `{"status":"saved","count":N}`

If `SCOREBOARD_API_TOKEN` is set, requests must include `Authorization: Bearer <token>`.

## UI wiring
Set the env vars before launching Streamlit:
```
$env:SCOREBOARD_SYNC_URL="http://localhost:8000/scoreboard"
$env:SCOREBOARD_SYNC_TOKEN="yoursecret"   # optional
$env:SCOREBOARD_USER_ID="user123"
streamlit run technic_v4/ui/technic_app.py
```

Reload button in the Scoreboard tab will pull from the API; saves happen on “Add to scoreboard,” clear, and add actions.

## Notes
- No DB; this is file-based. Good for local/dev; swap storage dir or add a DB layer if you need multi-instance durability.
- For production, front with HTTPS and real auth; keep the token secret.
