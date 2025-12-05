# Repository Guidelines

## Project Structure & Module Organization
- `technic_v4/`: Python core. `data_layer/` (prices, options, fundamentals, caches), `engine/` (scoring, trade planning, options selection), `scanner_core.py` (writes CSVs under `technic_v4/scanner_output/`), `api_contract.py` sample FastAPI payloads, `ui/technic_app.py` Streamlit surface.
- `scripts/`: operational helpers (`healthcheck.py`, `ingest_fundamentals.py`, `export_icons.py`).
- `technic_app/`: Flutter client; endpoints configured via `--dart-define` flags (see `technic_app/README.md`).
- Support files: caches in `data_cache/`, branding assets in `assets/`, API docs `API.md`, scoreboard outline `SCOREBOARD_API.md`.

## Build, Test, and Development Commands
- Python setup: `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`.
- Env check: `python scripts/healthcheck.py` to verify keys and network basics.
- Streamlit UI: `streamlit run technic_v4/ui/technic_app.py` (set `POLYGON_API_KEY`; optional `FMP_API_KEY`, `TECHNIC_API_KEY`, scoreboard sync envs).
- FastAPI: `uvicorn api:app --reload --port 8000` (optionally `scoreboard_service:app` for sync target).
- Scanner plumbing demo: `python technic_v4/api_contract.py` to exercise scan/output schema; prune generated CSVs before committing.
- Flutter: `cd technic_app && flutter pub get && flutter run --dart-define=TECHNIC_API_BASE=...`; `flutter test` for client checks.

## Coding Style & Naming Conventions
- Python: PEP8, 4-space indents, type hints, dataclasses for API payloads; snake_case for names, ALL_CAPS for constants. Keep functions pure where possible and wrap network calls with timeouts/retries.
- Data/outputs: keep `technic_v4/scanner_output/`, `data_cache/`, and exported assets out of commits; refresh universes via loaders instead of manual edits.
- Dart/Flutter: run `dart format .` and `flutter analyze`; prefer small composable widgets, const constructors, and snake_case file names under `lib/`.

## Testing Guidelines
- Run `python -m pytest tests technic_v4/tests` (warnings filtered by `pytest.ini`).
- For UI/API changes, hit `http://localhost:8000/health` and load the Streamlit app; confirm scoreboard CSVs still read/write locally.
- Avoid live-market regressions by using `scripts/healthcheck.py` and cached data in `data_cache/` when possible.

## Commit & Pull Request Guidelines
- Use short imperative subjects (e.g., "Add option scan params", "Tighten price fallback").
- In PRs, list commands run, env vars needed, and whether you touched `data_cache/` or `scanner_output/`; attach screenshots/gifs for Streamlit or Flutter UI tweaks and link issues/tasks.
- Do not commit secrets, `data_cache/`, generated CSVs, or `technic_app/build/` artifacts.

## Security & Configuration Tips
- Prefer env vars over `technic_v4/config.py`; never commit real keys. Set `POLYGON_API_KEY`, `FMP_API_KEY`, `TECHNIC_API_KEY`, and optional Supabase/scoreboard vars in your shell profile.
- Scrub logs before sharing; API responses may include ticker and price details.
