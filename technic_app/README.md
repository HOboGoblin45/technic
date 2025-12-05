# technic_app

Quant-first technic shell wired for live backend + Copilot.

## Configure backend endpoints (Step 1)

The Flutter client reads endpoints from compile-time defines. With the embedded API server in `technic_v4/ui/technic_app.py`, use the JSON port (`TECHNIC_API_PORT`, defaults to `8051`). Pass these to `flutter run` (or your IDE run config):

- `--dart-define=TECHNIC_API_BASE=http://localhost:8052` (Android emulator: `http://10.0.2.2:8052`)
- `--dart-define=TECHNIC_API_SCANNER=/api/scanner`
- `--dart-define=TECHNIC_API_MOVERS=/api/movers`
- `--dart-define=TECHNIC_API_IDEAS=/api/ideas`
- `--dart-define=TECHNIC_API_SCOREBOARD=/api/scoreboard`
- `--dart-define=TECHNIC_API_COPILOT=/api/copilot`

Example:

```bash
flutter run \
  --dart-define=TECHNIC_API_BASE=https://streamlit.yourdomain.com \
  --dart-define=TECHNIC_API_SCANNER=/api/scanner \
  --dart-define=TECHNIC_API_MOVERS=/api/movers \
  --dart-define=TECHNIC_API_IDEAS=/api/ideas \
  --dart-define=TECHNIC_API_SCOREBOARD=/api/scoreboard \
  --dart-define=TECHNIC_API_COPILOT=/api/copilot
```

If your API already returns arrays under `data`, the client will consume it. If keys differ, adjust the parsers in `lib/main.dart` near `fromJson` factories.
