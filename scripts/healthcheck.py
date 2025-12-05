"""
Quick health check for Technic.

Usage:
    python scripts/healthcheck.py
"""
from __future__ import annotations

import os
import sys

# Minimal key to fetch prices/news; keep required small so dev still runs
REQUIRED_KEYS = ["POLYGON_API_KEY"]
# Optional keys for extended features (Copilot, auth, scoreboard sync, fundamentals, Supabase)
OPTIONAL_KEYS = [
    "OPENAI_API_KEY",        # Copilot
    "TECHNIC_API_KEY",       # Backend auth (FastAPI)
    "FMP_API_KEY",           # Fundamentals
    "SCOREBOARD_SYNC_URL",   # Scoreboard sync endpoint
    "SCOREBOARD_SYNC_TOKEN", # Token sent to sync endpoint
    "SCOREBOARD_API_TOKEN",  # Scoreboard service auth
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_USER_ID",
]


def check_env():
    missing = []
    present = []
    for k in REQUIRED_KEYS:
        if os.getenv(k):
            present.append(k)
        else:
            missing.append(k)
    optional_present = [k for k in OPTIONAL_KEYS if os.getenv(k)]
    return missing, present, optional_present


def main():
    missing, present, optional_present = check_env()
    print("=== Technic health check ===")
    if present:
        print(f"Required env OK: {', '.join(present)}")
    if missing:
        print(f"Missing required env: {', '.join(missing)}")
    if optional_present:
        print(f"Optional env present: {', '.join(optional_present)}")
    else:
        print("Optional env present: none")
    print("\nNext steps:")
    print(" - Run the app: streamlit run technic_v4/ui/technic_app.py")
    print(" - Run tests: python -m pytest -q")
    if missing:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
