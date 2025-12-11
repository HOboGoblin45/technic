from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Repo root: .../technic-clean/
ROOT = Path(__file__).resolve().parents[1]


def _run_step(name: str, cmd: list[str]) -> None:
    """
    Run a single pipeline step as a subprocess from the repo root.
    If anything fails, raise SystemExit so the scheduler can see a non-zero exit code.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    pretty_cmd = " ".join(cmd)
    print(f"[PIPELINE {ts}] START {name}: {pretty_cmd}", flush=True)

    # Always run from the repo root so relative imports/paths inside scripts work.
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        fail_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        print(
            f"[PIPELINE {fail_ts}] FAILED {name} "
            f"with exit code {result.returncode}",
            flush=True,
        )
        raise SystemExit(result.returncode)

    done_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[PIPELINE {done_ts}] DONE {name}", flush=True)


def main() -> None:
    """
    Nightly Technic pipeline:
      1) Refresh events calendar from FMP.
      2) Refresh fundamentals cache from FMP.
      3) Run nightly_maintenance (alpha retrain, TFT export, scoreboard update).
      4) Run alpha/ICS backtest suite for fresh quality metrics.

    This script is designed to be run once per night by a scheduler.
    """
    start = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[PIPELINE {start}] Nightly pipeline starting from {ROOT}", flush=True)

    py = sys.executable

    # 1) Events calendar (FMP)
    _run_step(
        "build_events_calendar",
        [py, "scripts/build_events_calendar.py"],
    )

    # 2) Fundamentals cache (FMP)
    _run_step(
        "build_fundamentals_cache",
        [py, "scripts/build_fundamentals_cache.py"],
    )

    # 3) Core engine maintenance: alpha retrain, TFT/ONNX export, scoreboard
    _run_step(
        "nightly_maintenance",
        [py, "scripts/nightly_maintenance.py"],
    )

    # 4) Alpha + ICS backtest suite (optional but recommended)
    # This uses the dev tools under technic_v4/dev/backtest
    _run_step(
        "run_alpha_score_suite",
        [py, "-m", "technic_v4.dev.backtest.run_alpha_score_suite"],
    )

    # If you have a summarizer for the suite, call it here as a final step.
    try:
        _run_step(
            "summarize_alpha_score_suite",
            [py, "-m", "technic_v4.dev.backtest.summarize_alpha_score_suite"],
        )
    except Exception as exc:
        # Do not kill the pipeline if summarization fails; just log it.
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        print(
            f"[PIPELINE {ts}] WARNING summarize_alpha_score_suite failed: {exc}",
            flush=True,
        )

    end = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[PIPELINE {end}] Nightly pipeline COMPLETE", flush=True)


if __name__ == "__main__":
    main()
