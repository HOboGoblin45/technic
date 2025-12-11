from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List


def _default_data_path(root: Path) -> Path:
    """
    Pick the best available backtest dataset, preferring the richer
    training_data_v2 parquet if present, otherwise falling back to the
    replay_ics parquet from scan history.
    """
    candidates: List[Path] = [
        root / "data" / "training_data_v2.parquet",
        root / "technic_v4" / "scanner_output" / "history" / "replay_ics.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No backtest dataset found. Looked for: {', '.join(str(p) for p in candidates)}"
    )


def _ensure_output_dir(root: Path) -> Path:
    out_dir = root / "evaluation" / "alpha_history_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_one(
    data_path: Path,
    score_col: str,
    label_col: str,
    top_n: int,
    out_dir: Path,
) -> None:
    """
    Call the existing evaluate_alpha_history CLI for a single
    (score, label) combination and write a JSON summary.
    """
    out_path = out_dir / f"{score_col}__{label_col}.json"

    cmd = [
        sys.executable,
        "-m",
        "technic_v4.dev.backtest.evaluate_alpha_history",
        "--data",
        str(data_path),
        "--score",
        score_col,
        "--label",
        label_col,
        "--top-n",
        str(top_n),
        "--out",
        str(out_path),
        "--skip-regime",
    ]

    print(f"[SUITE] Running backtest for score='{score_col}', label='{label_col}'")
    print(f"[SUITE] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[SUITE] ERROR for {score_col} / {label_col}")
        print(result.stderr)
        return

    # Echo the key bits of stdout so we can see metrics in the console
    print(result.stdout)

    # Sanity‑check that the JSON file is valid
    try:
        payload = json.loads(out_path.read_text())
    except Exception as exc:
        print(f"[SUITE] WARNING: failed to parse JSON output at {out_path}: {exc}")
        return

    print(
        f"[SUITE] Saved summary for {score_col}/{label_col} "
        f"with keys={sorted(payload.keys())} -> {out_path}"
    )


def main() -> None:
    """
    Run a small battery of backtests:
      - InstitutionalCoreScore vs 5d/10d returns
      - TechRating vs 5d/10d
      - alpha_blend vs 5d/10d
      - AlphaScorePct vs 5d/10d
      - ml_alpha_z vs 5d/10d (if present)

    This is the "prove the edge" suite for roadmap step #1.
    """
    # Repo root: .../technic-clean/
    root = Path(__file__).resolve().parents[3]
    data_path = _default_data_path(root)
    out_dir = _ensure_output_dir(root)

    print(f"[SUITE] Using data: {data_path}")
    print(f"[SUITE] Output dir: {out_dir}")

    score_cols: List[str] = [
        "InstitutionalCoreScore",
        "TechRating",
        "alpha_blend",
        "AlphaScorePct",
        "ml_alpha_z",
    ]
    label_cols: List[str] = ["fwd_ret_5d", "fwd_ret_10d"]
    top_n = 20  # evaluate "top 20" style portfolios

    # Filter to scores that actually exist in the dataset
    import pandas as pd

    # Read a small sample to detect available columns; some datasets use lowercase 'symbol'
    df_head = pd.read_parquet(data_path).head()
    if "symbol" in df_head.columns and "Symbol" not in df_head.columns:
        df_head = df_head.rename(columns={"symbol": "Symbol"})
    available_cols = set(df_head.columns)
    effective_scores = [c for c in score_cols if c in available_cols]

    print(f"[SUITE] Effective score columns in data: {effective_scores}")
    print(f"[SUITE] Label columns in data: {label_cols}")

    for score in effective_scores:
        for label in label_cols:
            if label not in available_cols:
                print(f"[SUITE] Skipping label {label} (not in dataset)")
                continue
            _run_one(
                data_path=data_path,
                score_col=score,
                label_col=label,
                top_n=top_n,
                out_dir=out_dir,
            )

    print("[SUITE] Alpha/ICS backtest suite finished.")


if __name__ == "__main__":
    main()
