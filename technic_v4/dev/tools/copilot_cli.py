from __future__ import annotations

import argparse
import pandas as pd

from technic_v4.ui.generate_copilot_answer import generate_copilot_answer
from technic_v4.scanner_core import OUTPUT_DIR as SCAN_OUTPUT_DIR


def _load_row(symbol: str) -> pd.Series | None:
    """
    Load the first matching row for a symbol from the latest scan CSV.
    """
    path = SCAN_OUTPUT_DIR / "technic_scan_results.csv"
    if not path.exists():
        print(f"No scan results found at {path}")
        return None
    df = pd.read_csv(path)
    if "Symbol" not in df.columns:
        print("Scan results missing 'Symbol' column")
        return None
    sym = symbol.upper().strip()
    mask = df["Symbol"].astype(str).str.upper().str.strip() == sym
    sub = df.loc[mask]
    if sub.empty:
        print(f"No rows found for symbol={sym}")
        return None
    return sub.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Technic Copilot CLI")
    parser.add_argument("symbol", help="Ticker symbol, e.g. ODP")
    parser.add_argument(
        "-q",
        "--question",
        default="Explain this setup and suggest a simple trade idea for a novice trader.",
        help="Question to ask the Copilot",
    )
    args = parser.parse_args()

    row = _load_row(args.symbol)
    answer = generate_copilot_answer(question=args.question, row=row)
    print()
    print("=== COPILOT ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
