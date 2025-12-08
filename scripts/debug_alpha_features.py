import sys
from pathlib import Path

# --- Ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.engine import alpha_inference


def main():
    # Load the model bundle and see what it expects
    bundle = joblib.load("models/alpha/xgb_v1.pkl")
    features = bundle.get("features") or []
    print(f"Model expects {len(features)} features:")
    for i, name in enumerate(features, 1):
        print(f"  {i:2d}. {name}")

    # Run a small scan to keep this fast
    cfg = ScanConfig(
        max_symbols=200,
        lookback_days=90,
        min_tech_rating=0.0,
        trade_style="Short-term swing",
        allow_shorts=False,
        only_tradeable=False,
    )
    df, msg = run_scan(cfg)
    print("\nScan msg:", msg)
    print("Scan shape:", df.shape)
    print("Scan columns:", list(df.columns))

    # Check which expected features are missing from the scan DataFrame
    missing = [c for c in features if c not in df.columns]
    print(f"\nMissing features ({len(missing)}): {missing}")

    # Try calling score_alpha explicitly and see what happens
    alpha = alpha_inference.score_alpha(df)
    print("\nscore_alpha result type:", type(alpha))
    if alpha is None:
        print("score_alpha returned None")
    else:
        print("score_alpha length:", len(alpha))
        print("alpha head:")
        print(alpha.head())


if __name__ == "__main__":
    main()
