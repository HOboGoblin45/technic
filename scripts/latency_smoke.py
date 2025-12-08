from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.config.settings import get_settings
import time


def main():
    settings = get_settings()
    cfg = ScanConfig(
        max_symbols=5000,            # stress test large universes
        lookback_days=90,
        min_tech_rating=0.0,
        trade_style="Short-term swing",
        allow_shorts=False,
        only_tradeable=False,
    )
    start = time.time()
    df, msg = run_scan(cfg)
    elapsed = time.time() - start

    print(f"Scan finished in {elapsed:.2f}s with {len(df)} results")
    print(f"Message: {msg}")
    print(f"use_ray={settings.use_ray}, max_workers={settings.max_workers}")


if __name__ == "__main__":
    main()
