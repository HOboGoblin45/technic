from technic_v4.config.settings import get_settings
from technic_v4.scanner_core import run_scan, ScanConfig

def main():
    settings = get_settings()
    settings.use_ml_alpha = True
    settings.use_meta_alpha = False

    print(f"use_ml_alpha={settings.use_ml_alpha}, use_meta_alpha={settings.use_meta_alpha}")

    cfg = ScanConfig(
        max_symbols=100,
        lookback_days=150,
        min_tech_rating=0.0,
        trade_style="Short-term swing",
        only_tradeable=False,
    )

    df, status = run_scan(cfg)
    print("Status:", status)
    if df is None or df.empty:
        print("No results.")
        return

    cols = [c for c in ["Alpha5d", "Alpha10d", "AlphaScore", "ml_alpha_z", "alpha_blend"] if c in df.columns]
    print("Available alpha columns:", cols)

    print(df[["Symbol"] + cols].head(20))

if __name__ == "__main__":
    main()
