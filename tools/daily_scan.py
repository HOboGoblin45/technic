from datetime import datetime

from technic_v4.scanner_core import run_scan, ScanConfig


def main():
    cfg = ScanConfig(
        max_symbols=25,
        trade_style="Short-term swing",
        min_tech_rating=10.0,
    )
    df, status = run_scan(cfg)
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"[{today}] {status}")
    try:
        print(df[["Symbol", "Signal", "TechRating", "Entry", "Stop", "Target"]].head(15))
    except Exception:
        print(df.head(15))

    out_path = f"scanner_output/daily_scan_{today}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved daily scan to {out_path}")


if __name__ == "__main__":
    main()
