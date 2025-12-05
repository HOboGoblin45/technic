# technic/test_run_scan.py

from technic_v4.scanner_core import run_scan, ScanConfig

if __name__ == "__main__":
    config = ScanConfig(mode="fast", max_symbols=10, lookback_days=150)
    df, fname = run_scan(config)
    print(f"Scan complete. Output file: {fname}")
    print(df.head(20))
