import pandas as pd
from pathlib import Path

DATA_DIR = Path("technic_v4/engine/scanner_output/history")
DATA_DIR.mkdir(exist_ok=True)

def load_history():
    dfs = []
    for file in DATA_DIR.glob("scan_*.csv"):
        df = pd.read_csv(file)
        df["scan_date"] = file.stem.split("_")[1]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def bucket_backtest(df):
    df = df.dropna(subset=["InstitutionalCoreScore", "AlphaScore"])
    df["bucket"] = pd.qcut(df["InstitutionalCoreScore"], 5, labels=False)

    return df.groupby("bucket")["AlphaScore"].mean()

if __name__ == "__main__":
    df = load_history()
    if df.empty:
        print("No history yet.")
    else:
        results = bucket_backtest(df)
        print("ICS bucket returns:")
        print(results)
