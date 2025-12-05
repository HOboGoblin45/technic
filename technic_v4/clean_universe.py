import pandas as pd

INPUT_FILE = "ticker_universe.csv"
OUTPUT_FILE = "ticker_universe_cleaned.csv"

def main():
    # 1) Load the original universe
    df = pd.read_csv(INPUT_FILE)

    # 2) Drop rows with missing symbols
    df = df.dropna(subset=["symbol"])

    # 3) Normalize the symbol text
    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 4) Drop duplicate symbols, just in case
    df = df.drop_duplicates(subset=["symbol"])

    # 5) Sort for neatness
    df = df.sort_values("symbol").reset_index(drop=True)

    # 6) Save cleaned file
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned universe saved to {OUTPUT_FILE}")
    print(f"Total tickers: {len(df)}")

if __name__ == "__main__":
    main()
