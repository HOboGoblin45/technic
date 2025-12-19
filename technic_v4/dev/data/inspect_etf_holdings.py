from technic_v4.data_layer.etf_holdings import (
    load_etf_holdings,
    get_etf_holdings,
    get_asset_exposure,
)


def main() -> None:
    df = load_etf_holdings()
    print(f"Loaded ETF holdings: {len(df)} rows, {df['etf_symbol'].nunique()} ETFs")

    # Example: what does SPY hold?
    spy = get_etf_holdings("SPY", df)
    print("\nTop SPY holdings:")
    print(spy.head(10))

    # Example: which ETFs hold AAPL?
    aapl = get_asset_exposure("AAPL", df)
    print("\nTop ETFs holding AAPL:")
    print(aapl.head(10))


if __name__ == "__main__":
    main()
