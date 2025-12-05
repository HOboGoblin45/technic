from technic_v4.data_layer.polygon_client import get_stock_history_df

if __name__ == "__main__":
    df = get_stock_history_df("AAPL", days=30)

    if df is None:
        print("No data returned.")
    else:
        print(df.tail())
        print()
        print("Shape:", df.shape)
