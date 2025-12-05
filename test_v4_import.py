from technic_v4.universe_loader import load_universe

if __name__ == "__main__":
    universe = load_universe()
    print(f"Loaded {len(universe)} tickers.")
    print("First 10:", [u.symbol for u in universe[:10]])
