import joblib

path = "models/alpha/xgb_v1.pkl"
bundle = joblib.load(path)

print("Bundle keys:", list(bundle.keys()))
features = bundle.get("features") or []
print(f"Num features: {len(features)}")
for i, name in enumerate(features, 1):
    print(f"{i:2d}. {name}")
