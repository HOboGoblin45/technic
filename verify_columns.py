import pandas as pd

df = pd.read_csv('technic_v4/scanner_output/technic_scan_results.csv')

critical_cols = [
    'ATR_pct', 'Signal', 'TrendScore', 'MomentumScore', 
    'VolumeScore', 'VolatilityScore', 'OscillatorScore', 
    'BreakoutScore', 'ExplosivenessScore', 'RiskScore'
]

missing = [c for c in critical_cols if c not in df.columns]

print(f'Total columns: {len(df.columns)}')
print(f'Critical columns present: {len(critical_cols) - len(missing)}/{len(critical_cols)}')

if missing:
    print(f'Missing: {missing}')
else:
    print('✅ All critical columns present!')
    
print(f'\nSample values:')
print(f'  ATR_pct: {df["ATR_pct"].iloc[0]:.6f}')
print(f'  Signal: {df["Signal"].iloc[0]}')
print(f'  TrendScore: {df["TrendScore"].iloc[0]}')
print(f'  MomentumScore: {df["MomentumScore"].iloc[0]}')
print(f'  RiskScore: {df["RiskScore"].iloc[0]:.6f}')
