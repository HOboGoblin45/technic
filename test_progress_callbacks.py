"""
Test progress callbacks for Phase 3D-A
"""

import os
os.environ['REDIS_URL'] = 'redis://:ytvZ1OVXoGV4OenJH3GJYkDLeg2emqad@redis-12579.fcrce190.us-east-1-1.ec2.cloud.redislabs.com:12579/0'

from technic_v4.scanner_core import run_scan, ScanConfig

# Track progress updates
progress_updates = []

def progress_callback(stage, current, total, message, metadata):
    """Capture progress updates"""
    progress_pct = (current / total * 100) if total > 0 else 0
    progress_updates.append({
        'stage': stage,
        'current': current,
        'total': total,
        'message': message,
        'metadata': metadata,
        'progress_pct': progress_pct
    })
    
    # Print progress
    print(f"[PROGRESS] {stage.upper()}: {current}/{total} ({progress_pct:.1f}%) - {message}")
    if metadata and 'symbol' in metadata:
        print(f"           Symbol: {metadata['symbol']} | Sector: {metadata.get('sector', 'Unknown')}")

print("="*80)
print("TESTING PROGRESS CALLBACKS")
print("="*80)

# Run small scan with progress tracking
config = ScanConfig(
    max_symbols=10,
    lookback_days=90,
    trade_style="Short-term swing"
)

print("\n[TEST] Running scan with progress callbacks...")
print()

df, msg = run_scan(config, progress_cb=progress_callback)

print("\n" + "="*80)
print("PROGRESS CALLBACK RESULTS")
print("="*80)

print(f"\nTotal progress updates received: {len(progress_updates)}")

if progress_updates:
    print("\nFirst update:")
    print(f"  Stage: {progress_updates[0]['stage']}")
    print(f"  Message: {progress_updates[0]['message']}")
    print(f"  Progress: {progress_updates[0]['current']}/{progress_updates[0]['total']}")
    
    print("\nLast update:")
    print(f"  Stage: {progress_updates[-1]['stage']}")
    print(f"  Message: {progress_updates[-1]['message']}")
    print(f"  Progress: {progress_updates[-1]['current']}/{progress_updates[-1]['total']}")
    
    # Check if we got symbol-level updates
    symbol_updates = [u for u in progress_updates if 'symbol' in u.get('metadata', {})]
    print(f"\nSymbol-level updates: {len(symbol_updates)}")
    
    if symbol_updates:
        print("\nSample symbol updates:")
        for update in symbol_updates[:3]:
            meta = update['metadata']
            print(f"  - {meta['symbol']} ({meta.get('sector', 'Unknown')}): {update['progress_pct']:.1f}%")
    
    print("\n✅ Progress callbacks are working!")
else:
    print("\n⚠️  No progress updates received")

print(f"\nScan completed: {len(df)} results")
print(f"Status: {msg}")

print("\n" + "="*80)
