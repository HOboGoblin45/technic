"""
Generate Training Data for ML Models
Creates synthetic scan history for model training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import random
import numpy as np

from technic_v4.ml import ScanHistoryDB, ScanRecord, get_current_market_conditions


def generate_realistic_scan_data(num_scans: int = 150) -> list:
    """
    Generate realistic scan data for training
    
    Args:
        num_scans: Number of scan records to generate
    
    Returns:
        List of ScanRecord objects
    """
    print(f"Generating {num_scans} realistic scan records...")
    
    records = []
    base_date = datetime.now() - timedelta(days=90)
    
    # Sector options
    sectors_list = [
        ['Technology'],
        ['Healthcare'],
        ['Financial'],
        ['Technology', 'Healthcare'],
        ['Technology', 'Financial'],
        None  # All sectors
    ]
    
    # Market condition scenarios
    market_scenarios = [
        {'spy_trend': 'bullish', 'spy_volatility': 0.12, 'spy_momentum': 0.4},
        {'spy_trend': 'bullish', 'spy_volatility': 0.18, 'spy_momentum': 0.2},
        {'spy_trend': 'neutral', 'spy_volatility': 0.15, 'spy_momentum': 0.0},
        {'spy_trend': 'bearish', 'spy_volatility': 0.22, 'spy_momentum': -0.3},
        {'spy_trend': 'bearish', 'spy_volatility': 0.28, 'spy_momentum': -0.5},
    ]
    
    for i in range(num_scans):
        # Vary scan time across the day
        scan_date = base_date + timedelta(days=i * 90 / num_scans)
        scan_date = scan_date.replace(
            hour=random.randint(9, 16),
            minute=random.randint(0, 59)
        )
        
        # Random configuration
        max_symbols = random.choice([50, 75, 100, 150, 200])
        min_tech_rating = random.choice([10, 20, 30, 40, 50])
        min_dollar_vol = random.choice([1e6, 3e6, 5e6, 10e6])
        sectors = random.choice(sectors_list)
        lookback_days = random.choice([30, 60, 90, 120])
        use_alpha_blend = random.choice([True, False])
        
        config = {
            'max_symbols': max_symbols,
            'min_tech_rating': min_tech_rating,
            'min_dollar_vol': min_dollar_vol,
            'sectors': sectors,
            'lookback_days': lookback_days,
            'use_alpha_blend': use_alpha_blend,
            'enable_options': False,
            'profile': random.choice(['conservative', 'balanced', 'aggressive'])
        }
        
        # Market conditions
        market_scenario = random.choice(market_scenarios)
        market_conditions = {
            **market_scenario,
            'spy_return_5d': random.uniform(-0.05, 0.05),
            'spy_return_20d': random.uniform(-0.10, 0.10),
            'vix_level': 15 + random.uniform(-5, 10),
            'time_of_day': scan_date.hour,
            'day_of_week': scan_date.weekday(),
            'is_market_hours': 9 <= scan_date.hour <= 16
        }
        
        # Simulate results based on configuration
        # More symbols and lower thresholds = more results
        base_results = max_symbols * (1 - min_tech_rating / 100) * 0.5
        
        # Market conditions affect results
        if market_scenario['spy_trend'] == 'bullish':
            base_results *= 1.3
        elif market_scenario['spy_trend'] == 'bearish':
            base_results *= 0.7
        
        # Sector focus increases quality
        if sectors and len(sectors) <= 2:
            base_results *= 0.8  # Fewer but better quality
        
        # Add randomness
        result_count = int(base_results + random.uniform(-10, 10))
        result_count = max(0, min(result_count, max_symbols))
        
        # Signal rate depends on quality threshold
        signal_rate = 0.1 + (min_tech_rating / 100) * 0.3
        signals = int(result_count * signal_rate)
        
        results = {
            'count': result_count,
            'signals': signals
        }
        
        # Simulate duration based on complexity
        base_duration = max_symbols * 0.08  # 0.08s per symbol
        base_duration += lookback_days * 0.02  # Longer lookback = more time
        if use_alpha_blend:
            base_duration *= 1.2  # Alpha blend adds time
        if sectors:
            base_duration *= 0.9  # Sector filter reduces time
        
        # Add randomness
        duration = base_duration + random.uniform(-2, 2)
        duration = max(1.0, duration)
        
        performance = {
            'total_seconds': duration,
            'symbols_per_second': max_symbols / duration if duration > 0 else 0,
            'symbols_scanned': max_symbols,
            'symbols_returned': result_count
        }
        
        # Create record
        record = ScanRecord(
            scan_id=f"train_{i:04d}",
            timestamp=scan_date,
            config=config,
            results=results,
            performance=performance,
            market_conditions=market_conditions
        )
        
        records.append(record)
    
    return records


def main():
    """Generate and save training data"""
    print("="*60)
    print("TRAINING DATA GENERATION")
    print("="*60)
    
    # Generate data
    num_scans = 150
    records = generate_realistic_scan_data(num_scans)
    
    print(f"\n✓ Generated {len(records)} scan records")
    
    # Save to database
    db = ScanHistoryDB()
    
    print("\nSaving to database...")
    for i, record in enumerate(records):
        db.add_scan(record)
        if (i + 1) % 25 == 0:
            print(f"  Saved {i + 1}/{len(records)} records...")
    
    print(f"\n✓ Saved all records to database")
    
    # Show statistics
    stats = db.get_statistics()
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    print(f"Total scans: {stats['total_scans']}")
    print(f"Average results: {stats['avg_results']:.1f}")
    print(f"Average duration: {stats['avg_duration']:.1f}s")
    print(f"Total results: {stats['total_results']}")
    print(f"Total duration: {stats['total_duration']:.1f}s")
    
    if stats['date_range']:
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    print("\n✓ Training data generation complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/train_models.py")
    print("  2. Run: python scripts/validate_models.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
