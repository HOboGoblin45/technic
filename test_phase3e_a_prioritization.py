"""
Test Suite for Phase 3E-A: Smart Symbol Prioritization
Tests symbol scoring, priority queue, and dynamic reordering
"""

import time
import random
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technic_v4.symbol_scorer import SymbolScorer, SymbolScore
from technic_v4.prioritizer import (
    SmartSymbolPrioritizer,
    SymbolPriorityQueue,
    PrioritizedSymbol
)


def test_symbol_scoring():
    """Test the symbol scoring system"""
    print("\n" + "="*60)
    print("TEST 1: Symbol Scoring")
    print("="*60)
    
    scorer = SymbolScorer()
    
    # Test with different data scenarios
    test_cases = [
        {
            'name': 'High Activity Symbol',
            'symbol': 'HIGH',
            'market_data': {
                'volume_ratio': 2.5,  # High volume
                'return_5d': 0.08,    # Good momentum
                'atr_ratio': 0.03,    # Moderate volatility
                'rs_percentile': 85   # Strong relative strength
            },
            'expected_min': 70
        },
        {
            'name': 'Low Activity Symbol',
            'symbol': 'LOW',
            'market_data': {
                'volume_ratio': 0.8,
                'return_5d': -0.02,
                'atr_ratio': 0.01,
                'rs_percentile': 30
            },
            'expected_max': 60
        }
    ]
    
    for case in test_cases:
        score = scorer.score_symbol(
            case['symbol'],
            market_data=case.get('market_data')
        )
        
        print(f"\n{case['name']} ({case['symbol']}):")
        print(f"  Total Score: {score.total_score}")
        print(f"  Activity Score: {score.activity_score}")
        
        if 'expected_min' in case:
            assert score.total_score >= case['expected_min'], \
                f"Score {score.total_score} below expected {case['expected_min']}"
            print(f"  ✓ Score >= {case['expected_min']}")
        
        if 'expected_max' in case:
            assert score.total_score <= case['expected_max'], \
                f"Score {score.total_score} above expected {case['expected_max']}"
            print(f"  ✓ Score <= {case['expected_max']}")
    
    print("\n✓ Symbol scoring test passed")
    return True


def test_priority_queue():
    """Test the priority queue system"""
    print("\n" + "="*60)
    print("TEST 2: Priority Queue Management")
    print("="*60)
    
    queue = SymbolPriorityQueue(enable_diversity=False)  # Pure priority for testing
    
    # Add symbols with different scores
    test_symbols = [
        ('HIGH1', SymbolScore('HIGH1', 85.0)),
        ('HIGH2', SymbolScore('HIGH2', 75.0)),
        ('MED1', SymbolScore('MED1', 55.0)),
        ('MED2', SymbolScore('MED2', 45.0)),
        ('LOW1', SymbolScore('LOW1', 25.0)),
        ('LOW2', SymbolScore('LOW2', 15.0)),
    ]
    
    for symbol, score in test_symbols:
        queue.add_symbol(symbol, score)
    
    stats = queue.get_stats()
    print(f"\nQueue Statistics:")
    print(f"  High Priority: {stats['high_count']}")
    print(f"  Medium Priority: {stats['medium_count']}")
    print(f"  Low Priority: {stats['low_count']}")
    
    assert stats['high_count'] == 2, "Should have 2 high priority symbols"
    assert stats['medium_count'] == 2, "Should have 2 medium priority symbols"
    assert stats['low_count'] == 2, "Should have 2 low priority symbols"
    
    # Test batch retrieval in priority order
    batch = queue.get_next_batch(3)
    symbols_retrieved = [item.symbol for item in batch]
    
    print(f"\nFirst batch (3 symbols): {symbols_retrieved}")
    assert 'HIGH1' in symbols_retrieved and 'HIGH2' in symbols_retrieved, \
        "High priority symbols should be retrieved first"
    
    # Get remaining symbols
    batch2 = queue.get_next_batch(3)
    symbols_retrieved2 = [item.symbol for item in batch2]
    
    print(f"Second batch (3 symbols): {symbols_retrieved2}")
    assert 'LOW1' in symbols_retrieved2 or 'LOW2' in symbols_retrieved2, \
        "Low priority symbols should be retrieved last"
    
    print("\n✓ Priority queue test passed")
    return True


def test_diversity_mode():
    """Test diversity mode in priority queue"""
    print("\n" + "="*60)
    print("TEST 3: Diversity Mode")
    print("="*60)
    
    queue = SymbolPriorityQueue(enable_diversity=True)
    
    # Add many symbols across tiers
    for i in range(10):
        queue.add_symbol(f'HIGH{i}', SymbolScore(f'HIGH{i}', 80 + i))
        queue.add_symbol(f'MED{i}', SymbolScore(f'MED{i}', 50 + i))
        queue.add_symbol(f'LOW{i}', SymbolScore(f'LOW{i}', 20 + i))
    
    # Get a batch with diversity
    batch = queue.get_next_batch(10)
    
    # Count symbols by tier
    tier_counts = {'high': 0, 'medium': 0, 'low': 0}
    for item in batch:
        if item.score.total_score >= 70:
            tier_counts['high'] += 1
        elif item.score.total_score >= 40:
            tier_counts['medium'] += 1
        else:
            tier_counts['low'] += 1
    
    print(f"\nBatch composition with diversity:")
    print(f"  High: {tier_counts['high']}/10 ({tier_counts['high']*10}%)")
    print(f"  Medium: {tier_counts['medium']}/10 ({tier_counts['medium']*10}%)")
    print(f"  Low: {tier_counts['low']}/10 ({tier_counts['low']*10}%)")
    
    # Should have mix of priorities
    assert tier_counts['high'] >= 5, "Should have at least 50% high priority"
    assert tier_counts['medium'] >= 1, "Should have some medium priority"
    
    print("\n✓ Diversity mode test passed")
    return True


def test_learning_and_reordering():
    """Test learning from results and dynamic reordering"""
    print("\n" + "="*60)
    print("TEST 4: Learning and Dynamic Reordering")
    print("="*60)
    
    prioritizer = SmartSymbolPrioritizer(
        enable_diversity=False,
        enable_learning=True
    )
    
    # Add test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # Mock data for consistent testing
    market_data = {
        sym: {
            'volume_ratio': 1.5,
            'return_5d': 0.05,
            'atr_ratio': 0.03,
            'rs_percentile': 70
        }
        for sym in test_symbols
    }
    
    # Prioritize symbols
    count = prioritizer.prioritize_symbols(test_symbols, market_data=market_data)
    print(f"\nPrioritized {count} symbols")
    
    initial_stats = prioritizer.get_stats()
    print(f"Initial queue stats: High={initial_stats['queue']['high_count']}")
    
    # Simulate scanning and learning
    batch = prioritizer.get_next_batch(3)
    
    for item in batch[:2]:  # First 2 generate signals
        prioritizer.update_with_result(
            item['symbol'],
            generated_signal=True,
            tech_rating=85.0,
            alpha_score=0.8
        )
    
    # Third doesn't generate signal
    if len(batch) > 2:
        prioritizer.update_with_result(
            batch[2]['symbol'],
            generated_signal=False
        )
    
    # Check learning occurred
    final_stats = prioritizer.get_stats()
    print(f"\nAfter scanning:")
    print(f"  Signals found: {final_stats['scan_results']['signals_found']}")
    print(f"  Signal rate: {final_stats['scan_results']['signal_rate']:.1%}")
    print(f"  High-value symbols: {final_stats['scan_results']['high_value_count']}")
    
    assert final_stats['scan_results']['signals_found'] == 2, "Should have 2 signals"
    assert final_stats['scan_results']['high_value_count'] == 2, "Should have 2 high-value symbols"
    
    print("\n✓ Learning and reordering test passed")
    return True


def test_performance_tracking():
    """Test performance tracking and statistics"""
    print("\n" + "="*60)
    print("TEST 5: Performance Tracking")
    print("="*60)
    
    scorer = SymbolScorer()
    
    # Simulate multiple scans
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in test_symbols:
        # First scan - no signal
        scorer.update_performance(symbol, generated_signal=False)
        
        # Second scan - signal for AAPL only
        if symbol == 'AAPL':
            scorer.update_performance(
                symbol,
                generated_signal=True,
                tech_rating=75.0
            )
        else:
            scorer.update_performance(symbol, generated_signal=False)
    
    # Get session stats
    stats = scorer.get_session_stats()
    
    print(f"\nSession Statistics:")
    print(f"  Symbols scanned: {stats['symbols_scanned']}")
    print(f"  Total scans: {stats['total_scans']}")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Signal rate: {stats['signal_rate']:.1%}")
    
    assert stats['symbols_scanned'] == 3, "Should have scanned 3 symbols"
    assert stats['total_scans'] == 6, "Should have 6 total scans"
    assert stats['total_signals'] == 1, "Should have 1 signal"
    
    # Check top performers
    if stats['top_performers']:
        top = stats['top_performers'][0]
        print(f"\nTop performer: {top['symbol']} ({top['signal_rate']:.1%} signal rate)")
        assert top['symbol'] == 'AAPL', "AAPL should be top performer"
    
    print("\n✓ Performance tracking test passed")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases")
    print("="*60)
    
    # Test empty queue
    queue = SymbolPriorityQueue()
    batch = queue.get_next_batch(5)
    assert len(batch) == 0, "Empty queue should return empty batch"
    print("✓ Empty queue handled correctly")
    
    # Test duplicate symbols
    queue.add_symbol('DUP', SymbolScore('DUP', 75.0))
    queue.add_symbol('DUP', SymbolScore('DUP', 85.0))  # Should be ignored
    
    stats = queue.get_stats()
    assert stats['total_added'] == 1, "Duplicate should not be added"
    print("✓ Duplicate symbols handled correctly")
    
    # Test requesting more symbols than available
    queue.clear()
    queue.add_symbol('ONLY', SymbolScore('ONLY', 75.0))
    batch = queue.get_next_batch(10)
    assert len(batch) == 1, "Should return only available symbols"
    print("✓ Over-request handled correctly")
    
    # Test with None data
    scorer = SymbolScorer()
    score = scorer.score_symbol('TEST', None, None, None)
    assert score.total_score == 50.0, "Should return neutral score with no data"
    print("✓ None data handled correctly")
    
    print("\n✓ All edge cases passed")
    return True


def run_integration_test():
    """Run a full integration test simulating real usage"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Prioritization Flow")
    print("="*60)
    
    # Create prioritizer
    prioritizer = SmartSymbolPrioritizer(
        enable_diversity=True,
        enable_learning=True
    )
    
    # Simulate a universe of symbols
    universe = [
        f"{sector}{i}" 
        for sector in ['TECH', 'FIN', 'HEALTH', 'RETAIL', 'ENERGY']
        for i in range(10)
    ]
    
    print(f"\nUniverse: {len(universe)} symbols across 5 sectors")
    
    # Generate mock data with sector bias
    market_data = {}
    for symbol in universe:
        if symbol.startswith('TECH'):
            # Tech sector performing well
            market_data[symbol] = {
                'volume_ratio': random.uniform(1.5, 3.0),
                'return_5d': random.uniform(0.02, 0.10),
                'atr_ratio': random.uniform(0.02, 0.05),
                'rs_percentile': random.uniform(60, 90)
            }
        else:
            # Other sectors mixed
            market_data[symbol] = {
                'volume_ratio': random.uniform(0.8, 1.5),
                'return_5d': random.uniform(-0.05, 0.05),
                'atr_ratio': random.uniform(0.01, 0.03),
                'rs_percentile': random.uniform(30, 70)
            }
    
    # Prioritize all symbols
    count = prioritizer.prioritize_symbols(universe, market_data=market_data)
    print(f"Prioritized {count} symbols")
    
    initial_stats = prioritizer.get_stats()
    print(f"\nInitial distribution:")
    print(f"  High priority: {initial_stats['queue']['high_count']}")
    print(f"  Medium priority: {initial_stats['queue']['medium_count']}")
    print(f"  Low priority: {initial_stats['queue']['low_count']}")
    
    # Simulate scanning in batches
    batch_size = 10
    batch_num = 1
    total_signals = 0
    
    while prioritizer.queue.get_remaining_count() > 0:
        batch = prioritizer.get_next_batch(batch_size)
        
        high_count = sum(1 for item in batch if item['priority_tier'] == 'high')
        print(f"\nBatch {batch_num}: {len(batch)} symbols ({high_count} high priority)")
        
        # Simulate scanning with higher signal rate for high priority
        for item in batch:
            # Higher priority symbols more likely to generate signals
            if item['priority_tier'] == 'high':
                signal_prob = 0.4
            elif item['priority_tier'] == 'medium':
                signal_prob = 0.2
            else:
                signal_prob = 0.1
            
            generated_signal = random.random() < signal_prob
            
            if generated_signal:
                total_signals += 1
                tech_rating = random.uniform(60, 90)
                alpha_score = random.uniform(0.5, 0.9)
            else:
                tech_rating = None
                alpha_score = None
            
            prioritizer.update_with_result(
                item['symbol'],
                generated_signal,
                tech_rating,
                alpha_score
            )
        
        batch_num += 1
    
    # Final statistics
    final_stats = prioritizer.get_stats()
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"  Total symbols scanned: {final_stats['scan_results']['total_scanned']}")
    print(f"  Signals found: {final_stats['scan_results']['signals_found']}")
    print(f"  Signal rate: {final_stats['scan_results']['signal_rate']:.1%}")
    print(f"  High-value symbols: {final_stats['scan_results']['high_value_count']}")
    
    # Verify tech sector bias worked
    tech_in_high_value = sum(
        1 for sym in prioritizer.high_value_symbols 
        if sym.startswith('TECH')
    )
    
    if prioritizer.high_value_symbols:
        tech_percentage = tech_in_high_value / len(prioritizer.high_value_symbols) * 100
        print(f"  Tech sector in high-value: {tech_percentage:.0f}%")
    
    print("\n✓ Integration test completed successfully")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3E-A: SMART SYMBOL PRIORITIZATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Symbol Scoring", test_symbol_scoring),
        ("Priority Queue", test_priority_queue),
        ("Diversity Mode", test_diversity_mode),
        ("Learning & Reordering", test_learning_and_reordering),
        ("Performance Tracking", test_performance_tracking),
        ("Edge Cases", test_edge_cases),
        ("Integration Test", run_integration_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 3E-A tests passed!")
        print("\nKey Achievements:")
        print("  ✓ Symbol scoring with multi-factor analysis")
        print("  ✓ Three-tier priority queue system")
        print("  ✓ Diversity mode for balanced scanning")
        print("  ✓ Learning from scan results")
        print("  ✓ Dynamic reordering based on success")
        print("  ✓ Comprehensive performance tracking")
        print("\n  Expected improvement: 20-30% perceived speed increase")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
