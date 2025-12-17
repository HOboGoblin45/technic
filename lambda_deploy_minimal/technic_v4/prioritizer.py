"""
Symbol Prioritizer with Priority Queue
Phase 3E-A: Manages symbol prioritization and dynamic reordering
"""

import heapq
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import random

try:
    from technic_v4.symbol_scorer import SymbolScorer, SymbolScore
except ImportError:
    # For standalone testing
    from symbol_scorer import SymbolScorer, SymbolScore


@dataclass
class PrioritizedSymbol:
    """Symbol with priority for heap queue"""
    priority: float  # Negative for max-heap
    symbol: str
    score: SymbolScore
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        # Higher score = higher priority (use negative for max-heap)
        return self.priority < other.priority


class SymbolPriorityQueue:
    """
    Priority queue for symbols with dynamic reordering
    
    Features:
    - Three-tier priority system (high/medium/low)
    - Batch retrieval with mixed priorities
    - Dynamic reordering based on results
    - Random sampling for diversity
    """
    
    # Priority thresholds
    HIGH_PRIORITY_THRESHOLD = 70
    MEDIUM_PRIORITY_THRESHOLD = 40
    
    # Batch composition (for diversity)
    BATCH_COMPOSITION = {
        'high': 0.60,    # 60% from high priority
        'medium': 0.30,   # 30% from medium priority
        'low': 0.08,      # 8% from low priority
        'random': 0.02    # 2% random selection
    }
    
    def __init__(self, enable_diversity: bool = True):
        """
        Initialize priority queue
        
        Args:
            enable_diversity: Whether to mix priorities in batches
        """
        self.high_priority: List[PrioritizedSymbol] = []
        self.medium_priority: List[PrioritizedSymbol] = []
        self.low_priority: List[PrioritizedSymbol] = []
        self.processed: Set[str] = set()
        self.enable_diversity = enable_diversity
        
        # Track statistics
        self.stats = {
            'total_added': 0,
            'high_count': 0,
            'medium_count': 0,
            'low_count': 0,
            'processed_count': 0,
            'reorder_count': 0
        }
    
    def add_symbol(self, symbol: str, score: SymbolScore):
        """
        Add a symbol to the appropriate priority queue
        
        Args:
            symbol: Stock symbol
            score: Symbol score object
        """
        # Check if symbol already exists in any queue
        if symbol in self.processed:
            return  # Skip already processed symbols
        
        # Check if symbol is already in queues (for duplicate prevention)
        for item in self.high_priority + self.medium_priority + self.low_priority:
            if item.symbol == symbol:
                return  # Skip duplicate
        
        # Create prioritized symbol (negative score for max-heap)
        prioritized = PrioritizedSymbol(
            priority=-score.total_score,
            symbol=symbol,
            score=score
        )
        
        # Add to appropriate queue
        if score.total_score >= self.HIGH_PRIORITY_THRESHOLD:
            heapq.heappush(self.high_priority, prioritized)
            self.stats['high_count'] += 1
        elif score.total_score >= self.MEDIUM_PRIORITY_THRESHOLD:
            heapq.heappush(self.medium_priority, prioritized)
            self.stats['medium_count'] += 1
        else:
            heapq.heappush(self.low_priority, prioritized)
            self.stats['low_count'] += 1
        
        self.stats['total_added'] += 1
    
    def add_batch(self, symbols_with_scores: List[Tuple[str, SymbolScore]]):
        """
        Add multiple symbols at once
        
        Args:
            symbols_with_scores: List of (symbol, score) tuples
        """
        for symbol, score in symbols_with_scores:
            self.add_symbol(symbol, score)
    
    def get_next_batch(self, batch_size: int = 10) -> List[PrioritizedSymbol]:
        """
        Get next batch of symbols to process
        
        Args:
            batch_size: Number of symbols to retrieve
            
        Returns:
            List of prioritized symbols
        """
        batch = []
        
        if self.enable_diversity:
            # Mix priorities for diversity
            batch = self._get_diverse_batch(batch_size)
        else:
            # Pure priority order
            batch = self._get_priority_batch(batch_size)
        
        # Mark as processed
        for item in batch:
            self.processed.add(item.symbol)
            self.stats['processed_count'] += 1
        
        return batch
    
    def _get_priority_batch(self, batch_size: int) -> List[PrioritizedSymbol]:
        """Get batch in pure priority order"""
        batch = []
        
        # Take from high priority first
        while len(batch) < batch_size and self.high_priority:
            batch.append(heapq.heappop(self.high_priority))
        
        # Then medium priority
        while len(batch) < batch_size and self.medium_priority:
            batch.append(heapq.heappop(self.medium_priority))
        
        # Finally low priority
        while len(batch) < batch_size and self.low_priority:
            batch.append(heapq.heappop(self.low_priority))
        
        return batch
    
    def _get_diverse_batch(self, batch_size: int) -> List[PrioritizedSymbol]:
        """Get batch with mixed priorities for diversity"""
        batch = []
        
        # Calculate counts for each priority level
        high_count = int(batch_size * self.BATCH_COMPOSITION['high'])
        medium_count = int(batch_size * self.BATCH_COMPOSITION['medium'])
        low_count = int(batch_size * self.BATCH_COMPOSITION['low'])
        random_count = int(batch_size * self.BATCH_COMPOSITION['random'])
        
        # Adjust for rounding
        remaining = batch_size - (high_count + medium_count + low_count + random_count)
        high_count += remaining
        
        # Take from high priority
        for _ in range(min(high_count, len(self.high_priority))):
            if self.high_priority:
                batch.append(heapq.heappop(self.high_priority))
        
        # Take from medium priority
        for _ in range(min(medium_count, len(self.medium_priority))):
            if self.medium_priority:
                batch.append(heapq.heappop(self.medium_priority))
        
        # Take from low priority
        for _ in range(min(low_count, len(self.low_priority))):
            if self.low_priority:
                batch.append(heapq.heappop(self.low_priority))
        
        # Add random selections for exploration
        all_remaining = self._get_all_remaining()
        random_selections = min(random_count, len(all_remaining))
        if random_selections > 0:
            random_symbols = random.sample(all_remaining, random_selections)
            for sym in random_symbols:
                # Remove from original queue and add to batch
                self._remove_symbol(sym.symbol)
                batch.append(sym)
        
        # Fill any remaining slots from high priority
        while len(batch) < batch_size:
            if self.high_priority:
                batch.append(heapq.heappop(self.high_priority))
            elif self.medium_priority:
                batch.append(heapq.heappop(self.medium_priority))
            elif self.low_priority:
                batch.append(heapq.heappop(self.low_priority))
            else:
                break
        
        return batch
    
    def _get_all_remaining(self) -> List[PrioritizedSymbol]:
        """Get all remaining symbols across queues"""
        all_symbols = []
        all_symbols.extend(self.high_priority)
        all_symbols.extend(self.medium_priority)
        all_symbols.extend(self.low_priority)
        return all_symbols
    
    def _remove_symbol(self, symbol: str):
        """Remove a symbol from its queue"""
        # This is inefficient but used sparingly for random selection
        self.high_priority = [s for s in self.high_priority if s.symbol != symbol]
        heapq.heapify(self.high_priority)
        
        self.medium_priority = [s for s in self.medium_priority if s.symbol != symbol]
        heapq.heapify(self.medium_priority)
        
        self.low_priority = [s for s in self.low_priority if s.symbol != symbol]
        heapq.heapify(self.low_priority)
    
    def reorder_based_on_results(
        self,
        successful_symbols: List[str],
        boost_similar: bool = True
    ):
        """
        Dynamically reorder remaining symbols based on results
        
        Args:
            successful_symbols: Symbols that generated good signals
            boost_similar: Whether to boost similar symbols (same sector, etc.)
        """
        if not successful_symbols:
            return
        
        self.stats['reorder_count'] += 1
        
        # For now, just boost symbols that share characteristics
        # In production, this would analyze sector, market cap, etc.
        
        # Simple implementation: slightly boost all remaining high-priority symbols
        # when we find successful ones
        if self.high_priority:
            # Re-score high priority symbols with a small boost
            boosted = []
            while self.high_priority:
                sym = heapq.heappop(self.high_priority)
                # Give a 5-point boost
                sym.priority = min(sym.priority - 5, -100)  # Cap at 100
                boosted.append(sym)
            
            # Re-add to heap
            for sym in boosted:
                heapq.heappush(self.high_priority, sym)
    
    def get_remaining_count(self) -> int:
        """Get count of remaining symbols"""
        return (
            len(self.high_priority) +
            len(self.medium_priority) +
            len(self.low_priority)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            'remaining_high': len(self.high_priority),
            'remaining_medium': len(self.medium_priority),
            'remaining_low': len(self.low_priority),
            'total_remaining': self.get_remaining_count(),
            'diversity_enabled': self.enable_diversity
        }
    
    def clear(self):
        """Clear all queues"""
        self.high_priority.clear()
        self.medium_priority.clear()
        self.low_priority.clear()
        self.processed.clear()
        
        # Reset stats
        self.stats = {
            'total_added': 0,
            'high_count': 0,
            'medium_count': 0,
            'low_count': 0,
            'processed_count': 0,
            'reorder_count': 0
        }


class SmartSymbolPrioritizer:
    """
    Main prioritizer that combines scoring and queue management
    
    This is the main interface for the scanner to use
    """
    
    def __init__(
        self,
        enable_diversity: bool = True,
        enable_learning: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the smart prioritizer
        
        Args:
            enable_diversity: Mix priority levels in batches
            enable_learning: Learn from scan results
            cache_dir: Directory for historical data
        """
        self.scorer = SymbolScorer(cache_dir=cache_dir)
        self.queue = SymbolPriorityQueue(enable_diversity=enable_diversity)
        self.enable_learning = enable_learning
        
        # Track current scan performance
        self.scan_results = []
        self.high_value_symbols = []
    
    def prioritize_symbols(
        self,
        symbols: List[str],
        market_data: Optional[Dict[str, Dict[str, Any]]] = None,
        fundamental_data: Optional[Dict[str, Dict[str, Any]]] = None,
        technical_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> int:
        """
        Score and prioritize a list of symbols
        
        Args:
            symbols: List of symbols to prioritize
            market_data: Market data by symbol
            fundamental_data: Fundamental data by symbol
            technical_data: Technical data by symbol
            
        Returns:
            Number of symbols prioritized
        """
        symbols_with_scores = []
        
        for symbol in symbols:
            # Get data for this symbol
            sym_market = market_data.get(symbol) if market_data else None
            sym_fundamental = fundamental_data.get(symbol) if fundamental_data else None
            sym_technical = technical_data.get(symbol) if technical_data else None
            
            # Calculate score
            score = self.scorer.score_symbol(
                symbol,
                market_data=sym_market,
                fundamental_data=sym_fundamental,
                technical_data=sym_technical
            )
            
            symbols_with_scores.append((symbol, score))
        
        # Add to priority queue
        self.queue.add_batch(symbols_with_scores)
        
        return len(symbols_with_scores)
    
    def get_next_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get next batch of prioritized symbols
        
        Args:
            batch_size: Number of symbols to retrieve
            
        Returns:
            List of symbol info dictionaries
        """
        batch = self.queue.get_next_batch(batch_size)
        
        result = []
        for item in batch:
            result.append({
                'symbol': item.symbol,
                'priority_score': -item.priority,  # Convert back to positive
                'historical_score': item.score.historical_score,
                'activity_score': item.score.activity_score,
                'fundamental_score': item.score.fundamental_score,
                'technical_score': item.score.technical_score,
                'priority_tier': self._get_tier(item.score.total_score)
            })
        
        return result
    
    def _get_tier(self, score: float) -> str:
        """Get priority tier name"""
        if score >= self.queue.HIGH_PRIORITY_THRESHOLD:
            return 'high'
        elif score >= self.queue.MEDIUM_PRIORITY_THRESHOLD:
            return 'medium'
        else:
            return 'low'
    
    def update_with_result(
        self,
        symbol: str,
        generated_signal: bool,
        tech_rating: Optional[float] = None,
        alpha_score: Optional[float] = None
    ):
        """
        Update prioritizer with scan result
        
        Args:
            symbol: Symbol that was scanned
            generated_signal: Whether it generated a signal
            tech_rating: Technical rating if available
            alpha_score: Alpha score if available
        """
        # Track result
        self.scan_results.append({
            'symbol': symbol,
            'signal': generated_signal,
            'tech_rating': tech_rating,
            'alpha_score': alpha_score
        })
        
        # Update historical performance
        if self.enable_learning:
            self.scorer.update_performance(
                symbol,
                generated_signal,
                tech_rating,
                alpha_score
            )
        
        # Track high-value symbols
        if generated_signal and tech_rating and tech_rating > 70:
            self.high_value_symbols.append(symbol)
            
            # Reorder queue based on successful symbols
            if len(self.high_value_symbols) % 5 == 0:  # Every 5 high-value symbols
                self.queue.reorder_based_on_results(
                    self.high_value_symbols[-5:],
                    boost_similar=True
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        queue_stats = self.queue.get_stats()
        session_stats = self.scorer.get_session_stats()
        
        signal_count = sum(1 for r in self.scan_results if r['signal'])
        scan_count = len(self.scan_results)
        
        return {
            'queue': queue_stats,
            'session': session_stats,
            'scan_results': {
                'total_scanned': scan_count,
                'signals_found': signal_count,
                'signal_rate': signal_count / max(scan_count, 1),
                'high_value_count': len(self.high_value_symbols)
            }
        }
    
    def reset(self):
        """Reset for new scan"""
        self.queue.clear()
        self.scan_results.clear()
        self.high_value_symbols.clear()


if __name__ == "__main__":
    # Test the prioritizer
    try:
        from technic_v4.symbol_scorer import create_mock_market_data, create_mock_fundamental_data, create_mock_technical_data
    except ImportError:
        from symbol_scorer import create_mock_market_data, create_mock_fundamental_data, create_mock_technical_data
    
    print("Smart Symbol Prioritizer Test")
    print("=" * 60)
    
    # Create prioritizer
    prioritizer = SmartSymbolPrioritizer(enable_diversity=True)
    
    # Test symbols (would come from universe in production)
    test_symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN',
        'META', 'NVDA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'UNH', 'HD', 'MA',
        'DIS', 'BAC', 'NFLX', 'ADBE', 'CRM'
    ]
    
    # Generate mock data
    market_data = {sym: create_mock_market_data(sym) for sym in test_symbols}
    fundamental_data = {sym: create_mock_fundamental_data(sym) for sym in test_symbols}
    technical_data = {sym: create_mock_technical_data(sym) for sym in test_symbols}
    
    # Prioritize symbols
    count = prioritizer.prioritize_symbols(
        test_symbols,
        market_data=market_data,
        fundamental_data=fundamental_data,
        technical_data=technical_data
    )
    
    print(f"Prioritized {count} symbols")
    print(f"Queue stats: {prioritizer.queue.get_stats()}")
    
    # Get batches
    print("\n" + "=" * 60)
    print("Processing in priority order:")
    print("-" * 60)
    
    batch_num = 1
    while prioritizer.queue.get_remaining_count() > 0:
        batch = prioritizer.get_next_batch(batch_size=5)
        
        print(f"\nBatch {batch_num}:")
        for item in batch:
            tier_emoji = {'high': '🔥', 'medium': '⭐', 'low': '📊'}
            emoji = tier_emoji.get(item['priority_tier'], '📊')
            print(f"  {emoji} {item['symbol']:5s} - Score: {item['priority_score']:.1f} ({item['priority_tier']})")
            
            # Simulate scan result
            import random
            signal = random.random() > 0.7  # 30% chance of signal
            if signal:
                prioritizer.update_with_result(
                    item['symbol'],
                    generated_signal=True,
                    tech_rating=random.uniform(60, 90),
                    alpha_score=random.uniform(0.5, 0.9)
                )
        
        batch_num += 1
    
    # Final stats
    print("\n" + "=" * 60)
    print("Final Statistics:")
    stats = prioritizer.get_stats()
    print(f"Signals found: {stats['scan_results']['signals_found']}/{stats['scan_results']['total_scanned']}")
    print(f"Signal rate: {stats['scan_results']['signal_rate']:.1%}")
    print(f"High-value symbols: {stats['scan_results']['high_value_count']}")
