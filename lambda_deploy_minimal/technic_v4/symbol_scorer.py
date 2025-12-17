"""
Symbol Scorer for Smart Prioritization
Phase 3E-A: Scores symbols based on multiple factors to prioritize high-value opportunities
"""

import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path


@dataclass
class SymbolScore:
    """Represents a symbol's priority score with component breakdown"""
    symbol: str
    total_score: float
    historical_score: float = 0.0
    activity_score: float = 0.0
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'total_score': self.total_score,
            'historical_score': self.historical_score,
            'activity_score': self.activity_score,
            'fundamental_score': self.fundamental_score,
            'technical_score': self.technical_score,
            'last_updated': self.last_updated,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolScore':
        """Create from dictionary"""
        return cls(**data)


class SymbolScorer:
    """
    Scores symbols for prioritization based on multiple factors
    
    Scoring Components:
    - Historical Performance (40%): Past signal generation success
    - Market Activity (30%): Volume, momentum, volatility
    - Fundamentals (20%): Earnings, ratings, news
    - Technical Setup (10%): Patterns, indicators
    """
    
    # Scoring weights
    WEIGHTS = {
        'historical': 0.40,
        'activity': 0.30,
        'fundamental': 0.20,
        'technical': 0.10
    }
    
    # Score decay rate (scores decrease by 10% per week)
    DECAY_RATE = 0.10
    DECAY_PERIOD_DAYS = 7
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the symbol scorer
        
        Args:
            cache_dir: Directory for storing historical scores
        """
        self.cache_dir = cache_dir or Path("data_cache/symbol_scores")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load historical performance data
        self.historical_data = self._load_historical_data()
        
        # Track current session performance
        self.session_performance = {}
        
    def _load_historical_data(self) -> Dict[str, Dict[str, Any]]:
        """Load historical performance data from cache"""
        cache_file = self.cache_dir / "historical_scores.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_historical_data(self):
        """Save historical performance data to cache"""
        cache_file = self.cache_dir / "historical_scores.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception:
            pass
    
    def score_symbol(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
        technical_data: Optional[Dict[str, Any]] = None
    ) -> SymbolScore:
        """
        Calculate priority score for a symbol
        
        Args:
            symbol: Stock symbol
            market_data: Current market activity data
            fundamental_data: Fundamental metrics
            technical_data: Technical indicators
            
        Returns:
            SymbolScore with total and component scores
        """
        # Calculate component scores
        historical_score = self._calculate_historical_score(symbol)
        activity_score = self._calculate_activity_score(symbol, market_data)
        fundamental_score = self._calculate_fundamental_score(symbol, fundamental_data)
        technical_score = self._calculate_technical_score(symbol, technical_data)
        
        # Calculate weighted total
        total_score = (
            historical_score * self.WEIGHTS['historical'] +
            activity_score * self.WEIGHTS['activity'] +
            fundamental_score * self.WEIGHTS['fundamental'] +
            technical_score * self.WEIGHTS['technical']
        )
        
        # Create score object
        score = SymbolScore(
            symbol=symbol,
            total_score=round(total_score, 2),
            historical_score=round(historical_score, 2),
            activity_score=round(activity_score, 2),
            fundamental_score=round(fundamental_score, 2),
            technical_score=round(technical_score, 2),
            metadata={
                'has_market_data': market_data is not None,
                'has_fundamental_data': fundamental_data is not None,
                'has_technical_data': technical_data is not None
            }
        )
        
        return score
    
    def _calculate_historical_score(self, symbol: str) -> float:
        """
        Calculate score based on historical performance
        
        Factors:
        - Signal generation frequency
        - Signal quality (avg TechRating, AlphaScore)
        - Recent performance (time-weighted)
        - Success rate
        """
        if symbol not in self.historical_data:
            return 50.0  # Neutral score for new symbols
        
        hist = self.historical_data[symbol]
        score = 50.0
        
        # Signal generation frequency (0-30 points)
        signal_count = hist.get('signal_count', 0)
        scan_count = hist.get('scan_count', 1)
        signal_rate = signal_count / max(scan_count, 1)
        score += signal_rate * 30
        
        # Signal quality (0-30 points)
        avg_tech_rating = hist.get('avg_tech_rating', 50)
        score += (avg_tech_rating / 100) * 30
        
        # Recent performance boost (0-20 points)
        last_signal = hist.get('last_signal_time', 0)
        if last_signal:
            days_ago = (time.time() - last_signal) / 86400
            if days_ago < 7:
                score += 20 * (1 - days_ago / 7)
        
        # Apply time decay
        last_updated = hist.get('last_updated', time.time())
        days_old = (time.time() - last_updated) / 86400
        decay_factor = 1 - (self.DECAY_RATE * (days_old / self.DECAY_PERIOD_DAYS))
        decay_factor = max(0.5, min(1.0, decay_factor))  # Clamp between 0.5 and 1.0
        
        score *= decay_factor
        
        return min(100, max(0, score))
    
    def _calculate_activity_score(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate score based on current market activity
        
        Factors:
        - Volume surge (vs 20-day average)
        - Price momentum (5-day return)
        - Volatility (ATR-based)
        - Relative strength
        """
        if not market_data:
            return 50.0  # Neutral if no data
        
        score = 50.0
        
        # Volume surge (0-30 points)
        volume_ratio = market_data.get('volume_ratio', 1.0)  # Current vs 20-day avg
        if volume_ratio > 2.0:
            score += 30
        elif volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10
        
        # Price momentum (0-30 points)
        momentum_5d = market_data.get('return_5d', 0)
        if momentum_5d > 0.10:  # >10% gain
            score += 30
        elif momentum_5d > 0.05:  # >5% gain
            score += 20
        elif momentum_5d > 0.02:  # >2% gain
            score += 10
        elif momentum_5d < -0.05:  # >5% loss (contrarian)
            score += 15
        
        # Volatility (0-20 points) - moderate volatility preferred
        atr_ratio = market_data.get('atr_ratio', 1.0)  # ATR vs price
        if 0.02 < atr_ratio < 0.05:  # 2-5% daily range
            score += 20
        elif 0.01 < atr_ratio < 0.07:  # 1-7% daily range
            score += 10
        
        # Relative strength (0-20 points)
        rs_percentile = market_data.get('rs_percentile', 50)
        if rs_percentile > 80:
            score += 20
        elif rs_percentile > 60:
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_fundamental_score(self, symbol: str, fundamental_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate score based on fundamental factors
        
        Factors:
        - Earnings surprises
        - Analyst rating changes
        - News sentiment
        - Institutional activity
        """
        if not fundamental_data:
            return 50.0  # Neutral if no data
        
        score = 50.0
        
        # Earnings surprise (0-30 points)
        earnings_surprise = fundamental_data.get('earnings_surprise_pct', 0)
        if earnings_surprise > 10:
            score += 30
        elif earnings_surprise > 5:
            score += 20
        elif earnings_surprise > 0:
            score += 10
        
        # Rating changes (0-30 points)
        rating_change = fundamental_data.get('rating_change_30d', 0)
        if rating_change > 0:  # Upgrades
            score += rating_change * 10  # Max 3 upgrades = 30 points
        
        # News sentiment (0-20 points)
        news_sentiment = fundamental_data.get('news_sentiment', 0)  # -1 to 1
        score += (news_sentiment + 1) * 10  # Convert to 0-20
        
        # Institutional activity (0-20 points)
        inst_flow = fundamental_data.get('institutional_flow_30d', 0)
        if inst_flow > 0.05:  # >5% increase
            score += 20
        elif inst_flow > 0.02:  # >2% increase
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_technical_score(self, symbol: str, technical_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate score based on technical setup
        
        Factors:
        - Support/resistance proximity
        - Pattern completion
        - Indicator convergence
        - Trend strength
        """
        if not technical_data:
            return 50.0  # Neutral if no data
        
        score = 50.0
        
        # Support/resistance proximity (0-30 points)
        support_distance = technical_data.get('support_distance_pct', float('inf'))
        resistance_distance = technical_data.get('resistance_distance_pct', float('inf'))
        
        if support_distance < 2:  # Within 2% of support
            score += 20
        if resistance_distance > 5:  # >5% to resistance
            score += 10
        
        # Pattern completion (0-30 points)
        pattern_score = technical_data.get('pattern_score', 0)  # 0-100
        score += pattern_score * 0.3
        
        # Indicator convergence (0-20 points)
        bullish_indicators = technical_data.get('bullish_indicator_count', 0)
        total_indicators = technical_data.get('total_indicator_count', 1)
        convergence = bullish_indicators / max(total_indicators, 1)
        score += convergence * 20
        
        # Trend strength (0-20 points)
        trend_strength = technical_data.get('trend_strength', 0)  # -100 to 100
        score += abs(trend_strength) * 0.2
        
        return min(100, max(0, score))
    
    def update_performance(
        self,
        symbol: str,
        generated_signal: bool,
        tech_rating: Optional[float] = None,
        alpha_score: Optional[float] = None
    ):
        """
        Update historical performance data for a symbol
        
        Args:
            symbol: Stock symbol
            generated_signal: Whether symbol generated a signal
            tech_rating: Technical rating if signal generated
            alpha_score: Alpha score if signal generated
        """
        if symbol not in self.historical_data:
            self.historical_data[symbol] = {
                'signal_count': 0,
                'scan_count': 0,
                'total_tech_rating': 0,
                'avg_tech_rating': 50,
                'last_signal_time': None,
                'last_updated': time.time()
            }
        
        hist = self.historical_data[symbol]
        hist['scan_count'] += 1
        hist['last_updated'] = time.time()
        
        if generated_signal:
            hist['signal_count'] += 1
            hist['last_signal_time'] = time.time()
            
            if tech_rating is not None:
                hist['total_tech_rating'] += tech_rating
                hist['avg_tech_rating'] = hist['total_tech_rating'] / hist['signal_count']
        
        # Update session performance
        if symbol not in self.session_performance:
            self.session_performance[symbol] = {'signals': 0, 'scans': 0}
        
        self.session_performance[symbol]['scans'] += 1
        if generated_signal:
            self.session_performance[symbol]['signals'] += 1
        
        # Periodically save to disk
        if len(self.session_performance) % 10 == 0:
            self._save_historical_data()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        total_scans = sum(p['scans'] for p in self.session_performance.values())
        total_signals = sum(p['signals'] for p in self.session_performance.values())
        
        return {
            'symbols_scanned': len(self.session_performance),
            'total_scans': total_scans,
            'total_signals': total_signals,
            'signal_rate': total_signals / max(total_scans, 1),
            'top_performers': self._get_top_performers(5)
        }
    
    def _get_top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing symbols in current session"""
        performers = []
        for symbol, perf in self.session_performance.items():
            if perf['scans'] > 0:
                performers.append({
                    'symbol': symbol,
                    'signal_rate': perf['signals'] / perf['scans'],
                    'signals': perf['signals'],
                    'scans': perf['scans']
                })
        
        performers.sort(key=lambda x: x['signal_rate'], reverse=True)
        return performers[:n]
    
    def clear_old_data(self, days: int = 30):
        """Clear historical data older than specified days"""
        cutoff_time = time.time() - (days * 86400)
        
        symbols_to_remove = []
        for symbol, hist in self.historical_data.items():
            if hist.get('last_updated', 0) < cutoff_time:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.historical_data[symbol]
        
        if symbols_to_remove:
            self._save_historical_data()
            
        return len(symbols_to_remove)


def create_mock_market_data(symbol: str) -> Dict[str, Any]:
    """Create mock market data for testing"""
    import random
    
    return {
        'volume_ratio': random.uniform(0.5, 3.0),
        'return_5d': random.uniform(-0.15, 0.15),
        'atr_ratio': random.uniform(0.01, 0.08),
        'rs_percentile': random.uniform(20, 90)
    }


def create_mock_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Create mock fundamental data for testing"""
    import random
    
    return {
        'earnings_surprise_pct': random.uniform(-5, 15),
        'rating_change_30d': random.randint(-1, 2),
        'news_sentiment': random.uniform(-0.5, 0.8),
        'institutional_flow_30d': random.uniform(-0.05, 0.10)
    }


def create_mock_technical_data(symbol: str) -> Dict[str, Any]:
    """Create mock technical data for testing"""
    import random
    
    return {
        'support_distance_pct': random.uniform(0.5, 10),
        'resistance_distance_pct': random.uniform(1, 15),
        'pattern_score': random.uniform(0, 100),
        'bullish_indicator_count': random.randint(0, 5),
        'total_indicator_count': 5,
        'trend_strength': random.uniform(-80, 80)
    }


if __name__ == "__main__":
    # Test the symbol scorer
    scorer = SymbolScorer()
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print("Symbol Scoring Test")
    print("=" * 60)
    
    scores = []
    for symbol in test_symbols:
        # Generate mock data
        market_data = create_mock_market_data(symbol)
        fundamental_data = create_mock_fundamental_data(symbol)
        technical_data = create_mock_technical_data(symbol)
        
        # Calculate score
        score = scorer.score_symbol(
            symbol,
            market_data=market_data,
            fundamental_data=fundamental_data,
            technical_data=technical_data
        )
        
        scores.append(score)
        
        print(f"\n{symbol}:")
        print(f"  Total Score: {score.total_score}")
        print(f"  Historical: {score.historical_score}")
        print(f"  Activity: {score.activity_score}")
        print(f"  Fundamental: {score.fundamental_score}")
        print(f"  Technical: {score.technical_score}")
    
    # Sort by total score
    scores.sort(key=lambda x: x.total_score, reverse=True)
    
    print("\n" + "=" * 60)
    print("Priority Ranking:")
    for i, score in enumerate(scores, 1):
        print(f"{i}. {score.symbol}: {score.total_score}")
