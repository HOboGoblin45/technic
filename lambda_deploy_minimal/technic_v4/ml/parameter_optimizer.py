"""
Parameter Optimizer
Suggests optimal scan parameters based on user goals and market conditions
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from .scan_history import ScanHistoryDB
from .market_conditions import get_current_market_conditions


class ParameterOptimizer:
    """
    Suggest optimal scan parameters based on goals and market conditions
    
    Analyzes historical performance to recommend configurations.
    """
    
    def __init__(self, history_db: Optional[ScanHistoryDB] = None):
        """
        Initialize optimizer
        
        Args:
            history_db: Scan history database (creates new if None)
        """
        self.history_db = history_db or ScanHistoryDB()
    
    def suggest_quick_scan(self, market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Suggest parameters for quick scan (< 10s)
        
        Args:
            market_conditions: Current market conditions (fetches if None)
        
        Returns:
            Suggested configuration
        """
        if market_conditions is None:
            market_conditions = get_current_market_conditions()
        
        # Get hot sectors from recent history
        hot_sectors = self._get_hot_sectors(market_conditions)
        
        config = {
            'max_symbols': 50,
            'min_tech_rating': 30,
            'min_dollar_vol': 5e6,
            'sectors': hot_sectors[:2] if hot_sectors else ['Technology'],
            'lookback_days': 30,
            'use_alpha_blend': False,
            'enable_options': False,
            'profile': 'aggressive'
        }
        
        return {
            'config': config,
            'estimated_duration': 5.0,
            'estimated_results': 10,
            'reasoning': 'Quick scan optimized for speed with focus on active sectors'
        }
    
    def suggest_deep_scan(self, market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Suggest parameters for comprehensive scan
        
        Args:
            market_conditions: Current market conditions (fetches if None)
        
        Returns:
            Suggested configuration
        """
        if market_conditions is None:
            market_conditions = get_current_market_conditions()
        
        config = {
            'max_symbols': 200,
            'min_tech_rating': 10,
            'min_dollar_vol': 1e6,
            'sectors': None,  # All sectors
            'lookback_days': 90,
            'use_alpha_blend': True,
            'enable_options': True,
            'profile': 'balanced'
        }
        
        return {
            'config': config,
            'estimated_duration': 30.0,
            'estimated_results': 50,
            'reasoning': 'Comprehensive scan across all sectors with full analysis'
        }
    
    def suggest_optimal(
        self,
        goal: str = 'balanced',
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Suggest optimal parameters for goal
        
        Args:
            goal: 'speed', 'quality', or 'balanced'
            market_conditions: Current market conditions (fetches if None)
        
        Returns:
            Suggested configuration with reasoning
        """
        if market_conditions is None:
            market_conditions = get_current_market_conditions()
        
        if goal == 'speed':
            return self.suggest_quick_scan(market_conditions)
        elif goal == 'quality':
            return self.suggest_deep_scan(market_conditions)
        else:  # balanced
            return self._suggest_balanced(market_conditions)
    
    def _suggest_balanced(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest balanced parameters
        
        Args:
            market_conditions: Current market conditions
        
        Returns:
            Suggested configuration
        """
        # Adjust based on market conditions
        trend = market_conditions.get('spy_trend', 'neutral')
        volatility = market_conditions.get('spy_volatility', 0.15)
        
        # In volatile markets, be more selective
        if volatility > 0.25:
            min_rating = 25
            max_symbols = 75
        else:
            min_rating = 20
            max_symbols = 100
        
        # Get sectors performing well
        hot_sectors = self._get_hot_sectors(market_conditions)
        
        config = {
            'max_symbols': max_symbols,
            'min_tech_rating': min_rating,
            'min_dollar_vol': 3e6,
            'sectors': hot_sectors[:3] if hot_sectors else None,
            'lookback_days': 60,
            'use_alpha_blend': True,
            'enable_options': False,
            'profile': 'balanced'
        }
        
        reasoning = f"Balanced scan optimized for {trend} market with {volatility:.1%} volatility"
        
        return {
            'config': config,
            'estimated_duration': 15.0,
            'estimated_results': 25,
            'reasoning': reasoning
        }
    
    def _get_hot_sectors(self, market_conditions: Dict[str, Any]) -> List[str]:
        """
        Identify currently performing sectors
        
        Args:
            market_conditions: Current market conditions
        
        Returns:
            List of sector names sorted by performance
        """
        # Get recent scans
        recent_scans = self.history_db.get_recent_scans(limit=100)
        
        if not recent_scans:
            # Default sectors if no history
            return ['Technology', 'Healthcare', 'Financial']
        
        # Calculate signal rate by sector
        sector_stats = {}
        
        for scan in recent_scans:
            sectors = scan.config.get('sectors', [])
            if not sectors:
                continue
            
            signals = scan.results.get('signals', 0)
            count = scan.results.get('count', 1)
            signal_rate = signals / count if count > 0 else 0
            
            for sector in sectors:
                if sector not in sector_stats:
                    sector_stats[sector] = {'signals': 0, 'scans': 0}
                
                sector_stats[sector]['signals'] += signals
                sector_stats[sector]['scans'] += 1
        
        # Calculate average signal rate per sector
        sector_performance = []
        for sector, stats in sector_stats.items():
            if stats['scans'] >= 3:  # Minimum 3 scans
                avg_signal_rate = stats['signals'] / stats['scans']
                sector_performance.append((sector, avg_signal_rate))
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top sectors
        hot_sectors = [sector for sector, _ in sector_performance[:5]]
        
        return hot_sectors if hot_sectors else ['Technology', 'Healthcare', 'Financial']
    
    def analyze_config(
        self,
        config: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a configuration and provide feedback
        
        Args:
            config: Configuration to analyze
            market_conditions: Current market conditions (fetches if None)
        
        Returns:
            Analysis with warnings and suggestions
        """
        if market_conditions is None:
            market_conditions = get_current_market_conditions()
        
        warnings = []
        suggestions = []
        
        # Check max_symbols
        max_symbols = config.get('max_symbols', 100)
        if max_symbols < 20:
            warnings.append("Very low max_symbols may miss opportunities")
            suggestions.append("Consider increasing max_symbols to at least 50")
        elif max_symbols > 300:
            warnings.append("High max_symbols will increase scan time significantly")
            suggestions.append("Consider reducing max_symbols to 200 or less")
        
        # Check min_tech_rating
        min_rating = config.get('min_tech_rating', 10)
        if min_rating > 70:
            warnings.append("Very high tech rating threshold may yield no results")
            suggestions.append("Consider lowering min_tech_rating to 50 or below")
        
        # Check sector selection
        sectors = config.get('sectors')
        if sectors and len(sectors) > 5:
            warnings.append("Many sectors selected may dilute focus")
            suggestions.append("Consider focusing on 2-3 top-performing sectors")
        
        # Check lookback period
        lookback = config.get('lookback_days', 90)
        if lookback < 20:
            warnings.append("Short lookback period may miss important patterns")
            suggestions.append("Consider using at least 30 days lookback")
        elif lookback > 180:
            warnings.append("Long lookback period will increase processing time")
            suggestions.append("Consider reducing lookback to 90 days or less")
        
        # Market-specific warnings
        volatility = market_conditions.get('spy_volatility', 0.15)
        if volatility > 0.25 and min_rating < 20:
            warnings.append("Low quality threshold in volatile market may yield risky signals")
            suggestions.append("Consider increasing min_tech_rating in volatile conditions")
        
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'risk_level': self._assess_risk_level(config, market_conditions),
            'estimated_quality': self._estimate_quality(config)
        }
    
    def _assess_risk_level(self, config: Dict[str, Any], market_conditions: Dict[str, Any]) -> str:
        """
        Assess risk level of configuration
        
        Args:
            config: Configuration
            market_conditions: Market conditions
        
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        risk_score = 0
        
        # Low quality threshold increases risk
        if config.get('min_tech_rating', 10) < 20:
            risk_score += 1
        
        # Volatile market increases risk
        if market_conditions.get('spy_volatility', 0.15) > 0.25:
            risk_score += 1
        
        # Aggressive profile increases risk
        if config.get('profile') == 'aggressive':
            risk_score += 1
        
        if risk_score >= 2:
            return 'high'
        elif risk_score == 1:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_quality(self, config: Dict[str, Any]) -> str:
        """
        Estimate result quality
        
        Args:
            config: Configuration
        
        Returns:
            Quality level: 'low', 'medium', or 'high'
        """
        quality_score = 0
        
        # Higher tech rating improves quality
        if config.get('min_tech_rating', 10) >= 30:
            quality_score += 1
        
        # Alpha blend improves quality
        if config.get('use_alpha_blend', False):
            quality_score += 1
        
        # Longer lookback improves quality
        if config.get('lookback_days', 90) >= 60:
            quality_score += 1
        
        if quality_score >= 2:
            return 'high'
        elif quality_score == 1:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    # Test the parameter optimizer
    print("Testing Parameter Optimizer...")
    
    # Create optimizer
    optimizer = ParameterOptimizer()
    
    # Test quick scan suggestion
    print("\n1. Quick Scan Suggestion:")
    quick = optimizer.suggest_quick_scan()
    print(f"   Config: {quick['config']}")
    print(f"   Estimated duration: {quick['estimated_duration']}s")
    print(f"   Reasoning: {quick['reasoning']}")
    
    # Test deep scan suggestion
    print("\n2. Deep Scan Suggestion:")
    deep = optimizer.suggest_deep_scan()
    print(f"   Config: {deep['config']}")
    print(f"   Estimated duration: {deep['estimated_duration']}s")
    print(f"   Reasoning: {deep['reasoning']}")
    
    # Test balanced suggestion
    print("\n3. Balanced Scan Suggestion:")
    balanced = optimizer.suggest_optimal('balanced')
    print(f"   Config: {balanced['config']}")
    print(f"   Estimated duration: {balanced['estimated_duration']}s")
    print(f"   Reasoning: {balanced['reasoning']}")
    
    # Test config analysis
    print("\n4. Config Analysis:")
    test_config = {
        'max_symbols': 500,
        'min_tech_rating': 80,
        'lookback_days': 200
    }
    analysis = optimizer.analyze_config(test_config)
    print(f"   Warnings: {len(analysis['warnings'])}")
    for warning in analysis['warnings']:
        print(f"     - {warning}")
    print(f"   Risk level: {analysis['risk_level']}")
    print(f"   Quality: {analysis['estimated_quality']}")
    
    print("\n✓ Parameter Optimizer test complete!")
