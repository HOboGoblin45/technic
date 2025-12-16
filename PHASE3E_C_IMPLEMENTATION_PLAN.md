# Phase 3E-C: ML-Powered Scan Optimization - Implementation Plan

## Overview

**Goal**: Use machine learning to optimize scan parameters, predict outcomes, and auto-configure based on market conditions.

## Current Status

### Completed Phases
- ✅ Phase 3E-A: Smart Symbol Prioritization (57.4% improvement)
- ✅ Phase 3E-B: Incremental Results Streaming (99% faster first result)
- ✅ Combined improvement: >60% faster perceived completion

### What We Have
- Historical scan data (logs/recommendations.csv)
- Symbol performance tracking
- Priority scoring system
- Real-time streaming infrastructure

## Phase 3E-C Objectives

### 1. Pattern Learning
Learn which configurations work best:
- Which sectors perform best at different times
- Optimal technical thresholds for different market conditions
- Best scan parameters for different user goals

### 2. Auto-Configuration
Suggest optimal scan parameters:
- "Quick Scan" vs "Deep Scan" presets
- Market-aware configurations
- Time-of-day optimizations

### 3. Result Prediction
Estimate outcomes before scanning:
- Predict number of results
- Estimate scan duration
- Warn if parameters too restrictive/broad

## Implementation Strategy

### Component 1: Data Collection & Storage

#### Task 1.1: Scan History Database
```python
# technic_v4/ml/scan_history.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
import json
from pathlib import Path

@dataclass
class ScanRecord:
    """Record of a completed scan"""
    scan_id: str
    timestamp: datetime
    config: Dict[str, Any]  # ScanConfig as dict
    results: Dict[str, Any]  # Results summary
    performance: Dict[str, Any]  # Timing, throughput
    market_conditions: Dict[str, Any]  # SPY data, VIX, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scan_id': self.scan_id,
            'timestamp': self.timestamp.isoformat(),
            'config': self.config,
            'results': self.results,
            'performance': self.performance,
            'market_conditions': self.market_conditions
        }

class ScanHistoryDB:
    """Store and retrieve scan history"""
    
    def __init__(self, db_path: str = "data/scan_history.jsonl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def add_scan(self, record: ScanRecord):
        """Add scan record to database"""
        with open(self.db_path, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')
    
    def get_recent_scans(self, limit: int = 100) -> List[ScanRecord]:
        """Get recent scan records"""
        # Implementation
        pass
    
    def get_scans_by_config(self, config_filter: Dict) -> List[ScanRecord]:
        """Get scans matching config criteria"""
        # Implementation
        pass
```

#### Task 1.2: Market Condition Tracker
```python
# technic_v4/ml/market_conditions.py
def get_current_market_conditions() -> Dict[str, Any]:
    """
    Get current market conditions for ML features
    """
    # Get SPY data
    spy = data_engine.get_price_history("SPY", days=30)
    
    # Calculate features
    return {
        'spy_trend': calculate_trend(spy),
        'spy_volatility': calculate_volatility(spy),
        'vix_level': get_vix_level(),
        'market_breadth': calculate_breadth(),
        'time_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'is_market_hours': is_market_open()
    }
```

### Component 2: ML Models

#### Task 2.1: Result Count Predictor
```python
# technic_v4/ml/result_predictor.py
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class ResultCountPredictor:
    """
    Predict number of results based on scan config and market conditions
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False
    
    def train(self, scan_history: List[ScanRecord]):
        """Train model on historical scans"""
        X = []  # Features
        y = []  # Target (result count)
        
        for record in scan_history:
            features = self._extract_features(record)
            X.append(features)
            y.append(record.results['count'])
        
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True
    
    def predict(self, config: Dict, market_conditions: Dict) -> int:
        """Predict result count"""
        if not self.is_trained:
            return None
        
        features = self._extract_features_from_config(config, market_conditions)
        prediction = self.model.predict([features])[0]
        return int(max(0, prediction))
    
    def _extract_features(self, record: ScanRecord) -> List[float]:
        """Extract features from scan record"""
        return [
            record.config.get('max_symbols', 100),
            record.config.get('min_tech_rating', 10),
            len(record.config.get('sectors', [])),
            record.market_conditions.get('spy_trend', 0),
            record.market_conditions.get('spy_volatility', 0),
            record.market_conditions.get('time_of_day', 12),
            # ... more features
        ]
```

#### Task 2.2: Duration Predictor
```python
# technic_v4/ml/duration_predictor.py
class ScanDurationPredictor:
    """
    Predict scan duration based on config and system load
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False
    
    def predict(self, config: Dict, market_conditions: Dict) -> float:
        """Predict scan duration in seconds"""
        if not self.is_trained:
            # Use heuristic
            return config.get('max_symbols', 100) * 0.1  # 0.1s per symbol
        
        features = self._extract_features(config, market_conditions)
        prediction = self.model.predict([features])[0]
        return max(1.0, prediction)
```

#### Task 2.3: Parameter Optimizer
```python
# technic_v4/ml/parameter_optimizer.py
class ParameterOptimizer:
    """
    Suggest optimal scan parameters based on user goals
    """
    
    def suggest_quick_scan(self, market_conditions: Dict) -> Dict[str, Any]:
        """Suggest parameters for quick scan (< 10s)"""
        return {
            'max_symbols': 50,
            'min_tech_rating': 30,
            'sectors': self._get_hot_sectors(market_conditions),
            'lookback_days': 30
        }
    
    def suggest_deep_scan(self, market_conditions: Dict) -> Dict[str, Any]:
        """Suggest parameters for comprehensive scan"""
        return {
            'max_symbols': 200,
            'min_tech_rating': 10,
            'sectors': None,  # All sectors
            'lookback_days': 90
        }
    
    def suggest_optimal(
        self,
        goal: str,  # 'speed', 'quality', 'balanced'
        market_conditions: Dict
    ) -> Dict[str, Any]:
        """Suggest optimal parameters for goal"""
        if goal == 'speed':
            return self.suggest_quick_scan(market_conditions)
        elif goal == 'quality':
            return self.suggest_deep_scan(market_conditions)
        else:  # balanced
            return self._suggest_balanced(market_conditions)
    
    def _get_hot_sectors(self, market_conditions: Dict) -> List[str]:
        """Identify currently performing sectors"""
        # Analyze recent scan history
        # Return sectors with highest signal rates
        pass
```

### Component 3: API Integration

#### Task 3.1: ML-Enhanced Endpoints
```python
# api_ml_enhanced.py
from technic_v4.ml.result_predictor import ResultCountPredictor
from technic_v4.ml.duration_predictor import ScanDurationPredictor
from technic_v4.ml.parameter_optimizer import ParameterOptimizer

# Initialize ML models
result_predictor = ResultCountPredictor()
duration_predictor = ScanDurationPredictor()
param_optimizer = ParameterOptimizer()

@app.post("/scan/predict")
async def predict_scan_results(config: ScanRequest):
    """
    Predict scan outcomes before running
    """
    market_conditions = get_current_market_conditions()
    
    predicted_results = result_predictor.predict(
        config.dict(),
        market_conditions
    )
    
    predicted_duration = duration_predictor.predict(
        config.dict(),
        market_conditions
    )
    
    return {
        'predicted_results': predicted_results,
        'predicted_duration': predicted_duration,
        'confidence': 0.75,  # Model confidence
        'warnings': _generate_warnings(predicted_results, config)
    }

@app.get("/scan/suggest")
async def suggest_parameters(goal: str = 'balanced'):
    """
    Suggest optimal scan parameters
    """
    market_conditions = get_current_market_conditions()
    
    suggested_config = param_optimizer.suggest_optimal(
        goal,
        market_conditions
    )
    
    # Predict outcomes for suggested config
    predicted_results = result_predictor.predict(
        suggested_config,
        market_conditions
    )
    
    return {
        'suggested_config': suggested_config,
        'predicted_results': predicted_results,
        'reasoning': _explain_suggestion(suggested_config, market_conditions),
        'alternatives': {
            'quick': param_optimizer.suggest_quick_scan(market_conditions),
            'deep': param_optimizer.suggest_deep_scan(market_conditions)
        }
    }

def _generate_warnings(predicted_results: int, config: Dict) -> List[str]:
    """Generate warnings about scan configuration"""
    warnings = []
    
    if predicted_results == 0:
        warnings.append("Configuration may be too restrictive - no results expected")
    elif predicted_results > 100:
        warnings.append("Configuration may be too broad - consider narrowing filters")
    
    if config.get('min_tech_rating', 0) > 80:
        warnings.append("High tech rating threshold may limit results significantly")
    
    return warnings
```

### Component 4: Training Pipeline

#### Task 4.1: Model Training Script
```python
# scripts/train_ml_models.py
def train_all_models():
    """Train all ML models on historical data"""
    
    # Load scan history
    db = ScanHistoryDB()
    history = db.get_recent_scans(limit=1000)
    
    print(f"Training on {len(history)} historical scans...")
    
    # Train result predictor
    result_predictor = ResultCountPredictor()
    result_predictor.train(history)
    result_predictor.save('models/result_predictor.pkl')
    
    # Train duration predictor
    duration_predictor = ScanDurationPredictor()
    duration_predictor.train(history)
    duration_predictor.save('models/duration_predictor.pkl')
    
    # Evaluate models
    evaluate_models(result_predictor, duration_predictor, history)
    
    print("✓ Model training complete")

def evaluate_models(result_pred, duration_pred, test_data):
    """Evaluate model performance"""
    # Split data
    # Calculate metrics (MAE, RMSE, R²)
    # Generate evaluation report
    pass
```

## Implementation Timeline

### Week 1: Data Collection
- [ ] Implement ScanHistoryDB
- [ ] Add scan logging to scanner_core
- [ ] Implement market condition tracker
- [ ] Collect initial training data (100+ scans)

### Week 2: ML Models
- [ ] Implement ResultCountPredictor
- [ ] Implement ScanDurationPredictor
- [ ] Implement ParameterOptimizer
- [ ] Train initial models

### Week 3: API Integration
- [ ] Add `/scan/predict` endpoint
- [ ] Add `/scan/suggest` endpoint
- [ ] Implement warning system
- [ ] Add model confidence scores

### Week 4: Testing & Refinement
- [ ] Test prediction accuracy
- [ ] Refine feature engineering
- [ ] Optimize model parameters
- [ ] Deploy to production

## Success Criteria

### Prediction Accuracy
- Result count prediction: MAE < 10 results
- Duration prediction: MAE < 5 seconds
- Confidence calibration: 75%+ accuracy

### User Experience
- Helpful warnings reduce empty scans by 30%
- Suggested parameters improve signal rate by 20%
- Prediction time < 100ms

### Business Impact
- Reduce wasted scans by 25%
- Improve user satisfaction by 15%
- Increase scan efficiency by 20%

## Dependencies

### Python Packages
```txt
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0  # Model persistence
```

### Data Requirements
- Minimum 100 historical scans for training
- Market data (SPY, VIX)
- Continuous data collection

## Monitoring

### Model Performance Metrics
```python
metrics = {
    'prediction_accuracy': {
        'result_count_mae': 'mean absolute error',
        'duration_mae': 'mean absolute error',
        'confidence_calibration': 'accuracy at confidence level'
    },
    'usage': {
        'predictions_per_day': 'count',
        'suggestions_accepted': 'percentage',
        'warnings_heeded': 'percentage'
    },
    'business_impact': {
        'empty_scan_rate': 'percentage',
        'avg_signal_rate': 'percentage',
        'user_satisfaction': 'score'
    }
}
```

## Rollout Plan

### Phase 1: Shadow Mode (Week 1-2)
- Collect predictions but don't show to users
- Compare predictions to actual results
- Tune models based on real data

### Phase 2: Advisory Mode (Week 3-4)
- Show predictions as suggestions
- Users can ignore recommendations
- Collect feedback on usefulness

### Phase 3: Full Deployment (Week 5+)
- Predictions shown by default
- Auto-configuration available
- Continuous model improvement

## Future Enhancements

### Advanced Features
1. **Sector Rotation Detection**: Identify shifting sector performance
2. **Anomaly Detection**: Flag unusual market conditions
3. **Personalization**: Learn individual user preferences
4. **A/B Testing**: Optimize suggestions through experimentation

### Integration with Phase 4
- Distributed model training
- Real-time model updates
- Multi-region model deployment

## Conclusion

Phase 3E-C adds intelligence to the scanner, helping users:
- **Predict outcomes** before scanning
- **Optimize parameters** automatically
- **Avoid wasted scans** with warnings
- **Adapt to market conditions** dynamically

Combined with Phases 3E-A and 3E-B, this creates a truly intelligent, adaptive scanning system.
