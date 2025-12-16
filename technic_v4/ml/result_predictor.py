"""
Result Count Predictor
Predicts number of scan results based on config and market conditions
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import joblib

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

from .scan_history import ScanRecord


class ResultCountPredictor:
    """
    Predict number of results based on scan config and market conditions
    
    Uses Random Forest Regression to learn patterns from historical scans.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML predictions")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = []
        self.model_path = Path(model_path) if model_path else Path("models/result_predictor.pkl")
        
        # Try to load existing model
        if self.model_path.exists():
            self.load()
    
    def train(self, scan_history: List[ScanRecord]) -> Dict[str, float]:
        """
        Train model on historical scans
        
        Args:
            scan_history: List of historical scan records
        
        Returns:
            Dictionary with training metrics
        """
        if len(scan_history) < 10:
            raise ValueError("Need at least 10 historical scans for training")
        
        # Extract features and targets
        X = []
        y = []
        
        for record in scan_history:
            features = self._extract_features(record)
            X.append(features)
            y.append(record.results.get('count', 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return metrics
    
    def predict(
        self,
        config: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict result count
        
        Args:
            config: Scan configuration
            market_conditions: Current market conditions
        
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            return {
                'predicted_count': None,
                'confidence': 0.0,
                'error': 'Model not trained'
            }
        
        # Extract features
        features = self._extract_features_from_config(config, market_conditions)
        features_array = np.array([features])
        
        # Predict
        prediction = self.model.predict(features_array)[0]
        prediction = max(0, int(round(prediction)))
        
        # Estimate confidence based on feature importance
        confidence = self._estimate_confidence(features)
        
        return {
            'predicted_count': prediction,
            'confidence': confidence,
            'range': {
                'min': max(0, int(prediction * 0.7)),
                'max': int(prediction * 1.3)
            }
        }
    
    def _extract_features(self, record: ScanRecord) -> List[float]:
        """
        Extract features from scan record
        
        Args:
            record: Scan record
        
        Returns:
            List of feature values
        """
        config = record.config
        market = record.market_conditions
        
        return self._extract_features_from_config(config, market)
    
    def _extract_features_from_config(
        self,
        config: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[float]:
        """
        Extract features from config and market conditions
        
        Args:
            config: Scan configuration
            market_conditions: Market conditions
        
        Returns:
            List of feature values
        """
        features = [
            # Config features
            config.get('max_symbols', 100),
            config.get('min_tech_rating', 10),
            config.get('min_dollar_vol', 0) / 1e6,  # Normalize to millions
            len(config.get('sectors', [])) if config.get('sectors') else 0,
            len(config.get('industries', [])) if config.get('industries') else 0,
            config.get('lookback_days', 90),
            
            # Market features
            self._encode_trend(market_conditions.get('spy_trend', 'neutral')),
            market_conditions.get('spy_volatility', 0.15),
            market_conditions.get('spy_momentum', 0.0),
            market_conditions.get('spy_return_5d', 0.0),
            market_conditions.get('spy_return_20d', 0.0),
            market_conditions.get('vix_level', 20.0) if market_conditions.get('vix_level') else 20.0,
            
            # Time features
            market_conditions.get('time_of_day', 12),
            market_conditions.get('day_of_week', 2),
            1.0 if market_conditions.get('is_market_hours') else 0.0,
        ]
        
        self.feature_names = [
            'max_symbols', 'min_tech_rating', 'min_dollar_vol_millions',
            'num_sectors', 'num_industries', 'lookback_days',
            'trend_encoded', 'volatility', 'momentum',
            'return_5d', 'return_20d', 'vix_level',
            'time_of_day', 'day_of_week', 'is_market_hours'
        ]
        
        return features
    
    def _encode_trend(self, trend: str) -> float:
        """Encode trend as numeric value"""
        encoding = {
            'bullish': 1.0,
            'neutral': 0.0,
            'bearish': -1.0
        }
        return encoding.get(trend, 0.0)
    
    def _estimate_confidence(self, features: List[float]) -> float:
        """
        Estimate prediction confidence
        
        Args:
            features: Feature vector
        
        Returns:
            Confidence score (0-1)
        """
        if not self.is_trained:
            return 0.0
        
        # Use model's feature importances to estimate confidence
        try:
            importances = self.model.feature_importances_
            # Higher importance features contribute more to confidence
            confidence = min(0.95, 0.5 + np.mean(importances) * 2)
            return float(confidence)
        except:
            return 0.75  # Default confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }
    
    def save(self, path: Optional[str] = None):
        """
        Save model to disk
        
        Args:
            path: Path to save model (uses self.model_path if None)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, save_path)
    
    def load(self, path: Optional[str] = None):
        """
        Load model from disk
        
        Args:
            path: Path to load model from (uses self.model_path if None)
        """
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        data = joblib.dump(load_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']


if __name__ == "__main__":
    # Test the result predictor
    print("Testing Result Count Predictor...")
    
    if not SKLEARN_AVAILABLE:
        print("✗ scikit-learn not available")
        exit(1)
    
    # Create mock training data
    from datetime import datetime
    
    mock_records = []
    for i in range(50):
        record = ScanRecord(
            scan_id=f"test_{i}",
            timestamp=datetime.now(),
            config={
                'max_symbols': np.random.randint(50, 200),
                'min_tech_rating': np.random.randint(10, 50),
                'min_dollar_vol': np.random.randint(1, 10) * 1e6,
                'sectors': ['Technology'] if i % 2 == 0 else None,
                'lookback_days': 90
            },
            results={
                'count': np.random.randint(5, 50)
            },
            performance={
                'total_seconds': 10.0
            },
            market_conditions={
                'spy_trend': 'bullish' if i % 3 == 0 else 'neutral',
                'spy_volatility': 0.15 + np.random.random() * 0.1,
                'spy_momentum': np.random.random() - 0.5,
                'spy_return_5d': np.random.random() * 0.05,
                'spy_return_20d': np.random.random() * 0.1,
                'vix_level': 15 + np.random.random() * 10,
                'time_of_day': 12,
                'day_of_week': 2,
                'is_market_hours': True
            }
        )
        mock_records.append(record)
    
    # Train model
    predictor = ResultCountPredictor()
    metrics = predictor.train(mock_records)
    
    print(f"\n✓ Model trained on {len(mock_records)} records")
    print(f"  Test MAE: {metrics['test_mae']:.2f}")
    print(f"  Test R²: {metrics['test_r2']:.3f}")
    
    # Test prediction
    test_config = {
        'max_symbols': 100,
        'min_tech_rating': 30,
        'min_dollar_vol': 5e6,
        'sectors': ['Technology'],
        'lookback_days': 90
    }
    
    test_market = {
        'spy_trend': 'bullish',
        'spy_volatility': 0.18,
        'spy_momentum': 0.3,
        'spy_return_5d': 0.02,
        'spy_return_20d': 0.05,
        'vix_level': 18.0,
        'time_of_day': 14,
        'day_of_week': 2,
        'is_market_hours': True
    }
    
    prediction = predictor.predict(test_config, test_market)
    print(f"\n✓ Prediction: {prediction['predicted_count']} results")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Range: {prediction['range']['min']}-{prediction['range']['max']}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\n✓ Top 5 important features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in sorted_features:
        print(f"  {name}: {score:.3f}")
    
    print("\n✓ Result Count Predictor test complete!")
