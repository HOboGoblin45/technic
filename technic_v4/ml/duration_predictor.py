"""
Scan Duration Predictor
Predicts scan duration based on config and system load
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

from .scan_history import ScanRecord


class ScanDurationPredictor:
    """
    Predict scan duration based on config and system conditions
    
    Uses Random Forest Regression to learn timing patterns.
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
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = []
        self.model_path = Path(model_path) if model_path else Path("models/duration_predictor.pkl")
        
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
            y.append(record.performance.get('total_seconds', 0))
        
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
        Predict scan duration
        
        Args:
            config: Scan configuration
            market_conditions: Current market conditions
        
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            # Use heuristic
            estimated_duration = config.get('max_symbols', 100) * 0.1
            return {
                'predicted_seconds': estimated_duration,
                'confidence': 0.5,
                'method': 'heuristic'
            }
        
        # Extract features
        features = self._extract_features_from_config(config, market_conditions)
        features_array = np.array([features])
        
        # Predict
        prediction = self.model.predict(features_array)[0]
        prediction = max(1.0, prediction)
        
        # Estimate confidence
        confidence = 0.75  # Default confidence for trained model
        
        return {
            'predicted_seconds': float(prediction),
            'confidence': confidence,
            'method': 'ml_model',
            'range': {
                'min': float(prediction * 0.8),
                'max': float(prediction * 1.2)
            }
        }
    
    def _extract_features(self, record: ScanRecord) -> List[float]:
        """Extract features from scan record"""
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
            # Primary drivers of duration
            config.get('max_symbols', 100),
            config.get('lookback_days', 90),
            len(config.get('sectors', [])) if config.get('sectors') else 0,
            
            # Complexity factors
            config.get('min_tech_rating', 10) / 100.0,  # Normalize
            1.0 if config.get('use_alpha_blend', False) else 0.0,
            1.0 if config.get('enable_options', False) else 0.0,
            
            # Market activity (affects data fetching)
            market_conditions.get('spy_volatility', 0.15),
            1.0 if market_conditions.get('is_market_hours') else 0.0,
            
            # Time of day (affects system load)
            market_conditions.get('time_of_day', 12) / 24.0,  # Normalize
        ]
        
        self.feature_names = [
            'max_symbols', 'lookback_days', 'num_sectors',
            'min_tech_rating_norm', 'use_alpha_blend', 'enable_options',
            'volatility', 'is_market_hours', 'time_of_day_norm'
        ]
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, importances)
        }
    
    def save(self, path: Optional[str] = None):
        """Save model to disk"""
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
        """Load model from disk"""
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        data = joblib.load(load_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']


if __name__ == "__main__":
    # Test the duration predictor
    print("Testing Scan Duration Predictor...")
    
    if not SKLEARN_AVAILABLE:
        print("✗ scikit-learn not available")
        exit(1)
    
    # Create mock training data
    from datetime import datetime
    
    mock_records = []
    for i in range(50):
        max_symbols = np.random.randint(50, 200)
        lookback = np.random.choice([30, 60, 90])
        
        # Duration correlates with symbols and lookback
        base_duration = max_symbols * 0.08 + lookback * 0.05
        duration = base_duration + np.random.random() * 5
        
        record = ScanRecord(
            scan_id=f"test_{i}",
            timestamp=datetime.now(),
            config={
                'max_symbols': max_symbols,
                'lookback_days': lookback,
                'sectors': ['Technology'] if i % 2 == 0 else None,
                'min_tech_rating': np.random.randint(10, 50),
                'use_alpha_blend': i % 3 == 0,
                'enable_options': i % 4 == 0
            },
            results={'count': 20},
            performance={'total_seconds': duration},
            market_conditions={
                'spy_volatility': 0.15 + np.random.random() * 0.1,
                'is_market_hours': i % 2 == 0,
                'time_of_day': np.random.randint(9, 17)
            }
        )
        mock_records.append(record)
    
    # Train model
    predictor = ScanDurationPredictor()
    metrics = predictor.train(mock_records)
    
    print(f"\n✓ Model trained on {len(mock_records)} records")
    print(f"  Test MAE: {metrics['test_mae']:.2f}s")
    print(f"  Test R²: {metrics['test_r2']:.3f}")
    
    # Test prediction
    test_config = {
        'max_symbols': 100,
        'lookback_days': 90,
        'sectors': ['Technology'],
        'min_tech_rating': 30,
        'use_alpha_blend': True,
        'enable_options': False
    }
    
    test_market = {
        'spy_volatility': 0.18,
        'is_market_hours': True,
        'time_of_day': 14
    }
    
    prediction = predictor.predict(test_config, test_market)
    print(f"\n✓ Prediction: {prediction['predicted_seconds']:.1f}s")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Range: {prediction['range']['min']:.1f}s - {prediction['range']['max']:.1f}s")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\n✓ Feature importance:")
    for name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.3f}")
    
    print("\n✓ Scan Duration Predictor test complete!")
