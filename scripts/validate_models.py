"""
Validate ML Models
Phase 3E-C Week 3: Model Validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technic_v4.ml import (
    ScanHistoryDB,
    ResultCountPredictor,
    ScanDurationPredictor,
    get_current_market_conditions
)
import numpy as np


def validate_result_predictor(test_records: list) -> dict:
    """
    Validate result count predictor
    
    Args:
        test_records: Test scan records
    
    Returns:
        Validation metrics
    """
    print("\n" + "="*60)
    print("VALIDATING RESULT COUNT PREDICTOR")
    print("="*60)
    
    predictor = ResultCountPredictor()
    
    if not predictor.is_trained:
        print("\n✗ Model not trained. Run: python scripts/train_models.py")
        return {}
    
    print(f"\nTesting on {len(test_records)} records...")
    
    predictions = []
    actuals = []
    errors = []
    
    for record in test_records:
        pred = predictor.predict(record.config, record.market_conditions)
        predicted = pred['predicted_count']
        actual = record.results['count']
        
        predictions.append(predicted)
        actuals.append(actual)
        errors.append(abs(predicted - actual))
    
    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    mape = np.mean(np.abs(np.array(errors) / (np.array(actuals) + 1))) * 100
    
    # R² score
    ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {mae:.2f} results")
    print(f"  RMSE: {rmse:.2f} results")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²: {r2:.3f}")
    
    # Check targets
    print(f"\nTarget Achievement:")
    if mae < 10:
        print(f"  ✓ MAE < 10: {mae:.2f}")
    else:
        print(f"  ✗ MAE >= 10: {mae:.2f}")
    
    if r2 > 0.6:
        print(f"  ✓ R² > 0.6: {r2:.3f}")
    else:
        print(f"  ✗ R² <= 0.6: {r2:.3f}")
    
    # Show examples
    print(f"\nSample Predictions:")
    for i in range(min(5, len(test_records))):
        print(f"  Predicted: {predictions[i]:3d} | Actual: {actuals[i]:3d} | Error: {errors[i]:.1f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }


def validate_duration_predictor(test_records: list) -> dict:
    """
    Validate duration predictor
    
    Args:
        test_records: Test scan records
    
    Returns:
        Validation metrics
    """
    print("\n" + "="*60)
    print("VALIDATING DURATION PREDICTOR")
    print("="*60)
    
    predictor = ScanDurationPredictor()
    
    if not predictor.is_trained:
        print("\n✗ Model not trained. Run: python scripts/train_models.py")
        return {}
    
    print(f"\nTesting on {len(test_records)} records...")
    
    predictions = []
    actuals = []
    errors = []
    
    for record in test_records:
        pred = predictor.predict(record.config, record.market_conditions)
        predicted = pred['predicted_seconds']
        actual = record.performance['total_seconds']
        
        predictions.append(predicted)
        actuals.append(actual)
        errors.append(abs(predicted - actual))
    
    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    mape = np.mean(np.abs(np.array(errors) / (np.array(actuals) + 0.1))) * 100
    
    # R² score
    ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {mae:.2f} seconds")
    print(f"  RMSE: {rmse:.2f} seconds")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²: {r2:.3f}")
    
    # Check targets
    print(f"\nTarget Achievement:")
    if mae < 5:
        print(f"  ✓ MAE < 5s: {mae:.2f}s")
    else:
        print(f"  ✗ MAE >= 5s: {mae:.2f}s")
    
    if r2 > 0.7:
        print(f"  ✓ R² > 0.7: {r2:.3f}")
    else:
        print(f"  ✗ R² <= 0.7: {r2:.3f}")
    
    # Show examples
    print(f"\nSample Predictions:")
    for i in range(min(5, len(test_records))):
        print(f"  Predicted: {predictions[i]:5.1f}s | Actual: {actuals[i]:5.1f}s | Error: {errors[i]:.1f}s")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }


def test_live_predictions():
    """Test predictions with current market conditions"""
    print("\n" + "="*60)
    print("LIVE PREDICTION TEST")
    print("="*60)
    
    # Get current market conditions
    market_conditions = get_current_market_conditions()
    
    print(f"\nCurrent Market Conditions:")
    print(f"  Trend: {market_conditions.get('spy_trend', 'N/A')}")
    print(f"  Volatility: {market_conditions.get('spy_volatility', 0):.2%}")
    print(f"  Momentum: {market_conditions.get('spy_momentum', 0):.2f}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Quick Scan',
            'config': {
                'max_symbols': 50,
                'min_tech_rating': 30,
                'min_dollar_vol': 5e6,
                'sectors': ['Technology'],
                'lookback_days': 30,
                'use_alpha_blend': False
            }
        },
        {
            'name': 'Balanced Scan',
            'config': {
                'max_symbols': 100,
                'min_tech_rating': 25,
                'min_dollar_vol': 3e6,
                'sectors': ['Technology', 'Healthcare'],
                'lookback_days': 60,
                'use_alpha_blend': True
            }
        },
        {
            'name': 'Deep Scan',
            'config': {
                'max_symbols': 200,
                'min_tech_rating': 10,
                'min_dollar_vol': 1e6,
                'sectors': None,
                'lookback_days': 90,
                'use_alpha_blend': True
            }
        }
    ]
    
    result_predictor = ResultCountPredictor()
    duration_predictor = ScanDurationPredictor()
    
    print(f"\nPredictions for Different Scan Types:")
    print("-" * 60)
    
    for test in test_configs:
        print(f"\n{test['name']}:")
        
        result_pred = result_predictor.predict(test['config'], market_conditions)
        duration_pred = duration_predictor.predict(test['config'], market_conditions)
        
        print(f"  Expected results: {result_pred.get('predicted_count', 'N/A')}")
        print(f"  Expected duration: {duration_pred.get('predicted_seconds', 'N/A'):.1f}s")
        print(f"  Confidence: {result_pred.get('confidence', 0):.1%}")


def main():
    """Validate all ML models"""
    print("="*60)
    print("ML MODEL VALIDATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    db = ScanHistoryDB()
    all_records = db.get_recent_scans(limit=1000)
    
    if len(all_records) < 20:
        print(f"\n✗ Insufficient data: {len(all_records)} scans")
        return 1
    
    # Use last 20% as test set
    test_size = max(20, len(all_records) // 5)
    test_records = all_records[-test_size:]
    
    print(f"✓ Loaded {len(test_records)} test records")
    
    # Validate result predictor
    result_metrics = validate_result_predictor(test_records)
    
    # Validate duration predictor
    duration_metrics = validate_duration_predictor(test_records)
    
    # Test live predictions
    test_live_predictions()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if result_metrics:
        print(f"\nResult Count Predictor:")
        print(f"  MAE: {result_metrics['mae']:.2f} results")
        print(f"  R²: {result_metrics['r2']:.3f}")
        meets_target = result_metrics['mae'] < 10 and result_metrics['r2'] > 0.6
        print(f"  Status: {'✓ Production ready' if meets_target else '⚠ Needs tuning'}")
    
    if duration_metrics:
        print(f"\nDuration Predictor:")
        print(f"  MAE: {duration_metrics['mae']:.2f} seconds")
        print(f"  R²: {duration_metrics['r2']:.3f}")
        meets_target = duration_metrics['mae'] < 5 and duration_metrics['r2'] > 0.7
        print(f"  Status: {'✓ Production ready' if meets_target else '⚠ Needs tuning'}")
    
    print("\n✓ Model validation complete!")
    print("\nNext steps:")
    print("  1. Start API: python api_ml_enhanced.py")
    print("  2. Test API: python test_ml_api.py")
    print("  3. Deploy to production")
    
    return 0


if __name__ == "__main__":
    exit(main())
