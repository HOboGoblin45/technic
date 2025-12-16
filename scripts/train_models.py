"""
Train ML Models on Historical Data
Phase 3E-C Week 3: Model Training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technic_v4.ml import (
    ScanHistoryDB,
    ResultCountPredictor,
    ScanDurationPredictor
)
import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(predictor_name: str, metrics: dict, predictions: list, actuals: list):
    """
    Plot training results
    
    Args:
        predictor_name: Name of the predictor
        metrics: Training metrics
        predictions: Predicted values
        actuals: Actual values
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Predictions vs Actuals
    axes[0].scatter(actuals, predictions, alpha=0.5)
    axes[0].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{predictor_name}: Predictions vs Actuals')
    axes[0].grid(True, alpha=0.3)
    
    # Add metrics text
    text = f"MAE: {metrics['test_mae']:.2f}\nR²: {metrics['test_r2']:.3f}"
    axes[0].text(0.05, 0.95, text, transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Residuals
    residuals = np.array(predictions) - np.array(actuals)
    axes[1].scatter(actuals, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{predictor_name}: Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"models/{predictor_name.lower().replace(' ', '_')}_training.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()


def train_result_predictor(records: list) -> dict:
    """
    Train result count predictor
    
    Args:
        records: List of scan records
    
    Returns:
        Training metrics and predictions
    """
    print("\n" + "="*60)
    print("TRAINING RESULT COUNT PREDICTOR")
    print("="*60)
    
    predictor = ResultCountPredictor()
    
    print(f"\nTraining on {len(records)} records...")
    metrics = predictor.train(records)
    
    print(f"\nTraining Results:")
    print(f"  Training samples: {metrics['training_samples']}")
    print(f"  Test samples: {metrics['test_samples']}")
    print(f"  Train MAE: {metrics['train_mae']:.2f} results")
    print(f"  Test MAE: {metrics['test_mae']:.2f} results")
    print(f"  Train R²: {metrics['train_r2']:.3f}")
    print(f"  Test R²: {metrics['test_r2']:.3f}")
    
    # Check if meets targets
    if metrics['test_mae'] < 10:
        print(f"  ✓ MAE target met (< 10)")
    else:
        print(f"  ⚠ MAE above target: {metrics['test_mae']:.2f} (target: < 10)")
    
    if metrics['test_r2'] > 0.6:
        print(f"  ✓ R² target met (> 0.6)")
    else:
        print(f"  ⚠ R² below target: {metrics['test_r2']:.3f} (target: > 0.6)")
    
    # Get predictions for plotting
    predictions = []
    actuals = []
    for record in records[-50:]:  # Last 50 for visualization
        pred = predictor.predict(record.config, record.market_conditions)
        predictions.append(pred['predicted_count'])
        actuals.append(record.results['count'])
    
    # Plot results
    plot_training_results("Result Count Predictor", metrics, predictions, actuals)
    
    # Save model
    predictor.save()
    print(f"\n✓ Model saved to: {predictor.model_path}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\nTop 5 Important Features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in sorted_features:
        print(f"  {name:25s}: {score:.3f}")
    
    return metrics


def train_duration_predictor(records: list) -> dict:
    """
    Train duration predictor
    
    Args:
        records: List of scan records
    
    Returns:
        Training metrics and predictions
    """
    print("\n" + "="*60)
    print("TRAINING DURATION PREDICTOR")
    print("="*60)
    
    predictor = ScanDurationPredictor()
    
    print(f"\nTraining on {len(records)} records...")
    metrics = predictor.train(records)
    
    print(f"\nTraining Results:")
    print(f"  Training samples: {metrics['training_samples']}")
    print(f"  Test samples: {metrics['test_samples']}")
    print(f"  Train MAE: {metrics['train_mae']:.2f} seconds")
    print(f"  Test MAE: {metrics['test_mae']:.2f} seconds")
    print(f"  Train R²: {metrics['train_r2']:.3f}")
    print(f"  Test R²: {metrics['test_r2']:.3f}")
    
    # Check if meets targets
    if metrics['test_mae'] < 5:
        print(f"  ✓ MAE target met (< 5s)")
    else:
        print(f"  ⚠ MAE above target: {metrics['test_mae']:.2f}s (target: < 5s)")
    
    if metrics['test_r2'] > 0.7:
        print(f"  ✓ R² target met (> 0.7)")
    else:
        print(f"  ⚠ R² below target: {metrics['test_r2']:.3f} (target: > 0.7)")
    
    # Get predictions for plotting
    predictions = []
    actuals = []
    for record in records[-50:]:  # Last 50 for visualization
        pred = predictor.predict(record.config, record.market_conditions)
        predictions.append(pred['predicted_seconds'])
        actuals.append(record.performance['total_seconds'])
    
    # Plot results
    plot_training_results("Duration Predictor", metrics, predictions, actuals)
    
    # Save model
    predictor.save()
    print(f"\n✓ Model saved to: {predictor.model_path}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\nTop 5 Important Features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in sorted_features:
        print(f"  {name:25s}: {score:.3f}")
    
    return metrics


def main():
    """Train all ML models"""
    print("="*60)
    print("ML MODEL TRAINING")
    print("="*60)
    
    # Load historical data
    print("\nLoading historical scan data...")
    db = ScanHistoryDB()
    records = db.get_recent_scans(limit=1000)
    
    if len(records) < 50:
        print(f"\n✗ Insufficient data: {len(records)} scans (need at least 50)")
        print("\nPlease run: python scripts/generate_training_data.py")
        return 1
    
    print(f"✓ Loaded {len(records)} scan records")
    
    # Create models directory
    import os
    os.makedirs("models", exist_ok=True)
    
    # Train result predictor
    result_metrics = train_result_predictor(records)
    
    # Train duration predictor
    duration_metrics = train_duration_predictor(records)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nResult Count Predictor:")
    print(f"  Test MAE: {result_metrics['test_mae']:.2f} results")
    print(f"  Test R²: {result_metrics['test_r2']:.3f}")
    print(f"  Status: {'✓ Meets targets' if result_metrics['test_mae'] < 10 and result_metrics['test_r2'] > 0.6 else '⚠ Needs improvement'}")
    
    print(f"\nDuration Predictor:")
    print(f"  Test MAE: {duration_metrics['test_mae']:.2f} seconds")
    print(f"  Test R²: {duration_metrics['test_r2']:.3f}")
    print(f"  Status: {'✓ Meets targets' if duration_metrics['test_mae'] < 5 and duration_metrics['test_r2'] > 0.7 else '⚠ Needs improvement'}")
    
    print(f"\nModels saved to: models/")
    print(f"Training plots saved to: models/")
    
    print("\n✓ Model training complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/validate_models.py")
    print("  2. Start API: python api_ml_enhanced.py")
    print("  3. Test predictions: python test_ml_api.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
