"""
Test Suite for ML-Enhanced API
Phase 3E-C Week 2: API Integration Testing
"""

import requests
import json
import time
from typing import Dict, Any


BASE_URL = "http://localhost:8002"


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"ML Models:")
    print(f"  Result Predictor: {'✓ Trained' if data['ml_models']['result_predictor'] else '⚠ Not trained'}")
    print(f"  Duration Predictor: {'✓ Trained' if data['ml_models']['duration_predictor'] else '⚠ Not trained'}")
    print(f"Database: {data['database']['scans_recorded']} scans recorded")
    
    print("✓ Health check passed")
    return True


def test_market_conditions():
    """Test market conditions endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Market Conditions")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/market/conditions")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    conditions = data['conditions']
    
    print(f"Trend: {conditions.get('spy_trend', 'N/A')}")
    print(f"Volatility: {conditions.get('spy_volatility', 0):.2%}")
    print(f"Momentum: {conditions.get('spy_momentum', 0):.2f}")
    print(f"5-day return: {conditions.get('spy_return_5d', 0):.2%}")
    print(f"Market hours: {'Yes' if conditions.get('is_market_hours') else 'No'}")
    
    print("✓ Market conditions retrieved")
    return True


def test_predict_scan():
    """Test scan prediction endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Scan Prediction")
    print("="*60)
    
    # Test configuration
    config = {
        "max_symbols": 100,
        "min_tech_rating": 30,
        "min_dollar_vol": 5000000,
        "sectors": ["Technology", "Healthcare"],
        "lookback_days": 90,
        "use_alpha_blend": True,
        "profile": "balanced"
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/scan/predict",
        json=config
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    
    print(f"\nPredictions:")
    print(f"  Expected results: {data.get('predicted_results', 'N/A')}")
    print(f"  Expected duration: {data.get('predicted_duration', 'N/A')}s")
    print(f"  Confidence: {data.get('confidence', 0):.1%}")
    print(f"  Risk level: {data.get('risk_level', 'N/A')}")
    print(f"  Quality: {data.get('quality_estimate', 'N/A')}")
    
    if data.get('warnings'):
        print(f"\nWarnings ({len(data['warnings'])}):")
        for warning in data['warnings']:
            print(f"  ⚠ {warning}")
    
    if data.get('suggestions'):
        print(f"\nSuggestions ({len(data['suggestions'])}):")
        for suggestion in data['suggestions']:
            print(f"  💡 {suggestion}")
    
    print("\n✓ Prediction completed")
    return True


def test_suggest_parameters():
    """Test parameter suggestion endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Parameter Suggestions")
    print("="*60)
    
    for goal in ['speed', 'balanced', 'quality']:
        print(f"\n{goal.upper()} Scan:")
        print("-" * 40)
        
        response = requests.get(
            f"{BASE_URL}/scan/suggest",
            params={"goal": goal, "include_alternatives": False}
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        config = data['suggested_config']
        
        print(f"  Max symbols: {config.get('max_symbols')}")
        print(f"  Min tech rating: {config.get('min_tech_rating')}")
        print(f"  Sectors: {config.get('sectors', 'All')}")
        print(f"  Lookback: {config.get('lookback_days')} days")
        print(f"  Expected results: {data.get('predicted_results', 'N/A')}")
        print(f"  Expected duration: {data.get('predicted_duration', 'N/A')}s")
        print(f"  Reasoning: {data.get('reasoning', 'N/A')}")
    
    print("\n✓ All suggestions generated")
    return True


def test_history_stats():
    """Test history statistics endpoint"""
    print("\n" + "="*60)
    print("TEST 5: History Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/history/stats")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    
    print(f"Total scans: {data.get('total_scans', 0)}")
    print(f"Average results: {data.get('avg_results', 0):.1f}")
    print(f"Average duration: {data.get('avg_duration', 0):.1f}s")
    
    if data.get('date_range'):
        print(f"Date range: {data['date_range'].get('start', 'N/A')} to {data['date_range'].get('end', 'N/A')}")
    
    print("✓ History stats retrieved")
    return True


def test_model_status():
    """Test model status endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Model Status")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/models/status")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    
    print(f"Result Predictor:")
    print(f"  Trained: {data['result_predictor']['trained']}")
    if data['result_predictor']['feature_importance']:
        print(f"  Top features:")
        importance = data['result_predictor']['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, score in sorted_features:
            print(f"    {name}: {score:.3f}")
    
    print(f"\nDuration Predictor:")
    print(f"  Trained: {data['duration_predictor']['trained']}")
    if data['duration_predictor']['feature_importance']:
        print(f"  Top features:")
        importance = data['duration_predictor']['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, score in sorted_features:
            print(f"    {name}: {score:.3f}")
    
    print(f"\nTraining data available: {data.get('training_data_available', 0)} scans")
    
    print("✓ Model status retrieved")
    return True


def test_invalid_requests():
    """Test error handling"""
    print("\n" + "="*60)
    print("TEST 7: Error Handling")
    print("="*60)
    
    # Test invalid goal
    response = requests.get(
        f"{BASE_URL}/scan/suggest",
        params={"goal": "invalid"}
    )
    assert response.status_code == 400, "Should reject invalid goal"
    print("✓ Invalid goal rejected")
    
    # Test invalid config
    response = requests.post(
        f"{BASE_URL}/scan/predict",
        json={"max_symbols": -1}  # Invalid value
    )
    assert response.status_code == 422, "Should reject invalid config"
    print("✓ Invalid config rejected")
    
    print("✓ Error handling working")
    return True


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("ML-ENHANCED API TEST SUITE")
    print("="*60)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API is running: python api_ml_enhanced.py")
    
    # Wait for API to be ready
    print("\nChecking API availability...")
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ API is ready")
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                print(f"  Waiting for API... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("\n✗ API not available. Please start it first:")
                print("  python api_ml_enhanced.py")
                return False
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Market Conditions", test_market_conditions),
        ("Scan Prediction", test_predict_scan),
        ("Parameter Suggestions", test_suggest_parameters),
        ("History Statistics", test_history_stats),
        ("Model Status", test_model_status),
        ("Error Handling", test_invalid_requests)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except AssertionError as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"\n✗ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All ML API tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
