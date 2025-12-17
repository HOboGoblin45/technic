"""
Comprehensive Backend API Testing
Tests all endpoints with various scenarios
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8002"

def print_test_header(test_name):
    print("\n" + "="*60)
    print(f"TEST: {test_name}")
    print("="*60)

def print_result(success, message, data=None):
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {message}")
    if data:
        print(f"Response: {json.dumps(data, indent=2)}")

# Test 1: Health Check
print_test_header("Health Check")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    success = response.status_code == 200
    print_result(success, f"Status Code: {response.status_code}", response.json() if success else None)
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 2: Model Status
print_test_header("Model Status")
try:
    response = requests.get(f"{BASE_URL}/models/status", timeout=5)
    success = response.status_code == 200
    print_result(success, f"Status Code: {response.status_code}", response.json() if success else None)
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 3: Get Suggestions
print_test_header("Get Parameter Suggestions")
try:
    response = requests.get(f"{BASE_URL}/scan/suggest", timeout=5)
    success = response.status_code == 200
    print_result(success, f"Status Code: {response.status_code}", response.json() if success else None)
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 4: Predict Scan (with parameters)
print_test_header("Predict Scan Outcomes")
try:
    payload = {
        "sectors": ["Technology"],
        "min_tech_rating": 50.0,
        "max_symbols": 10
    }
    response = requests.post(f"{BASE_URL}/scan/predict", json=payload, timeout=10)
    success = response.status_code == 200
    print_result(success, f"Status Code: {response.status_code}", response.json() if success else None)
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 5: Execute Scan (with parameters)
print_test_header("Execute Scan")
try:
    payload = {
        "sectors": ["Technology"],
        "min_tech_rating": 50.0,
        "max_symbols": 5,
        "trade_style": "swing",
        "risk_profile": "moderate"
    }
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/scan/execute", json=payload, timeout=30)
    success = response.status_code == 200
    print_result(success, f"Status Code: {response.status_code}", response.json() if success else None)
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 6: Error Handling - Invalid Endpoint
print_test_header("Error Handling - Invalid Endpoint")
try:
    response = requests.get(f"{BASE_URL}/invalid/endpoint", timeout=5)
    success = response.status_code == 404
    print_result(success, f"Status Code: {response.status_code} (Expected 404)")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 7: Error Handling - Invalid Parameters
print_test_header("Error Handling - Invalid Parameters")
try:
    payload = {
        "sectors": "invalid",  # Should be a list
        "min_tech_rating": "invalid",  # Should be a number
    }
    response = requests.post(f"{BASE_URL}/scan/predict", json=payload, timeout=10)
    success = response.status_code in [400, 422]  # Bad request or validation error
    print_result(success, f"Status Code: {response.status_code} (Expected 400 or 422)")
    if not success:
        print(f"Response: {response.text}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Summary
print("\n" + "="*60)
print("API TESTING COMPLETE")
print("="*60)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Base URL: {BASE_URL}")
print("\nAll critical endpoints tested!")
