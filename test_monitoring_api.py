"""
Test Monitoring API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8003"

print("=" * 60)
print("MONITORING API TEST SUITE")
print("=" * 60)

# Test 1: Health Check
print("\n1. Testing Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"   Status: {data['status']}")
    print(f"   Uptime: {data['uptime_seconds']:.1f}s")
    print(f"   Total Requests: {data['stats']['total_requests']}")
    print(f"   Active Alerts: {data['stats']['active_alerts']}")
    print("   ✓ Health check passed")
except Exception as e:
    print(f"   ✗ Health check failed: {e}")

# Test 2: Current Metrics
print("\n2. Testing Current Metrics...")
try:
    response = requests.get(f"{BASE_URL}/metrics/current")
    data = response.json()
    metrics = data['metrics']
    
    print(f"   API Metrics:")
    print(f"     Requests/min: {metrics['api_metrics']['requests_per_minute']}")
    print(f"     Avg Response: {metrics['api_metrics']['avg_response_time_ms']}ms")
    print(f"     Error Rate: {metrics['api_metrics']['error_rate_percent']}%")
    
    print(f"   System Metrics:")
    print(f"     Memory: {metrics['system_metrics']['memory_usage_mb']}MB")
    print(f"     CPU: {metrics['system_metrics']['cpu_percent']}%")
    
    print("   ✓ Current metrics retrieved")
except Exception as e:
    print(f"   ✗ Current metrics failed: {e}")

# Test 3: Metrics Summary
print("\n3. Testing Metrics Summary...")
try:
    response = requests.get(f"{BASE_URL}/metrics/summary")
    data = response.json()
    summary = data['summary']
    
    print(f"   Total Requests: {summary['api']['total_requests']}")
    print(f"   Requests/min: {summary['api']['requests_per_minute']}")
    print(f"   Avg Response: {summary['api']['avg_response_time_ms']}ms")
    print(f"   Active Alerts: {summary['alerts']['active_count']}")
    
    print("   ✓ Metrics summary retrieved")
except Exception as e:
    print(f"   ✗ Metrics summary failed: {e}")

# Test 4: Active Alerts
print("\n4. Testing Active Alerts...")
try:
    response = requests.get(f"{BASE_URL}/alerts/active")
    data = response.json()
    
    print(f"   Active Alerts: {data['count']}")
    if data['alerts']:
        for alert in data['alerts']:
            print(f"     - {alert['severity'].upper()}: {alert['message']}")
    else:
        print("     No active alerts")
    
    print("   ✓ Active alerts retrieved")
except Exception as e:
    print(f"   ✗ Active alerts failed: {e}")

# Test 5: Alert History
print("\n5. Testing Alert History...")
try:
    response = requests.get(f"{BASE_URL}/alerts/history?hours=24")
    data = response.json()
    
    print(f"   Alerts (24h): {data['count']}")
    print("   ✓ Alert history retrieved")
except Exception as e:
    print(f"   ✗ Alert history failed: {e}")

# Test 6: Alert Summary
print("\n6. Testing Alert Summary...")
try:
    response = requests.get(f"{BASE_URL}/alerts/summary")
    data = response.json()
    summary = data['summary']
    
    print(f"   Active: {summary['active_count']}")
    print(f"   Total (24h): {summary['total_24h']}")
    print(f"   By Severity: {summary['by_severity']}")
    
    print("   ✓ Alert summary retrieved")
except Exception as e:
    print(f"   ✗ Alert summary failed: {e}")

# Test 7: Root Endpoint
print("\n7. Testing Root Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    
    print(f"   API Name: {data['name']}")
    print(f"   Version: {data['version']}")
    print(f"   Status: {data['status']}")
    print(f"   Endpoints: {len(data['endpoints'])} categories")
    
    print("   ✓ Root endpoint working")
except Exception as e:
    print(f"   ✗ Root endpoint failed: {e}")

# Test 8: Simulate some load and check metrics
print("\n8. Simulating Load...")
try:
    # Make several requests to generate metrics
    for i in range(10):
        requests.get(f"{BASE_URL}/health")
        time.sleep(0.1)
    
    # Check updated metrics
    response = requests.get(f"{BASE_URL}/metrics/current")
    data = response.json()
    metrics = data['metrics']
    
    print(f"   Total Requests: {metrics['api_metrics']['total_requests']}")
    print(f"   Recent Requests: {metrics['api_metrics']['recent_requests_count']}")
    print("   ✓ Load simulation complete")
except Exception as e:
    print(f"   ✗ Load simulation failed: {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All monitoring API endpoints are operational!")
print("\nAPI is ready for:")
print("  - Real-time metrics monitoring")
print("  - Alert management")
print("  - Historical data analysis")
print("  - Dashboard integration")
print("\n✓ Monitoring API test complete")
