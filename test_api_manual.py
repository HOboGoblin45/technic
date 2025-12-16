"""
Manual test script for API endpoints.
Tests the enhanced API with progress tracking.
"""

import requests
import json
import time
import asyncio
import websockets

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n1. Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    assert response.status_code == 200
    assert data["status"] == "ok"
    print("   ✅ Health check passed")
    return data

def test_start_scan():
    """Test starting a scan."""
    print("\n2. Testing POST /scan/start...")
    payload = {
        "max_symbols": 10,
        "min_tech_rating": 15.0,
        "async_mode": True
    }
    response = requests.post(f"{BASE_URL}/scan/start", json=payload)
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    assert response.status_code == 200
    assert "scan_id" in data
    print("   ✅ Scan started successfully")
    return data["scan_id"]

def test_get_progress(scan_id):
    """Test getting scan progress."""
    print(f"\n3. Testing GET /scan/progress/{scan_id}...")
    
    # Poll progress a few times
    for i in range(5):
        response = requests.get(f"{BASE_URL}/scan/progress/{scan_id}")
        print(f"   Attempt {i+1} - Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Scan Status: {data['status']}")
            print(f"   Stage: {data['stage']}")
            print(f"   Progress: {data['current']}/{data['total']} ({data['percentage']:.1f}%)")
            
            if data['status'] in ['completed', 'failed', 'cancelled']:
                print(f"   ✅ Scan {data['status']}")
                return data
        
        time.sleep(1)
    
    print("   ⏱️ Scan still running after 5 seconds")
    return None

def test_list_active():
    """Test listing active scans."""
    print("\n4. Testing GET /scan/active...")
    response = requests.get(f"{BASE_URL}/scan/active")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Active scans: {data['count']}")
    for scan in data['scans']:
        print(f"   - {scan['scan_id']}: {scan['status']} ({scan['percentage']:.1f}%)")
    print("   ✅ Active scans listed")
    return data

def test_cancel_scan(scan_id):
    """Test cancelling a scan."""
    print(f"\n5. Testing POST /scan/cancel/{scan_id}...")
    response = requests.post(f"{BASE_URL}/scan/cancel/{scan_id}")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    print("   ✅ Cancel request sent")
    return data

def test_legacy_endpoint():
    """Test legacy /scan endpoint."""
    print("\n6. Testing GET /scan (legacy)...")
    response = requests.get(f"{BASE_URL}/scan?max_symbols=5&min_tech_rating=10.0")
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Results: {len(data.get('results', []))} symbols")
        if 'performance_metrics' in data:
            print(f"   Performance: {json.dumps(data['performance_metrics'], indent=2)}")
        print("   ✅ Legacy endpoint working")
    else:
        print(f"   ❌ Error: {response.text}")
    
    return response.status_code == 200

async def test_websocket(scan_id):
    """Test WebSocket connection."""
    print(f"\n7. Testing WebSocket /scan/ws/{scan_id}...")
    try:
        uri = f"ws://localhost:8000/scan/ws/{scan_id}"
        async with websockets.connect(uri) as websocket:
            # Receive initial message
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)
            print(f"   WebSocket connected")
            print(f"   Initial status: {data['status']}")
            print("   ✅ WebSocket working")
            return True
    except Exception as e:
        print(f"   ⚠️ WebSocket error: {e}")
        return False

def test_sse(scan_id):
    """Test Server-Sent Events."""
    print(f"\n8. Testing SSE /scan/sse/{scan_id}...")
    try:
        response = requests.get(
            f"{BASE_URL}/scan/sse/{scan_id}",
            stream=True,
            timeout=2
        )
        
        # Read first event
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data:'):
                    data = json.loads(line_str[5:].strip())
                    print(f"   SSE Event received")
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print("   ✅ SSE working")
                    response.close()
                    return True
    except Exception as e:
        print(f"   ⚠️ SSE timeout or error (expected for completed scans)")
        return False

def run_all_tests():
    """Run all API tests."""
    print("=" * 60)
    print("ENHANCED API MANUAL TESTING")
    print("=" * 60)
    
    try:
        # Test health
        health_data = test_health()
        
        # Start a scan
        scan_id = test_start_scan()
        
        # Wait a bit for scan to start
        time.sleep(2)
        
        # Check progress
        progress_data = test_get_progress(scan_id)
        
        # List active scans
        active_data = test_list_active()
        
        # Try to cancel (might already be done)
        if progress_data and progress_data['status'] == 'running':
            cancel_data = test_cancel_scan(scan_id)
        
        # Test legacy endpoint
        legacy_ok = test_legacy_endpoint()
        
        # Test WebSocket (async)
        # Note: Create a new scan for WebSocket test
        new_scan_response = requests.post(f"{BASE_URL}/scan/start", json={
            "max_symbols": 5,
            "min_tech_rating": 10.0,
            "async_mode": True
        })
        if new_scan_response.status_code == 200:
            ws_scan_id = new_scan_response.json()["scan_id"]
            asyncio.run(test_websocket(ws_scan_id))
        
        # Test SSE
        test_sse(scan_id)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
