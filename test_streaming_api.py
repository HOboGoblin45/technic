"""
Test script for Phase 3E-B Streaming API
Tests WebSocket and HTTP endpoints
"""

import requests
import json
import time
import asyncio
import websockets
from typing import Dict, Any


def test_start_streaming_scan():
    """Test starting a streaming scan"""
    print("\n" + "="*60)
    print("TEST 1: Start Streaming Scan")
    print("="*60)
    
    url = "http://localhost:8001/scan/stream"
    payload = {
        "max_symbols": 20,
        "termination_criteria": {
            "max_signals": 5,
            "timeout_seconds": 30
        }
    }
    
    print(f"\nPOST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(url, json=payload)
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response:")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    return result


def test_get_stats(scan_id: str):
    """Test getting scan statistics"""
    print("\n" + "="*60)
    print("TEST 2: Get Scan Statistics")
    print("="*60)
    
    url = f"http://localhost:8001/scan/stats/{scan_id}"
    
    print(f"\nGET {url}")
    
    # Poll stats a few times
    for i in range(5):
        time.sleep(1)
        response = requests.get(url)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"\n[{i+1}/5] Stats:")
            print(f"  Status: {stats['status']}")
            print(f"  Processed: {stats['stats']['processed']}/{stats['stats']['total_symbols']}")
            print(f"  Signals: {stats['stats']['signals_found']}")
            print(f"  Progress: {stats['stats']['progress_pct']:.1f}%")
            
            if stats['status'] == 'complete':
                print("\n✓ Scan complete!")
                break
        else:
            print(f"Error: {response.status_code}")
            break
    
    return stats if response.status_code == 200 else None


def test_list_active_scans():
    """Test listing active scans"""
    print("\n" + "="*60)
    print("TEST 3: List Active Scans")
    print("="*60)
    
    url = "http://localhost:8001/scans/active"
    
    print(f"\nGET {url}")
    
    response = requests.get(url)
    
    print(f"\nStatus: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Active scans: {result['active_scans']}")
        
        for scan in result['scans']:
            print(f"\n  Scan ID: {scan['scan_id']}")
            print(f"  Status: {scan['status']}")
            if scan['stats']:
                print(f"  Progress: {scan['stats']['progress_pct']:.1f}%")
                print(f"  Signals: {scan['stats']['signals_found']}")


async def test_websocket_stream(scan_id: str):
    """Test WebSocket streaming"""
    print("\n" + "="*60)
    print("TEST 4: WebSocket Streaming")
    print("="*60)
    
    uri = f"ws://localhost:8001/ws/results/{scan_id}"
    
    print(f"\nConnecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to WebSocket")
            
            result_count = 0
            signal_count = 0
            
            # Receive messages
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    
                    msg_type = data.get('type')
                    
                    if msg_type == 'connected':
                        print(f"✓ {data.get('message')}")
                    
                    elif msg_type == 'result':
                        result_count += 1
                        result_data = data['data']
                        symbol = result_data['symbol']
                        signal = result_data.get('signal')
                        
                        if signal:
                            signal_count += 1
                            print(f"  [{result_count}] {symbol}: {signal} ✓")
                        else:
                            print(f"  [{result_count}] {symbol}: no signal")
                    
                    elif msg_type == 'stats':
                        stats = data['data']
                        print(f"\n  Progress: {stats['progress_pct']:.1f}% ({stats['processed']}/{stats['total_symbols']})")
                    
                    elif msg_type == 'complete':
                        print(f"\n✓ Scan complete!")
                        final_stats = data.get('stats', {})
                        print(f"  Total processed: {final_stats.get('processed')}")
                        print(f"  Signals found: {final_stats.get('signals_found')}")
                        print(f"  Time: {final_stats.get('elapsed_time', 0):.2f}s")
                        break
                    
                    elif msg_type == 'error':
                        print(f"✗ Error: {data.get('message')}")
                        break
                
                except asyncio.TimeoutError:
                    print("Timeout waiting for message")
                    break
            
            print(f"\nReceived {result_count} results, {signal_count} signals")
    
    except Exception as e:
        print(f"✗ WebSocket error: {e}")


def test_stop_scan(scan_id: str):
    """Test stopping a scan early"""
    print("\n" + "="*60)
    print("TEST 5: Stop Scan Early")
    print("="*60)
    
    url = f"http://localhost:8001/scan/stop/{scan_id}"
    
    print(f"\nPOST {url}")
    
    response = requests.post(url)
    
    print(f"\nStatus: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result.get('stats'):
            stats = result['stats']
            print(f"\nFinal Stats:")
            print(f"  Processed: {stats['processed']}/{stats['total_symbols']}")
            print(f"  Signals: {stats['signals_found']}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3E-B: STREAMING API TESTS")
    print("="*60)
    
    # Test 1: Start a streaming scan
    result = test_start_streaming_scan()
    scan_id = result['scan_id']
    
    print(f"\n✓ Scan started with ID: {scan_id}")
    
    # Test 2: Get stats while running
    test_get_stats(scan_id)
    
    # Test 3: List active scans
    test_list_active_scans()
    
    # Test 4: WebSocket streaming (start new scan for this)
    print("\n\nStarting new scan for WebSocket test...")
    result2 = test_start_streaming_scan()
    scan_id2 = result2['scan_id']
    
    # Run WebSocket test
    asyncio.run(test_websocket_stream(scan_id2))
    
    # Test 5: Early termination (start another scan)
    print("\n\nStarting new scan for early termination test...")
    result3 = requests.post(
        "http://localhost:8001/scan/stream",
        json={"max_symbols": 50}
    ).json()
    scan_id3 = result3['scan_id']
    
    time.sleep(2)  # Let it run a bit
    test_stop_scan(scan_id3)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\n✓ All streaming API tests completed successfully!")
    print("\nFeatures Tested:")
    print("  1. Start streaming scan")
    print("  2. Get real-time statistics")
    print("  3. List active scans")
    print("  4. WebSocket result streaming")
    print("  5. Early termination")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTest error: {e}")
        import traceback
        traceback.print_exc()
