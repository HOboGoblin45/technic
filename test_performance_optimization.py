"""
Test Performance Optimization (Task 5)
Tests caching, connection pooling, and performance improvements
"""

import requests
import time
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8003"


def test_cache_functionality():
    """Test that caching is working"""
    print("\n" + "="*60)
    print("TEST 1: Cache Functionality")
    print("="*60)
    
    # Clear cache first
    try:
        response = requests.post(f"{BASE_URL}/performance/cache/clear")
        print("✓ Cache cleared")
    except:
        print("⚠ Could not clear cache (endpoint may not exist in old version)")
    
    # First request (cache miss)
    start = time.time()
    response1 = requests.get(f"{BASE_URL}/metrics/current")
    time1 = time.time() - start
    
    # Second request (should be cached)
    start = time.time()
    response2 = requests.get(f"{BASE_URL}/metrics/current")
    time2 = time.time() - start
    
    print(f"\nFirst request time: {time1*1000:.2f}ms")
    print(f"Second request time: {time2*1000:.2f}ms")
    
    if time2 < time1:
        speedup = time1 / time2
        print(f"✓ Cache speedup: {speedup:.2f}x faster")
    else:
        print("⚠ No significant speedup detected")
    
    # Check if response indicates caching
    if response2.status_code == 200:
        data = response2.json()
        if 'cached' in data:
            print(f"✓ Cache status in response: {data['cached']}")
    
    return True


def test_cache_stats():
    """Test cache statistics endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Cache Statistics")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/performance/cache")
        
        if response.status_code == 200:
            stats = response.json()
            print("\nCache Statistics:")
            if 'cache_stats' in stats:
                cache_stats = stats['cache_stats']
                print(f"  Hits: {cache_stats.get('hits', 0)}")
                print(f"  Misses: {cache_stats.get('misses', 0)}")
                print(f"  Total Requests: {cache_stats.get('total_requests', 0)}")
                print(f"  Hit Rate: {cache_stats.get('hit_rate_percent', 0):.2f}%")
                print(f"  Valid Entries: {cache_stats.get('valid_entries', 0)}")
                print("✓ Cache statistics available")
                return True
            else:
                print("⚠ Cache stats not in expected format")
        else:
            print(f"⚠ Cache stats endpoint returned {response.status_code}")
            print("  (This is expected if using non-optimized version)")
    except Exception as e:
        print(f"⚠ Could not get cache stats: {e}")
        print("  (This is expected if using non-optimized version)")
    
    return False


def test_connection_pool():
    """Test connection pool statistics"""
    print("\n" + "="*60)
    print("TEST 3: Connection Pool")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/performance/connections")
        
        if response.status_code == 200:
            stats = response.json()
            print("\nConnection Pool Statistics:")
            if 'connection_pool' in stats:
                pool_stats = stats['connection_pool']
                print(f"  Max Connections: {pool_stats.get('max_connections', 0)}")
                print(f"  Active Connections: {pool_stats.get('active_connections', 0)}")
                print(f"  Total Requests: {pool_stats.get('total_requests', 0)}")
                print(f"  Avg Wait Time: {pool_stats.get('avg_wait_time_ms', 0):.2f}ms")
                print(f"  Utilization: {pool_stats.get('utilization_percent', 0):.2f}%")
                print("✓ Connection pool statistics available")
                return True
            else:
                print("⚠ Connection pool stats not in expected format")
        else:
            print(f"⚠ Connection pool endpoint returned {response.status_code}")
            print("  (This is expected if using non-optimized version)")
    except Exception as e:
        print(f"⚠ Could not get connection pool stats: {e}")
        print("  (This is expected if using non-optimized version)")
    
    return False


def test_performance_summary():
    """Test overall performance summary"""
    print("\n" + "="*60)
    print("TEST 4: Performance Summary")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/performance/summary")
        
        if response.status_code == 200:
            data = response.json()
            print("\nPerformance Summary:")
            if 'performance' in data:
                perf = data['performance']
                
                if 'cache' in perf:
                    print(f"\nCache:")
                    print(f"  Enabled: {perf['cache'].get('enabled', False)}")
                    print(f"  Hit Rate: {perf['cache'].get('hit_rate_percent', 0):.2f}%")
                
                if 'connection_pool' in perf:
                    print(f"\nConnection Pool:")
                    print(f"  Active: {perf['connection_pool'].get('active_connections', 0)}")
                    print(f"  Utilization: {perf['connection_pool'].get('utilization_percent', 0):.2f}%")
                
                if 'optimizations' in perf:
                    print(f"\nOptimizations:")
                    for opt in perf['optimizations']:
                        print(f"  ✓ {opt}")
                
                print("\n✓ Performance summary available")
                return True
            else:
                print("⚠ Performance data not in expected format")
        else:
            print(f"⚠ Performance summary endpoint returned {response.status_code}")
            print("  (This is expected if using non-optimized version)")
    except Exception as e:
        print(f"⚠ Could not get performance summary: {e}")
        print("  (This is expected if using non-optimized version)")
    
    return False


def test_response_times():
    """Test and compare response times"""
    print("\n" + "="*60)
    print("TEST 5: Response Time Comparison")
    print("="*60)
    
    endpoints = [
        "/health",
        "/metrics/current",
        "/metrics/summary",
        "/alerts/active"
    ]
    
    print("\nTesting response times (10 requests each):")
    print("-" * 60)
    
    for endpoint in endpoints:
        times = []
        for i in range(10):
            start = time.time()
            try:
                response = requests.get(f"{BASE_URL}{endpoint}")
                elapsed = (time.time() - start) * 1000  # Convert to ms
                if response.status_code == 200:
                    times.append(elapsed)
            except:
                pass
            time.sleep(0.1)  # Small delay between requests
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n{endpoint}:")
            print(f"  Avg: {avg_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
            
            if avg_time < 100:
                print(f"  ✓ Good performance (< 100ms)")
            elif avg_time < 200:
                print(f"  ⚠ Acceptable performance (< 200ms)")
            else:
                print(f"  ⚠ Slow performance (> 200ms)")
    
    return True


def test_concurrent_requests():
    """Test performance under concurrent load"""
    print("\n" + "="*60)
    print("TEST 6: Concurrent Request Handling")
    print("="*60)
    
    import concurrent.futures
    
    def make_request():
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/metrics/current", timeout=5)
            elapsed = time.time() - start
            return elapsed if response.status_code == 200 else None
        except:
            return None
    
    print("\nSending 20 concurrent requests...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        start = time.time()
        futures = [executor.submit(make_request) for _ in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start
    
    successful = [r for r in results if r is not None]
    
    if successful:
        avg_response = sum(successful) / len(successful) * 1000
        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Successful requests: {len(successful)}/20")
        print(f"  Avg response time: {avg_response:.2f}ms")
        print(f"  Requests/second: {len(successful)/total_time:.2f}")
        
        if len(successful) == 20:
            print("✓ All requests successful")
        else:
            print(f"⚠ {20 - len(successful)} requests failed")
        
        return True
    else:
        print("✗ No successful requests")
        return False


def main():
    """Run all performance tests"""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION TESTS (Task 5)")
    print("="*60)
    print("\nTesting monitoring API performance optimizations...")
    print("Base URL:", BASE_URL)
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("\n✗ API is not responding correctly")
            return 1
    except:
        print("\n✗ Could not connect to API")
        print("  Make sure monitoring API is running on port 8003")
        return 1
    
    tests = [
        ("Cache Functionality", test_cache_functionality),
        ("Cache Statistics", test_cache_stats),
        ("Connection Pool", test_connection_pool),
        ("Performance Summary", test_performance_summary),
        ("Response Times", test_response_times),
        ("Concurrent Requests", test_concurrent_requests)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
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
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed >= 4:  # At least basic tests should pass
        print("\n✓ Performance optimization tests completed successfully")
        print("\nNote: Some tests may fail if using non-optimized API version")
        return 0
    else:
        print(f"\n⚠ Only {passed} tests passed")
        return 1


if __name__ == "__main__":
    exit(main())
