"""
Load Test Scenarios
Predefined test scenarios for different load testing needs
"""

import requests
import time
from typing import Dict, List, Tuple
import concurrent.futures
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a load test"""
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    duration: float
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def print_summary(self):
        """Print test result summary"""
        print(f"\n{'='*80}")
        print(f"TEST RESULTS: {self.name}")
        print(f"{'='*80}")
        print(f"Duration: {self.duration:.2f}s")
        print(f"Total Requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests}")
        print(f"Failed: {self.failed_requests}")
        print(f"Success Rate: {self.get_success_rate():.1f}%")
        print(f"Avg Response Time: {self.avg_response_time:.2f}ms")
        print(f"Min Response Time: {self.min_response_time:.2f}ms")
        print(f"Max Response Time: {self.max_response_time:.2f}ms")
        print(f"Requests/sec: {self.requests_per_second:.2f}")
        print(f"{'='*80}\n")


def test_health_endpoint(base_url: str = "http://localhost:8000", num_requests: int = 100) -> TestResult:
    """
    Test health endpoint performance
    
    Args:
        base_url: Base URL of the API
        num_requests: Number of requests to make
    
    Returns:
        TestResult with performance metrics
    """
    print(f"\n[TEST] Health Endpoint - {num_requests} requests")
    
    response_times = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i in range(num_requests):
        try:
            req_start = time.time()
            response = requests.get(f"{base_url}/health", timeout=5)
            req_time = (time.time() - req_start) * 1000  # Convert to ms
            
            response_times.append(req_time)
            
            if response.status_code == 200:
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
            response_times.append(5000)  # Timeout value
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_requests}")
    
    duration = time.time() - start_time
    
    result = TestResult(
        name="Health Endpoint Test",
        total_requests=num_requests,
        successful_requests=successful,
        failed_requests=failed,
        avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
        min_response_time=min(response_times) if response_times else 0,
        max_response_time=max(response_times) if response_times else 0,
        requests_per_second=num_requests / duration if duration > 0 else 0,
        duration=duration
    )
    
    result.print_summary()
    return result


def test_scan_endpoint(base_url: str = "http://localhost:8000", num_scans: int = 10) -> TestResult:
    """
    Test scan endpoint performance
    
    Args:
        base_url: Base URL of the API
        num_scans: Number of scans to run
    
    Returns:
        TestResult with performance metrics
    """
    print(f"\n[TEST] Scan Endpoint - {num_scans} scans")
    
    response_times = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i in range(num_scans):
        try:
            params = {
                "max_symbols": 5,
                "sectors": "Technology",
                "min_tech_rating": 50
            }
            
            req_start = time.time()
            response = requests.post(f"{base_url}/scan", json=params, timeout=60)
            req_time = (time.time() - req_start) * 1000
            
            response_times.append(req_time)
            
            if response.status_code in [200, 202]:
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
            response_times.append(60000)  # Timeout value
        
        print(f"  Scan {i + 1}/{num_scans} completed")
    
    duration = time.time() - start_time
    
    result = TestResult(
        name="Scan Endpoint Test",
        total_requests=num_scans,
        successful_requests=successful,
        failed_requests=failed,
        avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
        min_response_time=min(response_times) if response_times else 0,
        max_response_time=max(response_times) if response_times else 0,
        requests_per_second=num_scans / duration if duration > 0 else 0,
        duration=duration
    )
    
    result.print_summary()
    return result


def test_cache_endpoint(base_url: str = "http://localhost:8000", num_requests: int = 50) -> TestResult:
    """
    Test cache status endpoint performance
    
    Args:
        base_url: Base URL of the API
        num_requests: Number of requests to make
    
    Returns:
        TestResult with performance metrics
    """
    print(f"\n[TEST] Cache Endpoint - {num_requests} requests")
    
    response_times = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i in range(num_requests):
        try:
            req_start = time.time()
            response = requests.get(f"{base_url}/cache/status", timeout=5)
            req_time = (time.time() - req_start) * 1000
            
            response_times.append(req_time)
            
            if response.status_code == 200:
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            failed += 1
            response_times.append(5000)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_requests}")
    
    duration = time.time() - start_time
    
    result = TestResult(
        name="Cache Endpoint Test",
        total_requests=num_requests,
        successful_requests=successful,
        failed_requests=failed,
        avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
        min_response_time=min(response_times) if response_times else 0,
        max_response_time=max(response_times) if response_times else 0,
        requests_per_second=num_requests / duration if duration > 0 else 0,
        duration=duration
    )
    
    result.print_summary()
    return result


def test_concurrent_scans(base_url: str = "http://localhost:8000", num_concurrent: int = 5) -> TestResult:
    """
    Test concurrent scan performance
    
    Args:
        base_url: Base URL of the API
        num_concurrent: Number of concurrent scans
    
    Returns:
        TestResult with performance metrics
    """
    print(f"\n[TEST] Concurrent Scans - {num_concurrent} simultaneous scans")
    
    def run_scan(scan_id: int) -> Tuple[int, float]:
        """Run a single scan"""
        try:
            params = {
                "max_symbols": 5,
                "sectors": "Technology",
                "min_tech_rating": 50
            }
            
            start = time.time()
            response = requests.post(f"{base_url}/scan", json=params, timeout=60)
            duration = (time.time() - start) * 1000
            
            return (response.status_code, duration)
        except Exception as e:
            return (500, 60000)
    
    start_time = time.time()
    
    # Run scans concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(run_scan, i) for i in range(num_concurrent)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    duration = time.time() - start_time
    
    # Process results
    response_times = [r[1] for r in results]
    successful = sum(1 for r in results if r[0] in [200, 202])
    failed = len(results) - successful
    
    result = TestResult(
        name="Concurrent Scans Test",
        total_requests=num_concurrent,
        successful_requests=successful,
        failed_requests=failed,
        avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
        min_response_time=min(response_times) if response_times else 0,
        max_response_time=max(response_times) if response_times else 0,
        requests_per_second=num_concurrent / duration if duration > 0 else 0,
        duration=duration
    )
    
    result.print_summary()
    return result


def run_all_tests(base_url: str = "http://localhost:8000") -> List[TestResult]:
    """
    Run all load test scenarios
    
    Args:
        base_url: Base URL of the API
    
    Returns:
        List of TestResult objects
    """
    print("\n" + "="*80)
    print("RUNNING ALL LOAD TEST SCENARIOS")
    print("="*80)
    
    results = []
    
    # Test 1: Health endpoint
    results.append(test_health_endpoint(base_url, num_requests=100))
    
    # Test 2: Cache endpoint
    results.append(test_cache_endpoint(base_url, num_requests=50))
    
    # Test 3: Scan endpoint
    results.append(test_scan_endpoint(base_url, num_scans=10))
    
    # Test 4: Concurrent scans
    results.append(test_concurrent_scans(base_url, num_concurrent=5))
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL TEST SUMMARY")
    print("="*80)
    
    total_requests = sum(r.total_requests for r in results)
    total_successful = sum(r.successful_requests for r in results)
    total_failed = sum(r.failed_requests for r in results)
    overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Total Requests: {total_requests}")
    print(f"Total Successful: {total_successful}")
    print(f"Total Failed: {total_failed}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    
    print("\nIndividual Test Results:")
    for result in results:
        print(f"  {result.name}: {result.get_success_rate():.1f}% success, {result.avg_response_time:.0f}ms avg")
    
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Run all tests
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Testing API at: {base_url}")
    
    try:
        results = run_all_tests(base_url)
        
        # Exit with error code if any tests failed significantly
        if any(r.get_success_rate() < 90 for r in results):
            print("⚠ Some tests had success rate below 90%")
            sys.exit(1)
        else:
            print("✓ All tests passed with >90% success rate")
            sys.exit(0)
    
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
