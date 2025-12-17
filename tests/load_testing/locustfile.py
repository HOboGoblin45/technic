"""
Locust Load Testing Configuration
Defines user behavior and load test scenarios for the Technic scanner
"""

from locust import HttpUser, task, between, events
import random
import time
from typing import Dict, Any


class ScannerUser(HttpUser):
    """
    Simulates a user interacting with the scanner API
    
    This user will:
    - Check health endpoints
    - Run scans with various parameters
    - Check cache status
    - Monitor performance metrics
    """
    
    # Wait between 1-5 seconds between tasks
    wait_time = between(1, 5)
    
    # Test data
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    
    def on_start(self):
        """Called when a user starts"""
        self.client.verify = False  # Disable SSL verification for local testing
        print(f"[USER] Starting load test user")
    
    @task(10)
    def health_check(self):
        """Check API health (most common operation)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    def cache_status(self):
        """Check cache status"""
        with self.client.get("/cache/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'available' in data:
                    response.success()
                else:
                    response.failure("Invalid cache status response")
            else:
                response.failure(f"Cache status failed: {response.status_code}")
    
    @task(3)
    def performance_metrics(self):
        """Get performance metrics"""
        with self.client.get("/performance/summary", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Performance metrics failed: {response.status_code}")
    
    @task(2)
    def quick_scan(self):
        """Run a quick scan with limited symbols"""
        params = {
            "max_symbols": 5,
            "sectors": random.choice(self.sectors),
            "min_tech_rating": 50
        }
        
        with self.client.post("/scan", json=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'results' in data or 'status' in data:
                    response.success()
                else:
                    response.failure("Invalid scan response")
            else:
                response.failure(f"Scan failed: {response.status_code}")
    
    @task(1)
    def full_scan(self):
        """Run a full sector scan"""
        params = {
            "max_symbols": 20,
            "sectors": random.choice(self.sectors),
            "min_tech_rating": 60
        }
        
        with self.client.post("/scan", json=params, catch_response=True, timeout=60) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 202:
                # Accepted - scan in progress
                response.success()
            else:
                response.failure(f"Full scan failed: {response.status_code}")
    
    @task(1)
    def symbol_lookup(self):
        """Look up a specific symbol"""
        symbol = random.choice(self.symbols)
        
        with self.client.get(f"/symbol/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Symbol not found is acceptable
                response.success()
            else:
                response.failure(f"Symbol lookup failed: {response.status_code}")


class QuickTest(HttpUser):
    """
    Quick test user for rapid validation
    Only performs lightweight operations
    """
    
    wait_time = between(0.5, 2)
    
    @task(5)
    def health_check(self):
        """Rapid health checks"""
        self.client.get("/health")
    
    @task(3)
    def cache_status(self):
        """Check cache"""
        self.client.get("/cache/status")
    
    @task(1)
    def quick_scan(self):
        """Minimal scan"""
        params = {"max_symbols": 3, "min_tech_rating": 70}
        self.client.post("/scan", json=params, timeout=30)


class StressTest(HttpUser):
    """
    Stress test user for maximum load
    Performs intensive operations rapidly
    """
    
    wait_time = between(0.1, 1)
    
    @task(3)
    def rapid_health_checks(self):
        """Rapid fire health checks"""
        for _ in range(5):
            self.client.get("/health")
            time.sleep(0.1)
    
    @task(2)
    def concurrent_scans(self):
        """Multiple concurrent scans"""
        params = {"max_symbols": 10, "min_tech_rating": 50}
        self.client.post("/scan", json=params, timeout=45)
    
    @task(1)
    def cache_operations(self):
        """Rapid cache operations"""
        self.client.get("/cache/status")
        self.client.post("/cache/clear")
        self.client.get("/cache/stats")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("\n" + "="*80)
    print("LOAD TEST STARTING")
    print("="*80)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("\n" + "="*80)
    print("LOAD TEST COMPLETE")
    print("="*80)
    
    # Print summary statistics
    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    
    if stats.total.num_requests > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure Rate: {failure_rate:.2f}%")
    
    print("="*80 + "\n")


# Custom event for tracking scan performance
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track individual requests"""
    if name == "/scan" and response_time > 5000:
        print(f"[SLOW SCAN] Scan took {response_time:.0f}ms")


if __name__ == "__main__":
    # This allows running the file directly for testing
    print("Locust configuration loaded successfully")
    print("\nAvailable user classes:")
    print("  - ScannerUser: Standard user behavior")
    print("  - QuickTest: Lightweight operations")
    print("  - StressTest: Maximum load testing")
    print("\nTo run load tests:")
    print("  locust -f tests/load_testing/locustfile.py --host=http://localhost:8000")
