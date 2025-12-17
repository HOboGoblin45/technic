"""
Load Test Runner Script
Automates running load tests and generating reports
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_quick_test(host: str = "http://localhost:8000"):
    """Run quick load test"""
    print("\n" + "="*80)
    print("RUNNING QUICK LOAD TEST")
    print("="*80)
    
    cmd = [
        "locust",
        "-f", "tests/load_testing/locustfile.py",
        "--host", host,
        "--users", "10",
        "--spawn-rate", "2",
        "--run-time", "30s",
        "--headless",
        "--only-summary"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Quick test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Quick test failed: {e}")
        return False


def run_stress_test(host: str = "http://localhost:8000"):
    """Run stress load test"""
    print("\n" + "="*80)
    print("RUNNING STRESS LOAD TEST")
    print("="*80)
    
    cmd = [
        "locust",
        "-f", "tests/load_testing/locustfile.py",
        "--host", host,
        "--users", "50",
        "--spawn-rate", "5",
        "--run-time", "60s",
        "--headless",
        "--only-summary"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Stress test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Stress test failed: {e}")
        return False


def run_scenario_tests(host: str = "http://localhost:8000"):
    """Run predefined test scenarios"""
    print("\n" + "="*80)
    print("RUNNING SCENARIO TESTS")
    print("="*80)
    
    cmd = ["python", "tests/load_testing/test_scenarios.py", host]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Scenario tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Scenario tests failed: {e}")
        return False


def run_web_ui(host: str = "http://localhost:8000"):
    """Launch Locust web UI for interactive testing"""
    print("\n" + "="*80)
    print("LAUNCHING LOCUST WEB UI")
    print("="*80)
    print(f"\nWeb UI will be available at: http://localhost:8089")
    print(f"Target host: {host}")
    print("\nPress Ctrl+C to stop\n")
    
    cmd = [
        "locust",
        "-f", "tests/load_testing/locustfile.py",
        "--host", host
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n✓ Web UI stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run load tests for Technic scanner")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="API host URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "stress", "scenarios", "web", "all"],
        default="all",
        help="Test mode to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TECHNIC SCANNER LOAD TESTING")
    print("="*80)
    print(f"Host: {args.host}")
    print(f"Mode: {args.mode}")
    print("="*80)
    
    results = []
    
    if args.mode == "web":
        run_web_ui(args.host)
        return 0
    
    if args.mode in ["quick", "all"]:
        results.append(("Quick Test", run_quick_test(args.host)))
    
    if args.mode in ["scenarios", "all"]:
        results.append(("Scenario Tests", run_scenario_tests(args.host)))
    
    if args.mode in ["stress", "all"]:
        results.append(("Stress Test", run_stress_test(args.host)))
    
    # Print summary
    print("\n" + "="*80)
    print("LOAD TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
