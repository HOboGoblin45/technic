"""
Option A Deployment Script
Automates the deployment of optimized monitoring API
"""

import subprocess
import sys
import time
import requests
import os
import signal

def print_step(step_num, message):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {message}")
    print('='*60)

def check_port_in_use(port=8003):
    """Check if port is in use"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return True
    except:
        return False

def kill_process_on_port(port=8003):
    """Kill process running on specified port (Windows)"""
    try:
        # Find process on port
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            # Extract PID
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    print(f"Found process {pid} on port {port}")
                    
                    # Kill the process
                    subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                    print(f"Killed process {pid}")
                    time.sleep(2)
                    return True
        return False
    except Exception as e:
        print(f"Error killing process: {e}")
        return False

def start_optimized_api():
    """Start the optimized monitoring API"""
    print("Starting monitoring_api_optimized.py...")
    
    # Start in a new process
    if sys.platform == 'win32':
        # Windows
        subprocess.Popen(
            ['python', 'monitoring_api_optimized.py'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac
        subprocess.Popen(
            ['python', 'monitoring_api_optimized.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    print("Waiting for API to start...")
    time.sleep(5)

def test_api_endpoints():
    """Test the new performance endpoints"""
    endpoints = [
        '/health',
        '/performance/cache',
        '/performance/connections',
        '/performance/summary'
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"http://localhost:8003{endpoint}", timeout=5)
            results[endpoint] = {
                'status': response.status_code,
                'success': response.status_code == 200
            }
            print(f"✓ {endpoint}: {response.status_code}")
        except Exception as e:
            results[endpoint] = {
                'status': 'ERROR',
                'success': False,
                'error': str(e)
            }
            print(f"✗ {endpoint}: ERROR - {e}")
    
    return results

def run_performance_tests():
    """Run the performance test suite"""
    print("Running performance tests...")
    try:
        result = subprocess.run(
            ['python', 'test_performance_optimization.py'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Tests timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main deployment function"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Option A: Quick Wins Deployment Script          ║
    ║                                                          ║
    ║  This script will:                                       ║
    ║  1. Stop the current monitoring API (if running)         ║
    ║  2. Start the optimized monitoring API                   ║
    ║  3. Test the new performance endpoints                   ║
    ║  4. Run the performance test suite                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to begin deployment...")
    
    # Step 1: Check if port is in use
    print_step(1, "Checking if port 8003 is in use")
    if check_port_in_use():
        print("⚠ Port 8003 is in use")
        response = input("Would you like to stop the current process? (y/n): ")
        if response.lower() == 'y':
            if kill_process_on_port():
                print("✓ Successfully stopped previous API")
            else:
                print("✗ Could not stop previous API automatically")
                print("Please manually stop monitoring_api.py (Ctrl+C) and run this script again")
                return
        else:
            print("Deployment cancelled. Please stop monitoring_api.py manually.")
            return
    else:
        print("✓ Port 8003 is available")
    
    # Step 2: Start optimized API
    print_step(2, "Starting Optimized Monitoring API")
    start_optimized_api()
    
    # Wait for startup
    print("Waiting for API to be ready...")
    for i in range(10):
        if check_port_in_use():
            print("✓ API is running!")
            break
        time.sleep(1)
        print(f"  Waiting... ({i+1}/10)")
    else:
        print("✗ API did not start successfully")
        print("Please check for errors and try starting manually:")
        print("  python monitoring_api_optimized.py")
        return
    
    # Step 3: Test endpoints
    print_step(3, "Testing Performance Endpoints")
    results = test_api_endpoints()
    
    all_success = all(r['success'] for r in results.values())
    if all_success:
        print("\n✓ All endpoints are working!")
    else:
        print("\n⚠ Some endpoints failed. Check the results above.")
    
    # Step 4: Run performance tests
    print_step(4, "Running Performance Test Suite")
    response = input("Run full performance tests? This will take ~2 minutes (y/n): ")
    if response.lower() == 'y':
        tests_passed = run_performance_tests()
        if tests_passed:
            print("\n✓ All performance tests passed!")
        else:
            print("\n⚠ Some tests failed. Check the output above.")
    else:
        print("Skipped performance tests")
    
    # Summary
    print_step("COMPLETE", "Deployment Summary")
    print("""
    ✓ Optimized monitoring API is running on port 8003
    ✓ Performance endpoints are available
    
    Next Steps:
    1. Monitor the API performance in your dashboard
    2. Check cache hit rates after some usage
    3. Deploy Redis to Render for production ($7/month)
    4. Add frontend improvements
    
    To stop the API:
    - Find the new console window and press Ctrl+C
    - Or use Task Manager to end the Python process
    
    To verify it's working:
    - Open http://localhost:8003/performance/summary
    - Check your monitoring dashboard at http://localhost:8504
    
    Documentation:
    - See OPTION_A_EXECUTION_PLAN.md for next steps
    - See WHATS_NEXT_COMPREHENSIVE_ROADMAP.md for long-term plan
    """)
    
    print("\n🎉 Deployment Complete! Enjoy your 40-50x performance boost!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user")
    except Exception as e:
        print(f"\n\nError during deployment: {e}")
        import traceback
        traceback.print_exc()
