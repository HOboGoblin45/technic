"""
Automated Performance Monitoring Script
Tracks key metrics and logs performance data
Path 1 Task 4: Performance Monitoring
"""

import requests
import time
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys

# Configuration
API_URL = "http://localhost:8003"
LOG_DIR = Path("logs/performance")
CHECK_INTERVAL = 60  # seconds
ALERT_THRESHOLDS = {
    'cache_hit_rate': 50.0,  # Alert if below 50%
    'avg_response_time': 1000,  # Alert if above 1000ms
    'error_rate': 5.0,  # Alert if above 5%
    'connection_utilization': 90.0  # Alert if above 90%
}


class PerformanceMonitor:
    """
    Automated performance monitoring with logging and alerts
    
    Features:
    - Periodic metric collection
    - CSV logging
    - Alert detection
    - Trend analysis
    - Summary reports
    """
    
    def __init__(self, api_url: str = API_URL, log_dir: Path = LOG_DIR):
        """
        Initialize performance monitor
        
        Args:
            api_url: Base URL for monitoring API
            log_dir: Directory for log files
        """
        self.api_url = api_url
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.metrics_log = self.log_dir / "performance_metrics.csv"
        self.alerts_log = self.log_dir / "alerts.log"
        self.summary_log = self.log_dir / "daily_summary.json"
        
        # Initialize CSV if needed
        if not self.metrics_log.exists():
            self._init_metrics_log()
    
    def _init_metrics_log(self):
        """Initialize CSV log file with headers"""
        headers = [
            'timestamp',
            'cache_hit_rate',
            'cache_hits',
            'cache_misses',
            'total_requests',
            'avg_response_time',
            'p95_response_time',
            'error_rate',
            'active_connections',
            'connection_reuse_rate',
            'speedup'
        ]
        
        with open(self.metrics_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def fetch_metrics(self) -> Dict[str, Any]:
        """Fetch current metrics from API"""
        metrics = {}
        
        try:
            # Cache stats
            response = requests.get(f"{self.api_url}/performance/cache", timeout=5)
            if response.status_code == 200:
                cache_data = response.json()
                metrics['cache'] = cache_data
            
            # Connection stats
            response = requests.get(f"{self.api_url}/performance/connections", timeout=5)
            if response.status_code == 200:
                conn_data = response.json()
                metrics['connections'] = conn_data
            
            # Performance summary
            response = requests.get(f"{self.api_url}/performance/summary", timeout=5)
            if response.status_code == 200:
                perf_data = response.json()
                metrics['performance'] = perf_data
            
            return metrics
        
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {}
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to CSV file"""
        if not metrics:
            return
        
        cache = metrics.get('cache', {})
        conn = metrics.get('connections', {})
        perf = metrics.get('performance', {})
        
        row = [
            datetime.now().isoformat(),
            cache.get('hit_rate', 0),
            cache.get('cache_hits', 0),
            cache.get('cache_misses', 0),
            cache.get('total_requests', 0),
            perf.get('response_times', {}).get('avg_cached', 0),
            perf.get('response_times', {}).get('avg_uncached', 0),
            0,  # error_rate (would need to be added to API)
            conn.get('active_connections', 0),
            conn.get('connection_reuse_rate', 0),
            perf.get('response_times', {}).get('speedup', '1.0x')
        ]
        
        with open(self.metrics_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        cache = metrics.get('cache', {})
        conn = metrics.get('connections', {})
        
        # Cache hit rate alert
        hit_rate = cache.get('hit_rate', 0)
        if hit_rate < ALERT_THRESHOLDS['cache_hit_rate']:
            alerts.append(
                f"LOW CACHE HIT RATE: {hit_rate:.1f}% "
                f"(threshold: {ALERT_THRESHOLDS['cache_hit_rate']}%)"
            )
        
        # Connection utilization alert
        active = conn.get('active_connections', 0)
        max_conn = conn.get('max_connections', 20)
        utilization = (active / max_conn * 100) if max_conn > 0 else 0
        
        if utilization > ALERT_THRESHOLDS['connection_utilization']:
            alerts.append(
                f"HIGH CONNECTION UTILIZATION: {utilization:.1f}% "
                f"(threshold: {ALERT_THRESHOLDS['connection_utilization']}%)"
            )
        
        return alerts
    
    def log_alerts(self, alerts: List[str]):
        """Log alerts to file"""
        if not alerts:
            return
        
        timestamp = datetime.now().isoformat()
        
        with open(self.alerts_log, 'a') as f:
            for alert in alerts:
                f.write(f"[{timestamp}] {alert}\n")
                print(f"🚨 ALERT: {alert}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate daily summary from logs"""
        if not self.metrics_log.exists():
            return {}
        
        # Read today's metrics
        today = datetime.now().date()
        metrics_today = []
        
        with open(self.metrics_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.fromisoformat(row['timestamp'])
                if timestamp.date() == today:
                    metrics_today.append(row)
        
        if not metrics_today:
            return {}
        
        # Calculate summary statistics
        hit_rates = [float(m['cache_hit_rate']) for m in metrics_today]
        response_times = [float(m['avg_response_time']) for m in metrics_today]
        
        summary = {
            'date': today.isoformat(),
            'checks_performed': len(metrics_today),
            'cache_hit_rate': {
                'avg': sum(hit_rates) / len(hit_rates),
                'min': min(hit_rates),
                'max': max(hit_rates)
            },
            'response_time': {
                'avg': sum(response_times) / len(response_times),
                'min': min(response_times),
                'max': max(response_times)
            },
            'total_requests': sum(int(m['total_requests']) for m in metrics_today)
        }
        
        # Save summary
        with open(self.summary_log, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_once(self):
        """Run a single monitoring check"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking performance...")
        
        # Fetch metrics
        metrics = self.fetch_metrics()
        
        if metrics:
            # Log metrics
            self.log_metrics(metrics)
            
            # Check for alerts
            alerts = self.check_alerts(metrics)
            if alerts:
                self.log_alerts(alerts)
            
            # Display summary
            cache = metrics.get('cache', {})
            print(f"  Cache Hit Rate: {cache.get('hit_rate', 0):.1f}%")
            print(f"  Total Requests: {cache.get('total_requests', 0):,}")
            print(f"  Alerts: {len(alerts)}")
        else:
            print("  ⚠️ Failed to fetch metrics")
    
    def run_continuous(self, interval: int = CHECK_INTERVAL):
        """Run continuous monitoring"""
        print(f"Starting performance monitoring (interval: {interval}s)")
        print(f"Logs: {self.log_dir}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_once()
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
            
            # Generate final summary
            summary = self.generate_summary()
            if summary:
                print("\n📊 Daily Summary:")
                print(f"  Checks: {summary['checks_performed']}")
                print(f"  Avg Hit Rate: {summary['cache_hit_rate']['avg']:.1f}%")
                print(f"  Avg Response Time: {summary['response_time']['avg']:.0f}ms")
                print(f"  Total Requests: {summary['total_requests']:,}")
            
            print(f"\nLogs saved to: {self.log_dir}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitoring")
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--api-url',
        default=API_URL,
        help=f'API URL (default: {API_URL})'
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = PerformanceMonitor(api_url=args.api_url)
    
    if args.once:
        # Run once
        monitor.run_once()
        
        # Show summary
        summary = monitor.generate_summary()
        if summary:
            print("\n📊 Summary:")
            print(json.dumps(summary, indent=2))
    else:
        # Run continuously
        monitor.run_continuous(interval=args.interval)


if __name__ == "__main__":
    main()
