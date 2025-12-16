"""
Test monitoring components
"""

import sys
import time

from technic_v4.monitoring.metrics_collector import MetricsCollector
from technic_v4.monitoring.alerts import AlertSystem

print("Testing Monitoring System")
print("=" * 60)

# Create alert system
alert_system = AlertSystem()

# Create metrics collector
collector = MetricsCollector()

# Simulate normal operation
print("\n1. Normal operation (no alerts)...")
for i in range(5):
    collector.record_request("/scan/predict", 150, 200)

metrics = collector.get_current_metrics()
alert_system.check_metrics(metrics)
print(f"   Active alerts: {len(alert_system.get_active_alerts())}")

# Simulate high error rate
print("\n2. Simulating high error rate...")
for i in range(10):
    collector.record_request("/scan/predict", 150, 500)  # Error status

metrics = collector.get_current_metrics()
alert_system.check_metrics(metrics)
print(f"   Active alerts: {len(alert_system.get_active_alerts())}")

# Simulate slow response
print("\n3. Simulating slow response time...")
collector.reset_metrics()
for i in range(10):
    collector.record_request("/scan/predict", 600, 200)  # Slow response

metrics = collector.get_current_metrics()
alert_system.check_metrics(metrics)
print(f"   Active alerts: {len(alert_system.get_active_alerts())}")

# Get alert summary
print("\n4. Alert Summary:")
print("=" * 60)
summary = alert_system.get_alert_summary()
print(f"Active alerts: {summary['active_count']}")
print(f"Total (24h): {summary['total_24h']}")
print(f"By severity: {summary['by_severity']}")

print("\n✓ Monitoring system test complete")
