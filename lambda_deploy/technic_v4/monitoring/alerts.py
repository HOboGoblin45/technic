"""
Alert System for ML API Monitoring
Detects issues and sends notifications
"""

import time
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """
    Alert rule configuration
    
    Example:
        rule = AlertRule(
            name="high_error_rate",
            condition=lambda metrics: metrics['api_metrics']['error_rate_percent'] > 5,
            severity=AlertSeverity.ERROR,
            message="Error rate is {error_rate}%"
        )
    """
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message: str
    cooldown_seconds: int = 300  # 5 minutes
    enabled: bool = True
    last_triggered: Optional[float] = None


@dataclass
class Alert:
    """Active alert"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metrics_snapshot: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[float] = None


class AlertSystem:
    """
    Monitors metrics and triggers alerts based on rules
    
    Features:
    - Configurable alert rules
    - Cooldown periods to prevent spam
    - Alert history
    - Multiple notification channels
    """
    
    def __init__(self):
        """Initialize alert system"""
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Add default rules
        self._add_default_rules()
        
        # Add default notification handler (console logging)
        self.add_notification_handler(self._console_notification)
    
    def _add_default_rules(self):
        """Add default alert rules"""
        
        # High error rate
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda m: m['api_metrics']['error_rate_percent'] > 5.0,
            severity=AlertSeverity.ERROR,
            message="High error rate: {error_rate}%",
            cooldown_seconds=300
        ))
        
        # Slow response time
        self.add_rule(AlertRule(
            name="slow_response_time",
            condition=lambda m: m['api_metrics']['avg_response_time_ms'] > 500,
            severity=AlertSeverity.WARNING,
            message="Slow response time: {response_time}ms",
            cooldown_seconds=300
        ))
        
        # Model accuracy degradation
        self.add_rule(AlertRule(
            name="model_accuracy_degradation",
            condition=lambda m: any(
                model.get('avg_mae', 0) > 15  # 50% worse than baseline
                for model in m['model_metrics'].values()
                if model.get('predictions_count', 0) > 10
            ),
            severity=AlertSeverity.WARNING,
            message="Model accuracy degraded: MAE > 15",
            cooldown_seconds=600
        ))
        
        # High memory usage
        self.add_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda m: m['system_metrics']['memory_usage_mb'] > 1000,
            severity=AlertSeverity.WARNING,
            message="High memory usage: {memory}MB",
            cooldown_seconds=300
        ))
        
        # High CPU usage
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda m: m['system_metrics']['cpu_percent'] > 80,
            severity=AlertSeverity.WARNING,
            message="High CPU usage: {cpu}%",
            cooldown_seconds=300
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler
        
        Args:
            handler: Function that takes an Alert and sends notification
        """
        self.notification_handlers.append(handler)
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """
        Check metrics against all rules and trigger alerts
        
        Args:
            metrics: Current metrics snapshot
        """
        now = time.time()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered and (now - rule.last_triggered) < rule.cooldown_seconds:
                continue
            
            # Evaluate condition
            try:
                if rule.condition(metrics):
                    self._trigger_alert(rule, metrics)
                    rule.last_triggered = now
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        # Format message with metrics
        message = rule.message
        try:
            message = message.format(
                error_rate=metrics['api_metrics']['error_rate_percent'],
                response_time=metrics['api_metrics']['avg_response_time_ms'],
                memory=metrics['system_metrics']['memory_usage_mb'],
                cpu=metrics['system_metrics']['cpu_percent']
            )
        except (KeyError, ValueError):
            pass  # Use original message if formatting fails
        
        # Create alert
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            timestamp=time.time(),
            metrics_snapshot=metrics
        )
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
        
        logger.warning(f"Alert triggered: {rule.name} - {message}")
    
    def resolve_alert(self, rule_name: str):
        """Mark an alert as resolved"""
        now = time.time()
        for alert in self.active_alerts:
            if alert.rule_name == rule_name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = now
                logger.info(f"Alert resolved: {rule_name}")
        
        # Remove resolved alerts from active list
        self.active_alerts = [a for a in self.active_alerts if not a.resolved]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [a for a in self.active_alerts if not a.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history
        
        Args:
            hours: Number of hours of history to return
        
        Returns:
            List of alerts from the specified time period
        """
        cutoff = time.time() - (hours * 3600)
        return [a for a in self.alert_history if a.timestamp >= cutoff]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts"""
        active = self.get_active_alerts()
        history_24h = self.get_alert_history(24)
        
        # Count by severity
        severity_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        
        for alert in history_24h:
            severity_counts[alert.severity.value] += 1
        
        return {
            'active_count': len(active),
            'total_24h': len(history_24h),
            'by_severity': severity_counts,
            'active_alerts': [
                {
                    'rule_name': a.rule_name,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp
                }
                for a in active
            ]
        }
    
    def _console_notification(self, alert: Alert):
        """Default notification handler - logs to console"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emoji.get(alert.severity, "")
        print(f"\n{emoji} ALERT [{alert.severity.value.upper()}]: {alert.message}")
        print(f"   Rule: {alert.rule_name}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")


# Example usage
if __name__ == "__main__":
    from technic_v4.monitoring.metrics_collector import MetricsCollector
    
    print("Testing Alert System")
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
    
    print("\n✓ Alert system test complete")
