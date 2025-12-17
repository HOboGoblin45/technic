"""
Metrics Collector for ML API Monitoring
Tracks API performance, model metrics, and system resources
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
from threading import Lock
import statistics


@dataclass
class RequestMetric:
    """Single API request metric"""
    timestamp: float
    endpoint: str
    duration_ms: float
    status_code: int
    error: Optional[str] = None


@dataclass
class ModelMetric:
    """Single model prediction metric"""
    timestamp: float
    model_name: str
    mae: float
    confidence: float
    prediction_count: int = 1


class MetricsCollector:
    """
    Collects and aggregates metrics for the ML API
    
    Features:
    - Request timing and throughput
    - Model performance tracking
    - System resource monitoring
    - Error rate calculation
    - Percentile calculations (P95, P99)
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            history_size: Number of recent metrics to keep in memory
        """
        self.history_size = history_size
        self.start_time = time.time()
        
        # Request metrics
        self.requests: deque = deque(maxlen=history_size)
        self.request_lock = Lock()
        
        # Model metrics
        self.model_predictions: Dict[str, deque] = {
            'result_predictor': deque(maxlen=history_size),
            'duration_predictor': deque(maxlen=history_size)
        }
        self.model_lock = Lock()
        
        # Aggregated counters
        self.total_requests = 0
        self.total_errors = 0
        self.endpoint_counts: Dict[str, int] = {}
        
        # System info
        self.process = psutil.Process()
    
    def record_request(
        self,
        endpoint: str,
        duration_ms: float,
        status_code: int,
        error: Optional[str] = None
    ):
        """
        Record an API request
        
        Args:
            endpoint: API endpoint path
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            error: Error message if request failed
        """
        with self.request_lock:
            metric = RequestMetric(
                timestamp=time.time(),
                endpoint=endpoint,
                duration_ms=duration_ms,
                status_code=status_code,
                error=error
            )
            self.requests.append(metric)
            
            # Update counters
            self.total_requests += 1
            if status_code >= 400:
                self.total_errors += 1
            
            self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
    
    def record_prediction(
        self,
        model_name: str,
        mae: float,
        confidence: float
    ):
        """
        Record a model prediction
        
        Args:
            model_name: Name of the model
            mae: Mean absolute error
            confidence: Prediction confidence score
        """
        with self.model_lock:
            if model_name not in self.model_predictions:
                self.model_predictions[model_name] = deque(maxlen=self.history_size)
            
            metric = ModelMetric(
                timestamp=time.time(),
                model_name=model_name,
                mae=mae,
                confidence=confidence
            )
            self.model_predictions[model_name].append(metric)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot
        
        Returns:
            Dictionary with all current metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            'timestamp': time.time(),
            'uptime_seconds': uptime,
            'api_metrics': self._get_api_metrics(),
            'model_metrics': self._get_model_metrics(),
            'system_metrics': self._get_system_metrics(),
            'endpoint_stats': self._get_endpoint_stats()
        }
    
    def _get_api_metrics(self) -> Dict[str, Any]:
        """Calculate API performance metrics"""
        with self.request_lock:
            if not self.requests:
                return {
                    'requests_per_minute': 0,
                    'avg_response_time_ms': 0,
                    'error_rate_percent': 0,
                    'p95_response_time_ms': 0,
                    'p99_response_time_ms': 0,
                    'total_requests': self.total_requests,
                    'total_errors': self.total_errors
                }
            
            # Calculate time window (last 60 seconds)
            now = time.time()
            recent_requests = [r for r in self.requests if now - r.timestamp <= 60]
            
            if not recent_requests:
                requests_per_minute = 0
                avg_response_time = 0
                error_rate = 0
                p95 = 0
                p99 = 0
            else:
                # Requests per minute
                time_span = max(1, now - recent_requests[0].timestamp)
                requests_per_minute = len(recent_requests) / (time_span / 60)
                
                # Average response time
                response_times = [r.duration_ms for r in recent_requests]
                avg_response_time = statistics.mean(response_times)
                
                # Error rate
                errors = sum(1 for r in recent_requests if r.status_code >= 400)
                error_rate = (errors / len(recent_requests)) * 100
                
                # Percentiles
                sorted_times = sorted(response_times)
                p95_idx = int(len(sorted_times) * 0.95)
                p99_idx = int(len(sorted_times) * 0.99)
                p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
                p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
            
            return {
                'requests_per_minute': round(requests_per_minute, 1),
                'avg_response_time_ms': round(avg_response_time, 1),
                'error_rate_percent': round(error_rate, 2),
                'p95_response_time_ms': round(p95, 1),
                'p99_response_time_ms': round(p99, 1),
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'recent_requests_count': len(recent_requests)
            }
    
    def _get_model_metrics(self) -> Dict[str, Any]:
        """Calculate model performance metrics"""
        with self.model_lock:
            metrics = {}
            
            for model_name, predictions in self.model_predictions.items():
                if not predictions:
                    metrics[model_name] = {
                        'predictions_count': 0,
                        'avg_mae': 0,
                        'avg_confidence': 0,
                        'last_updated': None
                    }
                    continue
                
                # Calculate averages
                maes = [p.mae for p in predictions]
                confidences = [p.confidence for p in predictions]
                
                metrics[model_name] = {
                    'predictions_count': len(predictions),
                    'avg_mae': round(statistics.mean(maes), 2),
                    'avg_confidence': round(statistics.mean(confidences), 2),
                    'min_mae': round(min(maes), 2),
                    'max_mae': round(max(maes), 2),
                    'last_updated': predictions[-1].timestamp
                }
            
            return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            return {
                'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 1),
                'cpu_percent': round(cpu_percent, 1),
                'num_threads': self.process.num_threads(),
                'disk_usage_percent': round(psutil.disk_usage('/').percent, 1)
            }
        except Exception as e:
            return {
                'memory_usage_mb': 0,
                'cpu_percent': 0,
                'num_threads': 0,
                'disk_usage_percent': 0,
                'error': str(e)
            }
    
    def _get_endpoint_stats(self) -> Dict[str, int]:
        """Get request counts by endpoint"""
        with self.request_lock:
            return dict(self.endpoint_counts)
    
    def get_history(
        self,
        minutes: int = 60,
        metric_type: str = 'api'
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics
        
        Args:
            minutes: Number of minutes of history to return
            metric_type: Type of metrics ('api', 'model', 'system')
        
        Returns:
            List of metric snapshots
        """
        cutoff_time = time.time() - (minutes * 60)
        
        if metric_type == 'api':
            with self.request_lock:
                return [
                    asdict(r) for r in self.requests
                    if r.timestamp >= cutoff_time
                ]
        elif metric_type == 'model':
            with self.model_lock:
                history = []
                for model_name, predictions in self.model_predictions.items():
                    history.extend([
                        asdict(p) for p in predictions
                        if p.timestamp >= cutoff_time
                    ])
                return history
        
        return []
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self.request_lock:
            self.requests.clear()
            self.total_requests = 0
            self.total_errors = 0
            self.endpoint_counts.clear()
        
        with self.model_lock:
            for predictions in self.model_predictions.values():
                predictions.clear()
        
        self.start_time = time.time()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Example usage
if __name__ == "__main__":
    # Create collector
    collector = MetricsCollector()
    
    # Simulate some requests
    print("Simulating API requests...")
    for i in range(10):
        collector.record_request(
            endpoint="/scan/predict",
            duration_ms=100 + i * 10,
            status_code=200 if i < 9 else 500
        )
        time.sleep(0.1)
    
    # Simulate model predictions
    print("Simulating model predictions...")
    for i in range(5):
        collector.record_prediction(
            model_name="result_predictor",
            mae=3.9 + i * 0.1,
            confidence=0.68 + i * 0.02
        )
    
    # Get current metrics
    print("\nCurrent Metrics:")
    print("=" * 60)
    metrics = collector.get_current_metrics()
    
    print(f"\nAPI Metrics:")
    for key, value in metrics['api_metrics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Metrics:")
    for model, stats in metrics['model_metrics'].items():
        print(f"  {model}:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    print(f"\nSystem Metrics:")
    for key, value in metrics['system_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\n✓ Metrics collector test complete")
