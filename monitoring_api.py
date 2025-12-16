"""
Monitoring API for ML System
Provides REST endpoints for metrics, alerts, and system health
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import asyncio
import time
import uvicorn

from technic_v4.monitoring import (
    MetricsCollector,
    get_metrics_collector,
    AlertSystem,
    AlertRule,
    AlertSeverity
)

# Initialize FastAPI app
app = FastAPI(
    title="ML Monitoring API",
    description="Real-time monitoring for ML-powered scan optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize monitoring components
metrics_collector = get_metrics_collector()
alert_system = AlertSystem()

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    print("=" * 60)
    print("ML Monitoring API")
    print("=" * 60)
    print("\nStarting monitoring service...")
    print("API Documentation: http://localhost:8003/docs")
    print("Health Check: http://localhost:8003/health")
    print("\nEndpoints:")
    print("  GET  /metrics/current - Current metrics snapshot")
    print("  GET  /metrics/history - Historical metrics")
    print("  GET  /metrics/summary - Aggregated summary")
    print("  GET  /alerts/active - Active alerts")
    print("  GET  /alerts/history - Alert history")
    print("  POST /alerts/configure - Configure alert rules")
    print("  GET  /health - Health check")
    print("  WS   /ws/metrics - WebSocket for real-time updates")
    print("=" * 60)


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns system status and uptime
    """
    metrics = metrics_collector.get_current_metrics()
    
    return {
        "status": "healthy",
        "uptime_seconds": metrics['uptime_seconds'],
        "monitoring": {
            "metrics_collector": "operational",
            "alert_system": "operational"
        },
        "stats": {
            "total_requests": metrics['api_metrics']['total_requests'],
            "active_alerts": len(alert_system.get_active_alerts())
        }
    }


@app.get("/metrics/current")
async def get_current_metrics():
    """
    Get current metrics snapshot
    
    Returns:
        Current metrics including API, model, and system metrics
    """
    try:
        metrics = metrics_collector.get_current_metrics()
        
        # Check for alerts
        alert_system.check_metrics(metrics)
        
        return {
            "success": True,
            "timestamp": time.time(),
            "metrics": metrics,
            "active_alerts": len(alert_system.get_active_alerts())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/history")
async def get_metrics_history(
    minutes: int = 60,
    metric_type: str = "api"
):
    """
    Get historical metrics
    
    Args:
        minutes: Number of minutes of history (default: 60)
        metric_type: Type of metrics ('api', 'model', 'system')
    
    Returns:
        Historical metrics for the specified time period
    """
    try:
        if metric_type not in ['api', 'model', 'system']:
            raise HTTPException(
                status_code=400,
                detail="metric_type must be 'api', 'model', or 'system'"
            )
        
        history = metrics_collector.get_history(minutes=minutes, metric_type=metric_type)
        
        return {
            "success": True,
            "metric_type": metric_type,
            "minutes": minutes,
            "count": len(history),
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get aggregated metrics summary
    
    Returns:
        Summary statistics and key metrics
    """
    try:
        metrics = metrics_collector.get_current_metrics()
        alert_summary = alert_system.get_alert_summary()
        
        return {
            "success": True,
            "timestamp": time.time(),
            "summary": {
                "uptime_seconds": metrics['uptime_seconds'],
                "api": {
                    "total_requests": metrics['api_metrics']['total_requests'],
                    "requests_per_minute": metrics['api_metrics']['requests_per_minute'],
                    "avg_response_time_ms": metrics['api_metrics']['avg_response_time_ms'],
                    "error_rate_percent": metrics['api_metrics']['error_rate_percent'],
                    "p95_response_time_ms": metrics['api_metrics']['p95_response_time_ms']
                },
                "models": {
                    name: {
                        "predictions": stats['predictions_count'],
                        "avg_mae": stats['avg_mae'],
                        "avg_confidence": stats['avg_confidence']
                    }
                    for name, stats in metrics['model_metrics'].items()
                },
                "system": metrics['system_metrics'],
                "alerts": alert_summary
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/active")
async def get_active_alerts():
    """
    Get all active alerts
    
    Returns:
        List of currently active alerts
    """
    try:
        active_alerts = alert_system.get_active_alerts()
        
        return {
            "success": True,
            "count": len(active_alerts),
            "alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "age_seconds": time.time() - alert.timestamp
                }
                for alert in active_alerts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/history")
async def get_alert_history(hours: int = 24):
    """
    Get alert history
    
    Args:
        hours: Number of hours of history (default: 24)
    
    Returns:
        Historical alerts for the specified time period
    """
    try:
        history = alert_system.get_alert_history(hours=hours)
        
        return {
            "success": True,
            "hours": hours,
            "count": len(history),
            "alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at
                }
                for alert in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/summary")
async def get_alert_summary():
    """
    Get alert summary statistics
    
    Returns:
        Summary of alerts by severity and status
    """
    try:
        summary = alert_system.get_alert_summary()
        
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/configure")
async def configure_alert(
    rule_name: str,
    enabled: Optional[bool] = None,
    cooldown_seconds: Optional[int] = None
):
    """
    Configure an alert rule
    
    Args:
        rule_name: Name of the alert rule
        enabled: Enable or disable the rule
        cooldown_seconds: Cooldown period in seconds
    
    Returns:
        Updated rule configuration
    """
    try:
        # Find the rule
        rule = next((r for r in alert_system.rules if r.name == rule_name), None)
        
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_name}' not found")
        
        # Update configuration
        if enabled is not None:
            rule.enabled = enabled
        
        if cooldown_seconds is not None:
            rule.cooldown_seconds = cooldown_seconds
        
        return {
            "success": True,
            "rule": {
                "name": rule.name,
                "enabled": rule.enabled,
                "severity": rule.severity.value,
                "cooldown_seconds": rule.cooldown_seconds
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/resolve/{rule_name}")
async def resolve_alert(rule_name: str):
    """
    Manually resolve an alert
    
    Args:
        rule_name: Name of the alert rule to resolve
    
    Returns:
        Success status
    """
    try:
        alert_system.resolve_alert(rule_name)
        
        return {
            "success": True,
            "message": f"Alert '{rule_name}' resolved"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics updates
    
    Sends metrics every 5 seconds
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get current metrics
            metrics = metrics_collector.get_current_metrics()
            alert_summary = alert_system.get_alert_summary()
            
            # Check for alerts
            alert_system.check_metrics(metrics)
            
            # Send update
            await websocket.send_json({
                "type": "metrics_update",
                "timestamp": time.time(),
                "metrics": metrics,
                "alerts": alert_summary
            })
            
            # Wait 5 seconds
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ML Monitoring API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "metrics": [
                "/metrics/current",
                "/metrics/history",
                "/metrics/summary"
            ],
            "alerts": [
                "/alerts/active",
                "/alerts/history",
                "/alerts/summary",
                "/alerts/configure",
                "/alerts/resolve/{rule_name}"
            ],
            "realtime": [
                "/ws/metrics"
            ],
            "health": [
                "/health"
            ]
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "monitoring_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
