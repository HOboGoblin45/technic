# Phase 2: Monitoring Dashboard - Implementation Plan

## Overview

Build a real-time monitoring dashboard for the ML-powered scan optimization system to track performance, detect issues, and provide actionable insights.

## Timeline: 2-3 Days

### Day 1: Core Infrastructure
- Metrics collection system
- Data storage & aggregation
- Basic API endpoints

### Day 2: Dashboard UI
- Real-time metrics display
- Performance graphs
- Alert system

### Day 3: Polish & Testing
- Advanced features
- Testing & validation
- Documentation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Monitoring Dashboard                    │
│                  (Port 8003)                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Metrics    │  │   Alerts     │  │   Graphs     │ │
│  │   Display    │  │   System     │  │   & Charts   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ Collect Metrics
                          │
┌─────────────────────────────────────────────────────────┐
│                    ML API (Port 8002)                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Metrics Middleware                                 │ │
│  │  - Request timing                                   │ │
│  │  - Model performance                                │ │
│  │  - Error tracking                                   │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Metrics Storage (SQLite/JSON)               │
│  - Time-series data                                      │
│  - Aggregated statistics                                 │
│  - Alert history                                         │
└─────────────────────────────────────────────────────────┘
```

## Phase 2A: Metrics Collection (Day 1, Morning)

### Task 1: Create Metrics Collector

**File:** `technic_v4/monitoring/metrics_collector.py`

**Features:**
- Request timing
- Model performance tracking
- Error rate monitoring
- Resource usage
- Cache statistics

**Metrics to Track:**
```python
{
    "api_metrics": {
        "requests_per_minute": 45,
        "avg_response_time_ms": 145,
        "error_rate_percent": 0.1,
        "p95_response_time_ms": 280,
        "p99_response_time_ms": 450
    },
    "model_metrics": {
        "result_predictor": {
            "predictions_count": 1250,
            "avg_mae": 3.9,
            "avg_confidence": 0.68,
            "last_updated": "2025-12-16T17:30:00"
        },
        "duration_predictor": {
            "predictions_count": 1250,
            "avg_mae": 0.55,
            "avg_confidence": 0.75,
            "last_updated": "2025-12-16T17:30:00"
        }
    },
    "system_metrics": {
        "uptime_seconds": 3600,
        "memory_usage_mb": 256,
        "cpu_percent": 15.5,
        "disk_usage_percent": 45.2
    },
    "cache_metrics": {
        "hit_rate_percent": 78.5,
        "total_hits": 980,
        "total_misses": 270,
        "cache_size_mb": 12.5
    }
}
```

### Task 2: Add Middleware to ML API

**File:** `api_ml_enhanced.py` (modify)

**Add:**
- Request timing middleware
- Metrics collection hooks
- Error tracking
- Performance logging

### Task 3: Create Metrics Storage

**File:** `technic_v4/monitoring/metrics_storage.py`

**Features:**
- Time-series data storage
- Aggregation functions
- Query interface
- Data retention policy

## Phase 2B: Dashboard Backend (Day 1, Afternoon)

### Task 4: Create Monitoring API

**File:** `monitoring_api.py`

**Endpoints:**
```python
GET  /metrics/current          # Current metrics snapshot
GET  /metrics/history          # Historical metrics
GET  /metrics/summary          # Aggregated summary
GET  /alerts/active            # Active alerts
GET  /alerts/history           # Alert history
POST /alerts/configure         # Configure alert rules
GET  /health                   # Dashboard health
```

### Task 5: Implement Alert System

**File:** `technic_v4/monitoring/alerts.py`

**Alert Types:**
- High error rate (> 5%)
- Slow response time (> 500ms)
- Model accuracy degradation (> 20% from baseline)
- Low cache hit rate (< 50%)
- High resource usage (> 80%)

**Alert Channels:**
- Console logging
- File logging
- Email (optional)
- Webhook (optional)

## Phase 2C: Dashboard Frontend (Day 2)

### Task 6: Create Dashboard UI

**File:** `monitoring_dashboard.py`

**Technology:** Streamlit or FastAPI + HTML/JS

**Sections:**

#### 1. Overview Panel
```
┌─────────────────────────────────────────────────┐
│  ML API Monitoring Dashboard                    │
├─────────────────────────────────────────────────┤
│  Status: ● HEALTHY          Uptime: 99.9%      │
│  Requests/min: 45           Errors: 0.1%       │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Response │  │  Model   │  │  Cache   │     │
│  │  145ms   │  │ MAE 3.9  │  │  78.5%   │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

#### 2. Model Performance
```
┌─────────────────────────────────────────────────┐
│  Model Performance                              │
├─────────────────────────────────────────────────┤
│  Result Predictor:                              │
│  ├─ MAE: 3.9 (target < 10) ✓                  │
│  ├─ R²: 0.934                                  │
│  ├─ Predictions: 1,250                         │
│  └─ Confidence: 68%                            │
│                                                  │
│  Duration Predictor:                            │
│  ├─ MAE: 0.55s (target < 5s) ✓                │
│  ├─ R²: 0.981                                  │
│  ├─ Predictions: 1,250                         │
│  └─ Confidence: 75%                            │
└─────────────────────────────────────────────────┘
```

#### 3. API Metrics
```
┌─────────────────────────────────────────────────┐
│  API Performance                                │
├─────────────────────────────────────────────────┤
│  Requests: 2,750 total                          │
│  ├─ Success: 2,747 (99.9%)                     │
│  └─ Errors: 3 (0.1%)                           │
│                                                  │
│  Response Times:                                │
│  ├─ Average: 145ms                             │
│  ├─ P95: 280ms                                 │
│  └─ P99: 450ms                                 │
│                                                  │
│  Throughput: 45 req/min                         │
└─────────────────────────────────────────────────┘
```

#### 4. Performance Graphs
- Request volume over time (line chart)
- Response time distribution (histogram)
- Model accuracy trends (line chart)
- Error rate over time (line chart)
- Cache hit rate (gauge)

#### 5. Active Alerts
```
┌─────────────────────────────────────────────────┐
│  Active Alerts                                  │
├─────────────────────────────────────────────────┤
│  ⚠ No active alerts                            │
│                                                  │
│  Recent Alerts (Last 24h):                      │
│  └─ None                                        │
└─────────────────────────────────────────────────┘
```

### Task 7: Add Real-Time Updates

**Features:**
- Auto-refresh every 5 seconds
- WebSocket for live updates (optional)
- Smooth animations
- Loading states

## Phase 2D: Advanced Features (Day 3)

### Task 8: Historical Analysis

**Features:**
- Time range selection
- Trend analysis
- Comparison views
- Export data (CSV/JSON)

### Task 9: Alert Configuration

**UI for:**
- Setting thresholds
- Enabling/disabling alerts
- Configuring channels
- Testing alerts

### Task 10: Admin Interface

**Features:**
- Model retraining trigger
- Cache management
- System diagnostics
- Configuration updates

## Implementation Steps

### Step 1: Create Monitoring Infrastructure
```bash
# Create monitoring module
mkdir -p technic_v4/monitoring
touch technic_v4/monitoring/__init__.py
touch technic_v4/monitoring/metrics_collector.py
touch technic_v4/monitoring/metrics_storage.py
touch technic_v4/monitoring/alerts.py
```

### Step 2: Implement Metrics Collection
```python
# In metrics_collector.py
class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_request(self, endpoint, duration_ms, status_code):
        # Record API request metrics
        pass
    
    def record_prediction(self, model_name, mae, confidence):
        # Record model prediction metrics
        pass
    
    def get_current_metrics(self):
        # Return current metrics snapshot
        pass
```

### Step 3: Add Middleware to ML API
```python
# In api_ml_enhanced.py
from technic_v4.monitoring import MetricsCollector

metrics = MetricsCollector()

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    metrics.record_request(request.url.path, duration_ms, response.status_code)
    return response
```

### Step 4: Create Monitoring API
```python
# monitoring_api.py
from fastapi import FastAPI
from technic_v4.monitoring import MetricsCollector

app = FastAPI(title="ML Monitoring Dashboard")
metrics = MetricsCollector()

@app.get("/metrics/current")
def get_current_metrics():
    return metrics.get_current_metrics()
```

### Step 5: Build Dashboard UI
```python
# monitoring_dashboard.py
import streamlit as st
import requests
import plotly.graph_objects as go

st.title("ML API Monitoring Dashboard")

# Fetch metrics
metrics = requests.get("http://localhost:8003/metrics/current").json()

# Display overview
col1, col2, col3 = st.columns(3)
col1.metric("Response Time", f"{metrics['api']['avg_response_time_ms']}ms")
col2.metric("Model MAE", f"{metrics['model']['result_predictor']['avg_mae']}")
col3.metric("Cache Hit Rate", f"{metrics['cache']['hit_rate_percent']}%")

# Add graphs
st.line_chart(metrics['history']['response_times'])
```

## Testing Plan

### Unit Tests
- Metrics collection accuracy
- Alert triggering logic
- Data aggregation functions

### Integration Tests
- End-to-end metrics flow
- Dashboard data display
- Alert delivery

### Performance Tests
- Metrics overhead < 1%
- Dashboard load time < 2s
- Real-time update latency < 100ms

## Success Criteria

- ✅ Dashboard loads in < 2 seconds
- ✅ Metrics update every 5 seconds
- ✅ All key metrics displayed
- ✅ Alerts trigger correctly
- ✅ < 1% overhead on ML API
- ✅ Historical data queryable
- ✅ Graphs render smoothly

## Deliverables

1. **Monitoring Module** (`technic_v4/monitoring/`)
   - metrics_collector.py
   - metrics_storage.py
   - alerts.py

2. **Monitoring API** (`monitoring_api.py`)
   - REST endpoints for metrics
   - Alert management

3. **Dashboard UI** (`monitoring_dashboard.py`)
   - Real-time metrics display
   - Performance graphs
   - Alert management

4. **Documentation**
   - Setup guide
   - User manual
   - API reference

5. **Tests** (`tests/monitoring/`)
   - Unit tests
   - Integration tests
   - Performance tests

## Next Steps

Ready to start implementation! Would you like me to:

**A) Start with Day 1 tasks** - Build metrics collection infrastructure
**B) See a detailed code example** - Show how a component will look
**C) Customize the plan** - Adjust features or timeline
**D) Begin implementation** - Start coding right away

Which would you prefer?
