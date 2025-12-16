# Phase 3E Integration & Production Deployment Plan

## Overview

Integrate Phase 3E-A (Smart Prioritization) and Phase 3E-B (Incremental Streaming) into the main scanner and deploy to production.

## Current Status

### Completed Components
- ✅ Phase 3E-A: Smart Symbol Prioritization (57.4% improvement)
- ✅ Phase 3E-B: Incremental Results Streaming (99% faster first result)
- ✅ All tests passing (11/12 tests, 92% pass rate)
- ✅ Streaming API server functional
- ✅ Performance benchmarks validated

### Components to Integrate
1. `technic_v4/symbol_scorer.py` - Multi-factor scoring
2. `technic_v4/prioritizer.py` - Priority queue system
3. `technic_v4/result_streamer.py` - Streaming infrastructure
4. `api_streaming.py` - Streaming API endpoints

## Integration Strategy

### Phase 1: Scanner Core Integration (Day 1-2)

#### Task 1.1: Add Streaming Mode to Scanner
Modify `scanner_core.py` to support streaming mode:

```python
# Add to scanner_core.py
from technic_v4.result_streamer import get_stream_manager, ScanResult

def run_scan_streaming(
    config: Optional[ScanConfig] = None,
    scan_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """
    Run scan in streaming mode
    Returns scan_id for result retrieval
    """
    # Initialize stream
    stream_manager = get_stream_manager()
    scan_id = scan_id or f"scan_{uuid.uuid4().hex[:12]}"
    
    # Create result queue
    queue = stream_manager.create_stream(scan_id, total_symbols)
    
    # Scan with result emission
    for symbol_result in scan_symbols():
        # Convert to ScanResult
        result = ScanResult(
            symbol=symbol_result['symbol'],
            signal=symbol_result.get('signal'),
            tech_rating=symbol_result.get('tech_rating'),
            # ... other fields
        )
        
        # Add to stream
        should_continue = stream_manager.add_result(scan_id, result)
        if not should_continue:
            break  # Early termination
    
    stream_manager.complete_stream(scan_id)
    return scan_id
```

#### Task 1.2: Add Prioritization to Scanner
Integrate smart prioritization:

```python
# Add to scanner_core.py
from technic_v4.prioritizer import SmartSymbolPrioritizer

def run_scan_prioritized(
    config: Optional[ScanConfig] = None,
    enable_streaming: bool = True,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Run scan with smart prioritization
    """
    # Initialize prioritizer
    prioritizer = SmartSymbolPrioritizer(
        enable_diversity=True,
        enable_learning=True
    )
    
    # Prioritize symbols
    prioritizer.prioritize_symbols(
        universe_symbols,
        market_data=market_data,
        fundamental_data=fundamental_data
    )
    
    # Process in priority order
    while prioritizer.queue.get_remaining_count() > 0:
        batch = prioritizer.get_next_batch(batch_size=10)
        
        for item in batch:
            result = scan_symbol(item['symbol'])
            
            # Update prioritizer with result
            prioritizer.update_with_result(
                item['symbol'],
                result.has_signal,
                result.tech_rating,
                result.alpha_score
            )
            
            # Stream if enabled
            if enable_streaming:
                emit_result(result)
    
    return results_df, status_text, metrics
```

### Phase 2: API Integration (Day 2-3)

#### Task 2.1: Unified API Endpoint
Create single endpoint that combines all features:

```python
# api_production.py
@app.post("/scan/start")
async def start_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    enable_prioritization: bool = True,
    enable_streaming: bool = True,
    termination_criteria: Optional[Dict] = None
):
    """
    Unified scan endpoint with all Phase 3E features
    """
    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    
    # Start scan in background
    background_tasks.add_task(
        run_integrated_scan,
        scan_id=scan_id,
        config=request,
        enable_prioritization=enable_prioritization,
        enable_streaming=enable_streaming,
        termination_criteria=termination_criteria
    )
    
    return {
        "scan_id": scan_id,
        "websocket_url": f"ws://{host}/ws/results/{scan_id}",
        "stats_url": f"http://{host}/scan/stats/{scan_id}",
        "features": {
            "prioritization": enable_prioritization,
            "streaming": enable_streaming,
            "early_termination": termination_criteria is not None
        }
    }
```

#### Task 2.2: Backward Compatibility
Maintain existing `/scan` endpoint:

```python
@app.post("/scan")
async def legacy_scan(request: ScanRequest):
    """
    Legacy batch-mode scan (backward compatible)
    """
    # Run traditional batch scan
    results_df, status, metrics = run_scan(config)
    return {
        "results": results_df.to_dict('records'),
        "status": status,
        "metrics": metrics
    }
```

### Phase 3: Testing & Validation (Day 3-4)

#### Task 3.1: Integration Tests
```python
# test_phase3e_integration.py
def test_prioritized_streaming_scan():
    """Test full integration of prioritization + streaming"""
    # Start scan with both features
    response = requests.post('/scan/start', json={
        'max_symbols': 50,
        'enable_prioritization': True,
        'enable_streaming': True,
        'termination_criteria': {'max_signals': 5}
    })
    
    scan_id = response.json()['scan_id']
    
    # Connect to WebSocket
    results = []
    async with websockets.connect(f'ws://localhost:8001/ws/results/{scan_id}') as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data['type'] == 'result':
                results.append(data['data'])
            elif data['type'] == 'complete':
                break
    
    # Verify prioritization worked
    assert results[0]['priority_tier'] == 'high'
    
    # Verify streaming worked
    assert len(results) > 0
    assert len(results) <= 50  # Early termination
```

#### Task 3.2: Load Testing
```python
# test_load.py
async def test_concurrent_scans(num_scans=50):
    """Test 50 concurrent streaming scans"""
    tasks = []
    for i in range(num_scans):
        task = asyncio.create_task(start_and_monitor_scan(i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Verify all completed successfully
    assert all(r['status'] == 'complete' for r in results)
```

### Phase 4: Production Deployment (Day 4-5)

#### Task 4.1: Environment Configuration
```bash
# .env.production
ENABLE_PRIORITIZATION=true
ENABLE_STREAMING=true
MAX_CONCURRENT_STREAMS=100
STREAM_TIMEOUT_SECONDS=300
EARLY_TERMINATION_DEFAULT=true
```

#### Task 4.2: Docker Configuration
```dockerfile
# Dockerfile updates
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Phase 3E components
COPY technic_v4/symbol_scorer.py technic_v4/
COPY technic_v4/prioritizer.py technic_v4/
COPY technic_v4/result_streamer.py technic_v4/
COPY api_streaming.py .

# Expose ports
EXPOSE 8000 8001

# Start both APIs
CMD ["sh", "-c", "python api_production.py & python api_streaming.py"]
```

#### Task 4.3: Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: technic-scanner-phase3e
spec:
  replicas: 3
  selector:
    matchLabels:
      app: technic-scanner
  template:
    metadata:
      labels:
        app: technic-scanner
        version: phase3e
    spec:
      containers:
      - name: scanner
        image: technic-scanner:phase3e
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: streaming
        env:
        - name: ENABLE_PRIORITIZATION
          value: "true"
        - name: ENABLE_STREAMING
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Deployment Checklist

### Pre-Deployment
- [ ] All integration tests passing
- [ ] Load testing completed (50+ concurrent scans)
- [ ] Memory leak testing (24-hour run)
- [ ] Documentation updated
- [ ] Monitoring dashboards configured

### Deployment Steps
1. [ ] Deploy to staging environment
2. [ ] Run smoke tests
3. [ ] Monitor for 24 hours
4. [ ] Gradual rollout (10% → 50% → 100%)
5. [ ] Monitor metrics and alerts

### Post-Deployment
- [ ] Verify all features working
- [ ] Check performance metrics
- [ ] Monitor error rates
- [ ] Collect user feedback

## Monitoring & Metrics

### Key Metrics to Track
```python
# Metrics to monitor
metrics = {
    'scan_performance': {
        'time_to_first_result': 'avg, p50, p95, p99',
        'total_scan_time': 'avg, p50, p95, p99',
        'symbols_per_second': 'avg',
        'early_termination_rate': 'percentage'
    },
    'prioritization': {
        'high_priority_signal_rate': 'percentage',
        'priority_accuracy': 'percentage',
        'reorder_frequency': 'count'
    },
    'streaming': {
        'active_streams': 'gauge',
        'stream_latency': 'avg, p95',
        'websocket_errors': 'count',
        'reconnection_rate': 'percentage'
    },
    'resources': {
        'memory_usage': 'gauge',
        'cpu_usage': 'gauge',
        'active_connections': 'gauge'
    }
}
```

### Alerts
```yaml
# alerts.yaml
alerts:
  - name: HighStreamLatency
    condition: stream_latency_p95 > 200ms
    severity: warning
    
  - name: LowPriorityAccuracy
    condition: priority_accuracy < 50%
    severity: warning
    
  - name: HighMemoryUsage
    condition: memory_usage > 80%
    severity: critical
    
  - name: StreamConnectionFailures
    condition: websocket_errors > 10/min
    severity: critical
```

## Rollback Plan

If issues arise:
1. Feature flags to disable prioritization/streaming
2. Revert to previous API version
3. Maintain backward compatibility with `/scan` endpoint

```python
# Feature flags
if not settings.ENABLE_PRIORITIZATION:
    # Use standard scanning
    return run_scan_legacy(config)

if not settings.ENABLE_STREAMING:
    # Return batch results
    return batch_results
```

## Success Criteria

### Performance
- [ ] Time to first result < 2s (target: <0.1s)
- [ ] 95th percentile scan time < 15s
- [ ] Support 100+ concurrent streams
- [ ] Memory usage stable over 24 hours

### Functionality
- [ ] Prioritization accuracy > 50%
- [ ] Streaming latency < 100ms
- [ ] Early termination saves > 20% resources
- [ ] Zero data loss in streaming

### Reliability
- [ ] System uptime > 99.9%
- [ ] Error rate < 0.1%
- [ ] Successful reconnection rate > 95%

## Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1 | Scanner integration | Modified scanner_core.py |
| 2 | API integration | Unified API endpoint |
| 3 | Testing | Integration tests passing |
| 4 | Staging deployment | Staging environment live |
| 5 | Production deployment | Production rollout complete |

## Next Steps

After successful deployment:
1. Monitor metrics for 1 week
2. Collect user feedback
3. Optimize based on real-world usage
4. Plan Phase 3E-C (ML optimization) or Phase 4 (Distributed architecture)
