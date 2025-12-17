# Technic Scanner - User Guide

## Welcome! 🎉

This guide will help you get the most out of the Technic Scanner system with all the new enhancements from Path 1 (Quick Wins & Polish).

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Using the Scanner](#using-the-scanner)
3. [Monitoring Performance](#monitoring-performance)
4. [Understanding Errors](#understanding-errors)
5. [Cache Dashboard](#cache-dashboard)
6. [Tips & Best Practices](#tips--best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Redis (optional, for caching)
- Polygon API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd technic-clean

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export POLYGON_API_KEY="your_api_key_here"
export REDIS_URL="redis://localhost:6379"  # Optional
```

### Quick Start

```bash
# Start the monitoring API
python monitoring_api_optimized.py

# In another terminal, run a scan
python -m technic_v4.scanner_core

# View the cache dashboard
streamlit run cache_dashboard.py
```

---

## Using the Scanner

### Basic Scan

The scanner analyzes stocks and provides recommendations based on technical and fundamental analysis.

**Command:**
```bash
python -m technic_v4.scanner_core
```

**What You'll See:**
1. **Progress Indicator** - Real-time progress with ETA
2. **Stage Information** - Current processing stage
3. **Cache Statistics** - Hit rate and performance
4. **Results** - Top stock recommendations

### Understanding Progress

The scanner goes through 5 stages:

1. **🔄 Initializing (5%)** - Setting up the scan
2. **📊 Loading Universe (5%)** - Loading stock symbols
3. **📡 Fetching Data (20%)** - Getting price data
4. **🔍 Scanning Symbols (65%)** - Analyzing stocks
5. **✅ Finalizing (5%)** - Preparing results

**Progress Display:**
```
🔍 Scanning Symbols
███████████████████████████████░ 75%

Status: Analyzing AAPL...    Progress: 75/100    ETA: 30s

📊 Stage Details ▼
✅ 🔄 Initializing (Complete)
✅ 📊 Loading Universe (Complete)
✅ 📡 Fetching Data (Complete)
🔵 🔍 Scanning Symbols (75.0%)
⚪ ✅ Finalizing (Pending)
```

### Scan Configuration

You can customize scans with various parameters:

```python
from technic_v4.scanner_core import ScanConfig

config = ScanConfig(
    max_symbols=50,           # Limit number of symbols
    sectors=["Technology"],   # Filter by sector
    min_tech_rating=15.0,     # Minimum technical rating
    lookback_days=60,         # Historical data period
    trade_style="swing"       # Trading style
)

results = run_scan(config=config)
```

---

## Monitoring Performance

### Real-Time Monitoring

**Start the monitoring script:**
```bash
python scripts/monitor_performance.py
```

**What it does:**
- Checks performance every 60 seconds
- Logs metrics to CSV
- Detects and alerts on issues
- Generates daily summaries

**Output:**
```
[2024-12-01 10:00:00] Checking performance...
  Cache Hit Rate: 75.3%
  Total Requests: 1,234
  Alerts: 0

[2024-12-01 10:01:00] Checking performance...
  Cache Hit Rate: 76.1%
  Total Requests: 1,289
  Alerts: 0
```

### One-Time Check

```bash
python scripts/monitor_performance.py --once
```

### Custom Interval

```bash
python scripts/monitor_performance.py --interval 30
```

### Viewing Logs

Performance logs are saved to `logs/performance/`:
- `performance_metrics.csv` - Detailed metrics
- `alerts.log` - Alert history
- `daily_summary.json` - Daily summaries

---

## Understanding Errors

### User-Friendly Error Messages

Instead of technical errors, you'll see clear, actionable messages:

**Example - API Rate Limit:**
```
⚠️ API rate limit exceeded

💡 Suggestion: Please wait 60 seconds and try again, or upgrade 
your API plan for higher limits

📋 Affected Symbols: AAPL, MSFT, GOOGL

[🔄 Retry]  [❌ Cancel]
```

### Error Types

1. **API Errors** - Connection, rate limits, timeouts
2. **Cache Errors** - Redis connection issues
3. **Data Errors** - Missing or invalid data
4. **Timeout Errors** - Operations taking too long
5. **Config Errors** - Invalid settings
6. **System Errors** - Unexpected issues

### Automatic Retry

Many errors are automatically retried with exponential backoff:
- Attempt 1: Immediate
- Attempt 2: 1 second delay
- Attempt 3: 2 seconds delay
- Attempt 4: 4 seconds delay

### Manual Retry

Click the **🔄 Retry** button to manually retry failed operations.

---

## Cache Dashboard

### Accessing the Dashboard

```bash
streamlit run cache_dashboard.py
```

Open your browser to `http://localhost:8501`

### Dashboard Sections

#### 1. Overview Metrics
- Cache Hit Rate (target: >70%)
- Total Requests
- Cached Items
- Performance Gain

#### 2. Cache Hit Rate Gauge
- Visual gauge showing current hit rate
- Color-coded zones (red/yellow/green)
- Delta vs target

#### 3. Hits vs Misses
- Pie chart showing distribution
- Percentage breakdown

#### 4. TTL Settings
- Time-to-live for each endpoint
- Bar chart visualization

#### 5. Connection Pool
- Active vs idle connections
- Utilization percentage
- Reuse rate

#### 6. Response Times
- Cached vs uncached comparison
- Speedup calculation

#### 7. Recommendations
- Optimization suggestions
- Priority-based (High/Medium/Good)
- Actionable steps

### Understanding Metrics

**Cache Hit Rate:**
- **>70%** - Excellent (green)
- **50-70%** - Good (yellow)
- **<50%** - Needs improvement (red)

**Response Times:**
- **Cached:** ~45ms (fast)
- **Uncached:** ~2,100ms (baseline)
- **Speedup:** 40-50x improvement

**Connection Pool:**
- **Active:** Currently in use
- **Idle:** Available for reuse
- **Reuse Rate:** >90% is excellent

---

## Tips & Best Practices

### Maximizing Cache Performance

1. **Run scans regularly** - Builds up cache
2. **Use consistent parameters** - Same settings = better cache hits
3. **Monitor hit rates** - Check dashboard daily
4. **Adjust TTL if needed** - Balance freshness vs performance

### Optimizing Scan Speed

1. **Limit symbols** - Start with 50-100 symbols
2. **Use sector filters** - Focus on specific sectors
3. **Shorter lookback** - 30-60 days is usually sufficient
4. **Enable Redis** - 2x speedup with caching

### Handling Errors

1. **Read the suggestion** - Follow the recommended action
2. **Check technical details** - If needed, expand for more info
3. **Use retry button** - For transient errors
4. **Report persistent issues** - Contact support if errors continue

### Performance Monitoring

1. **Run continuous monitoring** - In production
2. **Check logs daily** - Review performance trends
3. **Act on alerts** - Address issues promptly
4. **Review summaries** - Weekly performance review

---

## Troubleshooting

### Scanner is Slow

**Symptoms:** Scan takes >5 minutes

**Solutions:**
1. Check cache hit rate (should be >70%)
2. Reduce number of symbols
3. Verify Redis is running
4. Check internet connection speed

### Low Cache Hit Rate

**Symptoms:** Hit rate <50%

**Solutions:**
1. Run scans more frequently
2. Use consistent scan parameters
3. Increase TTL values
4. Check Redis memory capacity

### Connection Errors

**Symptoms:** "Unable to connect" errors

**Solutions:**
1. Check internet connection
2. Verify API key is valid
3. Check API rate limits
4. Try again in a few moments

### Redis Not Working

**Symptoms:** "Cache unavailable" warnings

**Solutions:**
1. Check Redis is running: `redis-cli ping`
2. Verify REDIS_URL environment variable
3. Check Redis connection: `python test_redis_connection.py`
4. Review Redis logs for errors

### High Memory Usage

**Symptoms:** System running slow, high RAM usage

**Solutions:**
1. Reduce max_symbols parameter
2. Clear Redis cache: `redis-cli FLUSHDB`
3. Restart the application
4. Check for memory leaks in logs

---

## Advanced Features

### Custom Scan Configurations

```python
# Aggressive growth scan
config = ScanConfig(
    max_symbols=100,
    sectors=["Technology", "Healthcare"],
    min_tech_rating=20.0,
    min_alpha_score=0.7,
    trade_style="swing"
)

# Conservative value scan
config = ScanConfig(
    max_symbols=50,
    min_tech_rating=15.0,
    min_fundamental_score=0.6,
    trade_style="position"
)
```

### Programmatic Access

```python
from technic_v4.scanner_core_enhanced import run_scan_enhanced

# With progress callback
def my_progress_callback(stage, current, total, message, metadata):
    print(f"{stage}: {current}/{total} - {message}")

results, status, metrics = run_scan_enhanced(
    config=config,
    progress_cb=my_progress_callback
)

print(f"Found {len(results)} opportunities")
print(f"Scan took {metrics['total_seconds']:.1f} seconds")
print(f"Cache hit rate: {metrics['cache_stats']['hit_rate']:.1f}%")
```

### API Integration

```python
import requests

# Get cache stats
response = requests.get("http://localhost:8003/performance/cache")
cache_stats = response.json()

# Get performance summary
response = requests.get("http://localhost:8003/performance/summary")
perf_summary = response.json()
```

---

## Getting Help

### Documentation

- **User Guide:** `docs/USER_GUIDE.md` (this file)
- **API Docs:** http://localhost:8003/docs
- **Implementation Plans:** `PATH1_QUICK_WINS_IMPLEMENTATION_PLAN.md`

### Support

- **GitHub Issues:** Report bugs and request features
- **Email:** support@example.com
- **Discord:** Join our community

### Useful Commands

```bash
# Check system status
python -c "from technic_v4.cache.redis_cache import redis_cache; print(redis_cache.get_stats())"

# View logs
tail -f logs/performance/performance_metrics.csv

# Clear cache
redis-cli FLUSHDB

# Test API
curl http://localhost:8003/health
```

---

## Changelog

### Version 1.1 (Path 1 Complete)

**New Features:**
- ✅ Multi-stage progress indicators
- ✅ Real-time cache dashboard
- ✅ Enhanced error handling with retry
- ✅ Automated performance monitoring
- ✅ Comprehensive user documentation

**Improvements:**
- 40-50x faster API responses (cached)
- User-friendly error messages
- Automatic retry mechanisms
- Performance tracking and alerts

---

## Quick Reference

### Common Commands

```bash
# Start monitoring API
python monitoring_api_optimized.py

# Run scanner
python -m technic_v4.scanner_core

# View cache dashboard
streamlit run cache_dashboard.py

# Monitor performance
python scripts/monitor_performance.py

# Test components
streamlit run test_loading_indicator.py
streamlit run components/ErrorHandler.py
```

### Key Metrics

| Metric | Target | Good | Needs Work |
|--------|--------|------|------------|
| Cache Hit Rate | >70% | 50-70% | <50% |
| Response Time (cached) | <100ms | <200ms | >200ms |
| Connection Reuse | >90% | >80% | <80% |
| Error Rate | <1% | <5% | >5% |

### Alert Thresholds

- Cache Hit Rate: <50%
- Response Time: >1000ms
- Error Rate: >5%
- Connection Utilization: >90%

---

## Conclusion

You now have all the tools to effectively use the Technic Scanner system! The new enhancements provide:

- **Better visibility** with progress indicators
- **Performance insights** via cache dashboard
- **Easier troubleshooting** with friendly errors
- **Automated monitoring** for peace of mind

Happy scanning! 📈🚀

---

*Last Updated: December 2024*  
*Version: 1.1 (Path 1 Complete)*
