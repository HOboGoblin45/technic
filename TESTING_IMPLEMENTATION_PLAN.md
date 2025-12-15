# Technic Testing Implementation Plan

**Goal**: Achieve 80%+ test coverage and production-ready quality assurance

**Timeline**: 2-3 weeks  
**Priority**: 🔴 CRITICAL - Blocking launch

---

## Phase 1: Unit Testing (Week 1)

### 1.1 Core Engine Tests (Days 1-3)

**Target Modules**:
- `technic_v4/engine/scoring.py` - TechRating calculation
- `technic_v4/engine/merit_engine.py` - MERIT score
- `technic_v4/engine/trade_planner.py` - Entry/Stop/Target
- `technic_v4/engine/factor_engine.py` - Factor computation
- `technic_v4/engine/portfolio_engine.py` - Risk-adjusted ranking

**Test Coverage Goals**:
- Scoring: 90%+ (critical business logic)
- MERIT: 90%+ (proprietary algorithm)
- Trade Planning: 85%+ (risk management)
- Factors: 80%+
- Portfolio: 80%+

### 1.2 Data Layer Tests (Days 4-5)

**Target Modules**:
- `technic_v4/data_engine.py` - Caching and data retrieval
- `technic_v4/data_layer/polygon_client.py` - API integration
- `technic_v4/data_layer/fundamentals.py` - Fundamental data

**Test Coverage Goals**:
- Data Engine: 85%+
- API Client: 80%+
- Fundamentals: 75%+

### 1.3 Scanner Core Tests (Days 6-7)

**Target Modules**:
- `technic_v4/scanner_core.py` - Main scanning logic
- Universe filtering
- Result validation

**Test Coverage Goals**:
- Scanner Core: 80%+

---

## Phase 2: Integration Testing (Week 2, Days 1-3)

### 2.1 API Endpoint Tests

**Endpoints to Test**:
- `GET /health` - Health check
- `GET /version` - Version info
- `POST /v1/scan` - Full scan with various configs
- `POST /v1/copilot` - Copilot responses
- `GET /v1/symbol/{ticker}` - Symbol details

**Test Scenarios**:
- Valid requests with expected responses
- Invalid requests with proper error handling
- Edge cases (empty universe, missing data)
- Authentication (valid/invalid API keys)
- Rate limiting
- Concurrent requests

### 2.2 Data Pipeline Tests

**Test Scenarios**:
- End-to-end data flow (API → Cache → Engine → Results)
- Cache hit/miss scenarios
- Data validation and error handling
- Missing data graceful degradation

---

## Phase 3: End-to-End Testing (Week 2, Days 4-5)

### 3.1 Critical User Flows

**Flows to Test**:
1. **First Scan Flow**
   - Open app → Run scan → View results → Tap result → View details
2. **Filter Flow**
   - Open filters → Select sector → Apply → Run scan → Verify filtered results
3. **Copilot Flow**
   - Tap "Ask Copilot" → Send question → Receive response
4. **Ideas Flow**
   - Run scan → Navigate to Ideas → View ideas → Save to watchlist
5. **Settings Flow**
   - Open settings → Change preferences → Verify persistence

### 3.2 Browser Testing (Streamlit UI)

**Test Scenarios**:
- Launch Streamlit app
- Run scan with various configurations
- Verify results display correctly
- Test Copilot integration
- Test export functionality

---

## Phase 4: Performance & Load Testing (Week 3, Days 1-2)

### 4.1 Performance Benchmarks

**Metrics to Measure**:
- Scan time (target: <90s for full universe)
- API response time (target: <2s)
- Memory usage (target: <4GB)
- CPU usage (target: <80%)

### 4.2 Load Testing

**Test Scenarios**:
- 10 concurrent users
- 50 concurrent users
- 100 concurrent users
- 500 concurrent users (target capacity)

**Tools**: Locust or Apache JMeter

---

## Phase 5: Security Testing (Week 3, Days 3-4)

### 5.1 Security Audit

**Tests**:
- API authentication bypass attempts
- SQL injection (if database used)
- XSS attacks
- API key exposure
- Rate limiting bypass
- Input validation

### 5.2 Dependency Audit

**Tools**:
- `pip-audit` for Python dependencies
- `npm audit` for Node dependencies (if any)
- Check for known vulnerabilities

---

## Phase 6: Regression Testing (Week 3, Day 5)

### 6.1 Automated Regression Suite

**Setup**:
- CI/CD integration (GitHub Actions)
- Run tests on every commit
- Block merges if tests fail

---

## Implementation Strategy

### Step 1: Setup Testing Infrastructure (Today)

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock
pip install locust  # For load testing
pip install pip-audit  # For security

# Create test directory structure
mkdir -p tests/{unit,integration,e2e,performance}
mkdir -p tests/unit/{engine,data_layer,scanner}
```

### Step 2: Write First Test Suite (Today)

**Start with highest-value tests**:
1. Scoring engine (most critical business logic)
2. MERIT engine (proprietary algorithm)
3. Trade planner (risk management)

### Step 3: Measure Coverage (Today)

```bash
# Run tests with coverage
pytest tests/ --cov=technic_v4 --cov-report=html --cov-report=term

# Target: 80%+ overall coverage
```

### Step 4: Iterate and Improve (This Week)

- Write tests for uncovered code
- Fix failing tests
- Refactor code for testability
- Document test patterns

---

## Success Criteria

### Week 1 Completion:
- ✅ 80%+ unit test coverage for core engine
- ✅ All unit tests passing
- ✅ Test infrastructure setup complete

### Week 2 Completion:
- ✅ All integration tests passing
- ✅ All E2E tests passing
- ✅ API endpoints fully tested

### Week 3 Completion:
- ✅ Load tested (500+ users)
- ✅ Security audit complete
- ✅ CI/CD pipeline active
- ✅ Regression suite automated

---

## Test Examples

### Example 1: Scoring Engine Test

```python
# tests/unit/engine/test_scoring.py
import pytest
import pandas as pd
from technic_v4.engine.scoring import compute_scores

def test_tech_rating_calculation():
    """Test TechRating is calculated correctly"""
    # Create sample data
    df = pd.DataFrame({
        'Close': [100, 101, 102, 103, 104],
        'Volume': [1000000] * 5,
        'High': [101, 102, 103, 104, 105],
        'Low': [99, 100, 101, 102, 103],
    })
    
    # Compute scores
    result = compute_scores(df)
    
    # Assertions
    assert 'TechRating' in result.columns
    assert result['TechRating'].iloc[-1] >= 0
    assert result['TechRating'].iloc[-1] <= 100
    assert not result['TechRating'].isna().any()

def test_risk_adjustment():
    """Test risk adjustment reduces score for volatile stocks"""
    # Test implementation
    pass
```

### Example 2: API Integration Test

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from technic_v4.api_server import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_scan_endpoint():
    """Test scan endpoint with valid request"""
    response = client.post("/v1/scan", json={
        "max_symbols": 10,
        "min_tech_rating": 0.0,
        "trade_style": "Short-term swing"
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 10
```

### Example 3: E2E Browser Test

```python
# tests/e2e/test_streamlit_ui.py
import pytest
from playwright.sync_api import Page, expect

def test_scan_flow(page: Page):
    """Test complete scan flow in Streamlit UI"""
    # Navigate to app
    page.goto("http://localhost:8501")
    
    # Wait for app to load
    page.wait_for_selector("text=Technic")
    
    # Click scan button
    page.click("button:has-text('Run Scan')")
    
    # Wait for results
    page.wait_for_selector("text=Results", timeout=120000)
    
    # Verify results displayed
    results = page.locator(".scan-result-card")
    expect(results).to_have_count_greater_than(0)
```

---

## Next Steps

1. **Today**: Setup testing infrastructure and write first test suite
2. **This Week**: Achieve 80%+ unit test coverage
3. **Next Week**: Complete integration and E2E tests
4. **Week 3**: Performance, load, and security testing

---

## Resources

- **pytest docs**: https://docs.pytest.org/
- **FastAPI testing**: https://fastapi.tiangolo.com/tutorial/testing/
- **Playwright**: https://playwright.dev/python/
- **Locust**: https://locust.io/

---

**Status**: Ready to begin implementation  
**First Task**: Setup testing infrastructure and write scoring engine tests
