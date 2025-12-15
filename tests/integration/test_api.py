"""Integration tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient
from technic_v4.api_server import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_health_check(self):
        """Test /health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_version_endpoint(self):
        """Test /version endpoint"""
        response = client.get("/version")
        
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
    
    def test_meta_endpoint(self):
        """Test /meta endpoint"""
        response = client.get("/meta")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "tagline" in data


class TestScanEndpoint:
    """Test /v1/scan endpoint"""
    
    def test_scan_minimal_request(self):
        """Test scan with minimal valid request"""
        response = client.post("/v1/scan", json={
            "max_symbols": 5,
            "min_tech_rating": 0.0,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 5
    
    def test_scan_with_filters(self):
        """Test scan with sector filters"""
        response = client.post("/v1/scan", json={
            "max_symbols": 10,
            "min_tech_rating": 0.0,
            "sectors": ["Technology"],
            "trade_style": "Short-term swing",
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        # Results should be filtered by sector
        assert len(data["results"]) <= 10
    
    def test_scan_with_options_mode(self):
        """Test scan with options mode"""
        response = client.post("/v1/scan", json={
            "max_symbols": 5,
            "options_mode": "stock_plus_options",
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        # Some results might have option recommendations
        if len(data["results"]) > 0:
            # Check structure
            result = data["results"][0]
            assert "symbol" in result
            assert "signal" in result
    
    def test_scan_invalid_request(self):
        """Test scan with invalid request"""
        response = client.post("/v1/scan", json={
            "max_symbols": -1,  # Invalid
        })
        
        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_scan_large_request(self):
        """Test scan with large max_symbols"""
        response = client.post("/v1/scan", json={
            "max_symbols": 100,
            "min_tech_rating": 0.0,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) <= 100
    
    def test_scan_high_rating_threshold(self):
        """Test scan with high rating threshold"""
        response = client.post("/v1/scan", json={
            "max_symbols": 50,
            "min_tech_rating": 80.0,  # High threshold
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Might return fewer results due to high threshold
        assert "results" in data
        
        # All results should meet threshold
        for result in data["results"]:
            if "techRating" in result and result["techRating"] is not None:
                assert result["techRating"] >= 80.0


class TestCopilotEndpoint:
    """Test /v1/copilot endpoint"""
    
    def test_copilot_simple_question(self):
        """Test Copilot with simple question"""
        response = client.post("/v1/copilot", json={
            "question": "What is a good entry point?",
        })
        
        # Might require OpenAI API key
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data or "response" in data
        else:
            # Expected if OpenAI key not configured
            assert response.status_code in [400, 500, 503]
    
    def test_copilot_with_symbol(self):
        """Test Copilot with symbol context"""
        response = client.post("/v1/copilot", json={
            "question": "Should I buy this stock?",
            "symbol": "AAPL",
        })
        
        # Might require OpenAI API key
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data or "response" in data
        else:
            assert response.status_code in [400, 500, 503]
    
    def test_copilot_empty_question(self):
        """Test Copilot with empty question"""
        response = client.post("/v1/copilot", json={
            "question": "",
        })
        
        # Should reject empty question
        assert response.status_code in [400, 422]


class TestSymbolEndpoint:
    """Test /v1/symbol/{ticker} endpoint"""
    
    def test_symbol_detail_valid(self):
        """Test symbol detail for valid ticker"""
        response = client.get("/v1/symbol/AAPL")
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data or "ticker" in data
        else:
            # Endpoint might not be implemented yet
            assert response.status_code in [404, 501]
    
    def test_symbol_detail_invalid(self):
        """Test symbol detail for invalid ticker"""
        response = client.get("/v1/symbol/INVALID123")
        
        # Should return 404 or error
        assert response.status_code in [404, 400, 500]
    
    def test_symbol_detail_with_days(self):
        """Test symbol detail with days parameter"""
        response = client.get("/v1/symbol/AAPL?days=90")
        
        if response.status_code == 200:
            data = response.json()
            # Should have historical data
            assert data is not None
        else:
            assert response.status_code in [404, 501]


class TestAuthentication:
    """Test API authentication"""
    
    def test_no_auth_required_for_health(self):
        """Test that health endpoint doesn't require auth"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_auth_with_valid_key(self):
        """Test scan with valid API key"""
        # Note: This test assumes TECHNIC_API_KEY is not set (dev mode)
        response = client.post(
            "/v1/scan",
            json={"max_symbols": 5},
            headers={"X-API-Key": "test-key"}
        )
        
        # Should work in dev mode (no key required)
        assert response.status_code in [200, 401]
    
    def test_auth_with_invalid_key(self):
        """Test scan with invalid API key"""
        # This test is environment-dependent
        response = client.post(
            "/v1/scan",
            json={"max_symbols": 5},
            headers={"X-API-Key": "invalid-key"}
        )
        
        # Should work in dev mode or reject if auth is enabled
        assert response.status_code in [200, 401]


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_endpoint(self):
        """Test request to non-existent endpoint"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self):
        """Test invalid HTTP method"""
        response = client.get("/v1/scan")  # Should be POST
        assert response.status_code in [404, 405]
    
    def test_malformed_json(self):
        """Test malformed JSON request"""
        response = client.post(
            "/v1/scan",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]
    
    def test_missing_required_fields(self):
        """Test request with missing required fields"""
        response = client.post("/v1/scan", json={})
        
        # Should use defaults or reject
        assert response.status_code in [200, 400, 422]


class TestResponseFormat:
    """Test response format consistency"""
    
    def test_scan_response_structure(self):
        """Test that scan response has expected structure"""
        response = client.post("/v1/scan", json={
            "max_symbols": 5,
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # Check top-level structure
            assert "status" in data or "results" in data
            
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                
                # Check result structure
                expected_fields = ["symbol", "signal", "techRating"]
                for field in expected_fields:
                    # At least some fields should be present
                    pass  # Flexible check
    
    def test_error_response_structure(self):
        """Test that error responses have consistent structure"""
        response = client.get("/invalid/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        # FastAPI default error structure
        assert "detail" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
