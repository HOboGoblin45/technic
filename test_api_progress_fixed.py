"""
Test suite for fixed enhanced API with progress tracking.
Tests Phase 3D-C implementation with async handling fixes.
"""

import json
import time
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import threading

# Import the fixed API
from api_enhanced_fixed import (
    app, 
    progress_store, 
    ScanRequest,
    create_progress_callback,
    run_scan_with_progress
)

client = TestClient(app)

class TestProgressTracking:
    """Test progress tracking functionality."""
    
    def setup_method(self):
        """Clear progress store before each test."""
        progress_store._store.clear()
        progress_store._websocket_connections.clear()
        progress_store._pending_notifications.clear()
    
    def test_health_endpoint(self):
        """Test health endpoint shows progress features."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"
        assert data["features"]["progress_tracking"] is True
        assert data["features"]["websocket"] is True
        assert "redis" in data["features"]
    
    def test_start_scan_async(self):
        """Test starting an async scan."""
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_scanner.ScanConfig = Mock(return_value=Mock())
            
            request_data = {
                "max_symbols": 10,
                "min_tech_rating": 15.0,
                "async_mode": True
            }
            
            response = client.post("/scan/start", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "pending"
            assert "progress_url" in data
            assert "websocket_url" in data
            
            # Verify progress entry was created
            scan_id = data["scan_id"]
            progress = progress_store.get(scan_id)
            assert progress is not None
            assert progress.status == "pending"
    
    def test_start_scan_sync(self):
        """Test starting a synchronous scan."""
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_scanner.ScanConfig = Mock(return_value=Mock())
            mock_df = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT'],
                'TechRating': [25.0, 30.0],
                'Signal': ['Long', 'Strong Long']
            })
            mock_scanner.run_scan = Mock(return_value=(mock_df, "Scan complete", {}))
            
            request_data = {
                "max_symbols": 10,
                "min_tech_rating": 15.0,
                "async_mode": False
            }
            
            response = client.post("/scan/start", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "completed"
    
    def test_get_progress(self):
        """Test getting scan progress."""
        # Create a progress entry
        scan_id = "test-scan-123"
        progress = progress_store.create(scan_id)
        progress_store.update(
            scan_id,
            status="running",
            stage="symbol_scanning",
            current=50,
            total=100,
            percentage=50.0,
            message="Scanning symbols...",
            eta_seconds=30.5,
            symbols_per_second=2.5
        )
        
        response = client.get(f"/scan/progress/{scan_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["scan_id"] == scan_id
        assert data["status"] == "running"
        assert data["stage"] == "symbol_scanning"
        assert data["current"] == 50
        assert data["total"] == 100
        assert data["percentage"] == 50.0
        assert data["eta_seconds"] == 30.5
        assert data["symbols_per_second"] == 2.5
    
    def test_get_progress_not_found(self):
        """Test getting progress for non-existent scan."""
        response = client.get("/scan/progress/non-existent-scan")
        assert response.status_code == 404
    
    def test_get_results_completed(self):
        """Test getting results for completed scan."""
        scan_id = "test-scan-456"
        progress = progress_store.create(scan_id)
        
        # Mock completed scan
        results = [
            {'Symbol': 'AAPL', 'TechRating': 25.0, 'Signal': 'Long', 'RewardRisk': 2.5},
            {'Symbol': 'MSFT', 'TechRating': 30.0, 'Signal': 'Strong Long', 'RewardRisk': 3.0}
        ]
        
        progress_store.update(
            scan_id,
            status="completed",
            message="Scan complete",
            results=results,
            metadata={'symbols_scanned': 100, 'symbols_returned': 2}
        )
        progress.completed_at = progress.started_at  # Set for testing
        
        response = client.get(f"/scan/results/{scan_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["scan_id"] == scan_id
        assert data["status"] == "completed"
        assert len(data["results"]) == 2
        assert data["results"][0]["ticker"] == "AAPL"
        assert data["performance_metrics"]["symbols_scanned"] == 100
    
    def test_get_results_not_completed(self):
        """Test getting results for incomplete scan."""
        scan_id = "test-scan-789"
        progress = progress_store.create(scan_id)
        progress_store.update(scan_id, status="running")
        
        response = client.get(f"/scan/results/{scan_id}")
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]
    
    def test_cancel_scan(self):
        """Test cancelling a running scan."""
        scan_id = "test-scan-cancel"
        progress = progress_store.create(scan_id)
        progress_store.update(scan_id, status="running")
        
        response = client.post(f"/scan/cancel/{scan_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "cancelling"
        
        # Verify cancel flag was set
        progress = progress_store.get(scan_id)
        assert progress.cancel_requested is True
    
    def test_cancel_completed_scan(self):
        """Test cancelling an already completed scan."""
        scan_id = "test-scan-done"
        progress = progress_store.create(scan_id)
        progress_store.update(scan_id, status="completed")
        
        response = client.post(f"/scan/cancel/{scan_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "completed"
        assert "cannot cancel" in data["message"]
    
    def test_list_active_scans(self):
        """Test listing active scans."""
        # Create multiple scans
        scan1 = progress_store.create("scan-1")
        progress_store.update("scan-1", status="running", stage="scanning", percentage=50.0)
        
        scan2 = progress_store.create("scan-2")
        progress_store.update("scan-2", status="pending")
        
        scan3 = progress_store.create("scan-3")
        progress_store.update("scan-3", status="completed")  # Should not be included
        
        response = client.get("/scan/active")
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 2
        assert len(data["scans"]) == 2
        
        scan_ids = [s["scan_id"] for s in data["scans"]]
        assert "scan-1" in scan_ids
        assert "scan-2" in scan_ids
        assert "scan-3" not in scan_ids
    
    def test_progress_callback(self):
        """Test progress callback updates store correctly."""
        scan_id = "test-callback"
        progress = progress_store.create(scan_id)
        
        callback = create_progress_callback(scan_id)
        
        # Simulate progress update
        callback(
            stage="symbol_scanning",
            current=25,
            total=100,
            message="Processing AAPL",
            metadata={
                'percentage': 25.0,
                'eta_seconds': 45.0,
                'symbols_per_second': 1.5,
                'symbol': 'AAPL'
            }
        )
        
        # Verify update
        progress = progress_store.get(scan_id)
        assert progress.status == "running"
        assert progress.stage == "symbol_scanning"
        assert progress.current == 25
        assert progress.total == 100
        assert progress.percentage == 25.0
        assert progress.eta_seconds == 45.0
        assert progress.symbols_per_second == 1.5
        assert progress.message == "Processing AAPL"
    
    def test_run_scan_with_progress_success(self):
        """Test successful scan execution with progress."""
        scan_id = "test-execution"
        progress = progress_store.create(scan_id)
        
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_config = Mock()
            mock_df = pd.DataFrame({
                'Symbol': ['AAPL'],
                'TechRating': [25.0]
            })
            mock_scanner.run_scan = Mock(return_value=(
                mock_df, 
                "Scan complete",
                {'symbols_scanned': 100, 'symbols_per_second': 10.0}
            ))
            
            # Run scan
            run_scan_with_progress(scan_id, mock_config)
            
            # Verify completion
            progress = progress_store.get(scan_id)
            assert progress.status == "completed"
            assert progress.message == "Scan complete"
            assert progress.results is not None
            assert len(progress.results) == 1
            assert progress.metadata['symbols_scanned'] == 100
    
    def test_run_scan_with_progress_error(self):
        """Test scan execution with error."""
        scan_id = "test-error"
        progress = progress_store.create(scan_id)
        
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_config = Mock()
            mock_scanner.run_scan = Mock(side_effect=Exception("Test error"))
            
            # Run scan
            run_scan_with_progress(scan_id, mock_config)
            
            # Verify error handling
            progress = progress_store.get(scan_id)
            assert progress.status == "failed"
            assert progress.error == "Test error"
            assert progress.completed_at is not None
    
    def test_run_scan_with_cancellation(self):
        """Test scan cancellation during execution."""
        scan_id = "test-cancel-during"
        progress = progress_store.create(scan_id)
        progress.cancel_requested = True
        
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_config = Mock()
            
            # Run scan
            run_scan_with_progress(scan_id, mock_config)
            
            # Verify cancellation
            progress = progress_store.get(scan_id)
            assert progress.status == "cancelled"
            assert "cancelled by user" in progress.message.lower()
    
    def test_legacy_scan_endpoint(self):
        """Test backward-compatible /scan endpoint."""
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            mock_scanner.ScanConfig = Mock(return_value=Mock())
            mock_df = pd.DataFrame({
                'Symbol': ['AAPL'],
                'TechRating': [25.0],
                'Signal': ['Long'],
                'RewardRisk': [2.5]
            })
            mock_scanner.run_scan = Mock(return_value=(mock_df, "Complete", {}))
            
            response = client.get("/scan?max_symbols=10&min_tech_rating=15.0")
            assert response.status_code == 200
            
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["ticker"] == "AAPL"

class TestPerformance:
    """Test performance aspects of progress tracking."""
    
    def setup_method(self):
        """Clear progress store before each test."""
        progress_store._store.clear()
        progress_store._pending_notifications.clear()
    
    def test_progress_callback_performance(self):
        """Test that progress callbacks are fast."""
        scan_id = "test-perf"
        progress = progress_store.create(scan_id)
        callback = create_progress_callback(scan_id)
        
        # Measure time for 1000 callbacks
        start = time.time()
        for i in range(1000):
            callback(
                stage="scanning",
                current=i,
                total=1000,
                message=f"Symbol {i}",
                metadata={'percentage': i/10.0}
            )
        elapsed = time.time() - start
        
        # Should be very fast (< 100ms for 1000 calls)
        assert elapsed < 0.5, f"Progress callbacks too slow: {elapsed:.3f}s for 1000 calls"
        
        # Verify last update
        progress = progress_store.get(scan_id)
        assert progress.current == 999
        assert progress.percentage == 99.9
    
    def test_concurrent_scans(self):
        """Test multiple concurrent scans."""
        scan_ids = []
        
        # Create 10 concurrent scans
        for i in range(10):
            scan_id = f"concurrent-{i}"
            progress = progress_store.create(scan_id)
            progress_store.update(scan_id, status="running", percentage=i*10)
            scan_ids.append(scan_id)
        
        # Verify all scans are tracked
        active = progress_store.list_active()
        assert len(active) == 10
        
        # Update all scans
        for i, scan_id in enumerate(scan_ids):
            progress_store.update(scan_id, percentage=(i+1)*10)
        
        # Verify updates
        for i, scan_id in enumerate(scan_ids):
            progress = progress_store.get(scan_id)
            assert progress.percentage == (i+1)*10

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def setup_method(self):
        """Clear progress store before each test."""
        progress_store._store.clear()
    
    def test_full_scan_workflow(self):
        """Test complete scan workflow from start to results."""
        with patch('api_enhanced_fixed.scanner_core') as mock_scanner:
            # Setup mocks
            mock_scanner.ScanConfig = Mock(return_value=Mock())
            mock_df = pd.DataFrame({
                'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
                'TechRating': [85.0, 78.0, 82.0],
                'Signal': ['Strong Long', 'Long', 'Strong Long'],
                'RewardRisk': [3.5, 2.8, 3.2],
                'Entry': [150.0, 2800.0, 300.0],
                'Stop': [145.0, 2750.0, 290.0],
                'Target': [165.0, 2950.0, 325.0]
            })
            mock_scanner.run_scan = Mock(return_value=(
                mock_df, 
                "Scan completed successfully",
                {
                    'symbols_scanned': 500,
                    'symbols_returned': 3,
                    'symbols_per_second': 25.0,
                    'cache_hit_rate': 0.75
                }
            ))
            
            # 1. Start scan
            start_response = client.post("/scan/start", json={
                "max_symbols": 500,
                "min_tech_rating": 70.0,
                "async_mode": False
            })
            assert start_response.status_code == 200
            scan_id = start_response.json()["scan_id"]
            
            # 2. Check progress
            progress_response = client.get(f"/scan/progress/{scan_id}")
            assert progress_response.status_code == 200
            progress_data = progress_response.json()
            assert progress_data["status"] == "completed"
            
            # 3. Get results
            results_response = client.get(f"/scan/results/{scan_id}")
            assert results_response.status_code == 200
            results_data = results_response.json()
            
            # Verify results
            assert len(results_data["results"]) == 3
            assert results_data["results"][0]["ticker"] == "AAPL"
            assert results_data["results"][0]["techRating"] == 85.0
            assert results_data["performance_metrics"]["symbols_scanned"] == 500
            assert results_data["performance_metrics"]["cache_hit_rate"] == 0.75

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
