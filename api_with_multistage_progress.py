"""
Enhanced FastAPI with multi-stage progress tracking support.
Integrates the Phase 3D-D multi-stage progress tracking implementation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

# Make sure technic_v4 is importable when running from repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    # Use the enhanced scanner with multi-stage progress tracking
    from technic_v4.scanner_core_enhanced import run_scan_enhanced, ScanConfig
    from technic_v4.engine import trade_planner
    from technic_v4.errors import ErrorType, ScanError
    from technic_v4.progress import ProgressTracker, MultiStageProgressTracker, format_time
except Exception as e:  # pragma: no cover - keep API alive even if imports fail
    print(f"Import error: {e}")
    run_scan_enhanced = None  # type: ignore
    ScanConfig = None  # type: ignore
    trade_planner = None  # type: ignore
    ErrorType = None  # type: ignore
    ScanError = None  # type: ignore
    ProgressTracker = None  # type: ignore
    MultiStageProgressTracker = None  # type: ignore
    format_time = None  # type: ignore

# Optional Redis for distributed progress storage
try:
    from technic_v4.cache.redis_cache import redis_cache
    REDIS_AVAILABLE = redis_cache.available if redis_cache else False
except ImportError:
    redis_cache = None
    REDIS_AVAILABLE = False

app = FastAPI(
    title="Technic API with Multi-Stage Progress", 
    version="0.3.0",
    description="Enhanced API with Phase 3D-D multi-stage progress tracking"
)

# -------------------------------------------------------------------
# Enhanced Progress Storage with Multi-Stage Support
# -------------------------------------------------------------------

class MultiStageScanProgress:
    """Stores multi-stage progress information for a scan."""
    
    def __init__(self, scan_id: str):
        self.scan_id = scan_id
        self.status = "pending"  # pending, running, completed, failed, cancelled
        
        # Multi-stage tracking
        self.current_stage = "initializing"
        self.stages = {
            "universe_loading": {"weight": 0.05, "current": 0, "total": 100, "progress": 0},
            "data_fetching": {"weight": 0.20, "current": 0, "total": 0, "progress": 0},
            "symbol_scanning": {"weight": 0.70, "current": 0, "total": 0, "progress": 0},
            "finalization": {"weight": 0.05, "current": 0, "total": 100, "progress": 0}
        }
        
        # Overall progress
        self.overall_progress = 0.0
        self.overall_eta = None
        
        # Stage-specific metrics
        self.stage_progress = 0.0
        self.stage_eta = None
        self.stage_throughput = None
        
        # General metrics
        self.message = "Scan queued"
        self.started_at = datetime.utcnow()
        self.completed_at = None
        self.error = None
        self.results = None
        self.performance_metrics = {}
        self.cancel_requested = False
        
    def update_stage(self, stage: str, current: int, total: int, metadata: dict = None):
        """Update progress for a specific stage."""
        if stage in self.stages:
            self.current_stage = stage
            self.stages[stage]["current"] = current
            self.stages[stage]["total"] = total
            self.stages[stage]["progress"] = (current / total * 100) if total > 0 else 0
            
            # Update from metadata if available
            if metadata:
                self.stage_progress = metadata.get('stage_progress_pct', self.stages[stage]["progress"])
                self.overall_progress = metadata.get('overall_progress_pct', self.calculate_overall_progress())
                self.stage_eta = metadata.get('stage_eta')
                self.overall_eta = metadata.get('overall_eta')
                self.stage_throughput = metadata.get('stage_throughput')
            else:
                self.overall_progress = self.calculate_overall_progress()
    
    def calculate_overall_progress(self) -> float:
        """Calculate overall progress based on weighted stages."""
        total = 0.0
        for stage_name, stage_data in self.stages.items():
            weight = stage_data["weight"]
            progress = stage_data["progress"] / 100.0  # Convert to 0-1 range
            total += weight * progress
        return total * 100  # Convert back to percentage
        
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        
        # Format ETAs
        overall_eta_str = format_time(self.overall_eta) if format_time and self.overall_eta else None
        stage_eta_str = format_time(self.stage_eta) if format_time and self.stage_eta else None
        
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stages": self.stages,
            "overall_progress": round(self.overall_progress, 1),
            "overall_eta": self.overall_eta,
            "overall_eta_formatted": overall_eta_str,
            "stage_progress": round(self.stage_progress, 1),
            "stage_eta": self.stage_eta,
            "stage_eta_formatted": stage_eta_str,
            "stage_throughput": self.stage_throughput,
            "message": self.message,
            "elapsed_seconds": elapsed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "performance_metrics": self.performance_metrics
        }

class MultiStageProgressStore:
    """Manages multi-stage scan progress storage."""
    
    def __init__(self):
        self._store: Dict[str, MultiStageScanProgress] = {}
        self._lock = threading.Lock()
        self._websocket_connections: Dict[str, List[WebSocket]] = {}
        self._pending_notifications: List[tuple] = []
        
    def create(self, scan_id: str) -> MultiStageScanProgress:
        """Create a new progress entry."""
        with self._lock:
            progress = MultiStageScanProgress(scan_id)
            self._store[scan_id] = progress
            return progress
    
    def get(self, scan_id: str) -> Optional[MultiStageScanProgress]:
        """Get progress for a scan."""
        return self._store.get(scan_id)
    
    def update(self, scan_id: str, **kwargs) -> Optional[MultiStageScanProgress]:
        """Update progress for a scan."""
        progress = self.get(scan_id)
        if progress:
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            # Queue WebSocket notification
            self._pending_notifications.append((scan_id, progress))
        return progress
    
    def update_stage_progress(self, scan_id: str, stage: str, current: int, total: int, 
                            message: str = "", metadata: dict = None) -> Optional[MultiStageScanProgress]:
        """Update stage-specific progress."""
        progress = self.get(scan_id)
        if progress:
            progress.update_stage(stage, current, total, metadata)
            progress.message = message if message else progress.message
            # Queue WebSocket notification
            self._pending_notifications.append((scan_id, progress))
        return progress
    
    def delete(self, scan_id: str) -> bool:
        """Delete progress entry."""
        with self._lock:
            if scan_id in self._store:
                del self._store[scan_id]
                return True
            return False
    
    def list_active(self) -> List[str]:
        """List active scan IDs."""
        return [
            scan_id for scan_id, progress in self._store.items()
            if progress.status in ("pending", "running")
        ]
    
    async def notify_websockets(self, scan_id: str, progress: MultiStageScanProgress):
        """Notify WebSocket clients of progress update."""
        if scan_id in self._websocket_connections:
            message = json.dumps(progress.to_dict())
            disconnected = []
            for ws in self._websocket_connections[scan_id]:
                try:
                    await ws.send_text(message)
                except:
                    disconnected.append(ws)
            # Clean up disconnected clients
            for ws in disconnected:
                self._websocket_connections[scan_id].remove(ws)
    
    def add_websocket(self, scan_id: str, websocket: WebSocket):
        """Add a WebSocket connection for a scan."""
        if scan_id not in self._websocket_connections:
            self._websocket_connections[scan_id] = []
        self._websocket_connections[scan_id].append(websocket)
    
    def remove_websocket(self, scan_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if scan_id in self._websocket_connections:
            if websocket in self._websocket_connections[scan_id]:
                self._websocket_connections[scan_id].remove(websocket)

# Global progress store
progress_store = MultiStageProgressStore()

# Thread pool for async scan execution
executor = ThreadPoolExecutor(max_workers=4)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class ScanRequest(BaseModel):
    max_symbols: int = Field(50, ge=1, le=500)
    min_tech_rating: float = Field(0.0, ge=0.0, le=100.0)
    sectors: Optional[List[str]] = None
    async_mode: bool = Field(True, description="Run scan asynchronously")

class ScanStartResponse(BaseModel):
    scan_id: str
    status: str
    message: str
    progress_url: str
    websocket_url: str
    sse_url: str

class MultiStageProgressResponse(BaseModel):
    scan_id: str
    status: str
    current_stage: str
    stages: Dict[str, Any]
    overall_progress: float
    overall_eta: Optional[float]
    overall_eta_formatted: Optional[str]
    stage_progress: float
    stage_eta: Optional[float]
    stage_eta_formatted: Optional[str]
    stage_throughput: Optional[float]
    message: str
    elapsed_seconds: float
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]
    performance_metrics: Dict[str, Any]

class ScanResultItem(BaseModel):
    ticker: Optional[str]
    signal: Optional[str]
    rrr: Optional[str]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    techRating: Optional[float]
    alphaScore: Optional[float]
    meritScore: Optional[float]
    sector: Optional[str]
    industry: Optional[str]

class ScanCompleteResponse(BaseModel):
    scan_id: str
    status: str
    results: List[ScanResultItem]
    performance_metrics: Dict[str, Any]
    stage_timings: Optional[Dict[str, float]]
    elapsed_seconds: float

# -------------------------------------------------------------------
# Multi-stage progress callback factory
# -------------------------------------------------------------------

def create_multistage_progress_callback(scan_id: str):
    """Create a multi-stage progress callback for the enhanced scanner."""
    
    def progress_callback(stage: str, current: int, total: int, message: str = "", metadata: dict = None):
        """Update multi-stage progress in the store."""
        progress_store.update_stage_progress(
            scan_id,
            stage=stage,
            current=current,
            total=total,
            message=message,
            metadata=metadata
        )
    
    return progress_callback

# -------------------------------------------------------------------
# Enhanced scan execution with multi-stage progress
# -------------------------------------------------------------------

def run_scan_with_multistage_progress(scan_id: str, config: Any) -> None:
    """Execute scan with multi-stage progress tracking."""
    progress = progress_store.get(scan_id)
    if not progress:
        return
    
    try:
        # Update status to running
        progress_store.update(scan_id, status="running", message="Starting enhanced scan...")
        
        # Create multi-stage progress callback
        progress_callback = create_multistage_progress_callback(scan_id)
        
        # Check if scan was cancelled
        if progress.cancel_requested:
            progress_store.update(
                scan_id,
                status="cancelled",
                message="Scan cancelled by user",
                completed_at=datetime.utcnow()
            )
            return
        
        # Run the enhanced scan with multi-stage progress tracking
        if run_scan_enhanced:
            df, status_text, performance_metrics = run_scan_enhanced(
                config=config, 
                progress_cb=progress_callback
            )
        else:
            # Fallback if enhanced scanner not available
            df = pd.DataFrame()
            status_text = "Enhanced scanner not available"
            performance_metrics = {}
        
        # Check if cancelled during execution
        if progress.cancel_requested:
            progress_store.update(
                scan_id,
                status="cancelled",
                message="Scan cancelled by user",
                completed_at=datetime.utcnow(),
                results=df.to_dict(orient="records") if not df.empty else []
            )
            return
        
        # Store results with performance metrics
        progress_store.update(
            scan_id,
            status="completed",
            message=status_text,
            completed_at=datetime.utcnow(),
            results=df.to_dict(orient="records") if not df.empty else [],
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        # Handle errors
        error_message = str(e)
        if isinstance(e, ScanError) and ScanError is not None:
            error_message = e.get_user_message()
        
        progress_store.update(
            scan_id,
            status="failed",
            message="Scan failed",
            error=error_message,
            completed_at=datetime.utcnow()
        )

# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Technic API with Multi-Stage Progress",
        "version": "0.3.0",
        "description": "Phase 3D-D implementation complete",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "start_scan": "POST /scan/start",
            "progress": "GET /scan/progress/{scan_id}",
            "results": "GET /scan/results/{scan_id}",
            "websocket": "WS /scan/ws/{scan_id}",
            "sse": "GET /scan/stream/{scan_id}",
            "active_scans": "GET /scan/active"
        }
    }

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.3.0",
        "features": {
            "multi_stage_progress": True,
            "websocket": True,
            "sse": True,
            "redis": REDIS_AVAILABLE,
            "enhanced_scanner": run_scan_enhanced is not None
        }
    }

@app.post("/scan/start", response_model=ScanStartResponse)
async def start_scan(request: ScanRequest) -> Dict[str, Any]:
    """
    Start a new scan with multi-stage progress tracking.
    
    Returns scan_id immediately for async tracking.
    """
    if not run_scan_enhanced:
        raise HTTPException(500, "Enhanced scanner unavailable")
    
    # Generate unique scan ID
    scan_id = str(uuid.uuid4())
    
    # Create progress entry
    progress = progress_store.create(scan_id)
    
    # Create scan config
    cfg = None
    if ScanConfig:
        cfg = ScanConfig(
            max_symbols=request.max_symbols,
            min_tech_rating=request.min_tech_rating,
            sectors=request.sectors
        )
    
    if request.async_mode:
        # Submit scan to executor
        executor.submit(run_scan_with_multistage_progress, scan_id, cfg)
        
        return {
            "scan_id": scan_id,
            "status": "pending",
            "message": "Scan started with multi-stage progress tracking",
            "progress_url": f"/scan/progress/{scan_id}",
            "websocket_url": f"/scan/ws/{scan_id}",
            "sse_url": f"/scan/stream/{scan_id}"
        }
    else:
        # Run synchronously (blocks until complete)
        run_scan_with_multistage_progress(scan_id, cfg)
        progress = progress_store.get(scan_id)
        
        return {
            "scan_id": scan_id,
            "status": progress.status,
            "message": progress.message,
            "progress_url": f"/scan/progress/{scan_id}",
            "websocket_url": f"/scan/ws/{scan_id}",
            "sse_url": f"/scan/stream/{scan_id}"
        }

@app.get("/scan/progress/{scan_id}", response_model=MultiStageProgressResponse)
def get_scan_progress(scan_id: str) -> Dict[str, Any]:
    """Get current multi-stage progress for a scan."""
    progress = progress_store.get(scan_id)
    if not progress:
        raise HTTPException(404, f"Scan {scan_id} not found")
    
    return progress.to_dict()

@app.get("/scan/results/{scan_id}", response_model=ScanCompleteResponse)
def get_scan_results(scan_id: str) -> Dict[str, Any]:
    """Get results for a completed scan."""
    progress = progress_store.get(scan_id)
    if not progress:
        raise HTTPException(404, f"Scan {scan_id} not found")
    
    if progress.status not in ("completed", "cancelled"):
        raise HTTPException(400, f"Scan {scan_id} is {progress.status}, not completed")
    
    # Format results
    results = []
    if progress.results:
        for row in progress.results:
            results.append({
                "ticker": row.get("Symbol"),
                "signal": row.get("Signal"),
                "rrr": f"R:R {row.get('RewardRisk'):.2f}" if pd.notna(row.get("RewardRisk")) else None,
                "entry": row.get("Entry"),
                "stop": row.get("Stop"),
                "target": row.get("Target"),
                "techRating": row.get("TechRating"),
                "alphaScore": row.get("AlphaScore"),
                "meritScore": row.get("MeritScore"),
                "sector": row.get("Sector"),
                "industry": row.get("Industry")
            })
    
    elapsed = (progress.completed_at - progress.started_at).total_seconds() if progress.completed_at else 0
    
    # Extract stage timings if available
    stage_timings = None
    if progress.performance_metrics and 'stage_timings' in progress.performance_metrics:
        stage_timings = progress.performance_metrics['stage_timings']
    
    return {
        "scan_id": scan_id,
        "status": progress.status,
        "results": results,
        "performance_metrics": progress.performance_metrics,
        "stage_timings": stage_timings,
        "elapsed_seconds": elapsed
    }

@app.post("/scan/cancel/{scan_id}")
def cancel_scan(scan_id: str) -> Dict[str, str]:
    """Cancel a running scan."""
    progress = progress_store.get(scan_id)
    if not progress:
        raise HTTPException(404, f"Scan {scan_id} not found")
    
    if progress.status not in ("pending", "running"):
        return {
            "scan_id": scan_id,
            "status": progress.status,
            "message": f"Scan is {progress.status}, cannot cancel"
        }
    
    # Set cancellation flag
    progress.cancel_requested = True
    progress_store.update(scan_id, cancel_requested=True)
    
    return {
        "scan_id": scan_id,
        "status": "cancelling",
        "message": "Cancellation requested"
    }

@app.get("/scan/active")
def list_active_scans() -> Dict[str, Any]:
    """List all active scans with multi-stage progress."""
    active_ids = progress_store.list_active()
    scans = []
    
    for scan_id in active_ids:
        progress = progress_store.get(scan_id)
        if progress:
            scans.append({
                "scan_id": scan_id,
                "status": progress.status,
                "current_stage": progress.current_stage,
                "overall_progress": progress.overall_progress,
                "message": progress.message
            })
    
    return {
        "count": len(scans),
        "scans": scans
    }

# -------------------------------------------------------------------
# WebSocket endpoint for real-time multi-stage progress
# -------------------------------------------------------------------

@app.websocket("/scan/ws/{scan_id}")
async def websocket_progress(websocket: WebSocket, scan_id: str):
    """WebSocket endpoint for real-time multi-stage progress updates."""
    await websocket.accept()
    
    # Check if scan exists
    progress = progress_store.get(scan_id)
    if not progress:
        await websocket.send_json({
            "error": f"Scan {scan_id} not found"
        })
        await websocket.close()
        return
    
    # Register WebSocket connection
    progress_store.add_websocket(scan_id, websocket)
    
    try:
        # Send initial status
        await websocket.send_json(progress.to_dict())
        
        # Keep connection alive and wait for completion
        while progress.status in ("pending", "running"):
            await asyncio.sleep(0.5)
            
            # Process any pending notifications
            if progress_store._pending_notifications:
                for sid, prog in progress_store._pending_notifications:
                    if sid == scan_id:
                        await websocket.send_json(prog.to_dict())
                progress_store._pending_notifications.clear()
            
            progress = progress_store.get(scan_id)
            if not progress:
                break
        
        # Send final status
        if progress:
            await websocket.send_json(progress.to_dict())
        
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up
        progress_store.remove_websocket(scan_id, websocket)

# -------------------------------------------------------------------
# Server-Sent Events (SSE) endpoint with multi-stage progress
# -------------------------------------------------------------------

async def multistage_event_generator(scan_id: str):
    """Generate SSE events for multi-stage scan progress."""
    progress = progress_store.get(scan_id)
    if not progress:
        yield f"data: {json.dumps({'error': f'Scan {scan_id} not found'})}\n\n"
        return
    
    # Send updates until completion
    last_update = None
    while progress.status in ("pending", "running"):
        current_update = progress.to_dict()
        
        # Only send if there's a change
        if current_update != last_update:
            yield f"data: {json.dumps(current_update)}\n\n"
            last_update = current_update
        
        await asyncio.sleep(0.5)
        progress = progress_store.get(scan_id)
        if not progress:
            break
    
    # Send final status
    if progress:
        yield f"data: {json.dumps(progress.to_dict())}\n\n"

@app.get("/scan/stream/{scan_id}")
async def sse_progress(scan_id: str):
    """Server-Sent Events endpoint for multi-stage progress updates."""
    return StreamingResponse(
        multistage_event_generator(scan_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

# -------------------------------------------------------------------
# Demo endpoint to showcase multi-stage progress
# -------------------------------------------------------------------

@app.get("/demo/progress")
async def demo_progress():
    """Demo endpoint showing multi-stage progress tracking capabilities."""
    return {
        "description": "Multi-stage progress tracking demo",
        "stages": [
            {
                "name": "universe_loading",
                "weight": "5%",
                "description": "Loading and filtering universe symbols"
            },
            {
                "name": "data_fetching",
                "weight": "20%",
                "description": "Batch fetching price data"
            },
            {
                "name": "symbol_scanning",
                "weight": "70%",
                "description": "Analyzing individual symbols"
            },
            {
                "name": "finalization",
                "weight": "5%",
                "description": "Post-processing and saving results"
            }
        ],
        "features": [
            "Real-time progress updates via WebSocket",
            "Server-Sent Events (SSE) streaming",
            "Stage-specific ETAs",
            "Overall progress calculation",
            "Throughput metrics",
            "Performance timing breakdown"
        ],
        "example_usage": {
            "start_scan": "POST /scan/start",
            "track_progress": "GET /scan/progress/{scan_id}",
            "websocket": "ws://localhost:8000/scan/ws/{scan_id}",
            "sse": "http://localhost:8000/scan/stream/{scan_id}"
        }
    }

# Convenience launcher
def _main() -> None:  # pragma: no cover
    import uvicorn
    print("Starting Technic API with Multi-Stage Progress Tracking (Phase 3D-D)")
    print("Documentation available at: http://localhost:8000/docs")
    print("Demo endpoint: http://localhost:8000/demo/progress")
    uvicorn.run("api_with_multistage_progress:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

if __name__ == "__main__":  # pragma: no cover
    _main()
