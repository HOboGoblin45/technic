"""
Enhanced FastAPI with progress tracking support for scanner operations.
Implements Phase 3D-C: API Progress Integration
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
    from technic_v4 import scanner_core
    from technic_v4.engine import trade_planner
    from technic_v4.errors import ErrorType, ScanError
    from technic_v4.progress import ProgressTracker, MultiStageProgressTracker
except Exception:  # pragma: no cover - keep API alive even if imports fail
    scanner_core = None  # type: ignore
    trade_planner = None  # type: ignore
    ErrorType = None  # type: ignore
    ScanError = None  # type: ignore
    ProgressTracker = None  # type: ignore
    MultiStageProgressTracker = None  # type: ignore

# Optional Redis for distributed progress storage
try:
    from technic_v4.cache.redis_cache import redis_cache
    REDIS_AVAILABLE = redis_cache.available if redis_cache else False
except ImportError:
    redis_cache = None
    REDIS_AVAILABLE = False

app = FastAPI(title="Technic API Enhanced", version="0.2.0")

# -------------------------------------------------------------------
# Progress Storage (in-memory fallback, Redis preferred)
# -------------------------------------------------------------------

class ScanProgress:
    """Stores progress information for a scan."""
    
    def __init__(self, scan_id: str):
        self.scan_id = scan_id
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.stage = "initializing"
        self.current = 0
        self.total = 0
        self.percentage = 0.0
        self.message = "Scan queued"
        self.eta_seconds = None
        self.symbols_per_second = None
        self.started_at = datetime.utcnow()
        self.completed_at = None
        self.error = None
        self.results = None
        self.metadata = {}
        self.cancel_requested = False
        
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "message": self.message,
            "eta_seconds": self.eta_seconds,
            "symbols_per_second": self.symbols_per_second,
            "elapsed_seconds": elapsed,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata
        }

class ProgressStore:
    """Manages scan progress storage."""
    
    def __init__(self):
        self._store: Dict[str, ScanProgress] = {}
        self._lock = threading.Lock()
        self._websocket_connections: Dict[str, List[WebSocket]] = {}
        
    def create(self, scan_id: str) -> ScanProgress:
        """Create a new progress entry."""
        with self._lock:
            progress = ScanProgress(scan_id)
            self._store[scan_id] = progress
            return progress
    
    def get(self, scan_id: str) -> Optional[ScanProgress]:
        """Get progress for a scan."""
        return self._store.get(scan_id)
    
    def update(self, scan_id: str, **kwargs) -> Optional[ScanProgress]:
        """Update progress for a scan."""
        progress = self.get(scan_id)
        if progress:
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            # Notify WebSocket clients
            asyncio.create_task(self._notify_websockets(scan_id, progress))
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
    
    async def _notify_websockets(self, scan_id: str, progress: ScanProgress):
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
progress_store = ProgressStore()

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

class ProgressResponse(BaseModel):
    scan_id: str
    status: str
    stage: str
    current: int
    total: int
    percentage: float
    message: str
    eta_seconds: Optional[float]
    symbols_per_second: Optional[float]
    elapsed_seconds: float
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]

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
    elapsed_seconds: float

# -------------------------------------------------------------------
# Progress callback factory
# -------------------------------------------------------------------

def create_progress_callback(scan_id: str):
    """Create a progress callback for the scanner."""
    
    def progress_callback(stage: str, current: int, total: int, message: str = "", metadata: dict = None):
        """Update progress in the store."""
        metadata = metadata or {}
        progress_store.update(
            scan_id,
            status="running",
            stage=stage,
            current=current,
            total=total,
            percentage=metadata.get('percentage', (current / total * 100) if total > 0 else 0),
            message=message,
            eta_seconds=metadata.get('eta_seconds'),
            symbols_per_second=metadata.get('symbols_per_second'),
            metadata=metadata
        )
    
    return progress_callback

# -------------------------------------------------------------------
# Scan execution
# -------------------------------------------------------------------

def run_scan_with_progress(scan_id: str, config: Any) -> None:
    """Execute scan with progress tracking."""
    progress = progress_store.get(scan_id)
    if not progress:
        return
    
    try:
        # Update status to running
        progress_store.update(scan_id, status="running", message="Starting scan...")
        
        # Create progress callback
        progress_callback = create_progress_callback(scan_id)
        
        # Check if scan was cancelled
        if progress.cancel_requested:
            progress_store.update(
                scan_id,
                status="cancelled",
                message="Scan cancelled by user",
                completed_at=datetime.utcnow()
            )
            return
        
        # Run the scan with progress tracking
        result = scanner_core.run_scan(config=config, progress_cb=progress_callback)
        
        # Handle different return formats
        if len(result) == 3:
            df, status_text, performance_metrics = result
        else:
            df, status_text = result
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
        
        # Store results
        progress_store.update(
            scan_id,
            status="completed",
            message=status_text,
            completed_at=datetime.utcnow(),
            results=df.to_dict(orient="records"),
            metadata=performance_metrics
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

@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "version": "0.2.0",
        "features": {
            "progress_tracking": True,
            "websocket": True,
            "redis": REDIS_AVAILABLE
        }
    }

@app.post("/scan/start", response_model=ScanStartResponse)
async def start_scan(request: ScanRequest) -> Dict[str, Any]:
    """
    Start a new scan with progress tracking.
    
    Returns scan_id immediately for async tracking.
    """
    if scanner_core is None:
        raise HTTPException(500, "scanner_core unavailable")
    
    # Generate unique scan ID
    scan_id = str(uuid.uuid4())
    
    # Create progress entry
    progress = progress_store.create(scan_id)
    
    # Create scan config
    cfg = scanner_core.ScanConfig(
        max_symbols=request.max_symbols,
        min_tech_rating=request.min_tech_rating,
        sectors=request.sectors
    )
    
    if request.async_mode:
        # Submit scan to executor
        executor.submit(run_scan_with_progress, scan_id, cfg)
        
        return {
            "scan_id": scan_id,
            "status": "pending",
            "message": "Scan started in background",
            "progress_url": f"/scan/progress/{scan_id}",
            "websocket_url": f"/scan/ws/{scan_id}"
        }
    else:
        # Run synchronously (blocks until complete)
        run_scan_with_progress(scan_id, cfg)
        progress = progress_store.get(scan_id)
        
        return {
            "scan_id": scan_id,
            "status": progress.status,
            "message": progress.message,
            "progress_url": f"/scan/progress/{scan_id}",
            "websocket_url": f"/scan/ws/{scan_id}"
        }

@app.get("/scan/progress/{scan_id}", response_model=ProgressResponse)
def get_scan_progress(scan_id: str) -> Dict[str, Any]:
    """Get current progress for a scan."""
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
    
    return {
        "scan_id": scan_id,
        "status": progress.status,
        "results": results,
        "performance_metrics": progress.metadata,
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
    """List all active scans."""
    active_ids = progress_store.list_active()
    scans = []
    
    for scan_id in active_ids:
        progress = progress_store.get(scan_id)
        if progress:
            scans.append({
                "scan_id": scan_id,
                "status": progress.status,
                "stage": progress.stage,
                "percentage": progress.percentage,
                "message": progress.message
            })
    
    return {
        "count": len(scans),
        "scans": scans
    }

# -------------------------------------------------------------------
# WebSocket endpoint for real-time progress
# -------------------------------------------------------------------

@app.websocket("/scan/ws/{scan_id}")
async def websocket_progress(websocket: WebSocket, scan_id: str):
    """WebSocket endpoint for real-time progress updates."""
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
# Server-Sent Events (SSE) endpoint
# -------------------------------------------------------------------

async def event_generator(scan_id: str):
    """Generate SSE events for scan progress."""
    progress = progress_store.get(scan_id)
    if not progress:
        yield f"data: {json.dumps({'error': f'Scan {scan_id} not found'})}\n\n"
        return
    
    # Send updates until completion
    while progress.status in ("pending", "running"):
        yield f"data: {json.dumps(progress.to_dict())}\n\n"
        await asyncio.sleep(0.5)
        progress = progress_store.get(scan_id)
        if not progress:
            break
    
    # Send final status
    if progress:
        yield f"data: {json.dumps(progress.to_dict())}\n\n"

@app.get("/scan/sse/{scan_id}")
async def sse_progress(scan_id: str):
    """Server-Sent Events endpoint for progress updates."""
    return StreamingResponse(
        event_generator(scan_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

# -------------------------------------------------------------------
# Backward compatibility: Original /scan endpoint
# -------------------------------------------------------------------

@app.get("/scan")
async def scan_legacy(
    max_symbols: int = Query(50, ge=1, le=500),
    min_tech_rating: float = Query(0.0, ge=0.0, le=100.0)
) -> Dict[str, Any]:
    """
    Legacy scan endpoint for backward compatibility.
    Runs synchronously and returns results directly.
    """
    request = ScanRequest(
        max_symbols=max_symbols,
        min_tech_rating=min_tech_rating,
        async_mode=False
    )
    
    # Start scan synchronously
    start_response = await start_scan(request)
    scan_id = start_response["scan_id"]
    
    # Get results
    results_response = get_scan_results(scan_id)
    
    # Clean up
    progress_store.delete(scan_id)
    
    return {
        "results": results_response["results"],
        "performance_metrics": results_response["performance_metrics"],
        "elapsed_seconds": results_response["elapsed_seconds"]
    }

# Convenience launcher
def _main() -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run("api_enhanced:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

if __name__ == "__main__":  # pragma: no cover
    _main()
