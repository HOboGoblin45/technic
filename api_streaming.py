"""
Streaming API for Phase 3E-B: Incremental Results Streaming
Provides WebSocket and SSE endpoints for real-time result streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import uuid
import time
from datetime import datetime

# Import our streaming infrastructure
from technic_v4.result_streamer import (
    get_stream_manager,
    ScanResult,
    StreamManager
)

# Import scanner (we'll use a mock for now)
from technic_v4.scanner_core import ScanConfig

app = FastAPI(title="Technic Scanner Streaming API", version="3E-B")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global stream manager
stream_manager = get_stream_manager()

# Active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}


# Request/Response Models
class StreamScanRequest(BaseModel):
    """Request to start a streaming scan"""
    sectors: Optional[List[str]] = None
    max_symbols: Optional[int] = 100
    min_tech_rating: Optional[float] = 10.0
    enable_prioritization: bool = True
    termination_criteria: Optional[Dict[str, Any]] = None


class StreamScanResponse(BaseModel):
    """Response when starting a streaming scan"""
    scan_id: str
    status: str
    message: str
    websocket_url: str
    sse_url: str
    stats_url: str
    stop_url: str


class ScanStatsResponse(BaseModel):
    """Current scan statistics"""
    scan_id: str
    status: str
    stats: Dict[str, Any]


# Helper Functions
async def simulate_scan_with_streaming(
    scan_id: str,
    config: ScanConfig,
    termination_criteria: Optional[Dict[str, Any]] = None
):
    """
    Simulate a scan with result streaming
    
    In production, this would call the actual scanner with streaming enabled
    """
    # Create stream
    total_symbols = config.max_symbols or 50
    queue = stream_manager.create_stream(
        scan_id,
        total_symbols,
        termination_criteria
    )
    
    # Simulate scanning symbols
    test_symbols = [f"SYM{i:03d}" for i in range(total_symbols)]
    
    for i, symbol in enumerate(test_symbols):
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate result (30% chance of signal)
        import random
        has_signal = random.random() < 0.3
        
        result = ScanResult(
            symbol=symbol,
            signal='BUY' if has_signal else None,
            tech_rating=random.uniform(70, 90) if has_signal else random.uniform(30, 60),
            alpha_score=random.uniform(0.6, 0.9) if has_signal else random.uniform(0.2, 0.5),
            entry=100.0 if has_signal else None,
            stop=95.0 if has_signal else None,
            target=110.0 if has_signal else None,
            priority_tier='high' if i < total_symbols // 3 else 'medium',
            priority_score=random.uniform(60, 90),
            batch_number=(i // 10) + 1
        )
        
        # Add to stream
        should_continue = stream_manager.add_result(scan_id, result)
        
        if not should_continue:
            print(f"[{scan_id}] Early termination triggered")
            break
    
    # Mark complete
    stream_manager.complete_stream(scan_id)
    print(f"[{scan_id}] Scan complete")


async def broadcast_result(scan_id: str, result: ScanResult):
    """Broadcast result to all connected WebSocket clients"""
    if scan_id in active_connections:
        message = {
            'type': 'result',
            'data': result.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all connections
        disconnected = []
        for websocket in active_connections[scan_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            active_connections[scan_id].remove(ws)


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Technic Scanner Streaming API",
        "version": "3E-B",
        "features": [
            "Real-time result streaming",
            "WebSocket support",
            "Server-Sent Events (SSE)",
            "Early termination",
            "Priority-based scanning"
        ],
        "endpoints": {
            "start_stream": "POST /scan/stream",
            "websocket": "WS /ws/results/{scan_id}",
            "sse": "GET /events/results/{scan_id}",
            "stats": "GET /scan/stats/{scan_id}",
            "stop": "POST /scan/stop/{scan_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_streams": len(stream_manager.get_active_streams())
    }


@app.post("/scan/stream", response_model=StreamScanResponse)
async def start_streaming_scan(
    request: StreamScanRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new streaming scan
    
    Results will be streamed in real-time via WebSocket or SSE
    """
    # Generate scan ID
    scan_id = f"scan_{uuid.uuid4().hex[:12]}"
    
    # Create scan config
    config = ScanConfig(
        sectors=request.sectors,
        max_symbols=request.max_symbols,
        min_tech_rating=request.min_tech_rating
    )
    
    # Start scan in background
    background_tasks.add_task(
        simulate_scan_with_streaming,
        scan_id,
        config,
        request.termination_criteria
    )
    
    # Return connection info
    base_url = "http://localhost:8001"  # Update with actual host
    
    return StreamScanResponse(
        scan_id=scan_id,
        status="started",
        message=f"Streaming scan started with ID: {scan_id}",
        websocket_url=f"ws://localhost:8001/ws/results/{scan_id}",
        sse_url=f"{base_url}/events/results/{scan_id}",
        stats_url=f"{base_url}/scan/stats/{scan_id}",
        stop_url=f"{base_url}/scan/stop/{scan_id}"
    )


@app.websocket("/ws/results/{scan_id}")
async def websocket_results(websocket: WebSocket, scan_id: str):
    """
    WebSocket endpoint for streaming results
    
    Clients connect here to receive real-time scan results
    """
    await websocket.accept()
    
    # Add to active connections
    if scan_id not in active_connections:
        active_connections[scan_id] = []
    active_connections[scan_id].append(websocket)
    
    try:
        # Get the stream
        queue = stream_manager.get_stream(scan_id)
        if not queue:
            await websocket.send_json({
                'type': 'error',
                'message': f'Scan {scan_id} not found'
            })
            await websocket.close()
            return
        
        # Subscribe to results
        async def send_result(result: ScanResult):
            try:
                await websocket.send_json({
                    'type': 'result',
                    'data': result.to_dict(),
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                print(f"Error sending result: {e}")
        
        # Send initial status
        await websocket.send_json({
            'type': 'connected',
            'scan_id': scan_id,
            'message': 'Connected to result stream'
        })
        
        # Stream results
        while not queue.is_complete():
            # Get pending results
            results = queue.get_results(max_count=10)
            
            for result in results:
                await websocket.send_json({
                    'type': 'result',
                    'data': result.to_dict(),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Send stats update
            stats = stream_manager.get_stats(scan_id)
            if stats:
                await websocket.send_json({
                    'type': 'stats',
                    'data': stats,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(0.1)
        
        # Send completion message
        final_stats = stream_manager.get_stats(scan_id)
        await websocket.send_json({
            'type': 'complete',
            'scan_id': scan_id,
            'stats': final_stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except WebSocketDisconnect:
        print(f"Client disconnected from {scan_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
    finally:
        # Remove from active connections
        if scan_id in active_connections and websocket in active_connections[scan_id]:
            active_connections[scan_id].remove(websocket)


@app.get("/events/results/{scan_id}")
async def sse_results(scan_id: str):
    """
    Server-Sent Events endpoint for streaming results
    
    Alternative to WebSocket for browsers that don't support it
    """
    async def event_generator():
        queue = stream_manager.get_stream(scan_id)
        if not queue:
            yield f"event: error\ndata: {json.dumps({'message': 'Scan not found'})}\n\n"
            return
        
        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'scan_id': scan_id})}\n\n"
        
        # Stream results
        while not queue.is_complete():
            # Get pending results
            results = queue.get_results(max_count=10)
            
            for result in results:
                data = json.dumps({
                    'type': 'result',
                    'data': result.to_dict()
                })
                yield f"event: result\ndata: {data}\n\n"
            
            # Send stats
            stats = stream_manager.get_stats(scan_id)
            if stats:
                data = json.dumps(stats)
                yield f"event: stats\ndata: {data}\n\n"
            
            await asyncio.sleep(0.1)
        
        # Send completion
        final_stats = stream_manager.get_stats(scan_id)
        data = json.dumps({'scan_id': scan_id, 'stats': final_stats})
        yield f"event: complete\ndata: {data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/scan/stats/{scan_id}", response_model=ScanStatsResponse)
async def get_scan_stats(scan_id: str):
    """Get current statistics for a scan"""
    stats = stream_manager.get_stats(scan_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    
    queue = stream_manager.get_stream(scan_id)
    status = "complete" if queue and queue.is_complete() else "running"
    
    return ScanStatsResponse(
        scan_id=scan_id,
        status=status,
        stats=stats
    )


@app.post("/scan/stop/{scan_id}")
async def stop_scan(scan_id: str):
    """
    Stop a running scan early
    
    Useful when user has found enough results
    """
    queue = stream_manager.get_stream(scan_id)
    
    if not queue:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    
    if queue.is_complete():
        return {
            "scan_id": scan_id,
            "status": "already_complete",
            "message": "Scan was already complete"
        }
    
    # Mark as complete
    stream_manager.complete_stream(scan_id)
    
    # Get final stats
    stats = stream_manager.get_stats(scan_id)
    
    return {
        "scan_id": scan_id,
        "status": "stopped",
        "message": "Scan stopped successfully",
        "stats": stats
    }


@app.delete("/scan/{scan_id}")
async def cleanup_scan(scan_id: str):
    """Clean up scan resources"""
    stream_manager.cleanup_stream(scan_id)
    
    # Remove WebSocket connections
    if scan_id in active_connections:
        del active_connections[scan_id]
    
    return {
        "scan_id": scan_id,
        "status": "cleaned_up",
        "message": "Scan resources freed"
    }


@app.get("/scans/active")
async def list_active_scans():
    """List all active scans"""
    active_scans = stream_manager.get_active_streams()
    
    scans_info = []
    for scan_id in active_scans:
        stats = stream_manager.get_stats(scan_id)
        queue = stream_manager.get_stream(scan_id)
        
        scans_info.append({
            'scan_id': scan_id,
            'status': 'complete' if queue and queue.is_complete() else 'running',
            'stats': stats
        })
    
    return {
        "active_scans": len(active_scans),
        "scans": scans_info
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Technic Scanner Streaming API - Phase 3E-B")
    print("=" * 60)
    print("\nStarting server on http://localhost:8001")
    print("\nEndpoints:")
    print("  POST   http://localhost:8001/scan/stream")
    print("  WS     ws://localhost:8001/ws/results/{scan_id}")
    print("  GET    http://localhost:8001/events/results/{scan_id}")
    print("  GET    http://localhost:8001/scan/stats/{scan_id}")
    print("  POST   http://localhost:8001/scan/stop/{scan_id}")
    print("\nDocs: http://localhost:8001/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
