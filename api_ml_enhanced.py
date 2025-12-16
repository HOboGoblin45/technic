"""
ML-Enhanced API for Scan Optimization
Phase 3E-C Week 2: API Integration with ML predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

# Import ML components
from technic_v4.ml import (
    ScanHistoryDB,
    ScanRecord,
    get_current_market_conditions,
    ResultCountPredictor,
    ScanDurationPredictor,
    ParameterOptimizer
)

# Import scanner
from technic_v4.scanner_core import ScanConfig, run_scan

# Initialize FastAPI app
app = FastAPI(
    title="Technic Scanner ML API",
    description="ML-powered scan optimization with predictions and suggestions",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
scan_history_db = ScanHistoryDB()
result_predictor = ResultCountPredictor()
duration_predictor = ScanDurationPredictor()
parameter_optimizer = ParameterOptimizer(scan_history_db)

# Try to load trained models
try:
    result_predictor.load()
    print("✓ Loaded trained result predictor")
except:
    print("⚠ Result predictor not trained yet")

try:
    duration_predictor.load()
    print("✓ Loaded trained duration predictor")
except:
    print("⚠ Duration predictor not trained yet")


# ============================================================================
# Request/Response Models
# ============================================================================

class ScanRequest(BaseModel):
    """Scan configuration request"""
    max_symbols: int = Field(100, ge=1, le=500)
    min_tech_rating: float = Field(10.0, ge=0, le=100)
    min_dollar_vol: float = Field(0, ge=0)
    sectors: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    lookback_days: int = Field(90, ge=20, le=365)
    use_alpha_blend: bool = False
    enable_options: bool = False
    profile: str = Field("balanced", pattern="^(conservative|balanced|aggressive)$")


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_results: Optional[int]
    predicted_duration: Optional[float]
    confidence: float
    warnings: List[str]
    suggestions: List[str]
    market_conditions: Dict[str, Any]
    risk_level: str
    quality_estimate: str


class SuggestionResponse(BaseModel):
    """Parameter suggestion response"""
    suggested_config: Dict[str, Any]
    predicted_results: Optional[int]
    predicted_duration: Optional[float]
    reasoning: str
    alternatives: Dict[str, Dict[str, Any]]
    market_conditions: Dict[str, Any]


class HistoryStatsResponse(BaseModel):
    """Scan history statistics"""
    total_scans: int
    date_range: Optional[Dict[str, str]]
    avg_results: float
    avg_duration: float
    total_results: int
    total_duration: float


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Technic Scanner ML API",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "ML-powered result prediction",
            "Scan duration estimation",
            "Intelligent parameter optimization",
            "Configuration analysis",
            "Historical performance tracking"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_models": {
            "result_predictor": result_predictor.is_trained,
            "duration_predictor": duration_predictor.is_trained
        },
        "database": {
            "scans_recorded": scan_history_db.get_statistics()['total_scans']
        }
    }


@app.post("/scan/predict", response_model=PredictionResponse)
async def predict_scan_results(request: ScanRequest):
    """
    Predict scan outcomes before running
    
    Returns predictions for result count, duration, and configuration analysis
    """
    try:
        # Get current market conditions
        market_conditions = get_current_market_conditions()
        
        # Convert request to config dict
        config = request.dict()
        
        # Predict result count
        result_prediction = result_predictor.predict(config, market_conditions)
        
        # Predict duration
        duration_prediction = duration_predictor.predict(config, market_conditions)
        
        # Analyze configuration
        analysis = parameter_optimizer.analyze_config(config, market_conditions)
        
        # Calculate overall confidence
        confidence = (
            result_prediction.get('confidence', 0.5) * 0.6 +
            duration_prediction.get('confidence', 0.5) * 0.4
        )
        
        return PredictionResponse(
            predicted_results=result_prediction.get('predicted_count'),
            predicted_duration=duration_prediction.get('predicted_seconds'),
            confidence=confidence,
            warnings=analysis['warnings'],
            suggestions=analysis['suggestions'],
            market_conditions=market_conditions,
            risk_level=analysis['risk_level'],
            quality_estimate=analysis['estimated_quality']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/scan/suggest", response_model=SuggestionResponse)
async def suggest_parameters(
    goal: str = "balanced",
    include_alternatives: bool = True
):
    """
    Suggest optimal scan parameters based on goal
    
    Args:
        goal: 'speed', 'quality', or 'balanced'
        include_alternatives: Include alternative configurations
    
    Returns:
        Suggested configuration with predictions
    """
    try:
        # Validate goal
        if goal not in ['speed', 'quality', 'balanced']:
            raise HTTPException(
                status_code=400,
                detail="Goal must be 'speed', 'quality', or 'balanced'"
            )
        
        # Get market conditions
        market_conditions = get_current_market_conditions()
        
        # Get suggestion
        suggestion = parameter_optimizer.suggest_optimal(goal, market_conditions)
        
        # Predict outcomes for suggested config
        result_prediction = result_predictor.predict(
            suggestion['config'],
            market_conditions
        )
        
        duration_prediction = duration_predictor.predict(
            suggestion['config'],
            market_conditions
        )
        
        # Get alternatives if requested
        alternatives = {}
        if include_alternatives:
            for alt_goal in ['speed', 'quality', 'balanced']:
                if alt_goal != goal:
                    alt_suggestion = parameter_optimizer.suggest_optimal(
                        alt_goal,
                        market_conditions
                    )
                    alternatives[alt_goal] = alt_suggestion
        
        return SuggestionResponse(
            suggested_config=suggestion['config'],
            predicted_results=result_prediction.get('predicted_count'),
            predicted_duration=duration_prediction.get('predicted_seconds'),
            reasoning=suggestion['reasoning'],
            alternatives=alternatives,
            market_conditions=market_conditions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")


@app.get("/market/conditions")
async def get_market_conditions():
    """
    Get current market conditions
    
    Returns real-time market state for ML features
    """
    try:
        conditions = get_current_market_conditions()
        return {
            "conditions": conditions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market conditions: {str(e)}"
        )


@app.get("/history/stats", response_model=HistoryStatsResponse)
async def get_history_stats():
    """
    Get scan history statistics
    
    Returns aggregate statistics from historical scans
    """
    try:
        stats = scan_history_db.get_statistics()
        return HistoryStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get history stats: {str(e)}"
        )


@app.get("/history/recent")
async def get_recent_scans(limit: int = 10):
    """
    Get recent scan records
    
    Args:
        limit: Number of recent scans to return (max 100)
    
    Returns:
        List of recent scan records
    """
    try:
        if limit > 100:
            limit = 100
        
        records = scan_history_db.get_recent_scans(limit=limit)
        
        return {
            "count": len(records),
            "scans": [record.to_dict() for record in records]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent scans: {str(e)}"
        )


@app.post("/scan/execute")
async def execute_scan_with_logging(request: ScanRequest):
    """
    Execute scan and log results to history
    
    Runs scan with provided config and stores results for ML training
    """
    try:
        # Get market conditions before scan
        market_conditions = get_current_market_conditions()
        
        # Convert request to ScanConfig
        config_dict = request.dict()
        scan_config = ScanConfig(**config_dict)
        
        # Run scan
        start_time = datetime.now()
        results_df, status_text, performance_metrics = run_scan(scan_config)
        
        # Create scan record
        scan_record = ScanRecord(
            scan_id=f"scan_{start_time.strftime('%Y%m%d_%H%M%S')}",
            timestamp=start_time,
            config=config_dict,
            results={
                'count': len(results_df),
                'signals': len(results_df[results_df.get('Signal', '') != '']) if 'Signal' in results_df.columns else 0
            },
            performance=performance_metrics,
            market_conditions=market_conditions
        )
        
        # Log to history
        scan_history_db.add_scan(scan_record)
        
        return {
            "scan_id": scan_record.scan_id,
            "status": status_text,
            "results_count": len(results_df),
            "performance": performance_metrics,
            "logged": True
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scan execution failed: {str(e)}"
        )


@app.post("/models/train")
async def train_models(min_samples: int = 50):
    """
    Train ML models on historical data
    
    Args:
        min_samples: Minimum number of samples required for training
    
    Returns:
        Training metrics and status
    """
    try:
        # Get recent scans
        records = scan_history_db.get_recent_scans(limit=1000)
        
        if len(records) < min_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(records)} scans (need {min_samples})"
            )
        
        # Train result predictor
        result_metrics = result_predictor.train(records)
        result_predictor.save()
        
        # Train duration predictor
        duration_metrics = duration_predictor.train(records)
        duration_predictor.save()
        
        return {
            "status": "success",
            "training_samples": len(records),
            "result_predictor": {
                "test_mae": result_metrics['test_mae'],
                "test_r2": result_metrics['test_r2']
            },
            "duration_predictor": {
                "test_mae": duration_metrics['test_mae'],
                "test_r2": duration_metrics['test_r2']
            },
            "models_saved": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@app.get("/models/status")
async def get_model_status():
    """
    Get ML model status and performance
    
    Returns information about trained models
    """
    try:
        # Get feature importance if models are trained
        result_importance = {}
        duration_importance = {}
        
        if result_predictor.is_trained:
            result_importance = result_predictor.get_feature_importance()
        
        if duration_predictor.is_trained:
            duration_importance = duration_predictor.get_feature_importance()
        
        return {
            "result_predictor": {
                "trained": result_predictor.is_trained,
                "feature_importance": result_importance
            },
            "duration_predictor": {
                "trained": duration_predictor.is_trained,
                "feature_importance": duration_importance
            },
            "training_data_available": scan_history_db.get_statistics()['total_scans']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Technic Scanner ML API")
    print("="*60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8002/docs")
    print("Health Check: http://localhost:8002/health")
    print("\nML Endpoints:")
    print("  POST /scan/predict - Predict scan outcomes")
    print("  GET  /scan/suggest - Get parameter suggestions")
    print("  POST /scan/execute - Run scan with logging")
    print("  POST /models/train - Train ML models")
    print("  GET  /models/status - Check model status")
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
