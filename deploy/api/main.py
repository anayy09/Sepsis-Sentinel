"""
FastAPI REST API for Sepsis Sentinel Model
Provides HTTP endpoints for sepsis prediction with real-time inference.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tritonclient.http as httpclient
import uvicorn
from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, Depends, 
    WebSocket, WebSocketDisconnect, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
triton_client = None
model_metadata = None
websocket_connections = []


class VitalSigns(BaseModel):
    """Vital signs data model."""
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="Heart rate (bpm)")
    systolic_bp: Optional[float] = Field(None, ge=50, le=300, description="Systolic BP (mmHg)")
    diastolic_bp: Optional[float] = Field(None, ge=30, le=200, description="Diastolic BP (mmHg)")
    mean_bp: Optional[float] = Field(None, ge=40, le=250, description="Mean BP (mmHg)")
    respiratory_rate: Optional[float] = Field(None, ge=5, le=60, description="Respiratory rate (/min)")
    temperature: Optional[float] = Field(None, ge=90, le=110, description="Temperature (°F)")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="Oxygen saturation (%)")
    gcs_total: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")


class LabValues(BaseModel):
    """Laboratory values data model."""
    wbc: Optional[float] = Field(None, ge=0, le=50, description="White blood cells (K/uL)")
    hemoglobin: Optional[float] = Field(None, ge=0, le=25, description="Hemoglobin (g/dL)")
    hematocrit: Optional[float] = Field(None, ge=0, le=70, description="Hematocrit (%)")
    platelets: Optional[float] = Field(None, ge=0, le=1000, description="Platelets (K/uL)")
    sodium: Optional[float] = Field(None, ge=100, le=180, description="Sodium (mEq/L)")
    potassium: Optional[float] = Field(None, ge=1, le=10, description="Potassium (mEq/L)")
    chloride: Optional[float] = Field(None, ge=80, le=130, description="Chloride (mEq/L)")
    bun: Optional[float] = Field(None, ge=0, le=200, description="BUN (mg/dL)")
    creatinine: Optional[float] = Field(None, ge=0, le=20, description="Creatinine (mg/dL)")
    glucose: Optional[float] = Field(None, ge=0, le=1000, description="Glucose (mg/dL)")
    lactate: Optional[float] = Field(None, ge=0, le=30, description="Lactate (mmol/L)")
    ph: Optional[float] = Field(None, ge=6.5, le=8.0, description="Blood pH")
    pco2: Optional[float] = Field(None, ge=10, le=100, description="PCO2 (mmHg)")
    po2: Optional[float] = Field(None, ge=30, le=600, description="PO2 (mmHg)")
    bicarbonate: Optional[float] = Field(None, ge=5, le=50, description="Bicarbonate (mEq/L)")


class WaveformFeatures(BaseModel):
    """Waveform-derived features data model."""
    ecg_hr_mean: Optional[float] = Field(None, description="ECG heart rate mean")
    ecg_hr_std: Optional[float] = Field(None, description="ECG heart rate std")
    abp_systolic_mean: Optional[float] = Field(None, description="ABP systolic mean")
    abp_diastolic_mean: Optional[float] = Field(None, description="ABP diastolic mean")
    resp_rate_mean: Optional[float] = Field(None, description="Respiratory rate mean")
    pleth_spo2_mean: Optional[float] = Field(None, description="Plethysmography SpO2 mean")


class Demographics(BaseModel):
    """Patient demographics data model."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    gender: str = Field(..., regex="^(M|F)$", description="Gender (M/F)")
    race: Optional[str] = Field(None, description="Race/ethnicity")
    weight: Optional[float] = Field(None, ge=1, le=300, description="Weight (kg)")
    height: Optional[float] = Field(None, ge=30, le=250, description="Height (cm)")


class AdmissionInfo(BaseModel):
    """Admission information data model."""
    admission_type: str = Field(..., description="Type of admission")
    admission_location: str = Field(..., description="Admission location")
    insurance: Optional[str] = Field(None, description="Insurance type")


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    patient_id: str = Field(..., description="Unique patient identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    
    # Patient data
    demographics: Demographics
    admission_info: AdmissionInfo
    
    # Time series data (last 72 time points)
    vitals: List[VitalSigns] = Field(..., max_items=72, description="Vital signs time series")
    labs: List[LabValues] = Field(..., max_items=72, description="Lab values time series")
    waveforms: List[WaveformFeatures] = Field(..., max_items=72, description="Waveform features")
    
    # Options
    return_explanations: bool = Field(False, description="Return model explanations")
    return_confidence: bool = Field(True, description="Return prediction confidence")
    
    @validator('vitals', 'labs', 'waveforms')
    def validate_time_series_length(cls, v):
        if len(v) < 24:  # Minimum 12 hours of data
            raise ValueError("Time series must contain at least 24 time points (12 hours)")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    requests: List[PredictionRequest] = Field(..., max_items=100, description="Batch requests")
    parallel_processing: bool = Field(True, description="Process requests in parallel")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    patient_id: str
    timestamp: datetime
    sepsis_probability: float = Field(..., ge=0, le=1, description="Sepsis probability [0-1]")
    sepsis_risk_level: str = Field(..., description="Risk level (low/medium/high/critical)")
    confidence_score: Optional[float] = Field(None, description="Model confidence")
    time_to_decision_hours: Optional[float] = Field(None, description="Predicted time to sepsis")
    
    # Model explanations (optional)
    feature_importance: Optional[Dict[str, float]] = None
    attention_weights: Optional[Dict[str, List[float]]] = None
    
    # Metadata
    model_version: str = "1.0.0"
    processing_time_ms: float
    request_id: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    model_status: str
    triton_status: str
    version: str = "1.0.0"


class ConnectionManager:
    """WebSocket connection manager for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Sepsis Sentinel API...")
    await initialize_triton_client()
    yield
    # Shutdown
    logger.info("Shutting down Sepsis Sentinel API...")
    if triton_client:
        triton_client.close()


# Create FastAPI application
app = FastAPI(
    title="Sepsis Sentinel API",
    description="Early sepsis prediction using multimodal AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# WebSocket manager
manager = ConnectionManager()


async def initialize_triton_client():
    """Initialize Triton Inference Server client."""
    global triton_client, model_metadata
    
    try:
        triton_url = "localhost:8000"  # Configure as needed
        triton_client = httpclient.InferenceServerClient(
            url=triton_url,
            verbose=False,
            connection_timeout=60.0,
            network_timeout=60.0
        )
        
        # Check server health
        if triton_client.is_server_live() and triton_client.is_server_ready():
            logger.info("Triton Inference Server is ready")
            
            # Get model metadata
            model_metadata = triton_client.get_model_metadata("sepsis_sentinel")
            logger.info("Model metadata retrieved successfully")
        else:
            raise Exception("Triton server is not ready")
            
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")
        triton_client = None


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key (placeholder implementation)."""
    # In production, implement proper authentication
    if credentials is None:
        return True  # Allow unauthenticated access for demo
    
    # Verify token/API key here
    return True


def preprocess_request(request: PredictionRequest) -> Dict[str, np.ndarray]:
    """Preprocess prediction request into model inputs."""
    # This is a simplified preprocessing step
    # In production, implement proper feature engineering and normalization
    
    # Extract demographics
    demographics = [
        request.demographics.age,
        1.0 if request.demographics.gender == "M" else 0.0,
        request.demographics.weight or 70.0,  # Default weight
        request.demographics.height or 170.0,  # Default height
    ]
    
    # Extract time series features (simplified)
    vital_features = []
    lab_features = []
    waveform_features = []
    
    for vital in request.vitals:
        vital_row = [
            vital.heart_rate or 80.0,
            vital.systolic_bp or 120.0,
            vital.diastolic_bp or 80.0,
            vital.respiratory_rate or 16.0,
            vital.temperature or 98.6,
            vital.spo2 or 98.0,
            vital.gcs_total or 15.0
        ]
        vital_features.append(vital_row)
    
    for lab in request.labs:
        lab_row = [
            lab.wbc or 7.0,
            lab.hemoglobin or 12.0,
            lab.platelets or 250.0,
            lab.sodium or 140.0,
            lab.potassium or 4.0,
            lab.creatinine or 1.0,
            lab.glucose or 100.0,
            lab.lactate or 2.0
        ]
        lab_features.append(lab_row)
    
    for waveform in request.waveforms:
        waveform_row = [
            waveform.ecg_hr_mean or 80.0,
            waveform.abp_systolic_mean or 120.0,
            waveform.resp_rate_mean or 16.0
        ]
        waveform_features.append(waveform_row)
    
    # Pad sequences to 72 time points if needed
    while len(vital_features) < 72:
        vital_features.append(vital_features[-1] if vital_features else [0.0] * 7)
    while len(lab_features) < 72:
        lab_features.append(lab_features[-1] if lab_features else [0.0] * 8)
    while len(waveform_features) < 72:
        waveform_features.append(waveform_features[-1] if waveform_features else [0.0] * 3)
    
    # Combine into model inputs (simplified feature engineering)
    # In production, use proper TFT and GNN feature extraction
    tft_features = np.concatenate([
        demographics,
        np.array(vital_features).flatten()[:252]  # Fit to TFT dimension
    ])
    
    # Pad to expected dimension
    if len(tft_features) < 256:
        tft_features = np.pad(tft_features, (0, 256 - len(tft_features)))
    else:
        tft_features = tft_features[:256]
    
    # Simple GNN features (normally would be graph-based)
    gnn_features = np.concatenate([
        np.array(lab_features).mean(axis=0),
        np.array(waveform_features).mean(axis=0)
    ])
    
    if len(gnn_features) < 64:
        gnn_features = np.pad(gnn_features, (0, 64 - len(gnn_features)))
    else:
        gnn_features = gnn_features[:64]
    
    return {
        'tft_features': tft_features.astype(np.float32),
        'gnn_features': gnn_features.astype(np.float32)
    }


async def run_inference(inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Run inference using Triton Inference Server."""
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client not available")
    
    try:
        # Prepare inputs for Triton
        triton_inputs = []
        
        tft_input = httpclient.InferInput("tft_features", inputs['tft_features'].shape, "FP32")
        tft_input.set_data_from_numpy(inputs['tft_features'].reshape(1, -1))
        triton_inputs.append(tft_input)
        
        gnn_input = httpclient.InferInput("gnn_features", inputs['gnn_features'].shape, "FP32")
        gnn_input.set_data_from_numpy(inputs['gnn_features'].reshape(1, -1))
        triton_inputs.append(gnn_input)
        
        # Prepare outputs
        triton_outputs = [
            httpclient.InferRequestedOutput("probabilities")
        ]
        
        # Run inference
        response = triton_client.infer("sepsis_sentinel", triton_inputs, outputs=triton_outputs)
        
        # Extract results
        probabilities = response.as_numpy("probabilities")
        sepsis_probability = float(probabilities[0])
        
        return {
            'sepsis_probability': sepsis_probability,
            'confidence_score': min(max(sepsis_probability, 1 - sepsis_probability) * 2, 1.0)
        }
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


def determine_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability < 0.2:
        return "low"
    elif probability < 0.5:
        return "medium"
    elif probability < 0.8:
        return "high"
    else:
        return "critical"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    triton_status = "healthy" if triton_client and triton_client.is_server_ready() else "unhealthy"
    model_status = "loaded" if model_metadata else "not_loaded"
    
    return HealthResponse(
        status="healthy" if triton_status == "healthy" and model_status == "loaded" else "unhealthy",
        timestamp=datetime.now(),
        model_status=model_status,
        triton_status=triton_status
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Single sepsis prediction endpoint."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        # Preprocess request
        inputs = preprocess_request(request)
        
        # Run inference
        results = await run_inference(inputs)
        
        # Create response
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            patient_id=request.patient_id,
            timestamp=datetime.now(),
            sepsis_probability=results['sepsis_probability'],
            sepsis_risk_level=determine_risk_level(results['sepsis_probability']),
            confidence_score=results['confidence_score'],
            processing_time_ms=processing_time,
            request_id=request_id
        )
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            request.patient_id,
            results['sepsis_probability'],
            processing_time
        )
        
        # Broadcast to WebSocket clients if high risk
        if response.sepsis_risk_level in ["high", "critical"]:
            background_tasks.add_task(
                broadcast_alert,
                response.dict()
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error for patient {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict_sepsis(
    request: BatchPredictionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Batch sepsis prediction endpoint."""
    if len(request.requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    start_time = time.time()
    
    try:
        if request.parallel_processing:
            # Process requests in parallel
            tasks = [predict_sepsis(req, BackgroundTasks()) for req in request.requests]
            responses = await asyncio.gather(*tasks)
        else:
            # Process requests sequentially
            responses = []
            for req in request.requests:
                response = await predict_sepsis(req, BackgroundTasks())
                responses.append(response)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": responses,
            "batch_size": len(request.requests),
            "total_processing_time_ms": processing_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for incoming data
            data = await websocket.receive_text()
            
            try:
                # Parse prediction request
                request_data = json.loads(data)
                request = PredictionRequest(**request_data)
                
                # Process prediction
                response = await predict_sepsis(request, BackgroundTasks())
                
                # Send response back
                await manager.send_personal_message(response.dict(), websocket)
                
            except Exception as e:
                error_response = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(error_response, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def log_prediction(patient_id: str, probability: float, processing_time: float):
    """Log prediction for monitoring and analytics."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "sepsis_probability": probability,
        "processing_time_ms": processing_time
    }
    
    # In production, save to database or monitoring system
    logger.info(f"Prediction logged: {log_entry}")


async def broadcast_alert(prediction_data: dict):
    """Broadcast high-risk alerts to connected WebSocket clients."""
    alert_message = {
        "type": "alert",
        "data": prediction_data,
        "timestamp": datetime.now().isoformat()
    }
    
    await manager.broadcast(alert_message)
    logger.info(f"Alert broadcasted for patient {prediction_data['patient_id']}")


@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    # In production, integrate with proper metrics collection
    return {
        "active_connections": len(manager.active_connections),
        "model_status": "healthy" if model_metadata else "unavailable",
        "uptime": "placeholder",
        "total_predictions": "placeholder"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info"
    )
