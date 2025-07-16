from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from ml_engine import ProductionMLEngine
from data_engine import ProductionDataEngine
import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LBO-ORACLE™ Production API",
    description="Elite LBO Analysis and Prediction System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ml_engine = ProductionMLEngine()
data_engine = ProductionDataEngine()

class DealRequest(BaseModel):
    industry: str
    ttm_revenue: float
    ttm_ebitda: float
    revenue_growth: float
    ebitda_margin: float
    entry_multiple: float
    leverage_ratio: float
    hold_period: int = 5

class PredictionResponse(BaseModel):
    irr_prediction: float
    uncertainty: float
    confidence_lower: float
    confidence_upper: float
    moic_prediction: float
    recommendation: str
    risk_score: float
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Starting LBO-ORACLE™ Production API...")
    
    try:
        # Try to load existing model
        ml_engine.load_model()
        logger.info("✅ Pre-trained model loaded")
    except:
        logger.info("🧠 No pre-trained model found, will train on first request")

@app.post("/predict", response_model=PredictionResponse)
async def predict_deal(request: DealRequest):
    """Main prediction endpoint"""
    start_time = datetime.now()
    
    try:
        # Ensure model is trained
        if ml_engine.model is None:
            logger.info("Training model for first request...")
            ml_engine.train(optimize_hp=False)  # Quick training for demo
        
        # Prepare features
        features = {
            'industry': request.industry,
            'ev_ebitda_multiple': request.entry_multiple,
            'leverage_ratio': request.leverage_ratio,
            'revenue_cagr': request.revenue_growth,
            'ebitda_margin': request.ebitda_margin
        }
        
        # Make prediction
        prediction = ml_engine.predict(features)
        
        # Calculate additional metrics
        moic = (1 + prediction['irr_prediction']) ** request.hold_period
        
        # Risk assessment
        risk_score = calculate_risk_score(request, prediction['irr_prediction'])
        
        # Generate recommendation
        recommendation = generate_recommendation(
            prediction['irr_prediction'], 
            risk_score, 
            moic
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            irr_prediction=prediction['irr_prediction'],
            uncertainty=prediction['uncertainty'],
            confidence_lower=prediction['confidence_lower'],
            confidence_upper=prediction['confidence_upper'],
            moic_prediction=moic,
            recommendation=recommendation,
            risk_score=risk_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain model with latest data"""
    background_tasks.add_task(retrain_background)
    return {"message": "Model retraining started"}

async def retrain_background():
    """Background task for retraining"""
    try:
        logger.info("Starting model retraining...")
        results = ml_engine.train(optimize_hp=True)
        logger.info(f"Retraining complete: {results}")
    except Exception as e:
        logger.error(f"Retraining failed: {e}")

@app.post("/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh market data"""
    background_tasks.add_task(refresh_data_background)
    return {"message": "Data refresh started"}

async def refresh_data_background():
    """Background task for data refresh"""
    try:
        logger.info("Starting data refresh...")
        await data_engine.fetch_sp500_data()
        await data_engine.generate_synthetic_deals(100)
        logger.info("Data refresh complete")
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ml_engine.model is not None,
        "version": "1.0.0"
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_loaded": ml_engine.model is not None,
        "device": str(ml_engine.device),
        "features": [
            "EV/EBITDA Multiple", "Leverage Ratio", "Revenue Growth",
            "EBITDA Margin", "Industry", "Interaction Terms"
        ]
    }

def calculate_risk_score(request: DealRequest, predicted_irr: float) -> float:
    """Calculate comprehensive risk score"""
    base_risk = 0.15
    
    # Leverage risk
    leverage_risk = max(0, (request.leverage_ratio - 0.65) * 0.3)
    
    # Multiple risk
    multiple_risk = max(0, (request.entry_multiple - 12) * 0.02)
    
    # Growth risk
    growth_risk = max(0, (0.05 - request.revenue_growth) * 0.8)
    
    # Return concentration risk
    return_risk = max(0, (predicted_irr - 0.25) * 0.2)
    
    total_risk = base_risk + leverage_risk + multiple_risk + growth_risk + return_risk
    return min(total_risk, 0.8)

def generate_recommendation(irr: float, risk_score: float, moic: float) -> str:
    """Generate investment recommendation"""
    if irr >= 0.25 and risk_score < 0.25 and moic >= 3.0:
        return "STRONG BUY - Exceptional risk-adjusted returns"
    elif irr >= 0.20 and risk_score < 0.35:
        return "BUY - Attractive returns meeting target thresholds"
    elif irr >= 0.15 and risk_score < 0.45:
        return "CONDITIONAL - Consider at improved terms"
    elif irr >= 0.12:
        return "HOLD - Marginal returns requiring value creation"
    else:
        return "PASS - Insufficient risk-adjusted returns"

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
