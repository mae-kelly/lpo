from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import numpy as np
from neural_architecture import LBOOracle
from training_pipeline import FinancialDataset
import pickle
from typing import Dict, List, Optional
import uvicorn

app = FastAPI(title="LBO-ORACLE™ API", version="1.0.0")

class DealInput(BaseModel):
    industry: str
    ttm_revenue: float
    ttm_ebitda: float
    revenue_growth: float
    ebitda_margin: float
    entry_multiple: float
    leverage_ratio: float
    hold_period: int

class ModelResponse(BaseModel):
    irr_prediction: float
    uncertainty: float
    confidence_interval: List[float]
    risk_factors: List[str]
    recommendation: str

class LBOOracleAPI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scalers = None
        self.load_model()
    
    def load_model(self):
        try:
            checkpoint = torch.load('best_model.pth', map_location=self.device)
            config = type('Config', (), checkpoint['config'])()
            
            self.model = LBOOracle(config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.scalers = checkpoint['scalers']
            
        except FileNotFoundError:
            raise RuntimeError("Model not found. Please train the model first.")
    
    def preprocess_input(self, deal_input: DealInput) -> Dict:
        market_cap = deal_input.ttm_revenue * 2.5
        enterprise_value = deal_input.ttm_ebitda * deal_input.entry_multiple
        debt_to_ebitda = enterprise_value * deal_input.leverage_ratio / deal_input.ttm_ebitda
        ev_to_revenue = enterprise_value / deal_input.ttm_revenue
        
        numerical_features = [
            market_cap, enterprise_value, deal_input.ttm_revenue, 
            deal_input.ttm_ebitda, deal_input.ttm_ebitda * 0.7,
            enterprise_value * deal_input.leverage_ratio, 
            deal_input.ttm_revenue * 0.1, market_cap / 1000000,
            deal_input.revenue_growth, deal_input.ebitda_margin,
            debt_to_ebitda, ev_to_revenue
        ]
        
        for i, (col, scaler) in enumerate(self.scalers['numerical'].items()):
            if i < len(numerical_features):
                numerical_features[i] = scaler.transform([[numerical_features[i]]])[0][0]
        
        sector_encoded = self.scalers['categorical']['sector'].transform([deal_input.industry])[0]
        industry_encoded = self.scalers['categorical']['industry'].transform([deal_input.industry])[0]
        
        numerical_features.extend([sector_encoded, industry_encoded])
        
        text_description = f"Company in {deal_input.industry} sector with {deal_input.ebitda_margin:.2%} EBITDA margin"
        
        return {
            'text_description': text_description,
            'numerical_features': numerical_features,
            'node_features': [deal_input.entry_multiple, deal_input.ebitda_margin, 
                            debt_to_ebitda, deal_input.revenue_growth]
        }
    
    def predict(self, deal_input: DealInput) -> ModelResponse:
        processed_input = self.preprocess_input(deal_input)
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        text_tokens = tokenizer(
            processed_input['text_description'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        batch = {
            'text_input': {
                'input_ids': text_tokens['input_ids'].to(self.device),
                'attention_mask': text_tokens['attention_mask'].to(self.device)
            },
            'numerical_input': torch.tensor([processed_input['numerical_features']], 
                                          dtype=torch.float32).to(self.device),
            'node_features': torch.tensor([processed_input['node_features']], 
                                        dtype=torch.float32).to(self.device),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], 
                                     dtype=torch.long).to(self.device),
            'temporal_sequence': torch.tensor([processed_input['numerical_features'][:12]], 
                                            dtype=torch.float32).unsqueeze(0).to(self.device),
            'forecast_input': torch.tensor([processed_input['numerical_features'][:8]], 
                                         dtype=torch.float32).to(self.device)
        }
        
        with torch.no_grad():
            outputs = self.model(batch)
            
            irr_pred = outputs['irr_prediction'].cpu().item()
            uncertainty = outputs['uncertainty'].cpu().item()
            
            confidence_interval = [
                irr_pred - 1.96 * uncertainty,
                irr_pred + 1.96 * uncertainty
            ]
            
            risk_factors = self.assess_risk_factors(deal_input, irr_pred)
            recommendation = self.generate_recommendation(deal_input, irr_pred, uncertainty)
            
            return ModelResponse(
                irr_prediction=irr_pred,
                uncertainty=uncertainty,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                recommendation=recommendation
            )
    
    def assess_risk_factors(self, deal_input: DealInput, predicted_irr: float) -> List[str]:
        risk_factors = []
        
        if deal_input.leverage_ratio > 0.65:
            risk_factors.append("High leverage ratio increases financial risk")
        
        if deal_input.entry_multiple > 12:
            risk_factors.append("High entry multiple limits return potential")
        
        if deal_input.revenue_growth < 0.03:
            risk_factors.append("Low growth profile constrains value creation")
        
        if deal_input.ebitda_margin < 0.15:
            risk_factors.append("Below-average margins indicate operational challenges")
        
        if predicted_irr < 0.20:
            risk_factors.append("Projected returns below target threshold")
        
        return risk_factors
    
    def generate_recommendation(self, deal_input: DealInput, predicted_irr: float, 
                              uncertainty: float) -> str:
        if predicted_irr >= 0.25 and uncertainty < 0.05:
            return "STRONG BUY - Attractive risk-adjusted returns with high confidence"
        elif predicted_irr >= 0.20 and uncertainty < 0.08:
            return "BUY - Meets target returns with acceptable risk profile"
        elif predicted_irr >= 0.15:
            return "HOLD - Consider at lower valuation or improved terms"
        else:
            return "PASS - Insufficient returns for risk profile"

oracle_api = LBOOracleAPI()

@app.post("/predict", response_model=ModelResponse)
async def predict_deal(deal_input: DealInput):
    try:
        result = oracle_api.predict(deal_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": oracle_api.model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
