import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
import wandb
from neural_architecture import LBOOracle, OptunaOptimizer
import json
from typing import Dict, List
import pickle

class FinancialDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_name: str = "microsoft/DialoGPT-medium"):
        self.db_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = self.load_and_process_data()
        self.scalers = self.fit_scalers()
        
    def load_and_process_data(self):
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            c.symbol, c.sector, c.industry, c.market_cap, c.enterprise_value, c.ev_ebitda,
            f.revenue, f.ebitda, f.net_income, f.total_debt, f.cash, f.shares_outstanding,
            f.date
        FROM companies c
        JOIN financials f ON c.symbol = f.symbol
        WHERE f.revenue > 0 AND f.ebitda > 0 AND c.ev_ebitda > 0 AND c.ev_ebitda < 50
        ORDER BY f.date DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])
        
        df['revenue_growth'] = df.groupby('symbol')['revenue'].pct_change()
        df['ebitda_margin'] = df['ebitda'] / df['revenue']
        df['debt_to_ebitda'] = df['total_debt'] / df['ebitda']
        df['ev_to_revenue'] = df['enterprise_value'] / df['revenue']
        
        numerical_features = [
            'market_cap', 'enterprise_value', 'revenue', 'ebitda', 'net_income',
            'total_debt', 'cash', 'shares_outstanding', 'revenue_growth',
            'ebitda_margin', 'debt_to_ebitda', 'ev_to_revenue'
        ]
        
        df[numerical_features] = df[numerical_features].fillna(0)
        
        synthetic_irr = self.calculate_synthetic_irr(df)
        df['target_irr'] = synthetic_irr
        
        return df.dropna()
    
    def calculate_synthetic_irr(self, df):
        base_irr = 0.15
        
        margin_factor = np.clip((df['ebitda_margin'] - 0.1) * 2, -0.1, 0.15)
        growth_factor = np.clip(df['revenue_growth'] * 0.5, -0.05, 0.1)
        leverage_factor = np.clip((5 - df['debt_to_ebitda']) * 0.02, -0.08, 0.08)
        multiple_factor = np.clip((15 - df['ev_ebitda']) * 0.005, -0.05, 0.05)
        
        sector_adjustments = {
            'Technology': 0.05,
            'Healthcare': 0.03,
            'Industrials': 0.01,
            'Consumer Discretionary': 0.0,
            'Consumer Staples': -0.01,
            'Energy': -0.02,
            'Financials': -0.01,
            'Materials': 0.0,
            'Real Estate': 0.02,
            'Communication Services': 0.02,
            'Utilities': -0.02
        }
        
        sector_factor = df['sector'].map(sector_adjustments).fillna(0)
        
        noise = np.random.normal(0, 0.02, len(df))
        
        synthetic_irr = (base_irr + margin_factor + growth_factor + 
                        leverage_factor + multiple_factor + sector_factor + noise)
        
        return np.clip(synthetic_irr, 0.05, 0.50)
    
    def fit_scalers(self):
        numerical_cols = [
            'market_cap', 'enterprise_value', 'revenue', 'ebitda', 'net_income',
            'total_debt', 'cash', 'shares_outstanding', 'revenue_growth',
            'ebitda_margin', 'debt_to_ebitda', 'ev_to_revenue'
        ]
        
        scalers = {}
        for col in numerical_cols:
            scaler = StandardScaler()
            scaler.fit(self.data[[col]])
            scalers[col] = scaler
        
        label_encoders = {}
        for col in ['sector', 'industry']:
            le = LabelEncoder()
            le.fit(self.data[col].astype(str))
            label_encoders[col] = le
        
        return {'numerical': scalers, 'categorical': label_encoders}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text_description = f"Company in {row['sector']} sector with {row['ebitda_margin']:.2%} EBITDA margin"
        text_tokens = self.tokenizer(
            text_description,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        numerical_features = []
        numerical_cols = [
            'market_cap', 'enterprise_value', 'revenue', 'ebitda', 'net_income',
            'total_debt', 'cash', 'shares_outstanding', 'revenue_growth',
            'ebitda_margin', 'debt_to_ebitda', 'ev_to_revenue'
        ]
        
        for col in numerical_cols:
            scaled_value = self.scalers['numerical'][col].transform([[row[col]]])[0][0]
            numerical_features.append(scaled_value)
        
        sector_encoded = self.scalers['categorical']['sector'].transform([str(row['sector'])])[0]
        industry_encoded = self.scalers['categorical']['industry'].transform([str(row['industry'])])[0]
        
        numerical_features.extend([sector_encoded, industry_encoded])
        
        node_features = torch.tensor([
            row['ev_ebitda'], row['ebitda_margin'], row['debt_to_ebitda'], 
            row['revenue_growth']
        ], dtype=torch.float32)
        
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        temporal_sequence = torch.tensor([
            numerical_features[-12:] if len(numerical_features) >= 12 
            else numerical_features + [0] * (12 - len(numerical_features))
        ], dtype=torch.float32).unsqueeze(0)
        
        return {
            'text_input': {
                'input_ids': text_tokens['input_ids'].squeeze(),
                'attention_mask': text_tokens['attention_mask'].squeeze()
            },
            'numerical_input': torch.tensor(numerical_features, dtype=torch.float32),
            'node_features': node_features.unsqueeze(0),
            'edge_index': edge_index,
            'temporal_sequence': temporal_sequence,
            'forecast_input': torch.tensor(numerical_features[:8], dtype=torch.float32),
            'targets': {
                'irr': torch.tensor(row['target_irr'], dtype=torch.float32)
            }
        }

class LBOTrainer:
    def __init__(self, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self.load_config(config_path) if config_path else self.default_config()
        
        wandb.init(project="lbo-oracle", config=self.config)
        
    def default_config(self):
        return {
            'node_features': 4,
            'graph_hidden_dim': 128,
            'graph_layers': 3,
            'temporal_dim': 256,
            'temporal_heads': 8,
            'forecast_input_dim': 8,
            'forecast_hidden_dim': 128,
            'forecast_output_dim': 32,
            'text_model': 'microsoft/DialoGPT-medium',
            'numerical_features': 14,
            'hidden_dim': 256,
            'num_heads': 8,
            'output_dim': 128,
            'combined_features': 512,
            'predictor_hidden': 256,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'weight_decay': 0.01
        }
    
    def load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def train(self, data_path: str):
        dataset = FinancialDataset(data_path)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        optimizer = OptunaOptimizer(LBOOracle, train_loader)
        best_params = optimizer.optimize(n_trials=50)
        
        self.config.update(best_params)
        
        model = LBOOracle(type('Config', (), self.config)()).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['num_epochs']
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_loss = self.validate_epoch(model, val_loader)
            
            scheduler.step()
            
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'scalers': dataset.scalers
                }, 'best_model.pth')
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return model
    
    def collate_fn(self, batch):
        collated = {}
        
        for key in batch[0].keys():
            if key == 'text_input':
                collated[key] = {
                    'input_ids': torch.stack([item[key]['input_ids'] for item in batch]),
                    'attention_mask': torch.stack([item[key]['attention_mask'] for item in batch])
                }
            elif key == 'targets':
                collated[key] = {
                    'irr': torch.stack([item[key]['irr'] for item in batch])
                }
            elif key in ['node_features', 'edge_index']:
                collated[key] = torch.cat([item[key] for item in batch], dim=0)
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) 
                    else {k2: v2.to(self.device) for k2, v2 in v.items()} 
                    if isinstance(v, dict) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = self.compute_loss(outputs, batch['targets'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model, val_loader):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) 
                        else {k2: v2.to(self.device) for k2, v2 in v.items()} 
                        if isinstance(v, dict) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch)
                loss = self.compute_loss(outputs, batch['targets'])
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def compute_loss(self, outputs, targets):
        mse_loss = nn.MSELoss()(outputs['irr_prediction'].squeeze(), targets['irr'])
        
        uncertainty_loss = -torch.distributions.Normal(
            outputs['irr_prediction'].squeeze(), 
            outputs['uncertainty'].squeeze()
        ).log_prob(targets['irr']).mean()
        
        return mse_loss + 0.1 * uncertainty_loss

if __name__ == "__main__":
    trainer = LBOTrainer()
    model = trainer.train('financial_data.db')
    print("Training completed successfully")
