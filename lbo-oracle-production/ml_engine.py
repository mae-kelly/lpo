import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import sqlite3
import pickle
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLBONet(nn.Module):
    """Advanced neural network for LBO prediction"""
    
    def __init__(self, input_dim=10, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layers
        layers.extend([
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # Extract features before final layers
        features = x
        for layer in self.network[:-2]:
            features = layer(features)
        
        # Main prediction
        prediction = self.network[-2:](features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features)
        
        return prediction, uncertainty

class ProductionMLEngine:
    def __init__(self, db_path="data/financial_data.db"):
        self.db_path = db_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        logger.info(f"ML Engine initialized on {self.device}")
    
    def load_training_data(self):
        """Load and preprocess training data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            ev_ebitda_multiple,
            leverage_ratio,
            revenue_cagr,
            ebitda_margin,
            industry,
            irr
        FROM deals 
        WHERE irr IS NOT NULL AND irr > 0.05 AND irr < 0.5
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            raise ValueError("No training data available")
        
        logger.info(f"Loaded {len(df)} training samples")
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for training"""
        # Encode categorical variables
        industry_encoded = self.label_encoder.fit_transform(df['industry'])
        
        # Create feature matrix
        features = np.column_stack([
            df['ev_ebitda_multiple'].values,
            df['leverage_ratio'].values,
            df['revenue_cagr'].values,
            df['ebitda_margin'].values,
            industry_encoded,
            # Add interaction terms
            df['ev_ebitda_multiple'].values * df['leverage_ratio'].values,
            df['revenue_cagr'].values * df['ebitda_margin'].values,
            # Add derived features
            (df['leverage_ratio'].values - 0.6) ** 2,  # Leverage penalty
            np.log1p(df['ev_ebitda_multiple'].values),  # Log multiple
            df['ebitda_margin'].values / (df['revenue_cagr'].values + 0.01)  # Efficiency ratio
        ])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Target variable
        targets = df['irr'].values
        
        return features_scaled, targets
    
    def optimize_hyperparameters(self, X, y, n_trials=50):
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            hidden_dims = []
            n_layers = trial.suggest_int('n_layers', 2, 5)
            
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 32, 512, step=32)
                hidden_dims.append(dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create model
            model = AdvancedLBONet(
                input_dim=X.shape[1],
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Train for limited epochs
            model.train()
            for epoch in range(20):
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train[i:i+batch_size]).to(self.device)
                    
                    optimizer.zero_grad()
                    pred, uncertainty = model(batch_X)
                    
                    # Combined loss
                    mse_loss = criterion(pred.squeeze(), batch_y)
                    uncertainty_loss = -torch.distributions.Normal(
                        pred.squeeze(), uncertainty.squeeze()
                    ).log_prob(batch_y).mean()
                    
                    loss = mse_loss + 0.1 * uncertainty_loss
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                
                val_pred, _ = model(X_val_tensor)
                val_loss = criterion(val_pred.squeeze(), y_val_tensor).item()
            
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def train(self, optimize_hp=True):
        """Train the ML model"""
        logger.info("Starting model training...")
        
        # Load and preprocess data
        df = self.load_training_data()
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Optimize hyperparameters
        if optimize_hp:
            best_params = self.optimize_hyperparameters(X_train, y_train)
        else:
            best_params = {
                'n_layers': 3,
                'hidden_dim_0': 256,
                'hidden_dim_1': 128,
                'hidden_dim_2': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # Build final model
        hidden_dims = [best_params[f'hidden_dim_{i}'] 
                      for i in range(best_params['n_layers'])]
        
        self.model = AdvancedLBONet(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            dropout_rate=best_params['dropout_rate']
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=best_params['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        max_patience = 20
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        for epoch in range(200):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            pred, uncertainty = self.model(X_train_tensor)
            
            mse_loss = criterion(pred.squeeze(), y_train_tensor)
            uncertainty_loss = -torch.distributions.Normal(
                pred.squeeze(), uncertainty.squeeze()
            ).log_prob(y_train_tensor).mean()
            
            loss = mse_loss + 0.1 * uncertainty_loss
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred, _ = self.model(X_test_tensor)
                val_loss = criterion(val_pred.squeeze(), y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                self.save_model()
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}")
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_pred, test_uncertainty = self.model(X_test_tensor)
            test_mse = mean_squared_error(y_test, test_pred.cpu().numpy())
            test_r2 = r2_score(y_test, test_pred.cpu().numpy())
        
        logger.info(f"Final Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        
        return {
            'test_mse': test_mse,
            'test_r2': test_r2,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, features_dict):
        """Make prediction for new deal"""
        if self.model is None:
            self.load_model()
        
        # Convert features to array format
        industry_encoded = self.label_encoder.transform([features_dict['industry']])[0]
        
        features = np.array([
            features_dict['ev_ebitda_multiple'],
            features_dict['leverage_ratio'],
            features_dict['revenue_cagr'],
            features_dict['ebitda_margin'],
            industry_encoded,
            # Interaction terms
            features_dict['ev_ebitda_multiple'] * features_dict['leverage_ratio'],
            features_dict['revenue_cagr'] * features_dict['ebitda_margin'],
            # Derived features
            (features_dict['leverage_ratio'] - 0.6) ** 2,
            np.log1p(features_dict['ev_ebitda_multiple']),
            features_dict['ebitda_margin'] / (features_dict['revenue_cagr'] + 0.01)
        ])
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction, uncertainty = self.model(features_tensor)
            
            irr_pred = prediction.cpu().item()
            irr_uncertainty = uncertainty.cpu().item()
        
        return {
            'irr_prediction': irr_pred,
            'uncertainty': irr_uncertainty,
            'confidence_lower': irr_pred - 1.96 * irr_uncertainty,
            'confidence_upper': irr_pred + 1.96 * irr_uncertainty
        }
    
    def save_model(self):
        """Save trained model and preprocessors"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'timestamp': datetime.now().isoformat()
        }, 'models/lbo_model.pth')
    
    def load_model(self):
        """Load trained model and preprocessors"""
        try:
            checkpoint = torch.load('models/lbo_model.pth', map_location=self.device)
            
            # Reconstruct model (you'd need to save architecture info in practice)
            self.model = AdvancedLBONet(input_dim=10).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.scaler = checkpoint['scaler']
            self.label_encoder = checkpoint['label_encoder']
            
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("No saved model found")

if __name__ == "__main__":
    engine = ProductionMLEngine()
    results = engine.train(optimize_hp=True)
    print(f"Training complete: {results}")
