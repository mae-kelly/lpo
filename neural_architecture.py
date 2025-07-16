import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from scipy.optimize import minimize
import optuna

class CausalGraphNet(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.layers.append(GCNConv(hidden_dim, 1))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.temporal_embedding = nn.Parameter(torch.randn(1000, d_model))
        
    def forward(self, x, temporal_mask=None):
        batch_size, seq_len, _ = x.size()
        
        temp_embed = self.temporal_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + temp_embed
        
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out(context)
        
        return output

class BayesianForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.mu_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.sigma_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x):
        mu = self.mu_layers(x)
        sigma = self.sigma_layers(x) + 1e-6
        
        return mu, sigma
    
    def sample(self, x, n_samples=100):
        mu, sigma = self.forward(x)
        
        dist = torch.distributions.Normal(mu, sigma)
        samples = dist.sample((n_samples,))
        
        return samples.mean(dim=0), samples.std(dim=0)

class MultiModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(config.text_model)
        self.numerical_encoder = nn.Sequential(
            nn.Linear(config.numerical_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, text_input, numerical_input):
        text_features = self.text_encoder(**text_input).last_hidden_state
        numerical_features = self.numerical_encoder(numerical_input)
        
        numerical_features = numerical_features.unsqueeze(1)
        
        fused_features, _ = self.fusion_layer(
            numerical_features, text_features, text_features
        )
        
        output = self.output_layer(fused_features.squeeze(1))
        return output

class LBOOracle(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.causal_graph = CausalGraphNet(
            config.node_features, 
            config.graph_hidden_dim, 
            config.graph_layers
        )
        
        self.temporal_attention = TemporalAttention(
            config.temporal_dim,
            config.temporal_heads
        )
        
        self.bayesian_forecaster = BayesianForecaster(
            config.forecast_input_dim,
            config.forecast_hidden_dim,
            config.forecast_output_dim
        )
        
        self.multimodal_encoder = MultiModalEncoder(config)
        
        self.irr_predictor = nn.Sequential(
            nn.Linear(config.combined_features, config.predictor_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.predictor_hidden, config.predictor_hidden),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden, 1)
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(config.combined_features, config.predictor_hidden),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden, 1),
            nn.Softplus()
        )
        
    def forward(self, batch):
        causal_features = self.causal_graph(
            batch['node_features'], 
            batch['edge_index']
        )
        
        temporal_features = self.temporal_attention(
            batch['temporal_sequence']
        )
        
        forecast_mu, forecast_sigma = self.bayesian_forecaster(
            batch['forecast_input']
        )
        
        multimodal_features = self.multimodal_encoder(
            batch['text_input'],
            batch['numerical_input']
        )
        
        combined_features = torch.cat([
            causal_features.flatten(1),
            temporal_features.mean(dim=1),
            forecast_mu,
            multimodal_features
        ], dim=1)
        
        irr_pred = self.irr_predictor(combined_features)
        uncertainty = self.uncertainty_estimator(combined_features)
        
        return {
            'irr_prediction': irr_pred,
            'uncertainty': uncertainty,
            'forecast_distribution': (forecast_mu, forecast_sigma),
            'attention_weights': temporal_features
        }

class OptunaOptimizer:
    def __init__(self, model_class, data_loader):
        self.model_class = model_class
        self.data_loader = data_loader
        
    def objective(self, trial):
        config = self.suggest_config(trial)
        model = self.model_class(config)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        )
        
        model.train()
        total_loss = 0
        
        for batch in self.data_loader:
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = self.compute_loss(outputs, batch['targets'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.data_loader)
    
    def suggest_config(self, trial):
        return type('Config', (), {
            'node_features': trial.suggest_int('node_features', 32, 256),
            'graph_hidden_dim': trial.suggest_int('graph_hidden_dim', 64, 512),
            'graph_layers': trial.suggest_int('graph_layers', 2, 8),
            'temporal_dim': trial.suggest_int('temporal_dim', 128, 1024),
            'temporal_heads': trial.suggest_int('temporal_heads', 4, 16),
            'forecast_input_dim': trial.suggest_int('forecast_input_dim', 32, 256),
            'forecast_hidden_dim': trial.suggest_int('forecast_hidden_dim', 64, 512),
            'forecast_output_dim': trial.suggest_int('forecast_output_dim', 16, 128),
            'combined_features': trial.suggest_int('combined_features', 256, 2048),
            'predictor_hidden': trial.suggest_int('predictor_hidden', 128, 1024),
        })()
    
    def compute_loss(self, outputs, targets):
        mse_loss = F.mse_loss(outputs['irr_prediction'], targets['irr'])
        
        uncertainty_loss = -torch.distributions.Normal(
            outputs['irr_prediction'], 
            outputs['uncertainty']
        ).log_prob(targets['irr']).mean()
        
        return mse_loss + 0.1 * uncertainty_loss
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params

