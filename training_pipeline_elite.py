import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sqlite3
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import wandb
from neural_architecture_elite import LBOOracleElite, EliteOptunaOptimizer, EliteTrainingPipeline
import json
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from datetime import datetime, timedelta
import warnings
import gc
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EliteConfig:
    concept_dim: int = 768
    concept_heads: int = 12
    rule_depth: int = 6
    market_dim: int = 256
    temporal_dim: int = 128
    max_horizon: int = 40
    company_features: int = 75
    industry_features: int = 35
    macro_features: int = 45
    deal_features: int = 60
    hidden_dim: int = 1024
    graph_heads: int = 16
    graph_layers: int = 8
    edge_dim: int = 64
    input_dim: int = 1024
    risk_factors: int = 25
    fusion_heads: int = 12
    output_dim: int = 3
    explanation_vocab_size: int = 2500
    memory_slots: int = 1000
    memory_dim: int = 512
    memory_heads: int = 8
    experience_dim: int = 256
    max_tasks: int = 50
    task_dim: int = 128
    num_experts: int = 8
    expert_hidden_dim: int = 512
    gate_hidden_dim: int = 256
    load_balancing_weight: float = 0.01
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    total_steps: int = 10000
    batch_size: int = 32
    num_epochs: int = 100

class EliteFinancialDataset(Dataset):
    def __init__(self, data_path: str, mode: str = 'train', config: EliteConfig = None):
        self.data_path = data_path
        self.mode = mode
        self.config = config or EliteConfig()
        
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        self.data = self.load_and_engineer_features()
        self.scalers = self.fit_advanced_scalers()
        
        self.synthetic_targets = self.generate_sophisticated_targets()
        
        logger.info(f"Loaded {len(self.data)} samples for {mode} mode")

    def load_and_engineer_features(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.data_path)
        
        query = '''
        WITH company_features AS (
            SELECT 
                c.*,
                LAG(c.market_cap, 4) OVER (PARTITION BY c.symbol ORDER BY c.last_updated) as market_cap_lag,
                LAG(c.ev_ebitda, 4) OVER (PARTITION BY c.symbol ORDER BY c.last_updated) as ev_ebitda_lag,
                LAG(c.revenue_growth_3y, 4) OVER (PARTITION BY c.symbol ORDER BY c.last_updated) as revenue_growth_lag
            FROM companies_comprehensive c
            WHERE c.market_cap > 100000000 AND c.ev_ebitda > 0 AND c.ev_ebitda < 25
        ),
        deal_features AS (
            SELECT 
                d.*,
                AVG(d.ev_ebitda_multiple) OVER (PARTITION BY d.sector) as sector_avg_multiple,
                STDDEV(d.ev_ebitda_multiple) OVER (PARTITION BY d.sector) as sector_std_multiple,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY d.ev_ebitda_multiple) OVER (PARTITION BY d.sector) as sector_q25_multiple,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY d.ev_ebitda_multiple) OVER (PARTITION BY d.sector) as sector_q75_multiple
            FROM ma_transactions_comprehensive d
            WHERE d.deal_value > 10000000 AND d.ev_ebitda_multiple > 0 AND d.ev_ebitda_multiple < 30
        ),
        macro_pivot AS (
            SELECT 
                date,
                MAX(CASE WHEN indicator_name = 'gdp_growth' THEN value END) as gdp_growth,
                MAX(CASE WHEN indicator_name = 'inflation_cpi' THEN value END) as inflation_cpi,
                MAX(CASE WHEN indicator_name = 'fed_funds_rate' THEN value END) as fed_funds_rate,
                MAX(CASE WHEN indicator_name = 'treasury_10y' THEN value END) as treasury_10y,
                MAX(CASE WHEN indicator_name = 'treasury_2y' THEN value END) as treasury_2y,
                MAX(CASE WHEN indicator_name = 'vix' THEN value END) as vix,
                MAX(CASE WHEN indicator_name = 'high_yield_spread' THEN value END) as high_yield_spread,
                MAX(CASE WHEN indicator_name = 'term_spread' THEN value END) as term_spread,
                MAX(CASE WHEN indicator_name = 'unemployment' THEN value END) as unemployment,
                MAX(CASE WHEN indicator_name = 'consumer_confidence' THEN value END) as consumer_confidence,
                MAX(CASE WHEN indicator_name = 'ism_manufacturing' THEN value END) as ism_manufacturing
            FROM macro_indicators_detailed 
            WHERE date >= date('now', '-5 years')
            GROUP BY date
        ),
        benchmarks AS (
            SELECT 
                sector,
                metric_name,
                median as sector_median,
                percentile_75 - percentile_25 as sector_iqr
            FROM industry_benchmarks
        )
        SELECT 
            cf.*,
            df.sector_avg_multiple, df.sector_std_multiple, df.sector_q25_multiple, df.sector_q75_multiple,
            mp.*,
            b1.sector_median as ev_ebitda_sector_median,
            b1.sector_iqr as ev_ebitda_sector_iqr,
            b2.sector_median as roe_sector_median,
            b2.sector_iqr as roe_sector_iqr,
            b3.sector_median as revenue_growth_sector_median,
            b3.sector_iqr as revenue_growth_sector_iqr
        FROM company_features cf
        LEFT JOIN deal_features df ON cf.sector = df.sector
        LEFT JOIN macro_pivot mp ON date(cf.last_updated) = mp.date
        LEFT JOIN benchmarks b1 ON cf.sector = b1.sector AND b1.metric_name = 'ev_ebitda'
        LEFT JOIN benchmarks b2 ON cf.sector = b2.sector AND b2.metric_name = 'roe'
        LEFT JOIN benchmarks b3 ON cf.sector = b3.sector AND b3.metric_name = 'revenue_growth_3y'
        WHERE cf.symbol IS NOT NULL 
        AND mp.gdp_growth IS NOT NULL
        ORDER BY cf.last_updated DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df = self.engineer_advanced_features(df)
        df = df.dropna(subset=['ev_ebitda', 'revenue_growth_3y', 'operating_margin'])
        
        return df

    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['momentum_score'] = (
            0.3 * np.clip((df['revenue_growth_3y'] - df['revenue_growth_sector_median']) / (df['revenue_growth_sector_iqr'] + 1e-6), -3, 3) +
            0.3 * np.clip((df['operating_margin'] - 0.15) / 0.1, -3, 3) +
            0.2 * np.clip((df['roic'] - 0.12) / 0.08, -3, 3) +
            0.2 * np.clip((1 / df['ev_ebitda'] - 1 / df['ev_ebitda_sector_median']) / (1 / df['ev_ebitda_sector_iqr'] + 1e-6), -3, 3)
        )
        
        df['quality_score'] = (
            0.25 * np.clip((df['roe'] - df['roe_sector_median']) / (df['roe_sector_iqr'] + 1e-6), -3, 3) +
            0.25 * np.clip((df['roic'] - 0.12) / 0.08, -3, 3) +
            0.25 * np.clip((df['operating_margin'] - df['operating_margin'].median()) / df['operating_margin'].std(), -3, 3) +
            0.25 * np.clip((df['debt_to_equity'] - df['debt_to_equity'].median()) / df['debt_to_equity'].std() * -1, -3, 3)
        )
        
        df['market_timing_score'] = (
            0.3 * np.clip((20 - df['vix']) / 10, -2, 2) +
            0.2 * np.clip((df['gdp_growth'] - 2.0) / 2.0, -2, 2) +
            0.2 * np.clip((60 - df['consumer_confidence']) / 20, -2, 2) +
            0.15 * np.clip((df['term_spread'] - 1.0) / 1.5, -2, 2) +
            0.15 * np.clip((8 - df['high_yield_spread']) / 4, -2, 2)
        )
        
        df['valuation_attractiveness'] = (
            0.4 * np.clip((df['ev_ebitda_sector_median'] - df['ev_ebitda']) / df['ev_ebitda_sector_iqr'], -3, 3) +
            0.3 * np.clip((df['price_to_book'].median() - df['price_to_book']) / df['price_to_book'].std(), -3, 3) +
            0.3 * np.clip((df['ev_revenue'].median() - df['ev_revenue']) / df['ev_revenue'].std(), -3, 3)
        )
        
        df['operational_leverage'] = df['operating_margin'] / (df['revenue_growth_3y'] + 0.01)
        df['financial_leverage'] = df['debt_to_equity'] / (df['roic'] + 0.01)
        df['growth_quality'] = df['revenue_growth_3y'] * df['operating_margin']
        df['efficiency_ratio'] = df['asset_turnover'] * df['operating_margin']
        
        df['sector_relative_ev_ebitda'] = (df['ev_ebitda'] - df['ev_ebitda_sector_median']) / (df['ev_ebitda_sector_iqr'] + 1e-6)
        df['sector_relative_growth'] = (df['revenue_growth_3y'] - df['revenue_growth_sector_median']) / (df['revenue_growth_sector_iqr'] + 1e-6)
        
        for lag in [1, 2, 3]:
            df[f'momentum_score_lag_{lag}'] = df.groupby('sector')['momentum_score'].shift(lag)
            df[f'quality_score_lag_{lag}'] = df.groupby('sector')['quality_score'].shift(lag)
        
        df['volatility_adjusted_return'] = df['relative_strength'] / (df['volatility_90d'] + 0.01)
        df['risk_adjusted_growth'] = df['revenue_growth_3y'] / (df['beta'] + 0.5)
        
        return df

    def generate_sophisticated_targets(self) -> Dict[str, np.ndarray]:
        base_irr = 0.18
        
        momentum_impact = np.clip(self.data['momentum_score'] * 0.08, -0.12, 0.15)
        quality_impact = np.clip(self.data['quality_score'] * 0.06, -0.08, 0.10)
        timing_impact = np.clip(self.data['market_timing_score'] * 0.05, -0.08, 0.08)
        valuation_impact = np.clip(self.data['valuation_attractiveness'] * 0.07, -0.10, 0.12)
        
        leverage_penalty = np.where(
            self.data['debt_to_equity'] > self.data['debt_to_equity'].quantile(0.75),
            -0.04 * np.log1p(self.data['debt_to_equity'] / self.data['debt_to_equity'].quantile(0.75)),
            0
        )
        
        sector_adjustments = {
            'Technology': 0.04, 'Healthcare': 0.02, 'Consumer Discretionary': 0.01,
            'Industrials': 0.00, 'Financials': -0.01, 'Energy': -0.02,
            'Consumer Staples': -0.01, 'Materials': 0.00, 'Real Estate': 0.01,
            'Communication Services': 0.02, 'Utilities': -0.02
        }
        sector_impact = self.data['sector'].map(sector_adjustments).fillna(0)
        
        macro_adjustment = (
            0.003 * (self.data['gdp_growth'] - 2.5) +
            -0.002 * (self.data['inflation_cpi'] - 2.0) +
            -0.001 * (self.data['fed_funds_rate'] - 2.0) +
            -0.0005 * (self.data['vix'] - 20)
        )
        
        interaction_effects = (
            0.02 * self.data['momentum_score'] * self.data['quality_score'] +
            0.015 * self.data['valuation_attractiveness'] * self.data['market_timing_score']
        )
        
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, len(self.data))
        
        synthetic_irr = (
            base_irr + momentum_impact + quality_impact + timing_impact + 
            valuation_impact + leverage_penalty + sector_impact + 
            macro_adjustment + interaction_effects + noise
        )
        
        synthetic_irr = np.clip(synthetic_irr, 0.02, 0.60)
        
        synthetic_multiple = (
            self.data['ev_ebitda_sector_median'] * 
            (1 + 0.3 * self.data['momentum_score'] + 0.2 * self.data['quality_score'] + 
             0.1 * self.data['market_timing_score'] + 0.15 * np.random.normal(0, 0.1, len(self.data)))
        )
        synthetic_multiple = np.clip(synthetic_multiple, 4.0, 25.0)
        
        risk_factors = np.maximum(
            0.05,
            0.12 - 0.03 * self.data['quality_score'] + 0.02 * self.data['financial_leverage'] + 
            0.01 * self.data['volatility_90d'].fillna(0.3) + 0.005 * self.data['beta'].fillna(1.2)
        )
        
        return {
            'irr': synthetic_irr,
            'exit_multiple': synthetic_multiple,
            'risk_factor': risk_factors
        }

    def fit_advanced_scalers(self) -> Dict:
        numerical_features = [
            'market_cap', 'enterprise_value', 'ev_ebitda', 'ev_revenue', 'price_to_book',
            'roe', 'roic', 'debt_to_equity', 'current_ratio', 'operating_margin', 'net_margin',
            'revenue_growth_3y', 'ebitda_growth_3y', 'beta', 'volatility_90d', 'relative_strength',
            'gdp_growth', 'inflation_cpi', 'fed_funds_rate', 'treasury_10y', 'vix',
            'momentum_score', 'quality_score', 'market_timing_score', 'valuation_attractiveness',
            'operational_leverage', 'financial_leverage', 'growth_quality', 'efficiency_ratio'
        ]
        
        scalers = {}
        
        for feature in numerical_features:
            if feature in self.data.columns:
                feature_data = self.data[feature].fillna(self.data[feature].median()).values.reshape(-1, 1)
                
                if feature in ['market_cap', 'enterprise_value']:
                    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
                else:
                    scaler = RobustScaler()
                
                scaler.fit(feature_data)
                scalers[feature] = scaler
        
        categorical_encoders = {}
        for col in ['sector', 'industry']:
            if col in self.data.columns:
                unique_values = self.data[col].fillna('Unknown').unique()
                encoder_dict = {val: idx for idx, val in enumerate(unique_values)}
                categorical_encoders[col] = encoder_dict
        
        return {
            'numerical': scalers,
            'categorical': categorical_encoders
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        company_description = self.create_rich_company_description(row)
        
        text_encoding = self.tokenizer(
            company_description,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        numerical_features = self.extract_scaled_features(row)
        
        graph_data = self.construct_graph_representation(row, idx)
        
        temporal_sequence = self.create_temporal_sequence(row)
        
        task_id = self.determine_task_id(row)
        
        targets = {
            'irr': torch.tensor(self.synthetic_targets['irr'][idx], dtype=torch.float32),
            'exit_multiple': torch.tensor(self.synthetic_targets['exit_multiple'][idx], dtype=torch.float32),
            'risk_factor': torch.tensor(self.synthetic_targets['risk_factor'][idx], dtype=torch.float32)
        }
        
        return {
            'input_features': torch.tensor(numerical_features, dtype=torch.float32),
            'financial_context': {
                'concept_ids': torch.randint(0, 1000, (20,), dtype=torch.long),
                'text_encoding': text_encoding
            },
            'market_context': torch.tensor(numerical_features[:self.config.market_dim], dtype=torch.float32),
            'temporal_context': temporal_sequence,
            'graph_data': graph_data,
            'task_id': torch.tensor(task_id, dtype=torch.long),
            'targets': targets,
            'experience_sequence': self.create_experience_sequence(row)
        }

    def create_rich_company_description(self, row) -> str:
        sector = row.get('sector', 'Unknown')
        industry = row.get('industry', 'Unknown')
        margin = row.get('operating_margin', 0) * 100
        growth = row.get('revenue_growth_3y', 0) * 100
        multiple = row.get('ev_ebitda', 0)
        
        description = (
            f"A {sector} company in the {industry} subsector with "
            f"{margin:.1f}% operating margins and {growth:.1f}% revenue growth. "
            f"Currently trading at {multiple:.1f}x EBITDA multiple. "
        )
        
        if row.get('momentum_score', 0) > 1:
            description += "Strong operational momentum and market positioning. "
        elif row.get('momentum_score', 0) < -1:
            description += "Facing operational challenges and market headwinds. "
        
        if row.get('quality_score', 0) > 1:
            description += "High-quality business with strong returns and efficiency metrics."
        elif row.get('quality_score', 0) < -1:
            description += "Lower-quality metrics with room for operational improvement."
        
        return description

    def extract_scaled_features(self, row) -> List[float]:
        features = []
        
        numerical_features = [
            'market_cap', 'enterprise_value', 'ev_ebitda', 'ev_revenue', 'price_to_book',
            'roe', 'roic', 'debt_to_equity', 'current_ratio', 'operating_margin', 'net_margin',
            'revenue_growth_3y', 'ebitda_growth_3y', 'beta', 'volatility_90d', 'relative_strength',
            'gdp_growth', 'inflation_cpi', 'fed_funds_rate', 'treasury_10y', 'vix',
            'momentum_score', 'quality_score', 'market_timing_score', 'valuation_attractiveness',
            'operational_leverage', 'financial_leverage', 'growth_quality', 'efficiency_ratio'
        ]
        
        for feature in numerical_features:
            if feature in self.scalers['numerical']:
                value = row.get(feature, 0)
                if pd.isna(value):
                    value = 0
                scaled_value = self.scalers['numerical'][feature].transform([[value]])[0][0]
                features.append(scaled_value)
            else:
                features.append(0.0)
        
        sector_encoded = self.scalers['categorical']['sector'].get(str(row.get('sector', 'Unknown')), 0)
        industry_encoded = self.scalers['categorical']['industry'].get(str(row.get('industry', 'Unknown')), 0)
        
        features.extend([sector_encoded / 20.0, industry_encoded / 50.0])
        
        while len(features) < self.config.input_dim:
            features.append(0.0)
        
        return features[:self.config.input_dim]

    def construct_graph_representation(self, row, idx) -> Dict:
        node_features = {
            'company': torch.tensor([[
                row.get('momentum_score', 0), row.get('quality_score', 0),
                row.get('valuation_attractiveness', 0), row.get('operational_leverage', 0)
            ]], dtype=torch.float32),
            'industry': torch.tensor([[
                row.get('sector_avg_multiple', 10), row.get('sector_std_multiple', 2),
                row.get('sector_q25_multiple', 8), row.get('sector_q75_multiple', 12)
            ]], dtype=torch.float32),
            'macro': torch.tensor([[
                row.get('gdp_growth', 2), row.get('inflation_cpi', 2),
                row.get('fed_funds_rate', 2), row.get('vix', 20)
            ]], dtype=torch.float32),
            'deal': torch.tensor([[
                row.get('ev_ebitda', 10), row.get('leverage_ratio', 0.6),
                row.get('revenue_growth_3y', 0.05), row.get('operating_margin', 0.15)
            ]], dtype=torch.float32)
        }
        
        edge_index = torch.tensor([
            [0, 1, 2, 3, 1, 2],
            [1, 2, 3, 0, 0, 1]
        ], dtype=torch.long)
        
        edge_attr = torch.randn(edge_index.size(1), self.config.edge_dim)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_counts': {'company': 1, 'industry': 1, 'macro': 1, 'deal': 1}
        }

    def create_temporal_sequence(self, row) -> torch.Tensor:
        sequence_length = 24
        
        base_features = [
            row.get('revenue_growth_3y', 0.05),
            row.get('operating_margin', 0.15),
            row.get('gdp_growth', 2.0),
            row.get('vix', 20.0)
        ]
        
        temporal_sequence = []
        for t in range(sequence_length):
            noise = np.random.normal(0, 0.1, len(base_features))
            trend = np.array([0.001 * t, -0.001 * t, 0.002 * t, -0.5 * t])
            features_t = np.array(base_features) + noise + trend
            temporal_sequence.append(features_t)
        
        return torch.tensor(temporal_sequence, dtype=torch.float32).unsqueeze(0)

    def create_experience_sequence(self, row) -> torch.Tensor:
        sequence_length = 12
        experience_dim = self.config.experience_dim
        
        base_experience = [
            row.get('momentum_score', 0), row.get('quality_score', 0),
            row.get('market_timing_score', 0), row.get('valuation_attractiveness', 0)
        ]
        
        experience_sequence = []
        for t in range(sequence_length):
            experience_t = base_experience + list(np.random.normal(0, 0.1, experience_dim - len(base_experience)))
            experience_sequence.append(experience_t)
        
        return torch.tensor(experience_sequence, dtype=torch.float32).unsqueeze(0)

    def determine_task_id(self, row) -> int:
        sector = row.get('sector', 'Unknown')
        
        sector_mapping = {
            'Technology': 0, 'Healthcare': 1, 'Financials': 2, 'Consumer Discretionary': 3,
            'Industrials': 4, 'Consumer Staples': 5, 'Energy': 6, 'Materials': 7,
            'Real Estate': 8, 'Communication Services': 9, 'Utilities': 10
        }
        
        return sector_mapping.get(sector, 11)

class EliteTrainingCoordinator:
    def __init__(self, config: EliteConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        wandb.init(
            project="lbo-oracle-elite",
            config=vars(config),
            name=f"elite_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Initialized Elite Training on {self.device}")

    def prepare_elite_datasets(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        full_dataset = EliteFinancialDataset(data_path, mode='full', config=self.config)
        
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

    def hyperparameter_optimization(self, train_loader: DataLoader, val_loader: DataLoader) -> EliteConfig:
        def objective(trial):
            config = EliteConfig(
                concept_dim=trial.suggest_categorical('concept_dim', [512, 768, 1024]),
                concept_heads=trial.suggest_categorical('concept_heads', [8, 12, 16]),
                hidden_dim=trial.suggest_categorical('hidden_dim', [512, 1024, 1536]),
                graph_layers=trial.suggest_int('graph_layers', 4, 12),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            )
            
            pipeline = EliteTrainingPipeline(config)
            
            best_val_loss = float('inf')
            for epoch in range(5):
                train_loss = pipeline.train_epoch(train_loader, epoch)
                val_metrics = pipeline.validate(val_loader)
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                
                trial.report(val_metrics['val_loss'], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_loss
        
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5)
        )
        
        study.optimize(objective, n_trials=20, timeout=3600)
        
        best_params = study.best_params
        
        optimized_config = EliteConfig(**{**vars(self.config), **best_params})
        
        logger.info(f"Optimization complete. Best validation loss: {study.best_value:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return optimized_config

    def train_elite_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                         test_loader: DataLoader) -> Dict:
        
        optimized_config = self.hyperparameter_optimization(train_loader, val_loader)
        
        pipeline = EliteTrainingPipeline(optimized_config)
        
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10
        
        for epoch in range(optimized_config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{optimized_config.num_epochs}")
            
            train_loss = pipeline.train_epoch(train_loader, epoch)
            val_metrics = pipeline.validate(val_loader)
            
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'learning_rate': pipeline.scheduler.get_last_lr()[0]
            })
            
            logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['val_loss']:.6f}")
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience = 0
                
                pipeline.save_elite_model(f'elite_model_best_epoch_{epoch}.pth')
                logger.info(f"New best model saved at epoch {epoch + 1}")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        test_metrics = pipeline.validate(test_loader)
        
        wandb.log({
            'test_loss': test_metrics['val_loss'],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae']
        })
        
        final_results = {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'optimized_config': optimized_config,
            'model_pipeline': pipeline
        }
        
        return final_results

async def main():
    print("🚀 Starting Elite LBO-Oracle Training Pipeline...")
    
    config = EliteConfig()
    coordinator = EliteTrainingCoordinator(config)
    
    print("📊 Loading and preparing elite datasets...")
    train_loader, val_loader, test_loader = coordinator.prepare_elite_datasets('elite_financial_data.db')
    
    print("🧠 Training elite neural architecture...")
    results = coordinator.train_elite_model(train_loader, val_loader, test_loader)
    
    print("✅ Elite training completed successfully!")
    print(f"📈 Best validation loss: {results['best_val_loss']:.6f}")
    print(f"🎯 Test MSE: {results['test_metrics']['mse']:.6f}")
    print(f"📊 Test MAE: {results['test_metrics']['mae']:.6f}")
    
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
