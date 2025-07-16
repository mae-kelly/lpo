import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sqlite3
import pickle
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class MassiveLBOMLTrainer:
    """
    ML trainer for massive LBO dataset
    Handles 10,000+ transactions across multiple data sources
    """
    
    def __init__(self, db_path='massive_lbo_database.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_and_process_massive_dataset(self) -> pd.DataFrame:
        """Load and process the massive LBO dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT target_company, acquirer_firm, deal_value_parsed, sector,
                       investment_date, data_source, confidence_score, raw_text
                FROM lbo_transactions 
                WHERE confidence_score > 0.5
                ORDER BY confidence_score DESC
            ''', conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} high-confidence transactions")
            
            # Process and enrich the dataset
            df = self.enrich_dataset(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich dataset with derived features and estimates"""
        
        # Parse investment year
        df['investment_year'] = pd.to_datetime(df['investment_date'], errors='coerce').dt.year
        
        # Create deal size categories
        df['deal_size_category'] = pd.cut(
            df['deal_value_parsed'], 
            bins=[0, 100e6, 500e6, 2e9, float('inf')],
            labels=['Small_Cap', 'Mid_Market', 'Large_Cap', 'Mega_Deal']
        )
        
        # Extract PE firm type
        df['firm_type'] = df['acquirer_firm'].apply(self.classify_pe_firm)
        
        # Estimate financial metrics
        df['estimated_revenue'] = df.apply(self.estimate_revenue, axis=1)
        df['estimated_ebitda'] = df.apply(self.estimate_ebitda, axis=1)
        df['estimated_ev_multiple'] = df.apply(self.estimate_ev_multiple, axis=1)
        df['estimated_leverage'] = df.apply(self.estimate_leverage, axis=1)
        df['estimated_irr'] = df.apply(self.estimate_irr, axis=1)
        
        # Market timing features
        df['market_cycle'] = df['investment_year'].apply(self.classify_market_cycle)
        
        # Data quality features
        df['source_quality'] = df['data_source'].apply(self.rate_source_quality)
        
        return df
    
    def classify_pe_firm(self, firm_name: str) -> str:
        """Classify PE firm by type/tier"""
        if pd.isna(firm_name):
            return 'Unknown'
        
        firm_lower = firm_name.lower()
        
        # Mega funds
        mega_funds = ['kkr', 'blackstone', 'apollo', 'carlyle', 'tpg']
        if any(fund in firm_lower for fund in mega_funds):
            return 'Mega_Fund'
        
        # Growth equity
        growth_funds = ['general atlantic', 'silver lake', 'vista', 'thoma bravo']
        if any(fund in firm_lower for fund in growth_funds):
            return 'Growth_Equity'
        
        # Middle market
        if any(word in firm_lower for word in ['capital', 'partners', 'equity']):
            return 'Middle_Market'
        
        return 'Other'
    
    def estimate_revenue(self, row) -> float:
        """Estimate revenue based on deal value and sector"""
        deal_value = row.get('deal_value_parsed', 0)
        sector = row.get('sector', 'Unknown')
        
        if deal_value == 0:
            return 0
        
        # Revenue multiples by sector (EV/Revenue)
        ev_revenue_multiples = {
            'Technology': 4.0,
            'Healthcare': 3.0,
            'Financial Services': 2.5,
            'Consumer': 1.8,
            'Industrial': 1.5,
            'Energy': 1.2,
            'Unknown': 2.0
        }
        
        multiple = ev_revenue_multiples.get(sector, 2.0)
        estimated_revenue = deal_value / multiple
        
        return estimated_revenue
    
    def estimate_ebitda(self, row) -> float:
        """Estimate EBITDA based on revenue and sector margins"""
        estimated_revenue = row.get('estimated_revenue', 0)
        sector = row.get('sector', 'Unknown')
        
        # EBITDA margins by sector
        ebitda_margins = {
            'Technology': 0.25,
            'Healthcare': 0.20,
            'Financial Services': 0.30,
            'Consumer': 0.15,
            'Industrial': 0.12,
            'Energy': 0.15,
            'Unknown': 0.18
        }
        
        margin = ebitda_margins.get(sector, 0.18)
        return estimated_revenue * margin
    
    def estimate_ev_multiple(self, row) -> float:
        """Estimate EV/EBITDA multiple"""
        deal_value = row.get('deal_value_parsed', 0)
        estimated_ebitda = row.get('estimated_ebitda', 0)
        sector = row.get('sector', 'Unknown')
        investment_year = row.get('investment_year', 2020)
        
        if estimated_ebitda <= 0:
            # Fallback to sector averages
            sector_multiples = {
                'Technology': 14.0,
                'Healthcare': 12.0,
                'Financial Services': 11.0,
                'Consumer': 10.0,
                'Industrial': 9.0,
                'Energy': 8.0,
                'Unknown': 11.0
            }
            base_multiple = sector_multiples.get(sector, 11.0)
        else:
            base_multiple = deal_value / estimated_ebitda
        
        # Market cycle adjustment
        if investment_year and investment_year >= 2020:
            base_multiple *= 1.15  # Higher multiples in recent years
        elif investment_year and investment_year <= 2010:
            base_multiple *= 0.85  # Lower multiples in earlier years
        
        return max(5.0, min(25.0, base_multiple))
    
    def estimate_leverage(self, row) -> float:
        """Estimate leverage ratio"""
        sector = row.get('sector', 'Unknown')
        deal_size = row.get('deal_value_parsed', 100e6)
        investment_year = row.get('investment_year', 2020)
        
        # Base leverage by sector
        base_leverage = {
            'Technology': 0.40,
            'Healthcare': 0.55,
            'Financial Services': 0.30,
            'Consumer': 0.60,
            'Industrial': 0.65,
            'Energy': 0.70,
            'Unknown': 0.60
        }
        
        leverage = base_leverage.get(sector, 0.60)
        
        # Size adjustment
        if deal_size > 1e9:
            leverage += 0.05  # Larger deals can support more leverage
        elif deal_size < 100e6:
            leverage -= 0.05
        
        # Time adjustment
        if investment_year and investment_year >= 2020:
            leverage += 0.03  # Higher leverage in recent years
        
        return max(0.20, min(0.85, leverage))
    
    def estimate_irr(self, row) -> float:
        """Estimate IRR based on deal characteristics"""
        ev_multiple = row.get('estimated_ev_multiple', 11.0)
        leverage = row.get('estimated_leverage', 0.60)
        sector = row.get('sector', 'Unknown')
        investment_year = row.get('investment_year', 2020)
        firm_type = row.get('firm_type', 'Middle_Market')
        confidence = row.get('confidence_score', 0.7)
        
        # Base IRR expectation
        base_irr = 0.18
        
        # Multiple factor (lower entry multiples = higher returns)
        multiple_factor = (12.0 - ev_multiple) * 0.006
        
        # Leverage factor (optimal around 55-65%)
        optimal_leverage = 0.60
        leverage_factor = -abs(leverage - optimal_leverage) * 0.08
        
        # Sector risk premiums
        sector_premiums = {
            'Technology': 0.025,
            'Healthcare': 0.015,
            'Financial Services': 0.005,
            'Consumer': 0.000,
            'Industrial': -0.005,
            'Energy': -0.015,
            'Unknown': 0.000
        }
        sector_factor = sector_premiums.get(sector, 0.0)
        
        # Firm type factor
        firm_factors = {
            'Mega_Fund': -0.01,  # Lower returns for mega funds
            'Growth_Equity': 0.01,
            'Middle_Market': 0.005,
            'Other': 0.000
        }
        firm_factor = firm_factors.get(firm_type, 0.0)
        
        # Vintage year factor
        if investment_year:
            if investment_year <= 2008 or investment_year >= 2020:
                vintage_factor = -0.02  # Challenging vintages
            elif 2009 <= investment_year <= 2015:
                vintage_factor = 0.02  # Good vintages
            else:
                vintage_factor = 0.00
        else:
            vintage_factor = 0.00
        
        # Confidence factor
        confidence_factor = (confidence - 0.7) * 0.01
        
        # Random noise for realism
        noise = np.random.normal(0, 0.012)
        
        irr = (base_irr + multiple_factor + leverage_factor + sector_factor + 
               firm_factor + vintage_factor + confidence_factor + noise)
        
        return max(0.05, min(0.50, irr))
    
    def classify_market_cycle(self, year) -> str:
        """Classify market cycle based on investment year"""
        if pd.isna(year):
            return 'Unknown'
        
        if year <= 2008:
            return 'Pre_Crisis'
        elif 2009 <= year <= 2015:
            return 'Post_Crisis'
        elif 2016 <= year <= 2019:
            return 'Expansion'
        elif year >= 2020:
            return 'COVID_Era'
        else:
            return 'Unknown'
    
    def rate_source_quality(self, source: str) -> float:
        """Rate data source quality"""
        source_ratings = {
            'pe_firm_portfolio': 0.9,
            'sec_edgar_filing': 0.85,
            'wikipedia_table': 0.8,
            'business_wire_pr': 0.7,
            'pe_firm_text_extraction': 0.6
        }
        
        return source_ratings.get(source, 0.5)
    
    def prepare_features_for_training(self, df: pd.DataFrame):
        """Prepare features for ML training"""
        if df.empty:
            logger.error("No data available for training")
            return None, None
        
        # Encode categorical variables
        categorical_columns = ['sector', 'firm_type', 'market_cycle', 'deal_size_category']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.encoders[col] = le
        
        # Select features for training
        feature_columns = [
            'deal_value_parsed', 'estimated_ev_multiple', 'estimated_leverage',
            'investment_year', 'confidence_score', 'source_quality',
            'sector_encoded', 'firm_type_encoded', 'market_cycle_encoded',
            'deal_size_category_encoded'
        ]
        
        # Handle missing values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        X = df[feature_columns].values
        y = df['estimated_irr'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        logger.info(f"Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def train_on_massive_dataset(self):
        """Train ML models on massive dataset"""
        print("🧠 Training ML models on massive LBO dataset...")
        
        df = self.load_and_process_massive_dataset()
        
        if df.empty:
            logger.error("No data available for training")
            return None
        
        print(f"📊 Dataset size: {len(df):,} transactions")
        
        X, y = self.prepare_features_for_training(df)
        
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model, 'mse': mse, 'mae': mae, 'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda k: results[k]['mse'])
        self.models['best'] = results[best_model_name]['model']
        
        # Feature importance analysis
        if hasattr(self.models['best'], 'feature_importances_'):
            self.analyze_feature_importance()
        
        # Save models
        self.save_models()
        
        print(f"✅ Training complete! Best model: {best_model_name}")
        print(f"📈 Final R²: {results[best_model_name]['r2']:.4f}")
        
        return results
    
    def analyze_feature_importance(self):
        """Analyze and report feature importance"""
        if 'best' not in self.models:
            return
        
        feature_names = [
            'deal_value', 'ev_multiple', 'leverage', 'investment_year',
            'confidence', 'source_quality', 'sector', 'firm_type',
            'market_cycle', 'deal_size_category'
        ]
        
        importances = self.models['best'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS:")
        print(importance_df.to_string(index=False))
    
    def save_models(self):
        """Save trained models and preprocessors"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open('massive_lbo_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Models saved successfully")

if __name__ == "__main__":
    trainer = MassiveLBOMLTrainer()
    trainer.train_on_massive_dataset()
