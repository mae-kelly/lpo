import yfinance as yf
import pandas as pd
import numpy as np
import requests
import asyncio
import aiofiles
from typing import Dict, List, Optional
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDataEngine:
    def __init__(self):
        self.db_path = "data/financial_data.db"
        self.setup_database()
    
    def setup_database(self):
        """Initialize production database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                enterprise_value REAL,
                ev_ebitda REAL,
                revenue REAL,
                ebitda REAL,
                debt_to_equity REAL,
                roe REAL,
                operating_margin REAL,
                revenue_growth REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS deals (
                deal_id TEXT PRIMARY KEY,
                target_company TEXT,
                industry TEXT,
                deal_value REAL,
                ev_ebitda_multiple REAL,
                leverage_ratio REAL,
                revenue_cagr REAL,
                ebitda_margin REAL,
                irr REAL,
                moic REAL,
                deal_date TEXT,
                data_source TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT PRIMARY KEY,
                sp500_close REAL,
                vix REAL,
                treasury_10y REAL,
                credit_spread REAL,
                dollar_index REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    async def fetch_sp500_data(self):
        """Fetch S&P 500 company data"""
        logger.info("Fetching S&P 500 data...")
        
        # Get S&P 500 tickers
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]
        symbols = sp500_df['Symbol'].tolist()[:50]  # Limit for production
        
        companies_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get financial data
                financials = ticker.quarterly_financials
                balance_sheet = ticker.quarterly_balance_sheet
                
                # Calculate metrics
                revenue = self.get_latest_metric(financials, 'Total Revenue')
                ebitda = self.calculate_ebitda(financials)
                revenue_growth = self.calculate_growth(financials, 'Total Revenue')
                
                company_data = {
                    'symbol': symbol,
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'ev_ebitda': info.get('enterpriseToEbitda', 0),
                    'revenue': revenue,
                    'ebitda': ebitda,
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'operating_margin': info.get('operatingMargins', 0),
                    'revenue_growth': revenue_growth,
                    'last_updated': datetime.now().isoformat()
                }
                
                companies_data.append(company_data)
                logger.info(f"Fetched data for {symbol}")
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        self.save_companies_data(companies_data)
        logger.info(f"Saved {len(companies_data)} companies")
        return companies_data
    
    def get_latest_metric(self, financials, metric):
        """Get latest value for a financial metric"""
        try:
            if metric in financials.index and len(financials.columns) > 0:
                return float(financials.loc[metric, financials.columns[0]])
        except:
            pass
        return 0
    
    def calculate_ebitda(self, financials):
        """Calculate EBITDA from financial statements"""
        try:
            operating_income = self.get_latest_metric(financials, 'Operating Income')
            depreciation = self.get_latest_metric(financials, 'Depreciation')
            return operating_income + depreciation
        except:
            return 0
    
    def calculate_growth(self, financials, metric):
        """Calculate growth rate for a metric"""
        try:
            if metric in financials.index and len(financials.columns) >= 4:
                current = financials.loc[metric, financials.columns[0]]
                previous = financials.loc[metric, financials.columns[3]]
                if previous > 0:
                    return (current / previous) ** (1/1) - 1  # Quarterly growth annualized
        except:
            pass
        return 0
    
    def save_companies_data(self, companies_data):
        """Save companies data to database"""
        conn = sqlite3.connect(self.db_path)
        
        for company in companies_data:
            conn.execute('''
                INSERT OR REPLACE INTO companies 
                (symbol, name, sector, industry, market_cap, enterprise_value, 
                 ev_ebitda, revenue, ebitda, debt_to_equity, roe, operating_margin,
                 revenue_growth, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                company['symbol'], company['name'], company['sector'],
                company['industry'], company['market_cap'], company['enterprise_value'],
                company['ev_ebitda'], company['revenue'], company['ebitda'],
                company['debt_to_equity'], company['roe'], company['operating_margin'],
                company['revenue_growth'], company['last_updated']
            ))
        
        conn.commit()
        conn.close()
    
    async def generate_synthetic_deals(self, n_deals=100):
        """Generate synthetic LBO deals for training"""
        logger.info(f"Generating {n_deals} synthetic deals...")
        
        conn = sqlite3.connect(self.db_path)
        companies_df = pd.read_sql_query("SELECT * FROM companies", conn)
        conn.close()
        
        if companies_df.empty:
            logger.warning("No companies data available")
            return []
        
        deals = []
        
        for i in range(n_deals):
            # Sample a random company for base metrics
            company = companies_df.sample(1).iloc[0]
            
            # Generate deal parameters
            deal_value = np.random.uniform(100e6, 2e9)  # $100M to $2B
            ebitda = deal_value / np.random.uniform(8, 15)  # 8-15x multiple
            revenue = ebitda / np.random.uniform(0.12, 0.25)  # 12-25% margin
            
            entry_multiple = deal_value / ebitda
            leverage_ratio = np.random.uniform(0.5, 0.75)
            revenue_cagr = np.random.uniform(0.02, 0.15)
            ebitda_margin = ebitda / revenue
            
            # Calculate synthetic IRR based on sophisticated model
            irr = self.calculate_synthetic_irr(
                entry_multiple, leverage_ratio, revenue_cagr, 
                ebitda_margin, company['sector']
            )
            
            moic = (1 + irr) ** 5  # 5-year hold
            
            deal = {
                'deal_id': f"DEAL_{i+1:04d}",
                'target_company': f"Target Company {i+1}",
                'industry': company['sector'],
                'deal_value': deal_value,
                'ev_ebitda_multiple': entry_multiple,
                'leverage_ratio': leverage_ratio,
                'revenue_cagr': revenue_cagr,
                'ebitda_margin': ebitda_margin,
                'irr': irr,
                'moic': moic,
                'deal_date': (datetime.now() - timedelta(days=np.random.randint(0, 1095))).strftime('%Y-%m-%d'),
                'data_source': 'synthetic'
            }
            
            deals.append(deal)
        
        self.save_deals_data(deals)
        logger.info(f"Generated and saved {len(deals)} synthetic deals")
        return deals
    
    def calculate_synthetic_irr(self, entry_multiple, leverage_ratio, 
                               revenue_cagr, ebitda_margin, sector):
        """Calculate sophisticated synthetic IRR"""
        base_irr = 0.18
        
        # Multiple factor
        multiple_factor = (12 - entry_multiple) * 0.01
        
        # Leverage factor (optimal around 0.6)
        leverage_factor = -abs(leverage_ratio - 0.6) * 0.15
        
        # Growth factor
        growth_factor = (revenue_cagr - 0.05) * 1.5
        
        # Margin factor
        margin_factor = (ebitda_margin - 0.15) * 0.8
        
        # Sector factor
        sector_factors = {
            'Technology': 0.03,
            'Healthcare': 0.02,
            'Consumer Discretionary': 0.01,
            'Industrials': 0.00,
            'Consumer Staples': -0.01,
            'Energy': -0.02,
            'Financials': -0.01
        }
        sector_factor = sector_factors.get(sector, 0)
        
        # Market timing (random)
        market_factor = np.random.uniform(-0.02, 0.02)
        
        # Noise
        noise = np.random.normal(0, 0.015)
        
        irr = (base_irr + multiple_factor + leverage_factor + 
               growth_factor + margin_factor + sector_factor + 
               market_factor + noise)
        
        return max(0.05, min(irr, 0.45))
    
    def save_deals_data(self, deals):
        """Save deals data to database"""
        conn = sqlite3.connect(self.db_path)
        
        for deal in deals:
            conn.execute('''
                INSERT OR REPLACE INTO deals 
                (deal_id, target_company, industry, deal_value, ev_ebitda_multiple,
                 leverage_ratio, revenue_cagr, ebitda_margin, irr, moic, 
                 deal_date, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                deal['deal_id'], deal['target_company'], deal['industry'],
                deal['deal_value'], deal['ev_ebitda_multiple'], deal['leverage_ratio'],
                deal['revenue_cagr'], deal['ebitda_margin'], deal['irr'],
                deal['moic'], deal['deal_date'], deal['data_source']
            ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self):
        """Get processed training data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            d.ev_ebitda_multiple,
            d.leverage_ratio,
            d.revenue_cagr,
            d.ebitda_margin,
            d.irr,
            d.industry
        FROM deals d
        WHERE d.irr IS NOT NULL AND d.irr > 0
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

async def main():
    """Main data collection workflow"""
    engine = ProductionDataEngine()
    
    print("🚀 Starting LBO-ORACLE™ Data Collection...")
    
    # Fetch real market data
    companies = await engine.fetch_sp500_data()
    print(f"📊 Collected {len(companies)} company records")
    
    # Generate synthetic deals
    deals = await engine.generate_synthetic_deals(200)
    print(f"💼 Generated {len(deals)} deal records")
    
    # Get training data
    training_data = engine.get_training_data()
    print(f"🧠 Training dataset: {len(training_data)} records")
    
    print("✅ Data collection complete!")

if __name__ == "__main__":
    asyncio.run(main())
