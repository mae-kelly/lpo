import yfinance as yf
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from typing import Dict, List
import json
import time
from fredapi import Fred
import quandl
from alpha_vantage.fundamentaldata import FundamentalData
import sqlite3
from datetime import datetime, timedelta
import os

class MarketDataPipeline:
    def __init__(self):
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY', ''))
        self.av_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.quandl_key = os.getenv('QUANDL_KEY', '')
        self.db_path = 'financial_data.db'
        self.init_database()
    
    def init_database(self):
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
                last_updated TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS financials (
                symbol TEXT,
                date TEXT,
                revenue REAL,
                ebitda REAL,
                net_income REAL,
                total_debt REAL,
                cash REAL,
                shares_outstanding REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                deal_id TEXT PRIMARY KEY,
                target_company TEXT,
                acquirer TEXT,
                deal_value REAL,
                ev_ebitda_multiple REAL,
                leverage_ratio REAL,
                sector TEXT,
                deal_date TEXT,
                exit_date TEXT,
                exit_multiple REAL,
                irr REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    async def fetch_sp500_universe(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500 = tables[0]
        symbols = sp500['Symbol'].tolist()
        
        companies_data = []
        async with aiohttp.ClientSession() as session:
            for symbol in symbols[:100]:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    companies_data.append({
                        'symbol': symbol,
                        'name': info.get('longName', ''),
                        'sector': info.get('sector', ''),
                        'industry': info.get('industry', ''),
                        'market_cap': info.get('marketCap', 0),
                        'enterprise_value': info.get('enterpriseValue', 0),
                        'ev_ebitda': info.get('enterpriseToEbitda', 0)
                    })
                    await asyncio.sleep(0.1)
                except:
                    continue
        
        self.save_companies_data(companies_data)
        return companies_data
    
    def fetch_financial_statements(self, symbols: List[str]):
        financials_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                income_stmt = ticker.quarterly_financials
                balance_sheet = ticker.quarterly_balance_sheet
                cash_flow = ticker.quarterly_cashflow
                
                for quarter in income_stmt.columns:
                    try:
                        revenue = income_stmt.loc['Total Revenue', quarter] if 'Total Revenue' in income_stmt.index else 0
                        ebitda = self.calculate_ebitda(income_stmt, quarter)
                        net_income = income_stmt.loc['Net Income', quarter] if 'Net Income' in income_stmt.index else 0
                        
                        total_debt = balance_sheet.loc['Total Debt', quarter] if 'Total Debt' in balance_sheet.index else 0
                        cash = balance_sheet.loc['Cash', quarter] if 'Cash' in balance_sheet.index else 0
                        shares = balance_sheet.loc['Share Issued', quarter] if 'Share Issued' in balance_sheet.index else 0
                        
                        financials_data.append({
                            'symbol': symbol,
                            'date': quarter.strftime('%Y-%m-%d'),
                            'revenue': float(revenue) if pd.notna(revenue) else 0,
                            'ebitda': float(ebitda) if pd.notna(ebitda) else 0,
                            'net_income': float(net_income) if pd.notna(net_income) else 0,
                            'total_debt': float(total_debt) if pd.notna(total_debt) else 0,
                            'cash': float(cash) if pd.notna(cash) else 0,
                            'shares_outstanding': float(shares) if pd.notna(shares) else 0
                        })
                    except:
                        continue
                
                time.sleep(0.2)
            except:
                continue
        
        self.save_financials_data(financials_data)
        return financials_data
    
    def calculate_ebitda(self, income_stmt, quarter):
        try:
            operating_income = income_stmt.loc['Operating Income', quarter] if 'Operating Income' in income_stmt.index else 0
            depreciation = income_stmt.loc['Depreciation', quarter] if 'Depreciation' in income_stmt.index else 0
            amortization = income_stmt.loc['Amortization', quarter] if 'Amortization' in income_stmt.index else 0
            
            ebitda = operating_income + depreciation + amortization
            return ebitda if pd.notna(ebitda) else 0
        except:
            return 0
    
    async def scrape_ma_transactions(self):
        urls = [
            "https://www.pitchbook.com/news/articles/private-equity-deal-multiples",
            "https://www.bain.com/insights/topics/private-equity-report/",
            "https://www.bcg.com/industries/private-equity",
        ]
        
        transactions = []
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(url) as response:
                        html = await response.text()
                        transactions.extend(self.parse_transaction_data(html))
                except:
                    continue
        
        return transactions
    
    def parse_transaction_data(self, html):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        transactions = []
        return transactions
    
    def fetch_macro_indicators(self):
        indicators = {
            'gdp': 'GDP',
            'inflation': 'CPIAUCSL',
            'unemployment': 'UNRATE', 
            'fed_funds': 'FEDFUNDS',
            'treasury_10y': 'GS10',
            'corporate_spreads': 'BAA10Y',
            'vix': 'VIXCLS'
        }
        
        macro_data = {}
        for name, series_id in indicators.items():
            try:
                data = self.fred.get_series(series_id, start='2000-01-01')
                macro_data[name] = data.to_dict()
            except:
                macro_data[name] = {}
        
        return macro_data
    
    def save_companies_data(self, data):
        conn = sqlite3.connect(self.db_path)
        for company in data:
            conn.execute('''
                INSERT OR REPLACE INTO companies 
                (symbol, name, sector, industry, market_cap, enterprise_value, ev_ebitda, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                company['symbol'], company['name'], company['sector'], 
                company['industry'], company['market_cap'], 
                company['enterprise_value'], company['ev_ebitda'],
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
    
    def save_financials_data(self, data):
        conn = sqlite3.connect(self.db_path)
        for record in data:
            conn.execute('''
                INSERT OR REPLACE INTO financials 
                (symbol, date, revenue, ebitda, net_income, total_debt, cash, shares_outstanding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['symbol'], record['date'], record['revenue'],
                record['ebitda'], record['net_income'], record['total_debt'],
                record['cash'], record['shares_outstanding']
            ))
        conn.commit()
        conn.close()
    
    def get_training_dataset(self):
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT c.symbol, c.sector, c.industry, c.ev_ebitda,
               f.revenue, f.ebitda, f.net_income, f.total_debt, f.cash
        FROM companies c
        JOIN financials f ON c.symbol = f.symbol
        WHERE f.revenue > 0 AND f.ebitda > 0
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

async def main():
    pipeline = MarketDataPipeline()
    
    print("Fetching S&P 500 universe...")
    companies = await pipeline.fetch_sp500_universe()
    
    print("Fetching financial statements...")
    symbols = [c['symbol'] for c in companies]
    financials = pipeline.fetch_financial_statements(symbols[:50])
    
    print("Fetching macro indicators...")
    macro = pipeline.fetch_macro_indicators()
    
    print("Generating training dataset...")
    dataset = pipeline.get_training_dataset()
    
    print(f"Dataset shape: {dataset.shape}")
    print("Data acquisition complete.")

if __name__ == "__main__":
    asyncio.run(main())
