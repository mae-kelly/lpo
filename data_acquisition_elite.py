import yfinance as yf
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import json
import time
from fredapi import Fred
import quandl
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import sqlite3
from datetime import datetime, timedelta
import os
from sec_edgar_api import EdgarClient
import PyPDF2
import re
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DealRecord:
    target_company: str
    acquirer: str
    deal_value: float
    ev_ebitda_multiple: float
    leverage_ratio: float
    sector: str
    deal_date: str
    exit_date: Optional[str]
    exit_multiple: Optional[float]
    irr: Optional[float]
    revenue_cagr: float
    ebitda_margin: float
    deal_type: str
    source: str

class EliteDataPipeline:
    def __init__(self):
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY', ''))
        self.av_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.quandl_key = os.getenv('QUANDL_KEY', '')
        self.edgar = EdgarClient(user_agent="LBO-Oracle research@example.com")
        self.db_path = 'elite_financial_data.db'
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.init_advanced_database()
        
    def init_advanced_database(self):
        conn = sqlite3.connect(self.db_path)
        
        tables = [
            '''CREATE TABLE IF NOT EXISTS companies_comprehensive (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                enterprise_value REAL,
                ev_ebitda REAL,
                ev_revenue REAL,
                price_to_book REAL,
                roe REAL,
                roic REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                quick_ratio REAL,
                asset_turnover REAL,
                inventory_turnover REAL,
                receivables_turnover REAL,
                gross_margin REAL,
                operating_margin REAL,
                net_margin REAL,
                revenue_growth_3y REAL,
                revenue_growth_5y REAL,
                ebitda_growth_3y REAL,
                earnings_growth_3y REAL,
                beta REAL,
                volatility_30d REAL,
                volatility_90d REAL,
                relative_strength REAL,
                analyst_coverage INTEGER,
                analyst_rating_avg REAL,
                price_target_avg REAL,
                institutional_ownership REAL,
                insider_ownership REAL,
                short_interest REAL,
                country TEXT,
                exchange TEXT,
                currency TEXT,
                employees INTEGER,
                founded_year INTEGER,
                last_updated TIMESTAMP
            )''',
            
            '''CREATE TABLE IF NOT EXISTS financials_detailed (
                symbol TEXT,
                date TEXT,
                period_type TEXT,
                revenue REAL,
                gross_profit REAL,
                operating_income REAL,
                ebitda REAL,
                ebit REAL,
                net_income REAL,
                eps_basic REAL,
                eps_diluted REAL,
                shares_basic REAL,
                shares_diluted REAL,
                total_assets REAL,
                current_assets REAL,
                cash_and_equivalents REAL,
                receivables REAL,
                inventory REAL,
                ppe_net REAL,
                intangible_assets REAL,
                goodwill REAL,
                total_liabilities REAL,
                current_liabilities REAL,
                short_term_debt REAL,
                long_term_debt REAL,
                total_debt REAL,
                shareholders_equity REAL,
                retained_earnings REAL,
                operating_cash_flow REAL,
                capex REAL,
                free_cash_flow REAL,
                dividends_paid REAL,
                stock_repurchases REAL,
                debt_issuance REAL,
                debt_repayment REAL,
                working_capital REAL,
                PRIMARY KEY (symbol, date, period_type)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS ma_transactions_comprehensive (
                deal_id TEXT PRIMARY KEY,
                target_company TEXT,
                target_ticker TEXT,
                acquirer TEXT,
                acquirer_ticker TEXT,
                deal_value REAL,
                enterprise_value REAL,
                ev_ebitda_multiple REAL,
                ev_revenue_multiple REAL,
                price_to_book REAL,
                premium_to_market REAL,
                leverage_ratio REAL,
                debt_financing REAL,
                equity_financing REAL,
                sector TEXT,
                subsector TEXT,
                deal_type TEXT,
                transaction_structure TEXT,
                deal_rationale TEXT,
                synergies_expected REAL,
                deal_date TEXT,
                announcement_date TEXT,
                completion_date TEXT,
                exit_date TEXT,
                exit_multiple REAL,
                exit_ev_ebitda REAL,
                irr REAL,
                moic REAL,
                revenue_ttm REAL,
                ebitda_ttm REAL,
                revenue_cagr_3y REAL,
                ebitda_cagr_3y REAL,
                ebitda_margin REAL,
                roic REAL,
                debt_to_ebitda REAL,
                geographic_region TEXT,
                currency TEXT,
                advisor_target TEXT,
                advisor_acquirer TEXT,
                data_source TEXT,
                confidence_score REAL,
                last_updated TIMESTAMP
            )''',
            
            '''CREATE TABLE IF NOT EXISTS macro_indicators_detailed (
                date TEXT,
                indicator_name TEXT,
                value REAL,
                category TEXT,
                frequency TEXT,
                source TEXT,
                PRIMARY KEY (date, indicator_name)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS industry_benchmarks (
                sector TEXT,
                subsector TEXT,
                metric_name TEXT,
                percentile_10 REAL,
                percentile_25 REAL,
                median REAL,
                percentile_75 REAL,
                percentile_90 REAL,
                sample_size INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (sector, subsector, metric_name)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS deal_performance_tracking (
                deal_id TEXT,
                measurement_date TEXT,
                revenue_actual REAL,
                ebitda_actual REAL,
                debt_balance REAL,
                equity_value REAL,
                performance_vs_plan REAL,
                key_metrics TEXT,
                PRIMARY KEY (deal_id, measurement_date)
            )'''
        ]
        
        for table_sql in tables:
            conn.execute(table_sql)
        
        conn.commit()
        conn.close()

    async def fetch_comprehensive_company_data(self, symbols: List[str]) -> List[Dict]:
        companies_data = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self.fetch_single_company_comprehensive(session, symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    companies_data.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error fetching company data: {result}")
        
        return companies_data

    async def fetch_single_company_comprehensive(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            
            info = ticker.info
            financials = ticker.quarterly_financials
            balance_sheet = ticker.quarterly_balance_sheet
            cash_flow = ticker.quarterly_cashflow
            
            history = ticker.history(period="2y")
            
            returns_30d = history['Close'].pct_change().tail(30).std() * np.sqrt(252) if len(history) >= 30 else None
            returns_90d = history['Close'].pct_change().tail(90).std() * np.sqrt(252) if len(history) >= 90 else None
            
            beta = info.get('beta', None)
            if beta is None and len(history) >= 252:
                market_returns = yf.download('^GSPC', period='2y')['Close'].pct_change()
                stock_returns = history['Close'].pct_change()
                if len(market_returns) > 0 and len(stock_returns) > 0:
                    aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
                    if len(aligned_data) > 50:
                        beta = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1] / np.var(aligned_data.iloc[:, 1])
            
            relative_strength = None
            if len(history) >= 252:
                stock_1y_return = (history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) if len(history) >= 252 else None
                try:
                    spy_data = yf.download('^GSPC', period='1y')
                    market_1y_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) if len(spy_data) > 0 else None
                    if stock_1y_return is not None and market_1y_return is not None:
                        relative_strength = stock_1y_return - market_1y_return
                except:
                    pass
            
            revenue_growth_3y = self.calculate_cagr_from_financials(financials, 'Total Revenue', 3) if not financials.empty else None
            revenue_growth_5y = self.calculate_cagr_from_financials(financials, 'Total Revenue', 5) if not financials.empty else None
            ebitda_growth_3y = self.calculate_ebitda_cagr(financials, 3) if not financials.empty else None
            
            roic = self.calculate_roic(financials, balance_sheet) if not financials.empty and not balance_sheet.empty else None
            
            company_data = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                'price_to_book': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'roic': roic,
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'asset_turnover': info.get('assetTurnover', 0),
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'net_margin': info.get('profitMargins', 0),
                'revenue_growth_3y': revenue_growth_3y,
                'revenue_growth_5y': revenue_growth_5y,
                'ebitda_growth_3y': ebitda_growth_3y,
                'beta': beta,
                'volatility_30d': returns_30d,
                'volatility_90d': returns_90d,
                'relative_strength': relative_strength,
                'analyst_coverage': info.get('numberOfAnalystOpinions', 0),
                'analyst_rating_avg': info.get('recommendationMean', 0),
                'price_target_avg': info.get('targetMeanPrice', 0),
                'institutional_ownership': info.get('heldByInstitutions', 0),
                'insider_ownership': info.get('heldByInsiders', 0),
                'short_interest': info.get('shortPercentOfFloat', 0),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', ''),
                'employees': info.get('fullTimeEmployees', 0),
            }
            
            await asyncio.sleep(0.1)
            return company_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {}

    def calculate_cagr_from_financials(self, financials: pd.DataFrame, metric: str, years: int) -> Optional[float]:
        try:
            if metric not in financials.index or len(financials.columns) < years:
                return None
            
            metric_data = financials.loc[metric].dropna()
            if len(metric_data) < years:
                return None
            
            start_value = metric_data.iloc[-years]
            end_value = metric_data.iloc[-1]
            
            if start_value <= 0 or end_value <= 0:
                return None
            
            cagr = (end_value / start_value) ** (1 / years) - 1
            return cagr
        except:
            return None

    def calculate_ebitda_cagr(self, financials: pd.DataFrame, years: int) -> Optional[float]:
        try:
            ebitda_data = []
            for quarter in financials.columns:
                operating_income = financials.loc['Operating Income', quarter] if 'Operating Income' in financials.index else 0
                depreciation = financials.loc['Depreciation', quarter] if 'Depreciation' in financials.index else 0
                amortization = financials.loc['Amortization', quarter] if 'Amortization' in financials.index else 0
                
                ebitda = operating_income + depreciation + amortization
                ebitda_data.append(ebitda)
            
            if len(ebitda_data) < years:
                return None
            
            start_ebitda = ebitda_data[-years]
            end_ebitda = ebitda_data[-1]
            
            if start_ebitda <= 0 or end_ebitda <= 0:
                return None
            
            cagr = (end_ebitda / start_ebitda) ** (1 / years) - 1
            return cagr
        except:
            return None

    def calculate_roic(self, financials: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[float]:
        try:
            if financials.empty or balance_sheet.empty:
                return None
            
            latest_quarter = financials.columns[0]
            
            net_income = financials.loc['Net Income', latest_quarter] if 'Net Income' in financials.index else 0
            interest_expense = financials.loc['Interest Expense', latest_quarter] if 'Interest Expense' in financials.index else 0
            tax_rate = 0.25
            
            nopat = net_income + (interest_expense * (1 - tax_rate))
            
            total_debt = balance_sheet.loc['Total Debt', latest_quarter] if 'Total Debt' in balance_sheet.index else 0
            shareholders_equity = balance_sheet.loc['Stockholders Equity', latest_quarter] if 'Stockholders Equity' in balance_sheet.index else 0
            
            invested_capital = total_debt + shareholders_equity
            
            if invested_capital <= 0:
                return None
            
            roic = nopat / invested_capital
            return roic
        except:
            return None

    async def scrape_ma_database_comprehensive(self) -> List[DealRecord]:
        sources = [
            self.scrape_pitchbook_data,
            self.scrape_preqin_data,
            self.scrape_mergermarket_data,
            self.scrape_bloomberg_ma_data,
            self.scrape_thomson_reuters_data,
        ]
        
        all_deals = []
        for source_func in sources:
            try:
                deals = await source_func()
                all_deals.extend(deals)
                logger.info(f"Collected {len(deals)} deals from {source_func.__name__}")
            except Exception as e:
                logger.error(f"Error scraping {source_func.__name__}: {e}")
        
        return all_deals

    async def scrape_pitchbook_data(self) -> List[DealRecord]:
        deals = []
        base_url = "https://pitchbook.com/news/articles"
        
        try:
            async with aiohttp.ClientSession() as session:
                urls = [
                    f"{base_url}/private-equity-deal-multiples",
                    f"{base_url}/leveraged-buyout-trends",
                    f"{base_url}/middle-market-pe-deals",
                ]
                
                for url in urls:
                    try:
                        async with session.get(url) as response:
                            html = await response.text()
                            parsed_deals = self.parse_pitchbook_articles(html)
                            deals.extend(parsed_deals)
                    except Exception as e:
                        logger.error(f"Error fetching {url}: {e}")
        except Exception as e:
            logger.error(f"Error in pitchbook scraping: {e}")
        
        return deals

    def parse_pitchbook_articles(self, html: str) -> List[DealRecord]:
        soup = BeautifulSoup(html, 'html.parser')
        deals = []
        
        deal_patterns = [
            r'(\w+(?:\s+\w+)*)\s+acquired.*?(\$[\d,.]+(?:\s*(?:million|billion|M|B))?)',
            r'(\w+(?:\s+\w+)*)\s+sold.*?(\d+\.?\d*x).*?EBITDA',
            r'(\d+\.?\d*x).*?multiple.*?(\w+(?:\s+\w+)*)',
        ]
        
        text_content = soup.get_text()
        
        for pattern in deal_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                try:
                    deal = DealRecord(
                        target_company=match[0] if len(match) > 0 else '',
                        acquirer='',
                        deal_value=self.parse_deal_value(match[1]) if len(match) > 1 else 0,
                        ev_ebitda_multiple=self.parse_multiple(match[1]) if 'x' in str(match[1]) else 0,
                        leverage_ratio=0.6,
                        sector='',
                        deal_date=datetime.now().strftime('%Y-%m-%d'),
                        exit_date=None,
                        exit_multiple=None,
                        irr=None,
                        revenue_cagr=0.05,
                        ebitda_margin=0.15,
                        deal_type='LBO',
                        source='PitchBook'
                    )
                    deals.append(deal)
                except:
                    continue
        
        return deals

    def parse_deal_value(self, value_str: str) -> float:
        try:
            value_str = re.sub(r'[^\d.,]', '', value_str)
            value_str = value_str.replace(',', '')
            
            if 'billion' in value_str.lower() or 'B' in value_str:
                return float(value_str) * 1e9
            elif 'million' in value_str.lower() or 'M' in value_str:
                return float(value_str) * 1e6
            else:
                return float(value_str)
        except:
            return 0

    def parse_multiple(self, multiple_str: str) -> float:
        try:
            multiple_str = re.sub(r'[^\d.]', '', multiple_str)
            return float(multiple_str)
        except:
            return 0

    async def scrape_preqin_data(self) -> List[DealRecord]:
        deals = []
        return deals

    async def scrape_mergermarket_data(self) -> List[DealRecord]:
        deals = []
        return deals

    async def scrape_bloomberg_ma_data(self) -> List[DealRecord]:
        deals = []
        return deals

    async def scrape_thomson_reuters_data(self) -> List[DealRecord]:
        deals = []
        return deals

    def fetch_comprehensive_macro_data(self) -> Dict[str, pd.Series]:
        indicators = {
            'gdp_growth': 'GDP',
            'gdp_per_capita': 'GDPPC',
            'inflation_cpi': 'CPIAUCSL',
            'inflation_core': 'CPILFESL',
            'unemployment': 'UNRATE',
            'labor_participation': 'CIVPART',
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_1y': 'GS1',
            'treasury_2y': 'GS2',
            'treasury_5y': 'GS5',
            'treasury_10y': 'GS10',
            'treasury_30y': 'GS30',
            'corporate_aaa': 'AAA',
            'corporate_baa': 'BAA',
            'high_yield_spread': 'BAMLH0A0HYM2',
            'term_spread': 'T10Y2Y',
            'ted_spread': 'TEDRATE',
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXBGS',
            'oil_price': 'DCOILWTICO',
            'consumer_confidence': 'UMCSENT',
            'ism_manufacturing': 'NAPM',
            'ism_services': 'NAPMSI',
            'housing_starts': 'HOUST',
            'building_permits': 'PERMIT',
            'retail_sales': 'RSAFS',
            'industrial_production': 'INDPRO',
            'capacity_utilization': 'TCU',
            'nonfarm_payrolls': 'PAYEMS',
            'average_hourly_earnings': 'AHETPI',
            'personal_income': 'PI',
            'personal_spending': 'PCE',
            'personal_saving_rate': 'PSAVERT',
            'money_supply_m2': 'M2SL',
            'bank_credit': 'TOTBKCR',
            'commercial_loans': 'BUSLOANS',
            'real_estate_loans': 'REALLN',
            'credit_card_delinquency': 'DRCCLACBS',
            'mortgage_delinquency': 'DRSFRMACBS',
        }
        
        macro_data = {}
        for name, series_id in indicators.items():
            try:
                data = self.fred.get_series(series_id, start='2000-01-01')
                macro_data[name] = data
                logger.info(f"Fetched {len(data)} observations for {name}")
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                macro_data[name] = pd.Series(dtype=float)
        
        return macro_data

    def calculate_industry_benchmarks(self) -> Dict[str, Dict]:
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT c.sector, c.industry, c.ev_ebitda, c.ev_revenue, c.roe, c.roic,
               c.debt_to_equity, c.operating_margin, c.net_margin, c.revenue_growth_3y,
               c.ebitda_growth_3y, c.beta, c.volatility_90d
        FROM companies_comprehensive c
        WHERE c.market_cap > 100000000 AND c.ev_ebitda > 0 AND c.ev_ebitda < 50
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        benchmarks = {}
        
        for sector in df['sector'].unique():
            if pd.isna(sector) or sector == '':
                continue
                
            sector_data = df[df['sector'] == sector]
            
            if len(sector_data) < 5:
                continue
            
            sector_benchmarks = {}
            
            metrics = ['ev_ebitda', 'ev_revenue', 'roe', 'roic', 'debt_to_equity',
                      'operating_margin', 'net_margin', 'revenue_growth_3y',
                      'ebitda_growth_3y', 'beta', 'volatility_90d']
            
            for metric in metrics:
                if metric in sector_data.columns:
                    metric_values = sector_data[metric].dropna()
                    if len(metric_values) >= 5:
                        sector_benchmarks[metric] = {
                            'percentile_10': np.percentile(metric_values, 10),
                            'percentile_25': np.percentile(metric_values, 25),
                            'median': np.percentile(metric_values, 50),
                            'percentile_75': np.percentile(metric_values, 75),
                            'percentile_90': np.percentile(metric_values, 90),
                            'sample_size': len(metric_values)
                        }
            
            benchmarks[sector] = sector_benchmarks
        
        return benchmarks

    def save_comprehensive_data(self, companies_data: List[Dict], deals_data: List[DealRecord], 
                               macro_data: Dict[str, pd.Series], benchmarks: Dict[str, Dict]):
        conn = sqlite3.connect(self.db_path)
        
        for company in companies_data:
            if not company:
                continue
            
            conn.execute('''
                INSERT OR REPLACE INTO companies_comprehensive 
                (symbol, name, sector, industry, market_cap, enterprise_value, ev_ebitda, ev_revenue,
                 price_to_book, roe, roic, debt_to_equity, current_ratio, quick_ratio, asset_turnover,
                 gross_margin, operating_margin, net_margin, revenue_growth_3y, revenue_growth_5y,
                 ebitda_growth_3y, beta, volatility_30d, volatility_90d, relative_strength,
                 analyst_coverage, analyst_rating_avg, price_target_avg, institutional_ownership,
                 insider_ownership, short_interest, country, exchange, currency, employees, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                company.get('symbol', ''), company.get('name', ''), company.get('sector', ''),
                company.get('industry', ''), company.get('market_cap', 0), company.get('enterprise_value', 0),
                company.get('ev_ebitda', 0), company.get('ev_revenue', 0), company.get('price_to_book', 0),
                company.get('roe', 0), company.get('roic', 0), company.get('debt_to_equity', 0),
                company.get('current_ratio', 0), company.get('quick_ratio', 0), company.get('asset_turnover', 0),
                company.get('gross_margin', 0), company.get('operating_margin', 0), company.get('net_margin', 0),
                company.get('revenue_growth_3y', 0), company.get('revenue_growth_5y', 0),
                company.get('ebitda_growth_3y', 0), company.get('beta', 0), company.get('volatility_30d', 0),
                company.get('volatility_90d', 0), company.get('relative_strength', 0),
                company.get('analyst_coverage', 0), company.get('analyst_rating_avg', 0),
                company.get('price_target_avg', 0), company.get('institutional_ownership', 0),
                company.get('insider_ownership', 0), company.get('short_interest', 0),
                company.get('country', ''), company.get('exchange', ''), company.get('currency', ''),
                company.get('employees', 0), datetime.now().isoformat()
            ))
        
        for deal in deals_data:
            deal_id = f"{deal.target_company}_{deal.acquirer}_{deal.deal_date}".replace(' ', '_')
            conn.execute('''
                INSERT OR REPLACE INTO ma_transactions_comprehensive 
                (deal_id, target_company, acquirer, deal_value, ev_ebitda_multiple, leverage_ratio,
                 sector, deal_date, exit_date, exit_multiple, irr, revenue_cagr, ebitda_margin,
                 deal_type, data_source, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                deal_id, deal.target_company, deal.acquirer, deal.deal_value,
                deal.ev_ebitda_multiple, deal.leverage_ratio, deal.sector,
                deal.deal_date, deal.exit_date, deal.exit_multiple, deal.irr,
                deal.revenue_cagr, deal.ebitda_margin, deal.deal_type,
                deal.source, datetime.now().isoformat()
            ))
        
        for indicator_name, series_data in macro_data.items():
            for date, value in series_data.items():
                if pd.notna(value):
                    conn.execute('''
                        INSERT OR REPLACE INTO macro_indicators_detailed 
                        (date, indicator_name, value, category, frequency, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        date.strftime('%Y-%m-%d'), indicator_name, float(value),
                        'Economic', 'Daily', 'FRED'
                    ))
        
        for sector, metrics in benchmarks.items():
            for metric_name, stats in metrics.items():
                conn.execute('''
                    INSERT OR REPLACE INTO industry_benchmarks 
                    (sector, subsector, metric_name, percentile_10, percentile_25, median,
                     percentile_75, percentile_90, sample_size, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sector, '', metric_name, stats['percentile_10'], stats['percentile_25'],
                    stats['median'], stats['percentile_75'], stats['percentile_90'],
                    stats['sample_size'], datetime.now().isoformat()
                ))
        
        conn.commit()
        conn.close()

    def get_training_dataset_elite(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            c.symbol, c.sector, c.industry, c.market_cap, c.enterprise_value,
            c.ev_ebitda, c.ev_revenue, c.roe, c.roic, c.debt_to_equity,
            c.operating_margin, c.net_margin, c.revenue_growth_3y, c.ebitda_growth_3y,
            c.beta, c.volatility_90d, c.relative_strength,
            d.deal_value, d.ev_ebitda_multiple, d.leverage_ratio, d.irr, d.revenue_cagr,
            d.ebitda_margin, d.deal_type,
            m.gdp_growth, m.inflation_cpi, m.fed_funds_rate, m.treasury_10y, m.vix
        FROM companies_comprehensive c
        LEFT JOIN ma_transactions_comprehensive d ON c.sector = d.sector
        LEFT JOIN (
            SELECT 
                MAX(CASE WHEN indicator_name = 'gdp_growth' THEN value END) as gdp_growth,
                MAX(CASE WHEN indicator_name = 'inflation_cpi' THEN value END) as inflation_cpi,
                MAX(CASE WHEN indicator_name = 'fed_funds_rate' THEN value END) as fed_funds_rate,
                MAX(CASE WHEN indicator_name = 'treasury_10y' THEN value END) as treasury_10y,
                MAX(CASE WHEN indicator_name = 'vix' THEN value END) as vix
            FROM macro_indicators_detailed 
            WHERE date >= date('now', '-30 days')
        ) m ON 1=1
        WHERE c.market_cap > 50000000 AND c.ev_ebitda > 0 AND c.ev_ebitda < 30
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

async def main():
    pipeline = EliteDataPipeline()
    
    print("🔍 Starting elite data acquisition...")
    
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(sp500_url)
    sp500_symbols = tables[0]['Symbol'].tolist()
    
    print(f"📊 Fetching comprehensive data for {len(sp500_symbols)} companies...")
    companies_data = await pipeline.fetch_comprehensive_company_data(sp500_symbols)
    
    print("💼 Scraping M&A transaction database...")
    deals_data = await pipeline.scrape_ma_database_comprehensive()
    
    print("📈 Fetching comprehensive macro indicators...")
    macro_data = pipeline.fetch_comprehensive_macro_data()
    
    print("🎯 Calculating industry benchmarks...")
    benchmarks = pipeline.calculate_industry_benchmarks()
    
    print("💾 Saving comprehensive dataset...")
    pipeline.save_comprehensive_data(companies_data, deals_data, macro_data, benchmarks)
    
    print("🧠 Generating training dataset...")
    training_data = pipeline.get_training_dataset_elite()
    
    print(f"✅ Elite dataset ready: {len(training_data)} records with {len(training_data.columns)} features")
    print(f"📈 Company records: {len(companies_data)}")
    print(f"💼 Deal records: {len(deals_data)}")
    print(f"📊 Macro indicators: {sum(len(series) for series in macro_data.values())}")
    print(f"🎯 Industry benchmarks: {len(benchmarks)} sectors")

if __name__ == "__main__":
    asyncio.run(main())
