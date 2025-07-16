import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import feedparser
import sqlite3
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import newspaper
import spacy
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MassiveLBODataPipeline:
    """
    MASSIVE LBO data scraper implementing all free public sources
    Target: 10,000+ real LBO transactions
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.db_path = 'massive_lbo_database.db'
        self.setup_database()
        
        # Load NLP model for deal extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("SpaCy model not found - install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # PE firms to scrape (comprehensive list)
        self.pe_firms = {
            'KKR': 'https://www.kkr.com/businesses/private-equity',
            'Blackstone': 'https://www.blackstone.com/our-businesses/private-equity/',
            'Apollo': 'https://www.apollo.com/strategies/private-equity',
            'Carlyle': 'https://www.carlyle.com/our-business/portfolio-companies',
            'TPG': 'https://www.tpg.com/platform/portfolio',
            'Bain Capital': 'https://www.baincapital.com/portfolio',
            'Vista Equity': 'https://www.vistaequitypartners.com/companies',
            'Thoma Bravo': 'https://www.thomabravo.com/companies',
            'General Atlantic': 'https://www.generalatlantic.com/portfolio',
            'Warburg Pincus': 'https://www.warburgpincus.com/portfolio',
            'Silver Lake': 'https://www.silverlake.com/portfolio',
            'Francisco Partners': 'https://www.franciscopartners.com/portfolio',
            'Leonard Green': 'https://www.leonardgreen.com/portfolio',
            'GTCR': 'https://www.gtcr.com/portfolio',
            'Advent International': 'https://www.adventinternational.com/portfolio',
            'CVC Capital': 'https://www.cvc.com/our-portfolio'
        }
        
    def setup_database(self):
        """Setup comprehensive database for massive data storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS lbo_transactions (
                transaction_id TEXT PRIMARY KEY,
                target_company TEXT,
                acquirer_firm TEXT,
                deal_value_text TEXT,
                deal_value_parsed REAL,
                enterprise_value REAL,
                revenue_ttm REAL,
                ebitda_ttm REAL,
                ev_ebitda_multiple REAL,
                leverage_ratio REAL,
                sector TEXT,
                subsector TEXT,
                geographic_region TEXT,
                investment_date TEXT,
                exit_date TEXT,
                exit_type TEXT,
                exit_value REAL,
                irr_calculated REAL,
                moic_calculated REAL,
                hold_period_years REAL,
                data_source TEXT,
                source_url TEXT,
                extraction_method TEXT,
                confidence_score REAL,
                raw_text TEXT,
                scraped_timestamp TIMESTAMP
            )
        ''')
        
        # Performance indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_firm ON lbo_transactions(acquirer_firm)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sector ON lbo_transactions(sector)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON lbo_transactions(investment_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_value ON lbo_transactions(deal_value_parsed)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with performance optimization")

    async def scrape_pe_firm_portfolios(self) -> List[Dict]:
        """Scrape ALL major PE firm portfolio pages"""
        all_deals = []
        
        print("🏢 Scraping PE firm portfolio pages...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for firm_name, url in self.pe_firms.items():
                task = self.scrape_single_firm_portfolio(session, firm_name, url)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_deals.extend(result)
                else:
                    logger.error(f"Error in portfolio scraping: {result}")
        
        logger.info(f"Scraped {len(all_deals)} deals from PE firm portfolios")
        return all_deals

    async def scrape_single_firm_portfolio(self, session, firm_name: str, url: str) -> List[Dict]:
        """Scrape individual PE firm portfolio"""
        deals = []
        
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for portfolio company listings
                portfolio_selectors = [
                    '.portfolio-company', '.company-card', '.investment-card',
                    '.portfolio-item', '[class*="portfolio"]', '[class*="company"]'
                ]
                
                for selector in portfolio_selectors:
                    companies = soup.select(selector)
                    if companies:
                        for company in companies:
                            deal = self.extract_deal_from_portfolio_card(company, firm_name, url)
                            if deal:
                                deals.append(deal)
                        break
                
                # Fallback: look for any text mentioning companies
                if not deals:
                    text_content = soup.get_text()
                    deals = self.extract_deals_from_text(text_content, firm_name, url)
                    
        except Exception as e:
            logger.error(f"Error scraping {firm_name}: {e}")
        
        return deals

    def extract_deal_from_portfolio_card(self, card_element, firm_name: str, source_url: str) -> Optional[Dict]:
        """Extract deal information from portfolio card HTML"""
        try:
            # Extract company name
            company_name = None
            name_selectors = ['h1', 'h2', 'h3', '.company-name', '.title', 'strong']
            for selector in name_selectors:
                name_elem = card_element.select_one(selector)
                if name_elem:
                    company_name = name_elem.get_text(strip=True)
                    break
            
            if not company_name:
                return None
            
            # Extract other details
            card_text = card_element.get_text()
            
            deal_data = {
                'target_company': company_name,
                'acquirer_firm': firm_name,
                'data_source': 'pe_firm_portfolio',
                'source_url': source_url,
                'extraction_method': 'html_card_parsing',
                'raw_text': card_text[:500],
                'scraped_timestamp': datetime.now().isoformat()
            }
            
            # Try to extract sector
            if self.nlp:
                doc = self.nlp(card_text.lower())
                sector = self.extract_sector_from_text(card_text)
                if sector:
                    deal_data['sector'] = sector
            
            # Try to extract dates
            date_patterns = [
                r'(19|20)\d{2}',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(19|20)\d{2}',
                r'\d{1,2}/\d{1,2}/(19|20)\d{2}'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, card_text.lower())
                if match:
                    deal_data['investment_date'] = match.group(0)
                    break
            
            deal_data['confidence_score'] = 0.8  # High confidence for PE firm data
            
            return deal_data
            
        except Exception as e:
            logger.error(f"Error extracting from portfolio card: {e}")
            return None

    def extract_deals_from_text(self, text: str, firm_name: str, source_url: str) -> List[Dict]:
        """Extract deals from unstructured text using NLP"""
        deals = []
        
        if not self.nlp:
            return deals
        
        # Split text into sentences
        sentences = text.split('.')
        
        for sentence in sentences:
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Look for acquisition/investment keywords
            if any(keyword in sentence.lower() for keyword in 
                  ['acquired', 'investment', 'portfolio', 'backed', 'funded']):
                
                # Extract potential company names (capitalized words)
                company_matches = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', sentence)
                
                for company in company_matches:
                    if len(company) > 3 and company not in ['The', 'And', 'For', 'Inc', 'LLC']:
                        deal = {
                            'target_company': company,
                            'acquirer_firm': firm_name,
                            'data_source': 'pe_firm_text_extraction',
                            'source_url': source_url,
                            'extraction_method': 'nlp_text_parsing',
                            'raw_text': sentence[:200],
                            'scraped_timestamp': datetime.now().isoformat(),
                            'confidence_score': 0.6
                        }
                        
                        sector = self.extract_sector_from_text(sentence)
                        if sector:
                            deal['sector'] = sector
                            
                        deals.append(deal)
        
        return deals[:20]  # Limit to top 20 to avoid noise

    def extract_sector_from_text(self, text: str) -> Optional[str]:
        """Extract sector from text using keyword matching"""
        sector_keywords = {
            'Technology': ['software', 'tech', 'digital', 'data', 'cloud', 'saas', 'platform'],
            'Healthcare': ['healthcare', 'medical', 'health', 'pharma', 'biotech', 'clinical'],
            'Financial Services': ['financial', 'bank', 'insurance', 'fintech', 'payments'],
            'Consumer': ['consumer', 'retail', 'brand', 'restaurant', 'food', 'beverage'],
            'Industrial': ['industrial', 'manufacturing', 'equipment', 'machinery'],
            'Energy': ['energy', 'oil', 'gas', 'renewable', 'power', 'utilities'],
            'Real Estate': ['real estate', 'property', 'reit', 'development'],
            'Media': ['media', 'entertainment', 'content', 'publishing', 'broadcasting']
        }
        
        text_lower = text.lower()
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return sector
        
        return None

    def scrape_business_wire_prs(self) -> List[Dict]:
        """Scrape Business Wire for PE deal announcements"""
        deals = []
        
        print("📰 Scraping Business Wire press releases...")
        
        # Search queries for different types of LBO announcements
        search_queries = [
            "private equity acquires",
            "leveraged buyout",
            "LBO transaction",
            "private equity investment",
            "buyout firm announces",
            "growth capital investment"
        ]
        
        for query in search_queries:
            try:
                # Search Business Wire
                search_url = f"https://www.businesswire.com/portal/site/home/search/?searchType=all&searchTerm={query.replace(' ', '+')}"
                
                response = self.session.get(search_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find press release links
                pr_links = soup.find_all('a', href=True)
                
                for link in pr_links[:10]:  # Limit to avoid overwhelming
                    href = link.get('href')
                    if 'businesswire.com' in href and 'news' in href:
                        deal = self.scrape_business_wire_article(href)
                        if deal:
                            deals.append(deal)
                            
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error searching Business Wire for '{query}': {e}")
        
        logger.info(f"Scraped {len(deals)} deals from Business Wire")
        return deals

    def scrape_business_wire_article(self, url: str) -> Optional[Dict]:
        """Scrape individual Business Wire press release"""
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            
            title = article.title
            text = article.text
            
            # Look for LBO-related content
            if not any(keyword in text.lower() for keyword in 
                      ['private equity', 'buyout', 'acquisition', 'investment']):
                return None
            
            deal_data = {
                'data_source': 'business_wire_pr',
                'source_url': url,
                'extraction_method': 'newspaper3k_parsing',
                'raw_text': text[:1000],
                'scraped_timestamp': datetime.now().isoformat()
            }
            
            # Extract deal details using regex
            # Company acquisition pattern
            acquisition_patterns = [
                r'([A-Z][a-zA-Z\s&]+?)\s+(?:acquires?|purchases?)\s+([A-Z][a-zA-Z\s&]+)',
                r'([A-Z][a-zA-Z\s&]+?)\s+(?:announces|completes)\s+(?:acquisition|purchase)\s+of\s+([A-Z][a-zA-Z\s&]+)',
                r'([A-Z][a-zA-Z\s&]+?)\s+invests\s+in\s+([A-Z][a-zA-Z\s&]+)'
            ]
            
            for pattern in acquisition_patterns:
                match = re.search(pattern, title + ' ' + text)
                if match:
                    deal_data['acquirer_firm'] = match.group(1).strip()
                    deal_data['target_company'] = match.group(2).strip()
                    break
            
            # Extract deal value
            value_patterns = [
                r'\$([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)',
                r'([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)\s*(?:dollar|transaction|deal)'
            ]
            
            for pattern in value_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    deal_data['deal_value_text'] = match.group(0)
                    try:
                        value = float(match.group(1).replace(',', ''))
                        multiplier = 1e9 if 'billion' in match.group(2).lower() else 1e6
                        deal_data['deal_value_parsed'] = value * multiplier
                    except:
                        pass
                    break
            
            # Extract date
            if article.publish_date:
                deal_data['investment_date'] = article.publish_date.strftime('%Y-%m-%d')
            
            # Extract sector
            sector = self.extract_sector_from_text(text)
            if sector:
                deal_data['sector'] = sector
            
            # Calculate confidence
            confidence = 0.5  # Base for press release
            if deal_data.get('target_company') and deal_data.get('acquirer_firm'):
                confidence += 0.3
            if deal_data.get('deal_value_parsed'):
                confidence += 0.2
                
            deal_data['confidence_score'] = confidence
            
            return deal_data if confidence > 0.6 else None
            
        except Exception as e:
            logger.error(f"Error scraping Business Wire article {url}: {e}")
            return None

    def scrape_wikipedia_lbo_lists(self) -> List[Dict]:
        """Scrape Wikipedia lists of LBOs and buyouts"""
        deals = []
        
        print("📚 Scraping Wikipedia LBO lists...")
        
        wikipedia_pages = [
            "https://en.wikipedia.org/wiki/List_of_largest_leveraged_buyouts",
            "https://en.wikipedia.org/wiki/List_of_private_equity_firms",
            "https://en.wikipedia.org/wiki/Category:Leveraged_buyouts",
            "https://en.wikipedia.org/wiki/Timeline_of_largest_leveraged_buyouts"
        ]
        
        for page_url in wikipedia_pages:
            try:
                response = self.session.get(page_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find tables with LBO data
                tables = soup.find_all('table', class_=['wikitable', 'sortable'])
                
                for table in tables:
                    table_deals = self.extract_deals_from_wikipedia_table(table, page_url)
                    deals.extend(table_deals)
                    
            except Exception as e:
                logger.error(f"Error scraping Wikipedia page {page_url}: {e}")
        
        logger.info(f"Scraped {len(deals)} deals from Wikipedia")
        return deals

    def extract_deals_from_wikipedia_table(self, table, source_url: str) -> List[Dict]:
        """Extract deals from Wikipedia table"""
        deals = []
        
        try:
            rows = table.find_all('tr')
            
            # Try to identify column headers
            headers = []
            if rows:
                header_row = rows[0]
                headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            # Map common column names
            column_mapping = {}
            for i, header in enumerate(headers):
                if any(word in header for word in ['company', 'target']):
                    column_mapping['company'] = i
                elif any(word in header for word in ['acquirer', 'buyer', 'firm']):
                    column_mapping['acquirer'] = i
                elif any(word in header for word in ['value', 'price', 'deal']):
                    column_mapping['value'] = i
                elif any(word in header for word in ['year', 'date']):
                    column_mapping['date'] = i
                elif any(word in header for word in ['sector', 'industry']):
                    column_mapping['sector'] = i
            
            # Extract data rows
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                
                deal_data = {
                    'data_source': 'wikipedia_table',
                    'source_url': source_url,
                    'extraction_method': 'table_parsing',
                    'scraped_timestamp': datetime.now().isoformat(),
                    'confidence_score': 0.9  # High confidence for Wikipedia
                }
                
                # Extract data based on column mapping
                for field, col_index in column_mapping.items():
                    if col_index < len(cells):
                        cell_text = cells[col_index].get_text(strip=True)
                        
                        if field == 'company':
                            deal_data['target_company'] = cell_text
                        elif field == 'acquirer':
                            deal_data['acquirer_firm'] = cell_text
                        elif field == 'value':
                            deal_data['deal_value_text'] = cell_text
                            # Try to parse value
                            value_match = re.search(r'([0-9,]+(?:\.[0-9]+)?)', cell_text)
                            if value_match:
                                try:
                                    value = float(value_match.group(1).replace(',', ''))
                                    if 'billion' in cell_text.lower():
                                        value *= 1e9
                                    elif 'million' in cell_text.lower():
                                        value *= 1e6
                                    deal_data['deal_value_parsed'] = value
                                except:
                                    pass
                        elif field == 'date':
                            deal_data['investment_date'] = cell_text
                        elif field == 'sector':
                            deal_data['sector'] = cell_text
                
                if deal_data.get('target_company'):
                    deals.append(deal_data)
                    
        except Exception as e:
            logger.error(f"Error extracting from Wikipedia table: {e}")
        
        return deals

    def scrape_sec_edgar_lbo_filings(self) -> List[Dict]:
        """Scrape SEC EDGAR for LBO-related filings"""
        deals = []
        
        print("📋 Scraping SEC EDGAR filings...")
        
        try:
            # Search EDGAR for LBO-related filings
            edgar_search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            
            # Search parameters for LBO-related filings
            search_params = [
                {"type": "8-K", "dateb": "", "count": "40"},
                {"type": "DEF 14A", "dateb": "", "count": "40"},
                {"type": "S-1", "dateb": "", "count": "40"}
            ]
            
            for params in search_params:
                try:
                    response = self.session.get(edgar_search_url, params=params)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find filing links
                    filing_links = soup.find_all('a', href=True)
                    
                    for link in filing_links[:10]:  # Limit processing
                        href = link.get('href')
                        if '/Archives/edgar/data/' in href:
                            filing_url = f"https://www.sec.gov{href}"
                            deal = self.scrape_sec_filing(filing_url)
                            if deal:
                                deals.append(deal)
                                
                    time.sleep(2)  # Rate limiting for SEC
                    
                except Exception as e:
                    logger.error(f"Error searching EDGAR with params {params}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in SEC EDGAR scraping: {e}")
        
        logger.info(f"Scraped {len(deals)} deals from SEC EDGAR")
        return deals

    def scrape_sec_filing(self, url: str) -> Optional[Dict]:
        """Scrape individual SEC filing for LBO information"""
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get text content
            text_content = soup.get_text()
            
            # Look for LBO-related keywords
            lbo_keywords = ['private equity', 'leveraged buyout', 'sponsor', 'acquisition', 'merger']
            if not any(keyword.lower() in text_content.lower() for keyword in lbo_keywords):
                return None
            
            deal_data = {
                'data_source': 'sec_edgar_filing',
                'source_url': url,
                'extraction_method': 'sec_filing_parsing',
                'raw_text': text_content[:1000],
                'scraped_timestamp': datetime.now().isoformat()
            }
            
            # Extract company name from filing
            company_match = re.search(r'COMPANY CONFORMED NAME:\s*([^\n]+)', text_content)
            if company_match:
                deal_data['target_company'] = company_match.group(1).strip()
            
            # Look for sponsor/PE firm mentions
            sponsor_patterns = [
                r'(?:sponsor|backed by|acquired by)\s+([A-Z][a-zA-Z\s&]+(?:Capital|Partners|Equity|Fund))',
                r'([A-Z][a-zA-Z\s&]+(?:Capital|Partners|Equity|Fund))\s+(?:owns|controls|acquired)'
            ]
            
            for pattern in sponsor_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    deal_data['acquirer_firm'] = match.group(1).strip()
                    break
            
            # Extract deal value
            value_patterns = [
                r'consideration.*?\$([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)',
                r'purchase price.*?\$([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)',
                r'transaction value.*?\$([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)'
            ]
            
            for pattern in value_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    deal_data['deal_value_text'] = match.group(0)
                    try:
                        value = float(match.group(1).replace(',', ''))
                        multiplier = 1e9 if 'billion' in match.group(2).lower() else 1e6
                        deal_data['deal_value_parsed'] = value * multiplier
                    except:
                        pass
                    break
            
            # Extract filing date
            date_match = re.search(r'FILED AS OF DATE:\s*([0-9]{8})', text_content)
            if date_match:
                date_str = date_match.group(1)
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                deal_data['investment_date'] = formatted_date
            
            # Calculate confidence
            confidence = 0.4  # Base for SEC filing
            if deal_data.get('target_company'):
                confidence += 0.3
            if deal_data.get('acquirer_firm'):
                confidence += 0.2
            if deal_data.get('deal_value_parsed'):
                confidence += 0.3
                
            deal_data['confidence_score'] = confidence
            
            return deal_data if confidence > 0.6 else None
            
        except Exception as e:
            logger.error(f"Error scraping SEC filing {url}: {e}")
            return None

    def save_deals_to_database(self, deals: List[Dict]):
        """Save all scraped deals to database"""
        if not deals:
            logger.warning("No deals to save")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        saved_count = 0
        for deal in deals:
            try:
                # Generate unique transaction ID
                transaction_id = f"{deal.get('target_company', 'UNKNOWN')}_{deal.get('acquirer_firm', 'UNKNOWN')}_{deal.get('data_source', 'UNKNOWN')}_{int(time.time())}"
                transaction_id = re.sub(r'[^A-Za-z0-9_]', '', transaction_id)[:100]
                
                conn.execute('''
                    INSERT OR REPLACE INTO lbo_transactions
                    (transaction_id, target_company, acquirer_firm, deal_value_text, deal_value_parsed,
                     sector, investment_date, data_source, source_url, extraction_method,
                     confidence_score, raw_text, scraped_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction_id,
                    deal.get('target_company'),
                    deal.get('acquirer_firm'),
                    deal.get('deal_value_text'),
                    deal.get('deal_value_parsed'),
                    deal.get('sector'),
                    deal.get('investment_date'),
                    deal.get('data_source'),
                    deal.get('source_url'),
                    deal.get('extraction_method'),
                    deal.get('confidence_score'),
                    deal.get('raw_text'),
                    deal.get('scraped_timestamp')
                ))
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving deal: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {saved_count} deals to database")

    async def run_massive_scraping_pipeline(self):
        """Run the complete massive data scraping pipeline"""
        print("🚀 Starting MASSIVE LBO Data Scraping Pipeline...")
        print("🎯 Target: 10,000+ real LBO transactions")
        
        all_deals = []
        
        # 1. PE Firm Portfolios (highest quality)
        print("\n🏢 Phase 1: PE Firm Portfolio Pages...")
        portfolio_deals = await self.scrape_pe_firm_portfolios()
        all_deals.extend(portfolio_deals)
        print(f"✅ Collected {len(portfolio_deals)} deals from PE portfolios")
        
        # 2. Business Wire Press Releases
        print("\n📰 Phase 2: Business Wire Press Releases...")
        pr_deals = self.scrape_business_wire_prs()
        all_deals.extend(pr_deals)
        print(f"✅ Collected {len(pr_deals)} deals from press releases")
        
        # 3. Wikipedia LBO Lists
        print("\n📚 Phase 3: Wikipedia LBO Lists...")
        wiki_deals = self.scrape_wikipedia_lbo_lists()
        all_deals.extend(wiki_deals)
        print(f"✅ Collected {len(wiki_deals)} deals from Wikipedia")
        
        # 4. SEC EDGAR Filings
        print("\n📋 Phase 4: SEC EDGAR Filings...")
        sec_deals = self.scrape_sec_edgar_lbo_filings()
        all_deals.extend(sec_deals)
        print(f"✅ Collected {len(sec_deals)} deals from SEC filings")
        
        # Save all deals
        print(f"\n💾 Saving {len(all_deals)} deals to database...")
        self.save_deals_to_database(all_deals)
        
        # Generate comprehensive report
        self.generate_massive_data_report()
        
        print(f"\n🎉 MASSIVE SCRAPING COMPLETE!")
        print(f"📊 Total deals collected: {len(all_deals)}")
        print(f"💾 Database: {self.db_path}")
        
        return all_deals

    def generate_massive_data_report(self):
        """Generate comprehensive report of scraped data"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM lbo_transactions", conn)
        conn.close()
        
        if df.empty:
            print("❌ No data in database")
            return
        
        report = f"""
🎯 MASSIVE LBO DATA PIPELINE REPORT
===================================

📊 DATABASE OVERVIEW:
- Total transactions: {len(df):,}
- Unique target companies: {df['target_company'].nunique():,}
- Unique PE firms: {df['acquirer_firm'].nunique():,}
- Date range: {df['investment_date'].min()} to {df['investment_date'].max()}

💰 DEAL VALUE ANALYSIS:
- Deals with parsed values: {df['deal_value_parsed'].notna().sum():,}
- Total deal value: ${df['deal_value_parsed'].sum()/1e12:.1f}T
- Average deal size: ${df['deal_value_parsed'].mean()/1e6:.1f}M
- Median deal size: ${df['deal_value_parsed'].median()/1e6:.1f}M

🏢 TOP PE FIRMS BY DEAL COUNT:
{df['acquirer_firm'].value_counts().head(10).to_string()}

🏭 SECTOR BREAKDOWN:
{df['sector'].value_counts().head(10).to_string()}

📡 DATA SOURCE QUALITY:
{df['data_source'].value_counts().to_string()}

🎯 CONFIDENCE ANALYSIS:
- High confidence (>0.8): {(df['confidence_score'] > 0.8).sum():,} deals
- Medium confidence (0.6-0.8): {((df['confidence_score'] >= 0.6) & (df['confidence_score'] <= 0.8)).sum():,} deals
- Lower confidence (<0.6): {(df['confidence_score'] < 0.6).sum():,} deals

✅ TRAINING READINESS: {len(df):,} transactions ready for ML training
        """
        
        print(report)
        
        with open('massive_lbo_report.txt', 'w') as f:
            f.write(report)
        
        # Show sample of best deals
        if len(df) > 0:
            print("\n🌟 SAMPLE HIGH-CONFIDENCE DEALS:")
            best_deals = df.nlargest(10, 'confidence_score')[
                ['target_company', 'acquirer_firm', 'deal_value_text', 'sector', 'data_source', 'confidence_score']
            ]
            print(best_deals.to_string(index=False))

if __name__ == "__main__":
    scraper = MassiveLBODataPipeline()
    asyncio.run(scraper.run_massive_scraping_pipeline())
