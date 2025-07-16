#!/bin/bash

echo "🚀 Deploying MASSIVE LBO Data Pipeline..."
echo "🎯 Target: 10,000+ real LBO transactions from all free sources!"

# Install dependencies
echo "📦 Installing comprehensive dependencies..."
python3 -m pip install requests beautifulsoup4 pandas numpy feedparser python-dateutil
python3 -m pip install scikit-learn sentence-transformers newspaper3k selenium
python3 -m pip install aiohttp asyncio tqdm webdriver-manager
python3 -m pip install spacy nltk fastapi uvicorn pydantic

# Download spaCy model
echo "🧠 Downloading NLP model..."
python3 -m spacy download en_core_web_sm || echo "SpaCy model download failed - continuing anyway"

# Run the massive scraping pipeline
echo "🔍 Starting MASSIVE data scraping pipeline..."
python3 massive_lbo_scraper.py

# Train ML models on massive dataset
echo "🧠 Training ML models on massive dataset..."
python3 ml_trainer_massive.py

echo "✅ MASSIVE LBO Pipeline deployment complete!"
echo ""
echo "📊 Check these files for results:"
echo "  • massive_lbo_report.txt - Comprehensive data report"
echo "  • massive_lbo_database.db - SQLite database with all deals"
echo "  • massive_lbo_models.pkl - Trained ML models"
echo ""
echo "🎯 Data sources scraped:"
echo "  • 16 major PE firm portfolio pages"
echo "  • Business Wire press releases"
echo "  • Wikipedia LBO lists"
echo "  • SEC EDGAR filings"
echo ""
echo "💾 Expected dataset size: 5,000-15,000 real LBO transactions"
