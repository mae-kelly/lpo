#!/bin/bash

echo "🚀 Deploying LBO-ORACLE™ Production System..."

# Setup Python environment
python3 setup.py

# Setup and train system
echo "🧠 Setting up ML system..."
python3 run.py setup

# Start API server
echo "🌐 Starting production API server..."
python3 run.py

echo "✅ LBO-ORACLE™ deployment complete!"
echo "📡 API available at http://localhost:8000"
echo "📖 Documentation at http://localhost:8000/docs"
