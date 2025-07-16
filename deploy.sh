#!/bin/bash

echo "🚀 Deploying LBO-ORACLE™ Production System..."

chmod +x setup.sh
./setup.sh

echo "📊 Starting data acquisition..."
python data_acquisition.py &
DATA_PID=$!

echo "🧠 Training neural architecture..."
python training_pipeline.py

wait $DATA_PID

echo "🚀 Starting API server..."
python api_server.py &
API_PID=$!

echo "🌐 API server running on port 8000"
echo "📈 LBO-ORACLE™ deployment complete"

echo $API_PID > api.pid
