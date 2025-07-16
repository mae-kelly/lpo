#!/bin/bash

echo "🚀 Deploying LBO-ORACLE™ Elite Production System..."
echo "🎓 Stanford PhD-Level Implementation"

chmod +x setup_advanced.sh
./setup_advanced.sh

echo "📊 Acquiring comprehensive financial data..."
python data_acquisition_elite.py &
DATA_PID=$!

echo "🧠 Training elite neural architecture..."
python training_pipeline_elite.py &
TRAINING_PID=$!

wait $DATA_PID
echo "✅ Data acquisition completed"

wait $TRAINING_PID
echo "✅ Neural architecture training completed"

echo "🚀 Starting elite API server..."
python api_server_elite.py &
API_PID=$!

echo "🌐 Elite API server running on port 8000"
echo "📈 LBO-ORACLE™ Elite deployment complete"
echo "🎯 Ready for institutional-grade analysis"

echo $API_PID > elite_api.pid
