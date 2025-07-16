#!/usr/bin/env python3

import asyncio
import subprocess
import sys
import os
from pathlib import Path

async def setup_system():
    """Setup the complete LBO-ORACLE™ system"""
    print("🚀 Setting up LBO-ORACLE™ Production System...")
    
    # Run setup
    subprocess.run([sys.executable, "setup.py"], check=True)
    
    # Collect data
    print("📊 Collecting market data...")
    from data_engine import main as data_main
    await data_main()
    
    # Train model
    print("🧠 Training ML model...")
    from ml_engine import ProductionMLEngine
    engine = ProductionMLEngine()
    results = engine.train(optimize_hp=False)  # Quick training for demo
    print(f"Training complete: {results}")
    
    print("✅ LBO-ORACLE™ setup complete!")
    print("🌐 Starting API server...")

def start_api():
    """Start the API server"""
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        asyncio.run(setup_system())
    else:
        start_api()
