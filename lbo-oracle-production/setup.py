#!/usr/bin/env python3

import subprocess
import sys
import os

def install_requirements():
    print("🚀 Installing LBO-ORACLE™ Production Requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def setup_environment():
    print("🔧 Setting up environment...")
    
    # Create directories
    dirs = ["data", "models", "reports", "logs"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create .env file
    with open(".env", "w") as f:
        f.write("# LBO-ORACLE™ Configuration\n")
        f.write("API_KEY=your_api_key_here\n")
        f.write("DEBUG=False\n")
        f.write("LOG_LEVEL=INFO\n")
    
    print("✅ Environment setup complete")

if __name__ == "__main__":
    install_requirements()
    setup_environment()
    print("🎉 LBO-ORACLE™ setup complete!")
