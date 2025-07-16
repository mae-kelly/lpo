#!/bin/bash

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers datasets accelerate bitsandbytes peft
pip install -q torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -q pandas numpy scipy scikit-learn optuna
pip install -q yfinance quandl alpha_vantage fredapi eikon
pip install -q beautifulsoup4 requests aiohttp asyncio
pip install -q fastapi uvicorn pydantic sqlalchemy
pip install -q wandb tensorboard matplotlib seaborn plotly
pip install -q causalml dowhy networkx pgmpy
pip install -q ta-lib python-telegram-bot schedule
pip install -q streamlit gradio chainlit
pip install -q openai anthropic cohere together

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lbo-oracle"
export TOKENIZERS_PARALLELISM=false

