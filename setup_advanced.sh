#!/bin/bash

pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0 bitsandbytes==0.41.0 peft==0.6.0
pip install -q torch-geometric==2.4.0 pyg-lib==0.3.0 torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2
pip install -q pandas==2.1.0 numpy==1.25.0 scipy==1.11.0 scikit-learn==1.3.0 optuna==3.4.0
pip install -q yfinance==0.2.22 quandl==3.7.0 fredapi==0.5.1 alpha-vantage==2.3.1 sec-edgar-api==1.0.0
pip install -q beautifulsoup4==4.12.0 requests==2.31.0 aiohttp==3.8.0 asyncio==3.4.3 lxml==4.9.0
pip install -q fastapi==0.103.0 uvicorn==0.23.0 pydantic==2.4.0 sqlalchemy==2.0.0
pip install -q wandb==0.15.0 tensorboard==2.14.0 matplotlib==3.7.0 seaborn==0.12.0 plotly==5.17.0
pip install -q causalml==0.15.0 dowhy==0.10.0 networkx==3.1.0 pgmpy==0.1.24
pip install -q ta-lib==0.4.28 python-telegram-bot==20.6.0 schedule==1.2.0
pip install -q streamlit==1.27.0 gradio==3.50.0 chainlit==0.7.0
pip install -q openai==0.28.0 anthropic==0.7.0 cohere==4.32.0 together==0.2.0
pip install -q sentence-transformers==2.2.0 faiss-cpu==1.7.4 chromadb==0.4.15
pip install -q PyPDF2==3.0.1 python-docx==0.8.11 openpyxl==3.1.0 xlsxwriter==3.1.0
pip install -q reportlab==4.0.4 fpdf2==2.7.6 weasyprint==60.0
pip install -q pymc==5.9.0 arviz==0.16.0 tensorflow==2.14.0 keras==2.14.0
pip install -q prophet==1.1.4 statsmodels==0.14.0 pmdarima==2.0.4
pip install -q xgboost==1.7.0 lightgbm==4.1.0 catboost==1.2.0
pip install -q shap==0.43.0 lime==0.2.0.1 interpret==0.4.3
pip install -q huggingface-hub==0.17.0 langchain==0.0.350 llama-index==0.9.0

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lbo-oracle-elite"
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

