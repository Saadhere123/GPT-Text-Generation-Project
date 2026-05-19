# GPT Text Generation Project

## Overview
This project generates human-like text using GPT-2 from Hugging Face Transformers.

## Features
- GPT-2 text generation
- Adjustable generation settings
- Streamlit UI
- FastAPI deployment
- Multiple prompts support
- Save outputs to file
- GPT-2 vs GPT-Neo comparison
- Fine-tuning support

## Python Version
Python 3.10 Recommended

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment (Windows)

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Main Model

```bash
python model.py
```

## Run Streamlit App

```bash
python -m streamlit run app.py
```

## Run FastAPI

```bash
python -m uvicorn api:app --reload
```

## Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- Streamlit
- FastAPI
