# Health AI WhatsApp Analyzer

A Python-based AI system for analyzing health-related WhatsApp conversations, providing risk scoring and personalized health recommendations using Retrieval-Augmented Generation (RAG).

## Overview

This project processes WhatsApp conversation data to:
- Extract and analyze health-related information
- Score health risks based on conversation content
- Generate personalized health recommendations using RAG
- Provide actionable insights from health conversations

## Project Structure

```
.
├── main.py                          # Entry point for the application
├── src/
│   ├── analyzer.py                  # Core analysis logic
│   ├── config.py                    # Configuration management
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── rag_recommender.py           # RAG-based recommendation engine
│   ├── risk_scorer.py               # Health risk scoring module
│   └── __init__.py
├── data/
│   ├── health_ai_whatsapp_100_conversations_long.txt  # Conversation data
│   ├── NDoH-guidelines.pdf          # Health guidelines document
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook for EDA
├── pyproject.toml                   # Project dependencies and metadata
├── .env                             # Environment variables 
├── .gitignore                       # Git ignore rules
└── .python-version                  # Python version specification
```

## Installation

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Healthcare-risk-assessment
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```
   Or with uv:
   ```bash
   uv pip install -e .
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Using Individual Modules

```python
from src.data_loader import load_conversations
from src.analyzer import analyze_conversations
from src.risk_scorer import score_risks
from src.rag_recommender import get_recommendations

# Load conversation data
conversations = load_conversations('data/health_ai_whatsapp_100_conversations_long.txt')

# Analyze conversations
analysis_results = analyze_conversations(conversations)

# Score health risks
risk_scores = score_risks(analysis_results)

# Get AI-powered recommendations
recommendations = get_recommendations(conversations, risk_scores)
```

### Jupyter Notebook Analysis

Open the exploratory analysis notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Module Descriptions

### [`src/config.py`](src/config.py)
Manages configuration settings including API keys, model parameters, and application constants.

### [`src/data_loader.py`](src/data_loader.py)
Handles loading and preprocessing WhatsApp conversation data from text files. Parses messages and extracts relevant metadata.

### [`src/analyzer.py`](src/analyzer.py)
Core analysis module that processes conversations to identify health-related topics, extract health indicators, and structure data for downstream analysis.

### [`src/risk_scorer.py`](src/risk_scorer.py)
Implements health risk scoring algorithms. Evaluates conversation content against health indicators to generate risk assessments.

### [`src/rag_recommender.py`](src/rag_recommender.py)
RAG-based recommendation engine that leverages conversation context and health data to generate personalized recommendations using retrieval-augmented generation.

## Data

The project includes sample WhatsApp conversation data:
- **[`data/health_ai_whatsapp_100_conversations_long.txt`](data/health_ai_whatsapp_100_conversations_long.txt)** - 100 extended health-related WhatsApp conversations
- **[`data/NDoH-guidelines.pdf`](data/NDoH-guidelines.pdf)** - Health guidelines document used for reference in recommendations and treatment plans.

## Development

### Running Analysis
```bash
jupyter notebook notebooks/analysis.ipynb
```

### Python Version
This project uses Python as specified in [`.python-version`](.python-version).

## Dependencies

See [`pyproject.toml`](pyproject.toml) for complete dependency list and project metadata.







