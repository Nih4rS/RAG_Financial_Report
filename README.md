# RAG_Financial_Report

## repo structure:
```
RAG_Financial_Report/
│── data/                     # Auto-updated financial data
│    ├── sec_filings/         # Extracted SEC reports (daily)
│    ├── yahoo_reports/       # Extracted Yahoo Finance data
│    ├── processed/           # Cleaned & tokenized data
│── notebooks/                # Google Colab notebooks for execution
│── models/                   # Fine-tuned financial models (LLM + RAG)
│── src/                      # Main automation scripts
│    ├── fetch_yahoo_finance.py   # Fetches yearly & quarterly reports
│    ├── fetch_sec_filings.py     # Scrapes SEC Edgar reports
│    ├── data_processing.py       # Cleans & structures data
│    ├── fine_tuning.py           # Fine-tunes LLM with QLoRA (Colab Pro)
│    ├── generate_report.py       # Generates AI-driven financial insights
│    ├── daily_refresh.py         # Automates daily data update to GitHub
│    ├── retrieval_rag.py         # RAG model for answering queries
│── dashboard/                   # Streamlit UI for financial queries
│    ├── app.py                   # Interactive stock market research tool
│── README.md                     # Project setup & usage guide
│── requirements.txt               # Dependencies
│── .gitignore                     # Ignore unnecessary files
```
