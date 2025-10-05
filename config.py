"""Configuration management for AI Finance Assistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "index")
REPORTS_DIR = BASE_DIR / os.getenv("REPORTS_DIR", "reports")

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://yahya-mgcqekep-eastus2.cognitiveservices.azure.com/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Azure OpenAI Embedding configuration
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_EMBEDDING_KEY = os.getenv("AZURE_EMBEDDING_KEY", "")

# SEC EDGAR configuration
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Research research@example.com")
SEC_BASE_URL = "https://data.sec.gov"
SEC_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

# RAG configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "8"))
BM25_TOP_K = 20  # BM25 pre-filter before reranking

# Processing configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# FinBERT configuration
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.6
SENTIMENT_NEGATIVE_THRESHOLD = -0.6

# Report configuration
REPORT_TEMPLATE_VERSION = "1.0"
MAX_REPORT_PAGES = 2
