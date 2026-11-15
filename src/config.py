"""Central config + env loading for the Myntra RAG chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project env file (holds KAGGLE_KEY and OPENAI_API_KEY)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "kaggle-rag-chatbot.env")

# --- Paths ---
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "products_clean.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "myntra_products"

# --- Models ---
EMBED_MODEL = "text-embedding-3-small"   # 1536-dim, cheap, strong
CHAT_MODEL = "gpt-4o-mini"
EMBED_BATCH = 500                        # texts per embedding request

# --- Retrieval ---
TOP_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add a line to kaggle-rag-chatbot.env:\n"
            "    OPENAI_API_KEY=sk-...\n"
        )
    return OPENAI_API_KEY
