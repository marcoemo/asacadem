"""Configuración central. Todo se puede sobreescribir con variables de entorno / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
PLAZOS_PATH = ROOT / "data" / "plazos.json"
STORAGE_DIR = Path(os.getenv("AGENTE_STORAGE_DIR", ROOT / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("AGENT_MODEL", "openai/gpt-6-luna")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")  # auto | openrouter | local
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Gestión de la ventana de contexto
MAX_MESSAGES_BEFORE_SUMMARY = int(os.getenv("MAX_MESSAGES_BEFORE_SUMMARY", "12"))
KEEP_LAST_MESSAGES = int(os.getenv("KEEP_LAST_MESSAGES", "6"))

RAG_TOP_K = 3
MEMORY_TOP_K = 3

DB_CHECKPOINTS = STORAGE_DIR / "checkpoints.db"   # memoria de corto plazo (por thread)
DB_APP = STORAGE_DIR / "app.db"                   # memoria de largo plazo + tickets
RAG_INDEX_PATH = STORAGE_DIR / "rag_index.json"
