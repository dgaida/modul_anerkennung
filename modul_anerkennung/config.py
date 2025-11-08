"""Globale Konfigurationen für das Modulanerkennungs-Tool."""
from pathlib import Path
from dotenv import load_dotenv
import os

# .env laden
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "secrets.env")

# Pfade
BASE_DIR: Path = Path(__file__).resolve().parent.parent
RAG_STORAGE_DIR: Path = BASE_DIR / "rag_storage"
OUTPUT_DIR: Path = BASE_DIR / "output"

# API Keys aus .env
API_KEY: str = os.getenv("LLM_API_KEY", "")
BASE_URL: str | None = os.getenv("LLM_BASE_URL")
