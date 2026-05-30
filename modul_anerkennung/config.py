"""Globale Konfigurationen für das Modulanerkennungs-Tool (Colab-kompatibel)."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Logger für die Konfiguration initialisieren
# Wir nutzen stderr, damit MCP-Kommunikation auf stdout nicht gestört wird
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. Umgebungserkennung
# ----------------------------------------------------------------------
def is_colab() -> bool:
    """Erkennt, ob das Skript in Google Colab ausgeführt wird."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


IN_COLAB = is_colab()

# ----------------------------------------------------------------------
# 2. Basisverzeichnis
# ----------------------------------------------------------------------
if IN_COLAB:
    # Colab: nutze das Arbeitsverzeichnis in /content
    BASE_DIR: Path = Path("/content")
else:
    # Lokal oder Entwicklungsumgebung
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Verzeichnisse anlegen
RAG_STORAGE_DIR: Path = BASE_DIR / "rag_storage"
OUTPUT_DIR: Path = BASE_DIR / "output"
RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 3. Umgebungsvariablen laden (.env + Colab userdata)
# ----------------------------------------------------------------------
# Zusätzliche Diagnostik
if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
    logger.info(f"[Config] Current Working Directory: {os.getcwd()}")
    logger.info(f"[Config] Calculated BASE_DIR: {BASE_DIR.absolute()}")

# Suche nach secrets.env oder .env
for env_name in ["secrets.env", ".env"]:
    dotenv_path = BASE_DIR / env_name
    if dotenv_path.exists():
        # Wir nutzen logger.info statt print, um MCP nicht zu korrumpieren
        logger.info(f"[Config] Found {env_name} at {dotenv_path.absolute()}")
        load_dotenv(dotenv_path=dotenv_path)
        if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
            logger.info(f"[Config] Loaded environment variables from {dotenv_path}")
    else:
        if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
            logger.info(f"[Config] {env_name} NOT found at {dotenv_path.absolute()}")

# Ensure MOCOGI_API_TOKEN is set if MOCOGI_API_KEY is provided
if not os.getenv("MOCOGI_API_TOKEN") and os.getenv("MOCOGI_API_KEY"):
    os.environ["MOCOGI_API_TOKEN"] = os.getenv("MOCOGI_API_KEY")

# Wenn Colab: versuche Keys aus google.colab.userdata zu holen
if IN_COLAB:
    try:
        from google.colab import userdata  # type: ignore

        for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "LLM_BASE_URL"]:
            try:
                value = userdata.get(key)
                if value:
                    os.environ[key] = value
            except Exception:
                continue
    except Exception:
        pass

# ----------------------------------------------------------------------
# 4. API Keys und Basis-URLs (für direkten Zugriff falls nötig)
# ----------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL")

# Backward compatibility and auto-detection for llm_client
API_KEY: str = GROQ_API_KEY or OPENAI_API_KEY or GEMINI_API_KEY
BASE_URL: str | None = LLM_BASE_URL

# Ensure API_KEY is set in environment for llm_client auto-detection
if not os.getenv("API_KEY") and API_KEY:
    os.environ["API_KEY"] = API_KEY

# ----------------------------------------------------------------------
# 5. Debug-Ausgabe (optional aktivierbar)
# ----------------------------------------------------------------------
if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
    logger.info(f"[Config] Running in Colab: {IN_COLAB}")
    logger.info(f"[Config] BASE_DIR: {BASE_DIR}")
    logger.info(f"[Config] RAG_STORAGE_DIR: {RAG_STORAGE_DIR}")
    logger.info(f"[Config] OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"[Config] OPENAI_API_KEY vorhanden: {bool(OPENAI_API_KEY)}")
    logger.info(f"[Config] GROQ_API_KEY vorhanden: {bool(GROQ_API_KEY)}")
    logger.info(f"[Config] GEMINI_API_KEY vorhanden: {bool(GEMINI_API_KEY)}")

    mocogi_token = os.getenv("MOCOGI_API_TOKEN")
    if mocogi_token:
        masked_token = mocogi_token[:4] + "..." + mocogi_token[-4:] if len(mocogi_token) > 8 else "***"
        logger.info(f"[Config] MOCOGI_API_TOKEN vorhanden: {masked_token}")
    else:
        logger.info("[Config] MOCOGI_API_TOKEN NICHT vorhanden")
