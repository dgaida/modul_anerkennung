"""Globale Konfigurationen für das Modulanerkennungs-Tool (Colab-kompatibel)."""

import os
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1. Basisverzeichnis (funktioniert sowohl lokal als auch in Colab)
# ----------------------------------------------------------------------

try:
    # Wenn Datei innerhalb des Pakets liegt (normaler Betrieb)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
except NameError:
    # In Colab: __file__ ist nicht definiert, daher fallback auf /content
    BASE_DIR: Path = Path("/content/modul_anerkennung")

# In Colab sicherstellen, dass Verzeichnisse existieren
RAG_STORAGE_DIR: Path = BASE_DIR / "rag_storage"
OUTPUT_DIR: Path = BASE_DIR / "output"
RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Environment laden (lokal oder Colab)
# ----------------------------------------------------------------------

# Standardpfad zur secrets.env-Datei
dotenv_path = BASE_DIR / "secrets.env"

# .env nur laden, wenn sie existiert
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

# In Colab kann man zusätzlich Variablen über `userdata` laden (optional)
try:
    from google.colab import userdata  # type: ignore

    # Übernimm Keys aus Colab-Umgebung, falls vorhanden
    for key in ["GROQ_API_KEY", "LLM_BASE_URL"]:
        if userdata.get(key):
            os.environ[key] = userdata.get(key)
except Exception:
    # Kein Colab, ignoriere einfach
    pass

# ----------------------------------------------------------------------
# 3. API Keys und URLs
# ----------------------------------------------------------------------

API_KEY: str = os.getenv("GROQ_API_KEY", "")
BASE_URL: str | None = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

# ----------------------------------------------------------------------
# 4. Debug-Ausgabe (optional, im Notebook sichtbar)
# ----------------------------------------------------------------------

if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
    print(f"[Config] BASE_DIR: {BASE_DIR}")
    print(f"[Config] RAG_STORAGE_DIR: {RAG_STORAGE_DIR}")
    print(f"[Config] OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"[Config] API_KEY vorhanden: {bool(API_KEY)}")
