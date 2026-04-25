# 🎓 Modul-Anerkennungstool (PAV Assistant)

Ein Python-Tool, das Prüfungsausschussvorsitzende (PAV) bei der **Anerkennung externer Studienleistungen** unterstützt.  
Das Tool verwendet **RAG (Retrieval-Augmented Generation)** und ein **LLM**, um Ähnlichkeiten zwischen externen Modulbeschreibungen und internen Modulhandbüchern zu erkennen.

---

## 🚀 Funktionen
- Model Context Protocol (MCP) Integration (Server & Client)

- Upload des eigenen Modulhandbuchs (PDF/Text)
- Upload externer Modulbeschreibungen
- Automatischer Vergleich mit RAG + Embeddings
- Anzeige ähnlicher Module in einer Gradio-GUI
- Begründungstexte für Anerkennung oder Ablehnung durch das LLM
- Nutzung eines `.env`-basierten Secrets-Systems für API-Keys

---

## 🧰 Installation

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

Erstelle eine Datei `secrets.env` im Projektverzeichnis:

```bash
LLM_API_KEY=dein_api_key
LLM_BASE_URL=https://api.openai.com/v1
```

---

## ▶️ Nutzung

Nach der Installation kannst du das Tool starten:

```bash
python -m modul_anerkennung.main
```

Oder direkt in einem Python-Skript:

```python
from modul_anerkennung.gui import launch_gui

launch_gui()
```

---


## 💻 Beispiel in Google Colab

Du kannst die Demo-Notebooks direkt in Google Colab ausprobieren:

| Notebook | Link |
|----------|------|
| Haupt-Demo (RAG) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/modul_anerkennung/blob/master/notebooks/colab_demo.ipynb) |
| MCP Server Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/modul_anerkennung/blob/master/notebooks/mcp_demo.ipynb) |
| MCP Client Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/modul_anerkennung/blob/master/notebooks/mcp_client_demo.ipynb) |

## 🔌 Model Context Protocol (MCP)

Das Projekt beinhaltet einen **MCP Server** und einen **MCP Client**, um Daten der TH Köln (Mocogi API) direkt in LLM-Workflows zu integrieren.

- **MCP Server**: `modul_anerkennung/mocogi_mcp.py` (basiert auf FastMCP)
- **MCP Client**: `modul_anerkennung/mcp_client.py`

### Nutzung des MCP Clients
```python
from modul_anerkennung.mcp_client import MocogiClient

async with MocogiClient() as client:
    programs = await client.list_study_programs()
    print(programs)
```

---

## 🧠 Lizenz

MIT License © 2025 Daniel Gaida
