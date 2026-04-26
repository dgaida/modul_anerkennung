# 🎓 Modul-Anerkennungstool (PAV Assistant)

Ein Python-Tool, das Prüfungsausschussvorsitzende (PAV) bei der **Anerkennung externer Studienleistungen** unterstützt.  
Das Tool verwendet ein **LLM**, um Ähnlichkeiten zwischen externen Modulbeschreibungen und internen Studiengängen der TH Köln über das **Model Context Protocol (MCP)** zu erkennen.

[![Version](https://img.shields.io/github/v/tag/dgaida/modul_anerkennung?label=version)](https://github.com/dgaida/modul_anerkennung/tags)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/modul_anerkennung/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/modul_anerkennung/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/modul_anerkennung/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/modul_anerkennung/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/modul_anerkennung/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/modul_anerkennung/actions/workflows/codeql.yml)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/modul_anerkennung/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/modul_anerkennung)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/modul_anerkennung/)


---

## 🚀 Funktionen  
- **MCP-basierte Suche**: Greift direkt auf die Mocogi-API der TH Köln zu, um aktuelle Moduldaten abzufragen.  
- **Automatisierte Analyse**: Extrahiert Modulname, ECTS und Suchbegriffe aus beliebigen Modulbeschreibungen.  
- **Intelligenter Vergleich**: Vergleicht externe Module mit internen Treffern und erstellt detaillierte Begründungsberichte (Ja/Nein/Vielleicht).  
- **Antrags-Generator**: Sammelt ausgewählte Anerkennungsvorschläge und generiert eine fertige Liste inkl. Begründungen für den Prüfungsausschuss.  
- **Render.com Ready**: Vorkonfiguriert für das Deployment auf Render.  

> **Hinweis zu RAG**: Die bisherige RAG-Funktionalität (Retrieval-Augmented Generation) ist weiterhin im Code vorhanden, wird aber in der aktuellen GUI zugunsten der direkten MCP-Abfrage nicht mehr primär genutzt.

---

## 🧰 Installation & Lokaler Start

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

Erstelle eine Datei `secrets.env` im Projektverzeichnis:

```bash
# Provider: openai, groq, oder gemini
LLM_PROVIDER=openai
OPENAI_API_KEY=dein_api_key
# Optional für TH Köln API
MOCOGI_API_TOKEN=dein_token
```

Starte die GUI:

```bash
python main.py
```

---

## ☁️ Deployment auf Render.com

1. Erstelle einen Web Service auf Render.  
2. Verbinde dieses Repository.  
3. Render nutzt automatisch die `render.yaml` und `requirements.txt`.  
4. Füge die Umgebungsvariablen (API Keys) im Render-Dashboard hinzu.  

---

## 🔌 Model Context Protocol (MCP)

Das Projekt nutzt einen **MCP Server**, um Daten der TH Köln (Mocogi API) direkt zu integrieren.

- **MCP Server**: `modul_anerkennung/mocogi_mcp.py`  
- **MCP Client**: `modul_anerkennung/mcp_client.py`  

---

## 🧠 Lizenz

MIT License © 2025 Daniel Gaida
