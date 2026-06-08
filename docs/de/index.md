# 🎓 Modul-Anerkennungstool (PAV Assistant)

![Version](https://img.shields.io/github/v/tag/dgaida/modul_anerkennung?label=version)
![Interrogate](https://img.shields.io/badge/docstrings-95%25-brightgreen.svg)
[![Tests](https://github.com/dgaida/modul_anerkennung/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/modul_anerkennung/actions/workflows/tests.yml)

Ein Python-Tool, das Prüfungsausschussvorsitzende (PAV) bei der **Anerkennung externer Studienleistungen** unterstützt. Das Tool verwendet Large Language Models (LLMs), um Ähnlichkeiten zwischen externen Modulbeschreibungen und internen Studiengängen der TH Köln über das **Model Context Protocol (MCP)** zu erkennen.

## Hauptfunktionen

* **🔌 MCP-basierte Suche**: Direkter Zugriff auf die Mocogi-API der TH Köln für aktuelle Moduldaten.  
* **🧠 Automatisierte Analyse**: Extraktion von Metadaten (ECTS, Keywords) aus Modulbeschreibungen.  
* **⚖️ Intelligenter Vergleich**: Detaillierte Begründungsberichte für Anerkennungsentscheidungen.  
* **📝 Antrags-Generator**: Erstellung von Listen für den Prüfungsausschuss.  
* **🌐 Bilingual**: Unterstützung für Dokumentation in Deutsch und Englisch.  

## Schnellstart

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
python main.py
```

Weitere Informationen finden Sie unter [Erste Schritte](getting-started.md).
