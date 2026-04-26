# Erste Schritte

Diese Anleitung führt Sie durch die ersten Schritte mit dem Modul-Anerkennungstool.

## Voraussetzungen

*   Python 3.10 oder höher
*   Ein API-Key für einen unterstützten LLM-Provider (OpenAI, Groq oder Gemini)
*   (Optional) Ein Mocogi API-Token der TH Köln

## Installation

Installieren Sie das Tool direkt von GitHub:

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

## Konfiguration

Erstellen Sie eine `secrets.env` Datei im Projektverzeichnis:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=ihr_api_key
MOCOGI_API_TOKEN=ihr_token
```

## Starten der Anwendung

Führen Sie die Hauptdatei aus, um die Gradio-GUI zu starten:

```bash
python main.py
```

Die Anwendung ist dann standardmäßig unter `http://127.0.0.1:7860` erreichbar.

## Beispiel-Workflow

1.  **Externe Beschreibung kopieren**: Fügen Sie den Text eines externen Moduls in das Analyse-Feld ein.
2.  **Analyse starten**: Klicken Sie auf "Modul analysieren". Das LLM extrahiert ECTS und Suchbegriffe.
3.  **Suche & Vergleich**: Das Tool sucht in der Mocogi-Datenbank nach Treffern und vergleicht diese automatisch.
4.  **Ergebnis prüfen**: Sehen Sie sich die generierte Begründung an und entscheiden Sie über die Anerkennung.
