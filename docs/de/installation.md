# Installation

Das Modul-Anerkennungstool kann auf verschiedene Arten installiert werden.

## Installation via Pip (Empfohlen)

Sie können das Paket direkt von GitHub installieren:

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

## Installation für Entwickler

Wenn Sie am Tool mitarbeiten möchten, klonen Sie das Repository und installieren es im Editier-Modus:

```bash
git clone https://github.com/dgaida/modul_anerkennung.git
cd modul_anerkennung
pip install -e .[test]
```

## Abhängigkeiten

Die wichtigsten Abhängigkeiten sind:

*   [`llm_client`](https://github.com/dgaida/llm_client): Schnittstelle zu LLM-Providern.
*   [`fastmcp`](https://github.com/jlowin/fastmcp): Framework für das Model Context Protocol.
*   [`gradio`](https://gradio.app): Framework für die Benutzeroberfläche.
*   [`raganything`](https://github.com/HKUDS/RAG-Anything): (Optional) Für RAG-Funktionalitäten.

## Docker (Geplant)

Eine Docker-Installation ist für zukünftige Versionen geplant.
