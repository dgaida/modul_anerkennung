# Docstring Compliance Report

Dieser Bericht listet Symbole auf, die derzeit nicht den Docstring-Standards entsprechen.

## Zusammenfassung
- **Aktuelle Abdeckung**: 72.7%
- **Ziel**: 95.0%
- **Fehlende Docstrings**: 18

## Fehlende Symbole & Empfohlene Fixes

### `modul_anerkennung/gui.py`
Das gesamte Modul und seine Funktionen fehlen.

**Symbol**: `launch_gui`
```python
def launch_gui():
    """Startet die Gradio-Weboberfläche für das Modul-Anerkennungstool.

    Konfiguriert die UI-Komponenten, definiert Event-Handler und startet
    den lokalen Webserver.
    """
```

### `modul_anerkennung/services.py`

**Symbol**: `RecognitionService.__init__`
```python
def __init__(self, llm_interface, mcp_client):
    """Initialisiert den RecognitionService.

    Args:
        llm_interface (LLMInterface): Die Schnittstelle zum LLM.
        mcp_client (MocogiClient): Der Client für den Mocogi MCP Server.
    """
```

### `modul_anerkennung/mocogi_mcp.py`

**Symbol**: `get_model`
```python
def get_model():
    """Lädt das Embedding-Modul verzögert (Lazy Loading).

    Returns:
        SentenceTransformer: Das geladene Modell für Text-Embeddings.
    """
```

### `modul_anerkennung/models.py`

**Symbol**: Modul Docstring
```python
"""Pydantic-Modelle für die Datenvalidierung und den Austausch zwischen Komponenten."""
```
