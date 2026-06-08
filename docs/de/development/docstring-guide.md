# Docstring Guide

In diesem Projekt verwenden wir den **Google Python Style Guide** für Docstrings. Dies stellt sicher, dass unsere API-Dokumentation konsistent, lesbar und durch `mkdocstrings` korrekt verarbeitbar ist.

## Standard-Format

Jeder Docstring sollte eine kurze Zusammenfassung enthalten, gefolgt von einer detaillierteren Beschreibung (falls erforderlich), den Argumenten, den Rückgabewerten und eventuell ausgelösten Ausnahmen.

```python
def analyze_module(text: str, model: str = "gpt-4") -> ModuleAnalysis:
    """Analysiert eine externe Modulbeschreibung.

    Diese Funktion nutzt ein LLM, um strukturierte Informationen aus einem
    unstrukturierten Text zu extrahieren.

    Args:
        text (str): Der Rohtext der Modulbeschreibung.
        model (str): Das zu verwendende LLM-Modell. Defaults to "gpt-4".

    Returns:
        ModuleAnalysis: Ein Pydantic-Modell mit den extrahierten Daten.

    Raises:
        ValueError: Wenn der Text leer ist oder das Modell nicht unterstützt wird.

    Example:
        >>> analysis = analyze_module("Mathematik 1, 5 ECTS...")
        >>> print(analysis.ects)
        5.0
    """
```

## Klassen

Klassen sollten einen Docstring direkt unter der Klassendefinition haben, der den Zweck der Klasse beschreibt.

```python
class RecognitionService:
    """Service-Layer für die Modul-Anerkennungslogik.

    Diese Klasse orchestriert den Workflow zwischen der Benutzeroberfläche,
    dem MCP-Client und dem LLM-Interface.
    """
```

## Best Practices

1. **Klarheit**: Verwenden Sie präzise Sprache.
2. **Typen**: Obwohl wir Python Type Hints verwenden, sollten die Typen im Docstring (optional) zur besseren Lesbarkeit wiederholt werden.
3. **Aktualität**: Aktualisieren Sie Docstrings, wenn Sie die Signatur oder das Verhalten einer Funktion ändern.
