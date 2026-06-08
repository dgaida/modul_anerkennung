# Entwicklung

Anleitungen für Entwickler, die zum Projekt beitragen möchten.

## Lokale Entwicklungsumgebung

1. Repository klonen
2. Virtuelle Umgebung erstellen: `python -m venv venv`
3. Abhängigkeiten installieren: `pip install -e .[test]`
4. Pre-commit Hooks installieren (geplant)

## Testen

Wir verwenden `pytest` für automatisierte Tests.

```bash
PYTHONPATH=. pytest tests
```

## Code-Stil

* Wir folgen **PEP 8**.
* Docstrings müssen dem **Google-Style** entsprechen (siehe [Docstring Guide](docstring-guide.md)).
* Verwenden Sie **Conventional Commits** für Commit-Nachrichten.

## Dokumentation bauen

Lokale Vorschau der Dokumentation:

```bash
mkdocs serve
```
