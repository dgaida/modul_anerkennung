# Architektur

Das Modul-Anerkennungstool ist als modulare Python-Anwendung konzipiert, die externe APIs (LLMs und Mocogi) integriert.

## Systemübersicht

```mermaid
graph TD
    User((Benutzer)) --> GUI[Gradio UI]
    GUI --> Service[RecognitionService]
    Service --> MCP[MocogiClient]
    Service --> LLM[LLM Interface]
    MCP --> MocogiAPI[Mocogi API / TH Köln]
    LLM --> OpenAI[OpenAI / Gemini / Groq]
    Service --> RAG[RAG Manager]
    RAG --> VectorDB[(Vector DB)]
```

## Datenfluss: Anerkennungsprozess

1. **Eingabe**: Der Benutzer gibt eine externe Modulbeschreibung ein.
1. **Extraktion**: Das LLM analysiert den Text und extrahiert Metadaten (Name, ECTS, Keywords).
1. **Suche**: Der `MocogiClient` nutzt das Model Context Protocol (MCP), um in der TH Köln Datenbank nach ähnlichen Modulen zu suchen.
1. **Vergleich**: Für die Top-Treffer generiert das LLM einen detaillierten Vergleichsbericht.
1. **Entscheidung**: Der Benutzer wählt Module für den Anerkennungsantrag aus.

## Kernkomponenten

* **RecognitionService**: Orchestriert die Logik zwischen GUI, MCP und LLM.
* **MocogiClient**: Kommuniziert über MCP mit dem Mocogi-Server.
* **LLMInterface**: Abstraktionsschicht für verschiedene Sprachmodelle über `llm_client`.
* **Models**: Pydantic-Modelle zur Validierung und Typisierung der Daten.
