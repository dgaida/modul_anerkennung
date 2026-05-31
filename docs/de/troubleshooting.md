# Troubleshooting

Häufig auftretende Probleme und deren Lösungen.

## 1. "Port 7860 already in use"

**Problem**: Die Gradio-GUI kann nicht starten, weil der Standard-Port belegt ist.
**Lösung**: Beenden Sie laufende Prozesse oder ändern Sie den Port in der `main.py`.
```bash
kill $(lsof -t -i :7860)
```

## 2. Authentifizierungsfehler bei Mocogi

**Problem**: Die Suche liefert keine Ergebnisse oder Fehlermeldungen (401 Unauthorized / 403 Forbidden).
**Lösung**:  
- Überprüfen Sie, ob `MOCOGI_API_TOKEN` in Ihrer `.env` korrekt gesetzt ist und das Präfix `Bearer` nicht doppelt vorhanden ist.  
- Der Token könnte abgelaufen sein. Loggen Sie sich im Browser bei Mocogi ein, kopieren Sie den aktuellen Token aus den Cookies (`access_token`) und aktualisieren Sie Ihre `.env` oder `secrets.env` Datei.  

## 3. LLM API Limit erreicht

**Problem**: Fehler `Rate limit reached` von OpenAI oder Groq.
**Lösung**: Wechseln Sie den Provider in der Konfiguration oder warten Sie einige Minuten.

## 4. MCP Server startet nicht

**Problem**: Der Client kann keine Verbindung zum MCP-Server herstellen.
**Lösung**: Stellen Sie sicher, dass alle Abhängigkeiten (`fastmcp`) installiert sind und `python -m modul_anerkennung.mocogi_mcp` manuell ausführbar ist.
