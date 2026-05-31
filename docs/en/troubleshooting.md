# Troubleshooting

Häufig auftretende Probleme und deren Lösungen.

## 1. "Port 7860 already in use"

**Problem**: Die Gradio-GUI kann nicht starten, weil der Standard-Port belegt ist.
**Lösung**: Beenden Sie laufende Prozesse oder ändern Sie den Port in der `main.py`.
```bash
kill $(lsof -t -i :7860)
```

## 2. Authentication Errors with Mocogi

**Problem**: Search returns no results or error messages (401 Unauthorized / 403 Forbidden).
**Lösung**:
- Check if `MOCOGI_API_TOKEN` is correctly set in your `.env` and that the `Bearer` prefix is not duplicated.
- The token might have expired. Log in to Mocogi in your browser, copy the current token from the cookies (`access_token`), and update your `.env` or `secrets.env` file.

## 3. LLM API Limit erreicht

**Problem**: Fehler `Rate limit reached` von OpenAI oder Groq.
**Lösung**: Wechseln Sie den Provider in der Konfiguration oder warten Sie einige Minuten.

## 4. MCP Server startet nicht

**Problem**: Der Client kann keine Verbindung zum MCP-Server herstellen.
**Lösung**: Stellen Sie sicher, dass alle Abhängigkeiten (`fastmcp`) installiert sind und `python -m modul_anerkennung.mocogi_mcp` manuell ausführbar ist.
