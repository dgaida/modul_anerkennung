# Konfiguration

Die Konfiguration erfolgt primär über Umgebungsvariablen oder eine `.env` / `secrets.env` Datei.

## Umgebungsvariablen

| Variable | Beschreibung | Standardwert |
|----------|--------------|--------------|
| `API_KEY` | Universeller API-Key (Provider wird automatisch erkannt) | - |
| `LLM_PROVIDER` | (Optional) Expliziter Provider (`openai`, `groq`, `gemini`) | - |
| `OPENAI_API_KEY` | API-Key für OpenAI | - |
| `GROQ_API_KEY` | API-Key für Groq | - |
| `GEMINI_API_KEY` | API-Key für Google Gemini | - |
| `MOCOGI_API_TOKEN` | Bearer Token für die Mocogi API (TH Köln) | - |
| `LOG_LEVEL` | Logging-Stufe (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `DEBUG_CONFIG` | Aktiviert Debug-Ausgaben für die Konfiguration | `false` |

## Bezug des Mocogi API-Tokens

Für den Schreibzugriff auf die Mocogi-API (z.B. Aktualisierung von Entwürfen) sowie den Zugriff auf Ihre eigenen Modul-Entwürfe wird ein `MOCOGI_API_TOKEN` benötigt. Diesen können Sie aus Ihrem Browser extrahieren, nachdem Sie sich bei Mocogi angemeldet haben:

1.  Öffnen Sie [Mocogi](https://module.gm.th-koeln.de/) im Browser und loggen Sie sich ein.  
2.  Öffnen Sie die **Entwicklertools** (F12 oder Rechtsklick -> Untersuchen).  
3.  Navigieren Sie zum Tab **Anwendung** (Application) oder **Speicher** (Storage).  
4.  Wählen Sie in der linken Seitenleiste unter **Cookies** die Adresse `https://module.gm.th-koeln.de` aus.  
5.  Suchen Sie in der Liste nach dem Cookie mit dem Namen `access_token`.  
6.  Kopieren Sie den Wert dieses Cookies und tragen Sie ihn als `MOCOGI_API_TOKEN` in Ihre Konfiguration ein.  

## Beispiel .env Datei

```env
# Entweder API_KEY (Provider-Auto-Erkennung)
API_KEY=sk-proj-xxxx...

# ODER provider-spezifisch
# OPENAI_API_KEY=sk-proj-xxxx...
# LLM_PROVIDER=openai
MOCOGI_API_TOKEN=your_bearer_token
DEBUG_CONFIG=true
```

## Google Colab

In Google Colab werden die Geheimnisse über `google.colab.userdata` bezogen. Stellen Sie sicher, dass die entsprechenden Schlüssel in Ihrem Google-Konto hinterlegt sind.
