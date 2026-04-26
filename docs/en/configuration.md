# Konfiguration

Die Konfiguration erfolgt primär über Umgebungsvariablen oder eine `.env` / `secrets.env` Datei.

## Umgebungsvariablen

| Variable | Beschreibung | Standardwert |
|----------|--------------|--------------|
| `LLM_PROVIDER` | Der zu verwendende Provider (`openai`, `groq`, `gemini`) | `openai` |
| `OPENAI_API_KEY` | API-Key für OpenAI | - |
| `GROQ_API_KEY` | API-Key für Groq | - |
| `GEMINI_API_KEY` | API-Key für Google Gemini | - |
| `MOCOGI_API_TOKEN` | Bearer Token für die Mocogi API (TH Köln) | - |
| `LOG_LEVEL` | Logging-Stufe (`DEBUG`, `INFO`, `WARNING`) | `INFO` |

## Beispiel .env Datei

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-xxxx...
MOCOGI_API_TOKEN=your_bearer_token
```

## Google Colab

In Google Colab werden die Geheimnisse über `google.colab.userdata` bezogen. Stellen Sie sicher, dass die entsprechenden Schlüssel in Ihrem Google-Konto hinterlegt sind.
