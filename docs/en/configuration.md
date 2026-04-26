# Configuration

Configuration is primarily done via environment variables or a `.env` / `secrets.env` file.

## Environment Variables

| Variable | Description | Default |
|----------|--------------|--------------|
| `API_KEY` | Universal API key (provider is automatically detected) | - |
| `LLM_PROVIDER` | (Optional) Explicit provider (`openai`, `groq`, `gemini`) | - |
| `OPENAI_API_KEY` | API key for OpenAI | - |
| `GROQ_API_KEY` | API key for Groq | - |
| `GEMINI_API_KEY` | API key for Google Gemini | - |
| `MOCOGI_API_TOKEN` | Bearer Token for the Mocogi API (TH Köln) | - |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`) | `INFO` |

## Example .env File

```env
# Either API_KEY (provider auto-detection)
API_KEY=sk-proj-xxxx...

# OR provider-specific
# OPENAI_API_KEY=sk-proj-xxxx...
# LLM_PROVIDER=openai
MOCOGI_API_TOKEN=your_bearer_token
```

## Google Colab

In Google Colab, secrets are retrieved via `google.colab.userdata`. Make sure the corresponding keys are stored in your Google account.
