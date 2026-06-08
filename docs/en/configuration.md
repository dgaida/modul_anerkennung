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
| `DEBUG_CONFIG` | Enables debug output for the configuration | `false` |

## Obtaining the Mocogi API Token

A `MOCOGI_API_TOKEN` is required for write access to the Mocogi API (e.g., updating drafts) and for accessing your own module drafts. You can extract this from your browser after logging into Mocogi:

1. Open [Mocogi](https://module.gm.th-koeln.de/) in your browser and log in.  
2. Open the **Developer Tools** (F12 or Right-click -> Inspect).  
3. Navigate to the **Application** or **Storage** tab.  
4. In the left sidebar under **Cookies**, select the address `https://module.gm.th-koeln.de`.  
5. Look for the cookie named `access_token` in the list.  
6. Copy the value of this cookie and enter it as `MOCOGI_API_TOKEN` in your configuration.  

## Example .env File

```env
# Either API_KEY (provider auto-detection)
API_KEY=sk-proj-xxxx...

# OR provider-specific
# OPENAI_API_KEY=sk-proj-xxxx...
# LLM_PROVIDER=openai
MOCOGI_API_TOKEN=your_bearer_token
DEBUG_CONFIG=true
```

## Google Colab

In Google Colab, secrets are retrieved via `google.colab.userdata`. Make sure the corresponding keys are stored in your Google account.
