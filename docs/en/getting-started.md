# Getting Started

This guide will walk you through the first steps with the module recognition tool.

## Prerequisites

* Python 3.10 oder höher  
* An API key for a supported LLM provider (OpenAI, Groq oder Gemini)  
* (Optional) Ein Mocogi API-Token der TH Köln  

## Installation

Install the tool directly from GitHub:

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

## Configuration

Create a `secrets.env` file in the project directory:

```env
API_KEY=ihr_api_key
MOCOGI_API_TOKEN=ihr_token
```

## Starting the Application

Run the main file to start the Gradio GUI:

```bash
python main.py
```

The application is then reachable by default at `http://127.0.0.1:7860`.

## Example Workflow

1. **Copy external description**: Paste the text of an external module into the analysis field.  
2. **Start analysis**: Click on "Analyze Module". The LLM extracts ECTS and search terms.  
3. **Search & Comparison**: The tool searches the Mocogi database for matches and compares them automatically.  
4. **Check result**: Look at the generated reasoning and decide on the recognition.  
