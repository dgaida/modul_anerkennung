# 🎓 Modul-Anerkennungstool (PAV Assistant)

Ein Python-Tool, das Prüfungsausschussvorsitzende (PAV) bei der **Anerkennung externer Studienleistungen** unterstützt.  
Das Tool verwendet **RAG (Retrieval-Augmented Generation)** und ein **LLM**, um Ähnlichkeiten zwischen externen Modulbeschreibungen und internen Modulhandbüchern zu erkennen.

---

## 🚀 Funktionen

- Upload des eigenen Modulhandbuchs (PDF/Text)
- Upload externer Modulbeschreibungen
- Automatischer Vergleich mit RAG + Embeddings
- Anzeige ähnlicher Module in einer Gradio-GUI
- Begründungstexte für Anerkennung oder Ablehnung durch das LLM
- Nutzung eines `.env`-basierten Secrets-Systems für API-Keys

---

## 🧰 Installation

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

Erstelle eine Datei `secrets.env` im Projektverzeichnis:

```bash
LLM_API_KEY=dein_api_key
LLM_BASE_URL=https://api.openai.com/v1
```

---

## ▶️ Nutzung

Nach der Installation kannst du das Tool starten:

```bash
python -m modul_anerkennung.main
```

Oder direkt in einem Python-Skript:

```python
from modul_anerkennung.gui import launch_gui

launch_gui()
```

---

## 💻 Beispiel in Google Colab

Siehe das mitgelieferte Notebook `colab_demo.ipynb`, das zeigt, wie man das Paket installiert und einsetzt.

---

## 🧠 Lizenz

MIT License © 2025 Daniel Gaida

