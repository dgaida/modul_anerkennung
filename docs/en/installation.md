# Installation

The module recognition tool can be installed in several ways.

## Installation via Pip (Recommended)

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/dgaida/modul_anerkennung.git
```

## Installation for Developers

If you want to contribute to the tool, clone the repository and install it in editable mode:

```bash
git clone https://github.com/dgaida/modul_anerkennung.git
cd modul_anerkennung
pip install -e .[test]
```

## Dependencies

The most important dependencies are:

*   [`llm_client`](https://github.com/dgaida/llm_client): Interface to LLM providers.  
*   [`fastmcp`](https://github.com/jlowin/fastmcp): Framework for the Model Context Protocol.  
*   [`gradio`](https://gradio.app): Framework for the user interface.  
*   [`raganything`](https://github.com/HKUDS/RAG-Anything): (Optional) For RAG functionalities.  

## Docker (Planned)

A Docker installation is planned for future versions.
