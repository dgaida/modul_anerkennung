# Agent Instructions

This repository is a module recognition tool that uses RAG and LLMs to help academic advisors evaluate external course modules for equivalence.

## Project Structure  
- `modul_anerkennung/`: Main package containing the logic for RAG, LLM interface, and GUI.  
- `notebooks/`: Demonstration notebooks for Google Colab.  
- `tests/`: Pytest test suite.  

## Development Guidelines  
- Follow PEP 8 coding standards.  
- Use `llm_client` for all LLM interactions.  
- Ensure Colab compatibility by checking for the environment and adjusting paths accordingly (see `modul_anerkennung/config.py`).  
- Use `secrets.env` or `.env` for local development secrets.  

## Skills

### github-repo-review
---
name: github-repo-review
description: >
  Perform a deep, holistic code review of a GitHub repository and propose specific, actionable
  improvements for maintainability, clarity, correctness, and long-term scalability.
  Use this skill whenever the user shares a GitHub repository link or codebase and asks for
  a review, audit, analysis, improvement suggestions, refactoring plan, or anything related to
  assessing code quality, structure, documentation, tests, CI/CD, or security.
  Also triggers when the user asks "what should I improve in my repo?", "review my project",
  "how can I make my code better?", or similar. Always use this skill before providing any
  code review or repository analysis — even for partial reviews of a single category.
---

### mkdocs-documentation
---
name: mkdocs-documentation
description: >
  Generate a complete, production-ready MkDocs documentation ecosystem for a Python GitHub
  repository. Use this skill whenever the user asks to set up, create, improve, or automate
  documentation for a Python project — including requests for MkDocs configuration, API docs,
  docstrings, GitHub Pages deployment, versioned docs, changelogs, or documentation CI/CD.
  Also triggers for: "document my project", "set up MkDocs", "auto-generate API docs",
  "publish docs to GitHub Pages", "add mkdocstrings", "write docstrings for my repo",
  "bilingual docs", or any request involving documentation quality metrics or coverage checks.
  Always use this skill before producing any MkDocs config, documentation structure,
  or docstring-related output — even for partial documentation tasks.
---

- Lösche keine Kommentare, debug Ausgaben (print/logging) oder Dokumentation grundlos, es sei denn der User fragt explizit danach oder die Dokumentation ist veraltet.

- **Hinweis zu API-Tools**: Das Tool `get_modules_by_po` liefert eventuell nicht zuverlässig alle Entwürfe (Drafts) für neue Prüfungsordnungen wie `inf_inf3`. In solchen Fällen sollte stattdessen `get_module_drafts` verwendet werden, um alle zugänglichen Entwürfe abzurufen. Bei Entwürfen ist darauf zu achten, dass die ID oft unter `id` (Draft-ID) oder `module.id` (Modul-ID) liegen kann; für Detailabfragen (`get_module_draft_details`) wird die Draft-ID benötigt.  
