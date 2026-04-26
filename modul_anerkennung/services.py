"""Service-Layer für die Modul-Anerkennungs-Logik."""

import json
import logging
from typing import List, Dict, Any, Tuple
from .llm_interface import LLMInterface
from .mcp_client import MocogiClient
from .models import ModuleAnalysis, ComparisonReport

logger = logging.getLogger(__name__)


class RecognitionService:
    """Service layer for module recognition logic."""

    def __init__(self, llm: LLMInterface = None):
        """Initialisiert den RecognitionService.

        Args:
            llm (LLMInterface, optional): Die Schnittstelle zum LLM. Falls None, wird
                eine neue Instanz von LLMInterface erstellt.
        """
        self.llm = llm or LLMInterface()

    async def analyze_module(self, text: str) -> ModuleAnalysis:
        """Analyzes an external module description to extract name, ECTS, and keywords."""
        if not text:
            raise ValueError("Keine Modulbeschreibung angegeben.")

        prompt = f"""Analysiere die folgende Modulbeschreibung und extrahiere:
1. Modulname
2. Anzahl ECTS (nur die Zahl)
3. 3-4 prägnante Suchbegriffe für eine semantische Suche.

Antworte ausschließlich im JSON-Format:
{{
  "name": "...",
  "ects": 5,
  "keywords": ["...", "...", "..."]
}}

Modulbeschreibung:
{text}"""

        response = await self.llm.achat([{"role": "user", "content": prompt}])
        return self.llm.extract_json(response, ModuleAnalysis)

    async def search_and_compare(
        self, po_id: str, keywords: str, max_ects: str, external_text: str
    ) -> List[Tuple[Dict[str, Any], ComparisonReport]]:
        """Searches for similar internal modules and compares them to the external module."""
        if not po_id:
            return []

        try:
            ects_val = float(max_ects) if max_ects else None
        except ValueError:
            ects_val = None

        async with MocogiClient() as client:
            modules = await client.call_tool(
                "search_modules",
                {"po_id": po_id, "search_term": keywords, "max_ects": ects_val},
            )

        # Process top 5 modules
        results = []
        for m in modules[:5]:
            comp = await self.perform_comparison(external_text, m)
            results.append((m, comp))

        return results

    async def perform_comparison(
        self, external_text: str, internal_module: Dict[str, Any]
    ) -> ComparisonReport:
        """Performs a detailed comparison between an external and an internal module."""
        internal_text = json.dumps(internal_module, indent=2)

        prompt = f"""Vergleiche die folgende externe Modulbeschreibung mit unserem internen Modul.

Externe Beschreibung:
{external_text}

Internes Modul:
{internal_text}

Erstelle einen detaillierten Vergleichsbericht.
Bestimme, ob das Modul anerkannt werden kann (Ja, Nein, Vielleicht).
Antworte im JSON-Format:
{{
  "decision": "Ja" | "Nein" | "Vielleicht",
  "reasoning": "Kurze Begründung",
  "report": "Ausführlicher Bericht"
}}
"""
        response = await self.llm.achat([{"role": "user", "content": prompt}])
        return self.llm.extract_json(response, ComparisonReport)
