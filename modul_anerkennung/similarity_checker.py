"""Vergleich von Modulbeschreibungen mittels RAG und LLM."""
from typing import List, Dict
from .rag_manager_light import LightRAGManager
from .llm_interface import LLMInterface


class SimilarityChecker:
    """Führt semantische Ähnlichkeitsprüfungen zwischen Modulen durch."""

    def __init__(self, rag: LightRAGManager, llm: LLMInterface):
        """Initialisiert den Ähnlichkeitsprüfer."""
        self.rag = rag
        self.llm = llm

    async def compare_modules(self, external_text: str) -> Dict[str, str]:
        """
        Vergleicht ein externes Modul mit den internen Modulen.
        Args:
            external_text (str): Beschreibung des externen Moduls.
        Returns:
            Dict[str, str]: Ergebnisse mit den besten Übereinstimmungen und Erklärungen.
        """
        results = await self.rag.query(external_text)

        if not results:
            return {"message": "Keine relevanten Module gefunden."}

        top_matches = results[:3]
        explanations = []

        for match in top_matches:
            messages = [
                {"role": "system", "content": "Du bist ein Fachprüfer für Modulbeschreibungen."},
                {"role": "user", "content": f"Erkläre die Unterschiede zwischen:\nExtern:\n{external_text}\n\nIntern:\n{match}"}
            ]
            explanation = self.llm.chat(messages)
            explanations.append(explanation)

        return {"matches": top_matches, "explanations": explanations}
