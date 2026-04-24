"""Vergleich von Modulbeschreibungen mittels RAG und LLM."""
from typing import Dict, Any
from .rag_manager import RAGManager
from .llm_interface import LLMInterface


class SimilarityChecker:
    """Führt semantische Ähnlichkeitsprüfungen zwischen Modulen durch."""

    def __init__(self, rag: RAGManager, llm: LLMInterface):
        """Initialisiert den Ähnlichkeitsprüfer."""
        self.rag = rag
        self.llm = llm

    async def compare_modules(self, external_text: str) -> Dict[str, Any]:
        """
        Vergleicht ein externes Modul mit den internen Modulen.
        Args:
            external_text (str): Beschreibung des externen Moduls.
        Returns:
            Dict[str, Any]: Ergebnisse mit den besten Übereinstimmungen und Erklärungen.
        """
        # RAG query returns a string in newer LightRAG/RAGAnything versions
        # but let's see how we can handle it.
        result = await self.rag.query(external_text)

        # If the result is a string, it's already the answer from RAG.
        # If it's a list, we can process it.

        if not result:
            return {"message": "Keine relevanten Module gefunden."}

        # If result is a string (common in LightRAG/RAGAnything aquery),
        # it already contains the explanation/answer.
        if isinstance(result, str):
            return {"matches": [], "explanation": result}

        # Fallback if it returns raw matches (depends on configuration/mode)
        top_matches = result[:3] if isinstance(result, list) else [result]
        explanations = []

        for match in top_matches:
            messages = [
                {"role": "system", "content": "Du bist ein Fachprüfer für Modulbeschreibungen."},
                {"role": "user", "content": f"Erkläre die Unterschiede zwischen:\nExtern:\n{external_text}\n\nIntern:\n{match}"}
            ]
            explanation = self.llm.chat(messages)
            explanations.append(explanation)

        return {"matches": top_matches, "explanations": explanations}
