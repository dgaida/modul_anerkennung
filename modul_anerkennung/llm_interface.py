"""Schnittstelle zum LLM über llm_client."""
from llm_client import LLMClient
from typing import List, Dict


class LLMInterface:
    """Wrapper für den universellen LLM-Client."""

    def __init__(self) -> None:
        """Initialisiert den LLM-Client."""
        self.client = LLMClient(api_choice="groq")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Führt eine Chat-Completion mit dem LLM aus.

        Args:
            messages (List[Dict[str, str]]): Nachrichtenverlauf für die LLM-Kommunikation.
        Returns:
            str: Antwort des LLM.
        """
        return self.client.chat_completion(messages)
