"""Schnittstelle zum LLM über llm_client."""

from llm_client import LLMClient
from typing import List, Dict, Any


class LLMInterface:
    """Wrapper für den universellen LLM-Client."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        """
        Initialisiert den LLM-Client.

        Args:
            provider (str, optional): Der zu verwendende Provider (z.B. "openai", "groq", "gemini").
                                    Falls None, wird versucht, den Provider automatisch zu erkennen.
            model (str, optional): Das zu verwendende Modell.
        """
        # LLMClient lädt automatisch Keys aus der Umgebung
        self.client = LLMClient(api_choice=provider, llm=model)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Führt eine Chat-Completion mit dem LLM aus.

        Args:
            messages (List[Dict[str, str]]): Nachrichtenverlauf für die LLM-Kommunikation.
            **kwargs: Zusätzliche Argumente für die Completion.
        Returns:
            str: Antwort des LLM.
        """
        return self.client.chat_completion(messages, **kwargs)

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Führt eine asynchrone Chat-Completion mit dem LLM aus.

        Args:
            messages (List[Dict[str, str]]): Nachrichtenverlauf für die LLM-Kommunikation.
            **kwargs: Zusätzliche Argumente für die Completion.
        Returns:
            str: Antwort des LLM.
        """
        # Stelle sicher, dass der Client asynchron arbeiten kann
        if not self.client.use_async:
            self.client.use_async = True
            # Wir müssen den Provider neu initialisieren, wenn wir auf async umstellen
            # In llm_client v0.3.0 wird dies intern gehandhabt, wenn achat_completion aufgerufen wird
        return await self.client.achat_completion(messages, **kwargs)
