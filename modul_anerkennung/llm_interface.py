"""Schnittstelle zum LLM über llm_client."""

import json
import re
from typing import List, Dict, Any, Type, TypeVar
from llm_client import LLMClient
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
            **kwargs: Zusätzliche Argumente for die Completion.
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
        if not self.client.use_async:
            self.client.use_async = True
        return await self.client.achat_completion(messages, **kwargs)

    def extract_json(self, text: str, model_class: Type[T]) -> T:
        """
        Extrahiert JSON aus einem Text und validiert es gegen eine Pydantic-Klasse.

        Args:
            text (str): Der Text, der JSON enthält.
            model_class (Type[T]): Die Pydantic-Klasse zur Validierung.

        Returns:
            T: Die validierte Instanz der model_class.
        """
        # Suche nach dem ersten { und dem letzten }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Kein JSON im LLM-Output gefunden: {text}")

        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Fehler beim Parsen oder Validieren des JSON: {e}\nRaw JSON: {json_str}"
            )
